//! Observational loaded-module discovery and bounded image parsing.
//!
//! [`Module`] is the only type in this module that changes loader ownership.
//! A [`LoadedModule`] is copied metadata and does not keep an image loaded.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::memory::{
    with_direct_memory_scope, Access, Address, AddressRange, DirectMemoryAccess, DirectMemoryScope,
    FaultContainedMemoryAccess, ForeignRef, ForeignReferenceError, MemoryError,
};

pub use crate::memory::ModuleId;

/// An explicitly loaded image. This owns exactly one loader reference.
pub struct Module {
    library: libloading::Library,
    path: PathBuf,
}

impl Module {
    /// Loads an image and acquires one loader reference.
    pub fn load(path: impl AsRef<OsStr>) -> Result<Self, ModuleError> {
        let path = PathBuf::from(path.as_ref());
        let library =
            // SAFETY: loading a library can run arbitrary initializers. That is the
            // explicit purpose of this owning, side-effecting API.
            unsafe { libloading::Library::new(&path) }.map_err(|source| ModuleError::Load {
                path: path.clone(),
                source,
            })?;
        Ok(Self { library, path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Looks up a symbol tied to this owner's lifetime.
    ///
    /// # Safety
    ///
    /// The caller must supply the symbol's exact type and obey its ABI and
    /// invocation contract.
    pub unsafe fn symbol<T>(&self, name: &[u8]) -> Result<libloading::Symbol<'_, T>, ModuleError> {
        // SAFETY: the type/ABI contract is delegated to this method's caller.
        unsafe { self.library.get(name) }.map_err(|source| ModuleError::Symbol {
            name: String::from_utf8_lossy(name).into_owned(),
            source,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageFormat {
    Pe,
    MachO64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SectionKind {
    Code,
    ReadOnlyData,
    Data,
    UninitializedData,
    Other,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Section {
    pub name: String,
    pub range: AddressRange,
    pub access: Access,
    pub kind: SectionKind,
}

impl Section {
    pub const fn is_executable(&self) -> bool {
        self.access.contains(Access::EXECUTE)
    }
}

/// Copied metadata for an image observed in a point-in-time snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoadedModule {
    pub id: ModuleId,
    pub path: PathBuf,
    /// Envelope beginning at the image header and ending at its highest
    /// mapped segment. On macOS this may contain gaps between segments.
    pub image: AddressRange,
    /// Exact mapped extents attributed to this image at snapshot time.
    pub mapped_ranges: Vec<AddressRange>,
    pub format: ImageFormat,
    pub sections: Vec<Section>,
}

impl LoadedModule {
    pub const fn base(&self) -> Address {
        self.image.start()
    }

    pub fn basename(&self) -> Option<&OsStr> {
        self.path.file_name()
    }

    pub fn contains_address(&self, address: Address) -> bool {
        self.mapped_ranges
            .iter()
            .any(|range| range.contains(address))
    }

    pub fn contains_range(&self, range: AddressRange) -> bool {
        self.mapped_ranges
            .iter()
            .any(|mapped| mapped.contains_range(range))
    }

    pub fn executable_sections(&self) -> impl Iterator<Item = &Section> {
        self.sections
            .iter()
            .filter(|section| section.is_executable())
    }

    pub fn section(&self, name: &str) -> Option<&Section> {
        self.sections.iter().find(|section| section.name == name)
    }

    /// Borrows one of this snapshot's sections as a foreign byte slice.
    ///
    /// The supplied `section` must be the actual reference returned by this
    /// module's [`Self::section`] or [`Self::executable_sections`] methods. A
    /// separately constructed or cloned `Section`, even with identical
    /// fields, is rejected.
    ///
    /// # Safety
    ///
    /// The complete section must remain one live allocation containing
    /// initialized bytes for `'a`, and it must remain mapped, readable,
    /// immovable, and free of mutation or data races that would violate an
    /// ordinary shared slice reference. In particular, the image must not be
    /// unloaded or reprotected until the returned lease is dropped.
    pub unsafe fn lease_section_scoped<'a, M: DirectMemoryAccess + ?Sized>(
        &self,
        scope: &'a DirectMemoryScope<'_, M>,
        section: &Section,
    ) -> Result<ForeignRef<'a, [u8]>, ModuleError> {
        if !self
            .sections
            .iter()
            .any(|owned| std::ptr::eq(owned, section))
        {
            return Err(ModuleError::SectionNotOwned);
        }
        if !self.contains_range(section.range) {
            return Err(ModuleError::SectionOutsideImage {
                section: section.range,
                image: self.image,
            });
        }
        if !section.access.contains(Access::READ) {
            return Err(ModuleError::Memory(MemoryError::PermissionDenied {
                address: section.range.start(),
                required: Access::READ,
                actual: section.access,
            }));
        }
        let len = section.range.len();
        if len > isize::MAX as usize {
            return Err(ModuleError::ByteLimit {
                kind: "section lease",
                actual: len,
                limit: isize::MAX as usize,
            });
        }
        // SAFETY: delegated to this method's documented caller contract. The
        // centralized capability constructor performs the range checks and
        // raw slice conversion.
        Ok(unsafe { scope.foreign_slice(section.range.start(), len)? })
    }

    pub fn scan_first_scoped<M: DirectMemoryAccess + ?Sized>(
        &self,
        scope: &DirectMemoryScope<'_, M>,
        section: &Section,
        pattern: &str,
        offset: i64,
        limits: ModuleLimits,
    ) -> Result<Option<Address>, ModuleError> {
        self.require_section(section)?;
        scan_section(section, pattern, offset, limits, |address, bytes| {
            scope.read_bytes_into(address, bytes)
        })
    }

    pub fn scan_all_scoped<M: DirectMemoryAccess + ?Sized>(
        &self,
        scope: &DirectMemoryScope<'_, M>,
        section: &Section,
        pattern: &str,
        offset: i64,
        limits: ModuleLimits,
    ) -> Result<Vec<Address>, ModuleError> {
        self.require_section(section)?;
        scan_section_all(section, pattern, offset, limits, |address, bytes| {
            scope.read_bytes_into(address, bytes)
        })
    }

    /// Scans by direct copying without a scope.
    ///
    /// # Safety
    /// The section must remain live, readable, and synchronized for the copy.
    pub unsafe fn scan_first_direct<M: DirectMemoryAccess + ?Sized>(
        &self,
        memory: &M,
        section: &Section,
        pattern: &str,
        offset: i64,
        limits: ModuleLimits,
    ) -> Result<Option<Address>, ModuleError> {
        self.require_section(section)?;
        // SAFETY: the caller establishes the direct-access contract.
        unsafe {
            with_direct_memory_scope(memory, |scope| {
                self.scan_first_scoped(scope, section, pattern, offset, limits)
            })
        }
    }

    /// Scans all matches by direct copying without a scope.
    ///
    /// # Safety
    /// The section must remain live, readable, and synchronized for the copy.
    pub unsafe fn scan_all_direct<M: DirectMemoryAccess + ?Sized>(
        &self,
        memory: &M,
        section: &Section,
        pattern: &str,
        offset: i64,
        limits: ModuleLimits,
    ) -> Result<Vec<Address>, ModuleError> {
        self.require_section(section)?;
        // SAFETY: the caller establishes the direct-access contract.
        unsafe {
            with_direct_memory_scope(memory, |scope| {
                self.scan_all_scoped(scope, section, pattern, offset, limits)
            })
        }
    }

    pub fn scan_first_fault_contained<M: FaultContainedMemoryAccess + ?Sized>(
        &self,
        memory: &M,
        section: &Section,
        pattern: &str,
        offset: i64,
        limits: ModuleLimits,
    ) -> Result<Option<Address>, ModuleError> {
        self.require_section(section)?;
        scan_section(section, pattern, offset, limits, |address, bytes| {
            memory.try_read_bytes_into(address, bytes)
        })
    }

    pub fn scan_all_fault_contained<M: FaultContainedMemoryAccess + ?Sized>(
        &self,
        memory: &M,
        section: &Section,
        pattern: &str,
        offset: i64,
        limits: ModuleLimits,
    ) -> Result<Vec<Address>, ModuleError> {
        self.require_section(section)?;
        scan_section_all(section, pattern, offset, limits, |address, bytes| {
            memory.try_read_bytes_into(address, bytes)
        })
    }

    fn require_section(&self, section: &Section) -> Result<(), ModuleError> {
        if self.contains_range(section.range) && self.sections.contains(section) {
            Ok(())
        } else {
            Err(ModuleError::SectionOutsideImage {
                section: section.range,
                image: self.image,
            })
        }
    }
}

/// Owned input returned by an observational platform enumerator.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModuleCandidate {
    pub id: ModuleId,
    pub path: PathBuf,
    pub image: AddressRange,
    /// Runtime slide for Mach-O virtual addresses. PE candidates use zero.
    pub slide: i64,
}

/// Enumerates already-loaded images without acquiring loader references.
pub trait ModuleSnapshotSource {
    fn loaded_module_candidates(
        &self,
        max_modules: usize,
    ) -> Result<Vec<ModuleCandidate>, ModuleError>;
}

/// Observes the images currently loaded in this process.
///
/// Enumeration is observational: it does not load images, acquire loader
/// references, or keep returned images resident. Consequently, a snapshot is
/// copied metadata only and callers must revalidate memory before using it.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProcessModuleSource;

impl ModuleSnapshotSource for ProcessModuleSource {
    fn loaded_module_candidates(
        &self,
        max_modules: usize,
    ) -> Result<Vec<ModuleCandidate>, ModuleError> {
        platform_modules::loaded_module_candidates(max_modules)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModuleLimits {
    pub max_modules: usize,
    pub max_sections: usize,
    pub max_header_bytes: usize,
    pub max_load_command_bytes: usize,
    pub max_section_bytes: usize,
    pub max_pattern_bytes: usize,
    pub max_results: usize,
}

impl Default for ModuleLimits {
    fn default() -> Self {
        Self {
            max_modules: 1_024,
            max_sections: 4_096,
            max_header_bytes: 1 << 20,
            max_load_command_bytes: 1 << 20,
            max_section_bytes: 64 << 20,
            max_pattern_bytes: 1 << 20,
            max_results: 16_384,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModuleMap {
    modules: Vec<LoadedModule>,
}

impl ModuleMap {
    pub fn modules(&self) -> &[LoadedModule] {
        &self.modules
    }

    pub fn module_containing(&self, address: Address) -> Option<&LoadedModule> {
        self.modules
            .iter()
            .find(|module| module.contains_address(address))
    }

    pub fn find_by_path(&self, path: &Path) -> Option<&LoadedModule> {
        self.modules
            .iter()
            .find(|module| paths_match(&module.path, path))
    }

    /// Finds a unique basename, reporting ambiguity instead of guessing.
    pub fn find_by_basename(&self, name: &OsStr) -> Result<Option<&LoadedModule>, ModuleError> {
        let mut matches = self
            .modules
            .iter()
            .filter(|module| names_match(module.basename(), name));
        let first = matches.next();
        if first.is_some() && matches.next().is_some() {
            return Err(ModuleError::AmbiguousName {
                name: name.to_string_lossy().into_owned(),
            });
        }
        Ok(first)
    }

    pub fn snapshot_scoped<M, S>(
        scope: &DirectMemoryScope<'_, M>,
        source: &S,
        limits: ModuleLimits,
    ) -> Result<Self, ModuleError>
    where
        M: DirectMemoryAccess + ?Sized,
        S: ModuleSnapshotSource + ?Sized,
    {
        snapshot(source, limits, |address, bytes| {
            scope.read_bytes_into(address, bytes)
        })
    }

    /// Observes loaded images with direct copies.
    ///
    /// # Safety
    /// Every enumerated image must remain loaded and readable for this call.
    pub unsafe fn snapshot_direct<M, S>(
        memory: &M,
        source: &S,
        limits: ModuleLimits,
    ) -> Result<Self, ModuleError>
    where
        M: DirectMemoryAccess + ?Sized,
        S: ModuleSnapshotSource + ?Sized,
    {
        // SAFETY: delegated to this method's caller.
        unsafe {
            with_direct_memory_scope(memory, |scope| Self::snapshot_scoped(scope, source, limits))
        }
    }

    pub fn snapshot_fault_contained<M, S>(
        memory: &M,
        source: &S,
        limits: ModuleLimits,
    ) -> Result<Self, ModuleError>
    where
        M: FaultContainedMemoryAccess + ?Sized,
        S: ModuleSnapshotSource + ?Sized,
    {
        snapshot(source, limits, |address, bytes| {
            memory.try_read_bytes_into(address, bytes)
        })
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ModuleError {
    #[error("failed to load {path}", path = path.display())]
    Load {
        path: PathBuf,
        #[source]
        source: libloading::Error,
    },
    #[error("failed to resolve symbol {name}")]
    Symbol {
        name: String,
        #[source]
        source: libloading::Error,
    },
    #[error("memory access failed")]
    Memory(#[from] MemoryError),
    #[error("foreign section lease validation failed")]
    ForeignReference(#[from] ForeignReferenceError),
    #[error("unsupported or invalid image magic {magic:#x}")]
    InvalidMagic { magic: u32 },
    #[error("image architecture {actual:#x} does not match this target")]
    ArchitectureMismatch { actual: u32 },
    #[error("malformed {context}")]
    Malformed { context: &'static str },
    #[error("{kind} count {actual} exceeds limit {limit}")]
    CountLimit {
        kind: &'static str,
        actual: usize,
        limit: usize,
    },
    #[error("{kind} byte count {actual} exceeds limit {limit}")]
    ByteLimit {
        kind: &'static str,
        actual: usize,
        limit: usize,
    },
    #[error("section {section:?} is outside image {image:?}")]
    SectionOutsideImage {
        section: AddressRange,
        image: AddressRange,
    },
    #[error("section is not owned by this module snapshot")]
    SectionNotOwned,
    #[error("module name {name} is ambiguous")]
    AmbiguousName { name: String },
    #[error("module observation operation {operation} failed with code {code}")]
    Platform { operation: &'static str, code: i64 },
    #[error("loaded module observation is unsupported on this platform")]
    UnsupportedPlatform,
    #[error("loaded image {index} has no {field}")]
    MissingImageField { index: usize, field: &'static str },
    #[error("loaded image {index} path exceeds {limit} units")]
    PathLimit { index: usize, limit: usize },
    #[error("invalid pattern: {message}")]
    Pattern { message: String },
    #[error("match count exceeds limit {limit}")]
    ResultLimit { limit: usize },
    #[error("address arithmetic overflow")]
    AddressOverflow,
}

#[cfg(target_os = "windows")]
mod platform_modules {
    use std::ffi::{c_void, OsString};
    use std::mem::{size_of, zeroed};
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    use std::path::PathBuf;
    use std::ptr;

    use super::{Address, ModuleCandidate, ModuleError, ModuleId};

    const MAX_PATH_UNITS: usize = 32_768;
    const MAX_ENUMERATION_ATTEMPTS: usize = 3;

    #[repr(C)]
    struct ModuleInfo {
        base: *mut c_void,
        image_size: u32,
        entry_point: *mut c_void,
    }

    // SAFETY: these declarations match the documented Win32 signatures. Each
    // call site validates buffer extents and the provenance of process/module
    // handles before crossing the FFI boundary.
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32EnumProcessModules(
            process: isize,
            modules: *mut isize,
            bytes: u32,
            needed: *mut u32,
        ) -> i32;
        fn K32GetModuleFileNameExW(
            process: isize,
            module: isize,
            filename: *mut u16,
            size: u32,
        ) -> u32;
        fn K32GetModuleInformation(
            process: isize,
            module: isize,
            information: *mut ModuleInfo,
            size: u32,
        ) -> i32;
        fn CompareStringOrdinal(
            first: *const u16,
            first_len: i32,
            second: *const u16,
            second_len: i32,
            ignore_case: i32,
        ) -> i32;
    }

    pub(super) fn loaded_module_candidates(
        max_modules: usize,
    ) -> Result<Vec<ModuleCandidate>, ModuleError> {
        // SAFETY: GetCurrentProcess returns a non-owning pseudo-handle which
        // must not be closed.
        let process = unsafe { GetCurrentProcess() };
        let modules = enumerate_modules(process, max_modules)?;
        let mut candidates = Vec::with_capacity(modules.len());
        for (index, module) in modules.into_iter().enumerate() {
            candidates.push(candidate(process, module, index)?);
        }
        Ok(candidates)
    }

    fn enumerate_modules(process: isize, max_modules: usize) -> Result<Vec<isize>, ModuleError> {
        let mut needed = 0u32;
        // SAFETY: a null buffer and zero size request only the required byte
        // count; `needed` is a valid output pointer.
        if unsafe { K32EnumProcessModules(process, ptr::null_mut(), 0, &mut needed) } == 0 {
            return Err(platform_error("K32EnumProcessModules(size)"));
        }

        for _ in 0..MAX_ENUMERATION_ATTEMPTS {
            let count = byte_count_to_modules(needed)?;
            if count > max_modules {
                return Err(ModuleError::CountLimit {
                    kind: "module",
                    actual: count,
                    limit: max_modules,
                });
            }
            if count == 0 {
                return Ok(Vec::new());
            }
            let mut modules = vec![0isize; count];
            let supplied = u32::try_from(
                modules
                    .len()
                    .checked_mul(size_of::<isize>())
                    .ok_or(ModuleError::AddressOverflow)?,
            )
            .map_err(|_| ModuleError::AddressOverflow)?;
            let mut returned = 0u32;
            // SAFETY: `modules` has `supplied` writable bytes and `returned`
            // is a valid output pointer. The pseudo-handle remains valid.
            if unsafe {
                K32EnumProcessModules(process, modules.as_mut_ptr(), supplied, &mut returned)
            } == 0
            {
                return Err(platform_error("K32EnumProcessModules"));
            }
            if returned <= supplied {
                modules.truncate(byte_count_to_modules(returned)?);
                return Ok(modules);
            }
            needed = returned;
        }
        Err(ModuleError::Platform {
            operation: "K32EnumProcessModules(changed repeatedly)",
            code: 0,
        })
    }

    fn byte_count_to_modules(bytes: u32) -> Result<usize, ModuleError> {
        let bytes = usize::try_from(bytes).map_err(|_| ModuleError::AddressOverflow)?;
        let width = size_of::<isize>();
        bytes
            .checked_add(width - 1)
            .map(|rounded| rounded / width)
            .ok_or(ModuleError::AddressOverflow)
    }

    fn candidate(
        process: isize,
        module: isize,
        index: usize,
    ) -> Result<ModuleCandidate, ModuleError> {
        // SAFETY: all-zero is a valid initial representation for MODULEINFO.
        let mut information: ModuleInfo = unsafe { zeroed() };
        // SAFETY: `information` is a correctly sized writable MODULEINFO and
        // `module` came from enumeration of `process`.
        if unsafe {
            K32GetModuleInformation(
                process,
                module,
                &mut information,
                size_of::<ModuleInfo>() as u32,
            )
        } == 0
        {
            return Err(platform_error("K32GetModuleInformation"));
        }
        let base = Address::new(information.base as usize);
        if base.is_null() || information.image_size == 0 {
            return Err(ModuleError::MissingImageField {
                index,
                field: "nonempty image range",
            });
        }
        let image = base.checked_range(information.image_size as usize)?;
        let path = module_path(process, module, index)?;
        // Detect unload/reload or handle reuse while the path was copied.
        // SAFETY: all-zero is a valid initial representation for MODULEINFO.
        let mut final_information: ModuleInfo = unsafe { zeroed() };
        // SAFETY: `final_information` is a correctly sized writable MODULEINFO
        // and `module` came from enumeration of `process`.
        if unsafe {
            K32GetModuleInformation(
                process,
                module,
                &mut final_information,
                size_of::<ModuleInfo>() as u32,
            )
        } == 0
        {
            return Err(platform_error("K32GetModuleInformation(recheck)"));
        }
        if final_information.base != information.base
            || final_information.image_size != information.image_size
        {
            return Err(ModuleError::Platform {
                operation: "module changed during observation",
                code: i64::try_from(index).unwrap_or(i64::MAX),
            });
        }
        Ok(ModuleCandidate {
            id: ModuleId(base.get() as u64),
            path,
            image,
            slide: 0,
        })
    }

    fn module_path(process: isize, module: isize, index: usize) -> Result<PathBuf, ModuleError> {
        let mut capacity = 260usize;
        loop {
            let mut buffer = vec![0u16; capacity];
            // SAFETY: `buffer` has `capacity` writable UTF-16 code units and
            // `module` came from enumeration of `process`.
            let written = unsafe {
                K32GetModuleFileNameExW(process, module, buffer.as_mut_ptr(), capacity as u32)
            } as usize;
            if written == 0 {
                return Err(platform_error("K32GetModuleFileNameExW"));
            }
            if written < capacity {
                buffer.truncate(written);
                return Ok(PathBuf::from(OsString::from_wide(&buffer)));
            }
            if capacity == MAX_PATH_UNITS {
                return Err(ModuleError::PathLimit {
                    index,
                    limit: MAX_PATH_UNITS,
                });
            }
            capacity = capacity.saturating_mul(2).min(MAX_PATH_UNITS);
        }
    }

    fn platform_error(operation: &'static str) -> ModuleError {
        ModuleError::Platform {
            operation,
            code: std::io::Error::last_os_error()
                .raw_os_error()
                .map(i64::from)
                .unwrap_or(-1),
        }
    }

    pub(super) fn os_strings_match(first: &std::ffi::OsStr, second: &std::ffi::OsStr) -> bool {
        let first: Vec<u16> = first.encode_wide().collect();
        let second: Vec<u16> = second.encode_wide().collect();
        let Ok(first_len) = i32::try_from(first.len()) else {
            return false;
        };
        let Ok(second_len) = i32::try_from(second.len()) else {
            return false;
        };
        // SAFETY: both pointers remain valid for their explicit lengths.
        unsafe {
            CompareStringOrdinal(first.as_ptr(), first_len, second.as_ptr(), second_len, 1) == 2
        }
    }
}

#[cfg(target_os = "macos")]
mod platform_modules {
    use std::ffi::{c_char, c_void, OsString};
    use std::os::unix::ffi::OsStringExt;
    use std::path::PathBuf;

    use mach2::kern_return::KERN_SUCCESS;
    use mach2::traps::mach_task_self;
    use mach2::vm::mach_vm_read_overwrite;
    use mach2::vm_types::{mach_vm_address_t, mach_vm_size_t};

    use super::{Address, ModuleCandidate, ModuleError, ModuleId};

    const MACHO_HEADER_SIZE: usize = 32;
    const LC_SEGMENT_64: u32 = 0x19;
    const MAX_LOAD_COMMAND_BYTES: usize = 1 << 20;
    const MAX_LOAD_COMMANDS: usize = 4_096;
    const MAX_PATH_BYTES: usize = 32_768;

    // SAFETY: these declarations match dyld's public process-image query ABI.
    // Each returned pointer is checked and its data is copied through Mach
    // before the observation is accepted.
    unsafe extern "C" {
        fn _dyld_image_count() -> u32;
        fn _dyld_get_image_header(index: u32) -> *const c_void;
        fn _dyld_get_image_vmaddr_slide(index: u32) -> isize;
        fn _dyld_get_image_name(index: u32) -> *const c_char;
    }

    pub(super) fn loaded_module_candidates(
        max_modules: usize,
    ) -> Result<Vec<ModuleCandidate>, ModuleError> {
        // SAFETY: dyld exposes a process-global observational image table.
        let count = unsafe { _dyld_image_count() } as usize;
        if count > max_modules {
            return Err(ModuleError::CountLimit {
                kind: "module",
                actual: count,
                limit: max_modules,
            });
        }
        let mut candidates = Vec::with_capacity(count);
        for index in 0..count {
            candidates.push(candidate(index)?);
        }
        // Detect index-table growth/shrinkage during collection. Per-image
        // identity is checked separately in `candidate`.
        // SAFETY: dyld exposes a process-global observational image table; the
        // result is used only to detect a concurrent table change.
        if unsafe { _dyld_image_count() } as usize != count {
            return Err(ModuleError::Platform {
                operation: "dyld image table changed during enumeration",
                code: 0,
            });
        }
        Ok(candidates)
    }

    fn candidate(index: usize) -> Result<ModuleCandidate, ModuleError> {
        let dyld_index = u32::try_from(index).map_err(|_| ModuleError::AddressOverflow)?;
        // SAFETY: `dyld_index` is below the image count captured immediately
        // before enumeration. We copy all returned data into owned storage.
        let header = unsafe { _dyld_get_image_header(dyld_index) };
        if header.is_null() {
            return Err(ModuleError::MissingImageField {
                index,
                field: "Mach-O header",
            });
        }
        // SAFETY: same dyld image-table contract as above.
        let name = unsafe { _dyld_get_image_name(dyld_index) };
        if name.is_null() {
            return Err(ModuleError::MissingImageField {
                index,
                field: "path",
            });
        }
        let path = bounded_path(name, index)?;
        // SAFETY: same dyld image-table contract as above.
        let slide = unsafe { _dyld_get_image_vmaddr_slide(dyld_index) } as i64;
        let image = image_range(header.cast::<u8>(), slide)?;
        // A concurrent unload can shift dyld indices. Refuse to combine fields
        // from different images into one candidate.
        // SAFETY: `dyld_index` was in range at the start of this observation;
        // the result is compared only as an opaque identity value.
        let final_header = unsafe { _dyld_get_image_header(dyld_index) };
        // SAFETY: same bounded dyld image-table observation as above; the
        // pointer is compared only and is not dereferenced here.
        let final_name = unsafe { _dyld_get_image_name(dyld_index) };
        // SAFETY: same bounded dyld image-table observation as above; the
        // scalar result is compared with the previously observed slide.
        let final_slide = unsafe { _dyld_get_image_vmaddr_slide(dyld_index) } as i64;
        if final_header != header || final_name != name || final_slide != slide {
            return Err(ModuleError::Platform {
                operation: "dyld image changed during observation",
                code: i64::try_from(index).unwrap_or(i64::MAX),
            });
        }
        let base = Address::new(header as usize);
        Ok(ModuleCandidate {
            id: ModuleId(base.get() as u64),
            path,
            image,
            slide,
        })
    }

    fn bounded_path(name: *const c_char, index: usize) -> Result<PathBuf, ModuleError> {
        let mut bytes = Vec::new();
        while bytes.len() < MAX_PATH_BYTES {
            let address = (name as usize)
                .checked_add(bytes.len())
                .ok_or(ModuleError::AddressOverflow)?;
            let page_remaining = 4_096 - (address & 4_095);
            let amount = page_remaining.min(MAX_PATH_BYTES - bytes.len());
            let mut chunk = vec![0u8; amount];
            read_self(address, &mut chunk, "mach_vm_read_overwrite(image path)")?;
            if let Some(nul) = chunk.iter().position(|byte| *byte == 0) {
                bytes.extend_from_slice(&chunk[..nul]);
                return Ok(PathBuf::from(OsString::from_vec(bytes)));
            }
            bytes.extend_from_slice(&chunk);
        }
        Err(ModuleError::PathLimit {
            index,
            limit: MAX_PATH_BYTES,
        })
    }

    fn image_range(header: *const u8, slide: i64) -> Result<super::AddressRange, ModuleError> {
        let mut header_bytes = [0u8; MACHO_HEADER_SIZE];
        read_self(
            header as usize,
            &mut header_bytes,
            "mach_vm_read_overwrite(Mach-O header)",
        )?;
        if u32_at(&header_bytes, 0)? != 0xfeed_facf {
            return Err(ModuleError::Malformed {
                context: "dyld Mach-O magic",
            });
        }
        let command_count = u32_at(&header_bytes, 16)? as usize;
        if command_count > MAX_LOAD_COMMANDS {
            return Err(ModuleError::CountLimit {
                kind: "Mach-O load command",
                actual: command_count,
                limit: MAX_LOAD_COMMANDS,
            });
        }
        let command_bytes = u32_at(&header_bytes, 20)? as usize;
        if command_bytes > MAX_LOAD_COMMAND_BYTES {
            return Err(ModuleError::ByteLimit {
                kind: "Mach-O load commands",
                actual: command_bytes,
                limit: MAX_LOAD_COMMAND_BYTES,
            });
        }
        if command_bytes == 0 && command_count != 0 {
            return Err(ModuleError::Malformed {
                context: "dyld Mach-O load-command byte count",
            });
        }
        let mut commands = vec![0u8; command_bytes];
        if command_bytes != 0 {
            let source = (header as usize)
                .checked_add(MACHO_HEADER_SIZE)
                .ok_or(ModuleError::AddressOverflow)?;
            read_self(
                source,
                &mut commands,
                "mach_vm_read_overwrite(Mach-O load commands)",
            )?;
        }

        let image_start = Address::new(header as usize);
        let mut image_end = image_start;
        let mut found_header_segment = false;
        let mut cursor = 0usize;
        for _ in 0..command_count {
            let prefix_end = cursor.checked_add(8).ok_or(ModuleError::AddressOverflow)?;
            if prefix_end > commands.len() {
                return Err(ModuleError::Malformed {
                    context: "dyld Mach-O load-command prefix",
                });
            }
            let command = u32_at(&commands[cursor..], 0)?;
            let command_size = u32_at(&commands[cursor..], 4)? as usize;
            if command_size < 8 {
                return Err(ModuleError::Malformed {
                    context: "dyld Mach-O cmdsize",
                });
            }
            let end = cursor
                .checked_add(command_size)
                .ok_or(ModuleError::AddressOverflow)?;
            if end > commands.len() {
                return Err(ModuleError::Malformed {
                    context: "dyld Mach-O command extent",
                });
            }
            if command == LC_SEGMENT_64 {
                let record = &commands[cursor..end];
                if record.len() < 72 {
                    return Err(ModuleError::Malformed {
                        context: "dyld Mach-O segment_64",
                    });
                }
                let vmaddr = u64_at(record, 24)?;
                let vmsize = u64_at(record, 32)?;
                let initprot = u32_at(record, 60)?;
                // Exclude __PAGEZERO and other deliberately unmapped segments.
                if vmsize != 0 && initprot != 0 {
                    let start = slid_address(vmaddr, slide)?;
                    let len = usize::try_from(vmsize).map_err(|_| ModuleError::AddressOverflow)?;
                    let segment_end = start.checked_add(len)?;
                    if start < image_start {
                        return Err(ModuleError::Malformed {
                            context: "mapped Mach-O segment before header",
                        });
                    }
                    found_header_segment |= start == image_start;
                    image_end = image_end.max(segment_end);
                }
            }
            cursor = end;
        }
        if cursor != commands.len() || image_end <= image_start || !found_header_segment {
            return Err(ModuleError::Malformed {
                context: "dyld Mach-O image extent",
            });
        }
        image_start
            .checked_range(image_end.get() - image_start.get())
            .map_err(Into::into)
    }

    fn read_self(
        address: usize,
        output: &mut [u8],
        operation: &'static str,
    ) -> Result<(), ModuleError> {
        let source =
            mach_vm_address_t::try_from(address).map_err(|_| ModuleError::AddressOverflow)?;
        let destination = mach_vm_address_t::try_from(output.as_mut_ptr() as usize)
            .map_err(|_| ModuleError::AddressOverflow)?;
        let size =
            mach_vm_size_t::try_from(output.len()).map_err(|_| ModuleError::AddressOverflow)?;
        let mut transferred = 0;
        // SAFETY: Mach contains invalid-source faults and writes at most
        // `size` bytes to the live caller-owned output buffer.
        let result = unsafe {
            mach_vm_read_overwrite(
                mach_task_self(),
                source,
                size,
                destination,
                &mut transferred,
            )
        };
        if result != KERN_SUCCESS {
            return Err(ModuleError::Platform {
                operation,
                code: i64::from(result),
            });
        }
        if transferred != size {
            return Err(ModuleError::Platform {
                operation,
                code: i64::try_from(transferred).unwrap_or(i64::MAX),
            });
        }
        Ok(())
    }

    fn slid_address(vmaddr: u64, slide: i64) -> Result<Address, ModuleError> {
        let value = i128::from(vmaddr) + i128::from(slide);
        let value = usize::try_from(value).map_err(|_| ModuleError::AddressOverflow)?;
        Ok(Address::new(value))
    }

    fn u32_at(bytes: &[u8], offset: usize) -> Result<u32, ModuleError> {
        let value = bytes
            .get(offset..offset + 4)
            .ok_or(ModuleError::Malformed {
                context: "dyld Mach-O u32 field",
            })?;
        Ok(u32::from_le_bytes(value.try_into().map_err(|_| {
            ModuleError::Malformed {
                context: "dyld Mach-O u32 field",
            }
        })?))
    }

    fn u64_at(bytes: &[u8], offset: usize) -> Result<u64, ModuleError> {
        let value = bytes
            .get(offset..offset + 8)
            .ok_or(ModuleError::Malformed {
                context: "dyld Mach-O u64 field",
            })?;
        Ok(u64::from_le_bytes(value.try_into().map_err(|_| {
            ModuleError::Malformed {
                context: "dyld Mach-O u64 field",
            }
        })?))
    }

    pub(super) fn os_strings_match(first: &std::ffi::OsStr, second: &std::ffi::OsStr) -> bool {
        first == second
    }
}

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
mod platform_modules {
    use super::{ModuleCandidate, ModuleError};

    pub(super) fn loaded_module_candidates(
        _max_modules: usize,
    ) -> Result<Vec<ModuleCandidate>, ModuleError> {
        Err(ModuleError::UnsupportedPlatform)
    }

    pub(super) fn os_strings_match(first: &std::ffi::OsStr, second: &std::ffi::OsStr) -> bool {
        first == second
    }
}

fn snapshot<S, R>(source: &S, limits: ModuleLimits, mut read: R) -> Result<ModuleMap, ModuleError>
where
    S: ModuleSnapshotSource + ?Sized,
    R: FnMut(Address, &mut [u8]) -> Result<(), MemoryError>,
{
    let candidates = source.loaded_module_candidates(limits.max_modules)?;
    if candidates.len() > limits.max_modules {
        return Err(ModuleError::CountLimit {
            kind: "module",
            actual: candidates.len(),
            limit: limits.max_modules,
        });
    }
    let mut modules = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        modules.push(parse_candidate(&candidate, limits, &mut read)?);
    }
    Ok(ModuleMap { modules })
}

fn parse_candidate<R>(
    candidate: &ModuleCandidate,
    limits: ModuleLimits,
    read: &mut R,
) -> Result<LoadedModule, ModuleError>
where
    R: FnMut(Address, &mut [u8]) -> Result<(), MemoryError>,
{
    require_contained(candidate.image, candidate.image.start().checked_range(4)?)?;
    let magic = read_array::<4, _>(read, candidate.image.start())?;
    if &magic[..2] == b"MZ" {
        parse_pe(candidate, limits, read)
    } else {
        let value = u32::from_le_bytes(magic);
        if value == 0xfeed_facf {
            parse_macho64(candidate, limits, read)
        } else {
            Err(ModuleError::InvalidMagic { magic: value })
        }
    }
}

fn parse_pe<R>(
    candidate: &ModuleCandidate,
    limits: ModuleLimits,
    read: &mut R,
) -> Result<LoadedModule, ModuleError>
where
    R: FnMut(Address, &mut [u8]) -> Result<(), MemoryError>,
{
    require_contained(candidate.image, candidate.image.start().checked_range(64)?)?;
    let dos = read_array::<64, _>(read, candidate.image.start())?;
    let nt_offset =
        usize::try_from(u32_at(&dos, 0x3c)?).map_err(|_| ModuleError::AddressOverflow)?;
    if nt_offset > limits.max_header_bytes {
        return Err(ModuleError::ByteLimit {
            kind: "PE header offset",
            actual: nt_offset,
            limit: limits.max_header_bytes,
        });
    }
    let nt = candidate.image.start().checked_add(nt_offset)?;
    require_contained(candidate.image, nt.checked_range(24)?)?;
    let coff = read_array::<24, _>(read, nt)?;
    if &coff[..4] != b"PE\0\0" {
        return Err(ModuleError::Malformed {
            context: "PE signature",
        });
    }
    let machine = u16_at(&coff, 4)? as u32;
    require_architecture(machine, true)?;
    let section_count = u16_at(&coff, 6)? as usize;
    require_count("PE section", section_count, limits.max_sections)?;
    let optional_size = u16_at(&coff, 20)? as usize;
    if optional_size < 60 || optional_size > limits.max_header_bytes {
        return Err(ModuleError::ByteLimit {
            kind: "PE optional header",
            actual: optional_size,
            limit: limits.max_header_bytes,
        });
    }
    let optional_address = nt.checked_add(24)?;
    require_contained(
        candidate.image,
        optional_address.checked_range(optional_size)?,
    )?;
    let mut optional = vec![0; optional_size];
    read(optional_address, &mut optional)?;
    let expected_magic = if cfg!(target_pointer_width = "64") {
        0x20b
    } else {
        0x10b
    };
    if u16_at(&optional, 0)? != expected_magic {
        return Err(ModuleError::Malformed {
            context: "PE optional-header magic",
        });
    }
    let image_size = u32_at(&optional, 56)? as usize;
    let image = candidate.image.start().checked_range(image_size)?;
    if !candidate.image.contains_range(image) {
        return Err(ModuleError::Malformed {
            context: "PE image size",
        });
    }
    let table_size = section_count
        .checked_mul(40)
        .ok_or(ModuleError::AddressOverflow)?;
    if table_size > limits.max_header_bytes {
        return Err(ModuleError::ByteLimit {
            kind: "PE section table",
            actual: table_size,
            limit: limits.max_header_bytes,
        });
    }
    let table_address = optional_address.checked_add(optional_size)?;
    let mut table = vec![0; table_size];
    if !table.is_empty() {
        require_contained(candidate.image, table_address.checked_range(table_size)?)?;
        read(table_address, &mut table)?;
    }
    let mut sections = Vec::with_capacity(section_count);
    for record in table.chunks_exact(40) {
        let virtual_size = u32_at(record, 8)? as usize;
        let raw_size = u32_at(record, 16)? as usize;
        let len = virtual_size.max(raw_size);
        if len == 0 {
            continue;
        }
        let relative = u32_at(record, 12)? as usize;
        let start = image.start().checked_add(relative)?;
        let range = start.checked_range(len)?;
        require_contained(image, range)?;
        let characteristics = u32_at(record, 36)?;
        let mut access = Access::empty();
        if characteristics & 0x4000_0000 != 0 {
            access |= Access::READ;
        }
        if characteristics & 0x8000_0000 != 0 {
            access |= Access::WRITE;
        }
        if characteristics & 0x2000_0000 != 0 {
            access |= Access::EXECUTE;
        }
        let kind = if characteristics & 0x20 != 0 {
            SectionKind::Code
        } else if characteristics & 0x80 != 0 {
            SectionKind::UninitializedData
        } else if access.contains(Access::WRITE) {
            SectionKind::Data
        } else if access.contains(Access::READ) {
            SectionKind::ReadOnlyData
        } else {
            SectionKind::Other
        };
        sections.push(Section {
            name: fixed_name(&record[..8]),
            range,
            access,
            kind,
        });
    }
    Ok(LoadedModule {
        id: candidate.id,
        path: candidate.path.clone(),
        image,
        mapped_ranges: vec![image],
        format: ImageFormat::Pe,
        sections,
    })
}

fn parse_macho64<R>(
    candidate: &ModuleCandidate,
    limits: ModuleLimits,
    read: &mut R,
) -> Result<LoadedModule, ModuleError>
where
    R: FnMut(Address, &mut [u8]) -> Result<(), MemoryError>,
{
    require_contained(candidate.image, candidate.image.start().checked_range(32)?)?;
    let header = read_array::<32, _>(read, candidate.image.start())?;
    let cpu = u32_at(&header, 4)?;
    require_architecture(cpu, false)?;
    let command_count = u32_at(&header, 16)? as usize;
    require_count("Mach-O load command", command_count, limits.max_sections)?;
    let command_bytes = u32_at(&header, 20)? as usize;
    if command_bytes > limits.max_load_command_bytes {
        return Err(ModuleError::ByteLimit {
            kind: "Mach-O load commands",
            actual: command_bytes,
            limit: limits.max_load_command_bytes,
        });
    }
    let commands_address = candidate.image.start().checked_add(32)?;
    if command_bytes != 0 {
        let commands_range = commands_address.checked_range(command_bytes)?;
        require_contained(candidate.image, commands_range)?;
    } else if command_count != 0 {
        return Err(ModuleError::Malformed {
            context: "Mach-O load-command byte count",
        });
    }
    let mut commands = vec![0; command_bytes];
    if !commands.is_empty() {
        read(commands_address, &mut commands)?;
    }
    let mut cursor = 0usize;
    let mut sections = Vec::new();
    let mut mapped_ranges = Vec::new();
    let mut total_sections = 0usize;
    for _ in 0..command_count {
        let prefix_end = cursor.checked_add(8).ok_or(ModuleError::AddressOverflow)?;
        if prefix_end > commands.len() {
            return Err(ModuleError::Malformed {
                context: "Mach-O load-command prefix",
            });
        }
        let command = u32_at(&commands[cursor..], 0)?;
        let command_size = u32_at(&commands[cursor..], 4)? as usize;
        if command_size < 8 {
            return Err(ModuleError::Malformed {
                context: "Mach-O cmdsize",
            });
        }
        let end = cursor
            .checked_add(command_size)
            .ok_or(ModuleError::AddressOverflow)?;
        if end > commands.len() {
            return Err(ModuleError::Malformed {
                context: "Mach-O command extent",
            });
        }
        let record = &commands[cursor..end];
        if command == 0x19 {
            if command_size < 72 {
                return Err(ModuleError::Malformed {
                    context: "Mach-O segment_64",
                });
            }
            let vmaddr = u64_at(record, 24)?;
            let vmsize = u64_at(record, 32)?;
            let initprot = u32_at(record, 60)?;
            let section_count = u32_at(record, 64)? as usize;
            total_sections = total_sections
                .checked_add(section_count)
                .ok_or(ModuleError::AddressOverflow)?;
            require_count("Mach-O section", total_sections, limits.max_sections)?;
            let section_bytes = section_count
                .checked_mul(80)
                .ok_or(ModuleError::AddressOverflow)?;
            let required = 72usize
                .checked_add(section_bytes)
                .ok_or(ModuleError::AddressOverflow)?;
            if required > command_size {
                return Err(ModuleError::Malformed {
                    context: "Mach-O section records",
                });
            }
            let access = macho_access(initprot);
            let segment_start =
                Address::new(usize::try_from(vmaddr).map_err(|_| ModuleError::AddressOverflow)?)
                    .checked_add_signed(candidate.slide)?;
            let segment_len = usize::try_from(vmsize).map_err(|_| ModuleError::AddressOverflow)?;
            // __PAGEZERO and other permissionless segments are declarations,
            // not mapped extents owned by the loaded image.
            let segment = if segment_len != 0 && !access.is_empty() {
                let segment = segment_start.checked_range(segment_len)?;
                require_contained(candidate.image, segment)?;
                Some(segment)
            } else {
                None
            };
            for section_record in record[72..required].chunks_exact(80) {
                let address = usize::try_from(u64_at(section_record, 32)?)
                    .map_err(|_| ModuleError::AddressOverflow)?;
                let len = usize::try_from(u64_at(section_record, 40)?)
                    .map_err(|_| ModuleError::AddressOverflow)?;
                if len == 0 {
                    continue;
                }
                let start = Address::new(address).checked_add_signed(candidate.slide)?;
                let range = start.checked_range(len)?;
                require_contained(candidate.image, range)?;
                let segment = segment.ok_or(ModuleError::Malformed {
                    context: "Mach-O section in unmapped segment",
                })?;
                require_contained(segment, range)?;
                let flags = u32_at(section_record, 64)?;
                let name = fixed_name(&section_record[..16]);
                let kind = if flags & (0x8000_0000 | 0x0000_0400) != 0
                    || access.contains(Access::EXECUTE)
                {
                    SectionKind::Code
                } else if access.contains(Access::WRITE) {
                    SectionKind::Data
                } else if access.contains(Access::READ) {
                    SectionKind::ReadOnlyData
                } else {
                    SectionKind::Other
                };
                sections.push(Section {
                    name,
                    range,
                    access,
                    kind,
                });
            }
            if let Some(segment) = segment {
                mapped_ranges.push(segment);
            }
        }
        cursor = end;
    }
    if cursor != commands.len() {
        return Err(ModuleError::Malformed {
            context: "Mach-O load-command byte count",
        });
    }
    Ok(LoadedModule {
        id: candidate.id,
        path: candidate.path.clone(),
        image: candidate.image,
        mapped_ranges,
        format: ImageFormat::MachO64,
        sections,
    })
}

fn scan_section<R>(
    section: &Section,
    pattern: &str,
    offset: i64,
    limits: ModuleLimits,
    mut read: R,
) -> Result<Option<Address>, ModuleError>
where
    R: FnMut(Address, &mut [u8]) -> Result<(), MemoryError>,
{
    if pattern.split_ascii_whitespace().next().is_none() {
        return Err(ModuleError::Pattern {
            message: "pattern is empty".into(),
        });
    }
    let pattern = parse_pattern(pattern, limits.max_pattern_bytes)?;
    let bytes = copy_section(section, limits, &mut read)?;
    let result = matching_positions(&bytes, &pattern).next();
    result
        .map(|position| match_address(section, position, offset))
        .transpose()
}

fn scan_section_all<R>(
    section: &Section,
    pattern: &str,
    offset: i64,
    limits: ModuleLimits,
    mut read: R,
) -> Result<Vec<Address>, ModuleError>
where
    R: FnMut(Address, &mut [u8]) -> Result<(), MemoryError>,
{
    if pattern.split_ascii_whitespace().next().is_none() {
        return Err(ModuleError::Pattern {
            message: "pattern is empty".into(),
        });
    }
    let pattern = parse_pattern(pattern, limits.max_pattern_bytes)?;
    let bytes = copy_section(section, limits, &mut read)?;
    let mut results = Vec::new();
    for position in matching_positions(&bytes, &pattern) {
        if results.len() == limits.max_results {
            return Err(ModuleError::ResultLimit {
                limit: limits.max_results,
            });
        }
        results.push(match_address(section, position, offset)?);
    }
    Ok(results)
}

fn parse_pattern(pattern: &str, max_pattern_bytes: usize) -> Result<Vec<Option<u8>>, ModuleError> {
    if pattern.len() > max_pattern_bytes {
        return Err(ModuleError::ByteLimit {
            kind: "pattern",
            actual: pattern.len(),
            limit: max_pattern_bytes,
        });
    }
    let mut parsed = Vec::new();
    for token in pattern.split_ascii_whitespace() {
        if token == "?" {
            parsed.push(None);
        } else {
            let byte = u8::from_str_radix(token, 16).map_err(|_| ModuleError::Pattern {
                message: format!("{token:?} is not an 8-bit hexadecimal byte or ?"),
            })?;
            parsed.push(Some(byte));
        }
    }
    if parsed.is_empty() {
        Err(ModuleError::Pattern {
            message: "pattern is empty".into(),
        })
    } else {
        Ok(parsed)
    }
}

fn matching_positions<'a>(
    bytes: &'a [u8],
    pattern: &'a [Option<u8>],
) -> impl Iterator<Item = usize> + 'a {
    bytes
        .windows(pattern.len())
        .enumerate()
        .filter_map(|(position, window)| {
            pattern
                .iter()
                .zip(window)
                .all(|(expected, actual)| expected.is_none_or(|expected| expected == *actual))
                .then_some(position)
        })
}

fn copy_section<R>(
    section: &Section,
    limits: ModuleLimits,
    read: &mut R,
) -> Result<Vec<u8>, ModuleError>
where
    R: FnMut(Address, &mut [u8]) -> Result<(), MemoryError>,
{
    let len = section.range.len();
    if len > limits.max_section_bytes {
        return Err(ModuleError::ByteLimit {
            kind: "section",
            actual: len,
            limit: limits.max_section_bytes,
        });
    }
    let mut bytes = vec![0; len];
    read(section.range.start(), &mut bytes)?;
    Ok(bytes)
}

fn match_address(section: &Section, position: usize, offset: i64) -> Result<Address, ModuleError> {
    let raw_match = section.range.start().checked_add(position)?;
    let raw_match = u64::try_from(raw_match.get()).map_err(|_| ModuleError::AddressOverflow)?;
    let plan_start = scan_plan::apply_match_offset(raw_match, offset)
        .map_err(|_| ModuleError::AddressOverflow)?;
    usize::try_from(plan_start)
        .map(Address::new)
        .map_err(|_| ModuleError::AddressOverflow)
}

fn read_array<const N: usize, R>(read: &mut R, address: Address) -> Result<[u8; N], ModuleError>
where
    R: FnMut(Address, &mut [u8]) -> Result<(), MemoryError>,
{
    let mut value = [0; N];
    read(address, &mut value)?;
    Ok(value)
}

fn u16_at(bytes: &[u8], offset: usize) -> Result<u16, ModuleError> {
    let value = bytes
        .get(offset..offset + 2)
        .ok_or(ModuleError::Malformed {
            context: "u16 field",
        })?;
    Ok(u16::from_le_bytes([value[0], value[1]]))
}

fn u32_at(bytes: &[u8], offset: usize) -> Result<u32, ModuleError> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or(ModuleError::Malformed {
            context: "u32 field",
        })?;
    Ok(u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
}

fn u64_at(bytes: &[u8], offset: usize) -> Result<u64, ModuleError> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or(ModuleError::Malformed {
            context: "u64 field",
        })?;
    Ok(u64::from_le_bytes(value.try_into().map_err(|_| {
        ModuleError::Malformed {
            context: "u64 field",
        }
    })?))
}

fn fixed_name(bytes: &[u8]) -> String {
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

fn names_match(candidate: Option<&OsStr>, requested: &OsStr) -> bool {
    let Some(candidate) = candidate else {
        return false;
    };
    platform_modules::os_strings_match(candidate, requested)
}

fn paths_match(candidate: &Path, requested: &Path) -> bool {
    platform_modules::os_strings_match(candidate.as_os_str(), requested.as_os_str())
}

fn require_count(kind: &'static str, actual: usize, limit: usize) -> Result<(), ModuleError> {
    if actual > limit {
        Err(ModuleError::CountLimit {
            kind,
            actual,
            limit,
        })
    } else {
        Ok(())
    }
}

fn require_contained(image: AddressRange, section: AddressRange) -> Result<(), ModuleError> {
    if image.contains_range(section) {
        Ok(())
    } else {
        Err(ModuleError::SectionOutsideImage { section, image })
    }
}

fn macho_access(protection: u32) -> Access {
    let mut access = Access::empty();
    if protection & 1 != 0 {
        access |= Access::READ;
    }
    if protection & 2 != 0 {
        access |= Access::WRITE;
    }
    if protection & 4 != 0 {
        access |= Access::EXECUTE;
    }
    access
}

fn require_architecture(actual: u32, pe: bool) -> Result<(), ModuleError> {
    let expected = if pe {
        if cfg!(target_arch = "x86_64") {
            0x8664
        } else if cfg!(target_arch = "aarch64") {
            0xaa64
        } else {
            actual
        }
    } else if cfg!(target_arch = "x86_64") {
        0x0100_0007
    } else if cfg!(target_arch = "aarch64") {
        0x0100_000c
    } else {
        actual
    };
    if actual == expected {
        Ok(())
    } else {
        Err(ModuleError::ArchitectureMismatch { actual })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }
    fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn pe_fixture() -> (Vec<u8>, ModuleCandidate) {
        let mut bytes = vec![0u8; 0x1000];
        bytes[..2].copy_from_slice(b"MZ");
        write_u32(&mut bytes, 0x3c, 0x80);
        bytes[0x80..0x84].copy_from_slice(b"PE\0\0");
        write_u16(
            &mut bytes,
            0x84,
            if cfg!(target_arch = "aarch64") {
                0xaa64
            } else {
                0x8664
            },
        );
        write_u16(&mut bytes, 0x86, 1);
        write_u16(&mut bytes, 0x94, 0x70);
        write_u16(
            &mut bytes,
            0x98,
            if cfg!(target_pointer_width = "64") {
                0x20b
            } else {
                0x10b
            },
        );
        write_u32(&mut bytes, 0x98 + 56, 0x1000);
        let section = 0x98 + 0x70;
        bytes[section..section + 5].copy_from_slice(b".text");
        write_u32(&mut bytes, section + 8, 4);
        write_u32(&mut bytes, section + 12, 0x200);
        write_u32(&mut bytes, section + 16, 4);
        write_u32(&mut bytes, section + 36, 0x6000_0020);
        bytes[0x200..0x204].copy_from_slice(&[0xaa, 0xbb, 0xaa, 0xbb]);
        let base = Address::new(bytes.as_ptr() as usize);
        let candidate = ModuleCandidate {
            id: ModuleId(1),
            path: PathBuf::from("fixture.dll"),
            image: base.checked_range(bytes.len()).unwrap(),
            slide: 0,
        };
        (bytes, candidate)
    }

    #[test]
    fn parses_bounded_pe_and_scans_all_matches() {
        let (bytes, candidate) = pe_fixture();
        let mut read = |address: Address, output: &mut [u8]| {
            let offset = address.get() - bytes.as_ptr() as usize;
            output.copy_from_slice(&bytes[offset..offset + output.len()]);
            Ok(())
        };
        let module = parse_candidate(&candidate, ModuleLimits::default(), &mut read).unwrap();
        assert_eq!(module.sections[0].name, ".text");
        let found = scan_section_all(
            &module.sections[0],
            "aa bb",
            0,
            ModuleLimits::default(),
            &mut read,
        )
        .unwrap();
        assert_eq!(
            found,
            vec![
                candidate.image.start().checked_add(0x200).unwrap(),
                candidate.image.start().checked_add(0x202).unwrap()
            ]
        );
    }

    #[test]
    fn rejects_section_outside_pe_image() {
        let (mut bytes, candidate) = pe_fixture();
        write_u32(&mut bytes, 0x98 + 0x70 + 12, 0xfff);
        let mut read = |address: Address, output: &mut [u8]| {
            let offset = address.get() - bytes.as_ptr() as usize;
            output.copy_from_slice(&bytes[offset..offset + output.len()]);
            Ok(())
        };
        assert!(matches!(
            parse_candidate(&candidate, ModuleLimits::default(), &mut read),
            Err(ModuleError::SectionOutsideImage { .. })
        ));
    }

    #[test]
    fn rejects_pe_header_read_outside_candidate() {
        let (bytes, mut candidate) = pe_fixture();
        candidate.image = candidate.image.start().checked_range(0x98).unwrap();
        let mut read = |address: Address, output: &mut [u8]| {
            let offset = address.get() - bytes.as_ptr() as usize;
            output.copy_from_slice(&bytes[offset..offset + output.len()]);
            Ok(())
        };
        assert!(matches!(
            parse_candidate(&candidate, ModuleLimits::default(), &mut read),
            Err(ModuleError::SectionOutsideImage { .. })
        ));
    }

    #[test]
    fn parses_bounded_macho_section() {
        let mut bytes = vec![0u8; 0x1000];
        let base = bytes.as_ptr() as usize;
        write_u32(&mut bytes, 0, 0xfeed_facf);
        write_u32(
            &mut bytes,
            4,
            if cfg!(target_arch = "aarch64") {
                0x0100_000c
            } else {
                0x0100_0007
            },
        );
        write_u32(&mut bytes, 16, 1);
        write_u32(&mut bytes, 20, 152);
        write_u32(&mut bytes, 32, 0x19);
        write_u32(&mut bytes, 36, 152);
        bytes[40..46].copy_from_slice(b"__TEXT");
        write_u64(&mut bytes, 56, base as u64);
        write_u64(&mut bytes, 64, 0x1000);
        write_u32(&mut bytes, 92, 5);
        write_u32(&mut bytes, 96, 1);
        bytes[104..110].copy_from_slice(b"__text");
        write_u64(&mut bytes, 136, (base + 0x200) as u64);
        write_u64(&mut bytes, 144, 4);
        write_u32(&mut bytes, 168, 0x8000_0400);
        let candidate = ModuleCandidate {
            id: ModuleId(9),
            path: PathBuf::from("fixture.dylib"),
            image: Address::new(base).checked_range(bytes.len()).unwrap(),
            slide: 0,
        };
        let mut read = |address: Address, output: &mut [u8]| {
            let offset = address.get() - base;
            output.copy_from_slice(&bytes[offset..offset + output.len()]);
            Ok(())
        };
        let module = parse_candidate(&candidate, ModuleLimits::default(), &mut read).unwrap();
        assert_eq!(module.format, ImageFormat::MachO64);
        assert_eq!(module.sections[0].name, "__text");
        assert!(module.sections[0].is_executable());
    }

    #[test]
    fn rejects_macho_command_extent() {
        let mut bytes = vec![0u8; 0x100];
        let base = bytes.as_ptr() as usize;
        write_u32(&mut bytes, 0, 0xfeed_facf);
        write_u32(
            &mut bytes,
            4,
            if cfg!(target_arch = "aarch64") {
                0x0100_000c
            } else {
                0x0100_0007
            },
        );
        write_u32(&mut bytes, 16, 1);
        write_u32(&mut bytes, 20, 8);
        write_u32(&mut bytes, 32, 0x19);
        write_u32(&mut bytes, 36, 72);
        let candidate = ModuleCandidate {
            id: ModuleId(10),
            path: PathBuf::from("broken.dylib"),
            image: Address::new(base).checked_range(bytes.len()).unwrap(),
            slide: 0,
        };
        let mut read = |address: Address, output: &mut [u8]| {
            let offset = address.get() - base;
            output.copy_from_slice(&bytes[offset..offset + output.len()]);
            Ok(())
        };
        assert!(matches!(
            parse_candidate(&candidate, ModuleLimits::default(), &mut read),
            Err(ModuleError::Malformed {
                context: "Mach-O command extent"
            })
        ));
    }

    #[test]
    fn enforces_cumulative_macho_section_limit() {
        let mut bytes = vec![0u8; 0x1000];
        let base = bytes.as_ptr() as usize;
        write_u32(&mut bytes, 0, 0xfeed_facf);
        write_u32(
            &mut bytes,
            4,
            if cfg!(target_arch = "aarch64") {
                0x0100_000c
            } else {
                0x0100_0007
            },
        );
        write_u32(&mut bytes, 16, 2);
        write_u32(&mut bytes, 20, 464);
        for (command, segment_start, section_start) in [
            (32usize, base, base + 0x200),
            (264, base + 0x800, base + 0xa00),
        ] {
            write_u32(&mut bytes, command, 0x19);
            write_u32(&mut bytes, command + 4, 232);
            write_u64(&mut bytes, command + 24, segment_start as u64);
            write_u64(&mut bytes, command + 32, 0x800);
            write_u32(&mut bytes, command + 60, 5);
            write_u32(&mut bytes, command + 64, 2);
            for (section, address) in [section_start, section_start + 0x100]
                .into_iter()
                .enumerate()
            {
                let record = command + 72 + section * 80;
                write_u64(&mut bytes, record + 32, address as u64);
                write_u64(&mut bytes, record + 40, 4);
            }
        }
        let candidate = ModuleCandidate {
            id: ModuleId(11),
            path: PathBuf::from("sections.dylib"),
            image: Address::new(base).checked_range(bytes.len()).unwrap(),
            slide: 0,
        };
        let mut read = |address: Address, output: &mut [u8]| {
            let offset = address.get() - base;
            output.copy_from_slice(&bytes[offset..offset + output.len()]);
            Ok(())
        };
        let limits = ModuleLimits {
            max_sections: 2,
            ..ModuleLimits::default()
        };
        assert!(matches!(
            parse_candidate(&candidate, limits, &mut read),
            Err(ModuleError::CountLimit {
                kind: "Mach-O section",
                actual: 4,
                limit: 2,
            })
        ));
    }

    #[test]
    fn enforces_result_limit_and_checked_offset() {
        let (bytes, candidate) = pe_fixture();
        let section = Section {
            name: ".text".into(),
            range: candidate
                .image
                .start()
                .checked_add(0x200)
                .unwrap()
                .checked_range(4)
                .unwrap(),
            access: Access::READ | Access::EXECUTE,
            kind: SectionKind::Code,
        };
        let limits = ModuleLimits {
            max_results: 1,
            ..ModuleLimits::default()
        };
        let mut read = |address: Address, output: &mut [u8]| {
            let offset = address.get() - bytes.as_ptr() as usize;
            output.copy_from_slice(&bytes[offset..offset + output.len()]);
            Ok(())
        };
        assert!(matches!(
            scan_section_all(&section, "aa bb", 0, limits, &mut read),
            Err(ModuleError::ResultLimit { limit: 1 })
        ));

        assert_eq!(
            match_address(&section, 2, -1).unwrap(),
            section.range.start().checked_add(1).unwrap()
        );
        assert!(matches!(
            match_address(&section, 0, i64::MIN),
            Err(ModuleError::AddressOverflow)
        ));
    }

    #[test]
    fn enforces_pattern_byte_limit_before_parsing() {
        let limits = ModuleLimits {
            max_pattern_bytes: 2,
            ..ModuleLimits::default()
        };
        let section = Section {
            name: ".text".into(),
            range: Address::new(0x1000).checked_range(4).unwrap(),
            access: Access::READ | Access::EXECUTE,
            kind: SectionKind::Code,
        };
        let mut read = |_address: Address, _output: &mut [u8]| Ok(());
        assert!(matches!(
            scan_section(&section, "aa bb", 0, limits, &mut read),
            Err(ModuleError::ByteLimit {
                kind: "pattern",
                actual: 5,
                limit: 2,
            })
        ));
    }

    struct Candidates(Vec<ModuleCandidate>);
    impl ModuleSnapshotSource for Candidates {
        fn loaded_module_candidates(&self, _: usize) -> Result<Vec<ModuleCandidate>, ModuleError> {
            Ok(self.0.clone())
        }
    }

    #[test]
    fn basename_lookup_reports_ambiguity() {
        let (bytes, first) = pe_fixture();
        let mut second = first.clone();
        second.id = ModuleId(2);
        second.path = PathBuf::from("other/fixture.dll");
        let source = Candidates(vec![first, second]);
        let mut read = |address: Address, output: &mut [u8]| {
            let offset = address.get() - bytes.as_ptr() as usize;
            output.copy_from_slice(&bytes[offset..offset + output.len()]);
            Ok(())
        };
        let map = snapshot(&source, ModuleLimits::default(), &mut read).unwrap();
        assert!(matches!(
            map.find_by_basename(OsStr::new("fixture.dll")),
            Err(ModuleError::AmbiguousName { .. })
        ));
    }

    #[test]
    fn module_lookup_excludes_image_envelope_gaps() {
        let image = Address::new(0x1000).checked_range(0x2000).unwrap();
        let module = LoadedModule {
            id: ModuleId(17),
            path: PathBuf::from("discontiguous.dylib"),
            image,
            mapped_ranges: vec![
                Address::new(0x1000).checked_range(0x100).unwrap(),
                Address::new(0x2000).checked_range(0x100).unwrap(),
            ],
            format: ImageFormat::MachO64,
            sections: Vec::new(),
        };
        let map = ModuleMap {
            modules: vec![module],
        };
        assert!(map.module_containing(Address::new(0x1050)).is_some());
        assert!(map.module_containing(Address::new(0x1800)).is_none());
    }

    #[cfg(any(target_os = "windows", target_os = "macos"))]
    #[test]
    fn section_lease_borrows_owned_section_and_rejects_clone() {
        use crate::memory::ProcessMemory;

        let (bytes, candidate) = pe_fixture();
        let mut read = |address: Address, output: &mut [u8]| {
            let offset = address.get() - bytes.as_ptr() as usize;
            output.copy_from_slice(&bytes[offset..offset + output.len()]);
            Ok(())
        };
        let mut module = parse_candidate(&candidate, ModuleLimits::default(), &mut read).unwrap();
        let copied = module.sections[0].clone();
        let memory = ProcessMemory::new();

        // SAFETY: `bytes` owns the complete immutable section storage and
        // remains live and immovable until both leases are dropped.
        unsafe {
            memory.with_scope(|scope| {
                let lease = module
                    .lease_section_scoped(scope, &module.sections[0])
                    .unwrap();
                assert_eq!(&*lease, &[0xaa, 0xbb, 0xaa, 0xbb]);
                assert!(matches!(
                    module.lease_section_scoped(scope, &copied),
                    Err(ModuleError::SectionNotOwned)
                ));
                module.sections[0].access = Access::EXECUTE;
                assert!(matches!(
                    module.lease_section_scoped(scope, &module.sections[0]),
                    Err(ModuleError::Memory(MemoryError::PermissionDenied {
                        required: Access::READ,
                        actual: Access::EXECUTE,
                        ..
                    }))
                ));
            });
        }
    }

    #[cfg(any(target_os = "windows", target_os = "macos"))]
    #[test]
    fn process_source_observes_loaded_images() {
        let source = ProcessModuleSource;
        let candidates = source.loaded_module_candidates(4_096).unwrap();
        assert!(!candidates.is_empty());
        assert!(candidates.iter().all(|candidate| {
            !candidate.path.as_os_str().is_empty()
                && candidate.image.start().get() as u64 == candidate.id.0
        }));
    }

    #[cfg(any(target_os = "windows", target_os = "macos"))]
    #[test]
    fn process_source_honors_module_limit() {
        assert!(matches!(
            ProcessModuleSource.loaded_module_candidates(0),
            Err(ModuleError::CountLimit {
                kind: "module",
                limit: 0,
                ..
            })
        ));
    }
}
