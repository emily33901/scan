//! Observational loaded-module discovery and bounded image parsing.
//!
//! [`Module`] is the only type in this module that changes loader ownership.
//! A [`LoadedModule`] is copied metadata and does not keep an image loaded.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::memory::{
    with_direct_memory_scope, Access, Address, AddressRange, DirectMemoryAccess, DirectMemoryScope,
    FaultContainedMemoryAccess, MemoryError,
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
        // SAFETY: loading a library can run arbitrary initializers. That is the
        // explicit purpose of this owning, side-effecting API.
        let library =
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
    pub image: AddressRange,
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

    pub fn executable_sections(&self) -> impl Iterator<Item = &Section> {
        self.sections
            .iter()
            .filter(|section| section.is_executable())
    }

    pub fn section(&self, name: &str) -> Option<&Section> {
        self.sections.iter().find(|section| section.name == name)
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
        if self.image.contains_range(section.range) && self.sections.contains(section) {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModuleLimits {
    pub max_modules: usize,
    pub max_sections: usize,
    pub max_header_bytes: usize,
    pub max_load_command_bytes: usize,
    pub max_section_bytes: usize,
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
            .find(|module| module.image.contains(address))
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
    #[error("module name {name} is ambiguous")]
    AmbiguousName { name: String },
    #[error("invalid pattern: {message}")]
    Pattern { message: String },
    #[error("match count exceeds limit {limit}")]
    ResultLimit { limit: usize },
    #[error("address arithmetic overflow")]
    AddressOverflow,
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
            require_count("Mach-O section", section_count, limits.max_sections)?;
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
            let segment_start =
                Address::new(usize::try_from(vmaddr).map_err(|_| ModuleError::AddressOverflow)?)
                    .checked_add_signed(candidate.slide)?;
            let segment_len = usize::try_from(vmsize).map_err(|_| ModuleError::AddressOverflow)?;
            if segment_len != 0 {
                require_contained(candidate.image, segment_start.checked_range(segment_len)?)?;
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
    let pattern = parse_pattern(pattern)?;
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
    let pattern = parse_pattern(pattern)?;
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

fn parse_pattern(pattern: &str) -> Result<Vec<Option<u8>>, ModuleError> {
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
    section
        .range
        .start()
        .checked_add(position)?
        .checked_add_signed(offset)
        .map_err(ModuleError::from)
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
    if cfg!(target_os = "windows") {
        candidate
            .to_string_lossy()
            .eq_ignore_ascii_case(&requested.to_string_lossy())
    } else {
        candidate == requested
    }
}

fn paths_match(candidate: &Path, requested: &Path) -> bool {
    if cfg!(target_os = "windows") {
        candidate
            .to_string_lossy()
            .eq_ignore_ascii_case(&requested.to_string_lossy())
    } else {
        candidate == requested
    }
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
}
