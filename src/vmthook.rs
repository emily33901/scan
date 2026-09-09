mod memory;
pub mod thunk;

use std::{
    collections::HashMap,
    error::Error,
    fmt,
    mem::{align_of, size_of},
    ptr,
    sync::{
        atomic::{AtomicPtr, AtomicU64, Ordering},
        Arc, OnceLock,
    },
};

use memory::{
    checked_pointer_address, ensure_executable, ensure_readable, ensure_writable, MemoryMap,
    ProcessMemoryMap,
};
use parking_lot::Mutex;
use thunk::{DispatchControl, ThunkablePreHook, TrampolineError, TrampolineStorage};

static NEXT_HOOK_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HookOptions {
    pub name: Option<String>,
    pub max_vtable_slots: usize,
}

impl Default for HookOptions {
    fn default() -> Self {
        Self {
            name: None,
            max_vtable_slots: 4096,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HookError {
    NullPointer {
        what: &'static str,
    },
    MisalignedPointer {
        what: &'static str,
        address: usize,
    },
    UnreadableObject {
        address: usize,
    },
    UnwritableObject {
        address: usize,
    },
    UnreadableVtable {
        address: usize,
    },
    AddressOverflow {
        what: &'static str,
    },
    TerminatorLimit {
        max_slots: usize,
    },
    TerminatorMappingBoundary {
        slot: usize,
    },
    InvalidPrefix {
        address: usize,
    },
    NullOriginal {
        slot: usize,
    },
    NonExecutableOriginal {
        slot: usize,
        address: usize,
    },
    SlotOutOfRange {
        slot: usize,
        len: usize,
    },
    DuplicateHook {
        instance: usize,
        slot: usize,
        existing: HookId,
    },
    VptrConflict {
        expected: usize,
        actual: usize,
    },
    InstanceRestored,
    HookNotFound(HookId),
    UnsupportedTarget,
    JitInitialization(String),
    JitCompilation(String),
    JitFinalization(String),
}

impl fmt::Display for HookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NullPointer { what } => write!(f, "{what} pointer is null"),
            Self::MisalignedPointer { what, address } => {
                write!(f, "{what} pointer {address:#x} is not pointer-aligned")
            }
            Self::UnreadableObject { address } => {
                write!(f, "object vptr cell at {address:#x} is not readable")
            }
            Self::UnwritableObject { address } => {
                write!(f, "object vptr cell at {address:#x} is not writable")
            }
            Self::UnreadableVtable { address } => {
                write!(f, "vtable memory at {address:#x} is not readable")
            }
            Self::AddressOverflow { what } => write!(f, "{what} address arithmetic overflow"),
            Self::TerminatorLimit { max_slots } => write!(
                f,
                "vtable has no null terminator within the {max_slots}-slot limit"
            ),
            Self::TerminatorMappingBoundary { slot } => write!(
                f,
                "vtable mapping boundary reached at slot {slot} before its null terminator"
            ),
            Self::InvalidPrefix { address } => {
                write!(f, "vtable ABI prefix at {address:#x} is unreadable")
            }
            Self::NullOriginal { slot } => write!(f, "original function in slot {slot} is null"),
            Self::NonExecutableOriginal { slot, address } => write!(
                f,
                "original function in slot {slot} at {address:#x} is not executable"
            ),
            Self::SlotOutOfRange { slot, len } => {
                write!(f, "vtable slot {slot} is outside table length {len}")
            }
            Self::DuplicateHook {
                instance,
                slot,
                existing,
            } => write!(
                f,
                "vtable slot {slot} on instance {instance:#x} is already hooked by {}",
                existing.0
            ),
            Self::VptrConflict { expected, actual } => write!(
                f,
                "instance vptr changed: expected {expected:#x}, found {actual:#x}"
            ),
            Self::InstanceRestored => {
                f.write_str("restored hook instances cannot currently be hooked again")
            }
            Self::HookNotFound(id) => write!(f, "hook {} was not found", id.0),
            Self::UnsupportedTarget => f.write_str("vtable hooking is unsupported on this target"),
            Self::JitInitialization(message) => {
                write!(f, "failed to initialize trampoline JIT: {message}")
            }
            Self::JitCompilation(message) => write!(f, "failed to compile trampoline: {message}"),
            Self::JitFinalization(message) => write!(f, "failed to finalize trampoline: {message}"),
        }
    }
}

impl Error for HookError {}

pub type HookResult<T> = std::result::Result<T, HookError>;

fn map_trampoline_error(error: TrampolineError) -> HookError {
    match error {
        TrampolineError::Initialization(message) => HookError::JitInitialization(message),
        TrampolineError::Compilation(message) => HookError::JitCompilation(message),
        TrampolineError::Finalization(message) => HookError::JitFinalization(message),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HookId(u64);

impl HookId {
    fn next() -> Self {
        Self(NEXT_HOOK_ID.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HookInfo {
    pub hook_id: HookId,
    pub instance: usize,
    pub slot: usize,
    pub name: Option<String>,
    pub diagnostics: HookDiagnostics,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HookStatus {
    Ready,
    Running,
    BusyBypass,
    Disabled,
    Panicked,
    Restored,
}

impl HookStatus {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Running,
            2 => Self::BusyBypass,
            3 => Self::Disabled,
            4 => Self::Panicked,
            5 => Self::Restored,
            _ => Self::Ready,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HookDiagnostics {
    pub hook_id: HookId,
    pub enabled: bool,
    pub invocations: u64,
    pub bypasses: u64,
    pub panics: u64,
    pub last_status: HookStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VtableAbi {
    Msvc64,
    Itanium64,
}

impl VtableAbi {
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    fn current() -> HookResult<Self> {
        Ok(Self::Msvc64)
    }

    #[cfg(all(
        target_os = "macos",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    fn current() -> HookResult<Self> {
        // DEFERRED(vmthook): arm64e pointer authentication needs validation in Ableton.
        Ok(Self::Itanium64)
    }

    #[cfg(not(any(
        all(target_os = "windows", target_arch = "x86_64"),
        all(
            target_os = "macos",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    fn current() -> HookResult<Self> {
        Err(HookError::UnsupportedTarget)
    }

    const fn prefix_entries(self) -> usize {
        match self {
            Self::Msvc64 => 1,
            Self::Itanium64 => 2,
        }
    }
}

#[derive(Debug)]
struct ClonedVtable {
    storage: Box<[*const ()]>,
    prefix_entries: usize,
}

impl ClonedVtable {
    unsafe fn copy_from<M: MemoryMap>(
        original_table: *const *const (),
        table_len: usize,
        abi: VtableAbi,
        map: &M,
    ) -> HookResult<Self> {
        let prefix_entries = abi.prefix_entries();
        let mut entries = Vec::with_capacity(prefix_entries + table_len);
        let pointer_size = size_of::<*const ()>();

        for distance in (1..=prefix_entries).rev() {
            let offset = distance
                .checked_mul(pointer_size)
                .ok_or(HookError::AddressOverflow {
                    what: "vtable prefix",
                })?;
            let address = (original_table as usize).checked_sub(offset).ok_or(
                HookError::AddressOverflow {
                    what: "vtable prefix",
                },
            )?;
            ensure_readable(map, address, pointer_size)
                .map_err(|_| HookError::InvalidPrefix { address })?;
            entries.push(unsafe { ptr::read(address as *const *const ()) });
        }

        for index in 0..table_len {
            let address =
                checked_pointer_address(original_table as usize, index).map_err(|_| {
                    HookError::AddressOverflow {
                        what: "vtable entry",
                    }
                })?;
            ensure_readable(map, address, pointer_size)
                .map_err(|_| HookError::UnreadableVtable { address })?;
            entries.push(unsafe { ptr::read(address as *const *const ()) });
        }

        Ok(Self {
            storage: entries.into_boxed_slice(),
            prefix_entries,
        })
    }

    fn address_point(&self) -> *const *const () {
        // SAFETY: construction always reserves `prefix_entries` elements first.
        unsafe { self.storage.as_ptr().add(self.prefix_entries) }
    }

    fn function_entry_mut(&mut self, index: usize) -> &mut *const () {
        &mut self.storage[self.prefix_entries + index]
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Instance(*mut ());
unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

struct InstanceState {
    instance: *mut (),
    original_table: *const *const (),
    table_len: usize,
    abi: VtableAbi,
    generations: Vec<ClonedVtable>,
    slots: HashMap<usize, HookId>,
    restored: bool,
}

impl std::fmt::Debug for InstanceState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InstanceState")
            .field("instance", &self.instance)
            .field("original_table", &self.original_table)
            .field("generations", &self.generations.len())
            .field("hooked_slots", &self.slots.keys())
            .finish()
    }
}

impl InstanceState {
    unsafe fn count_funcs<M: MemoryMap>(
        table: *const *const (),
        max_slots: usize,
        map: &M,
    ) -> HookResult<usize> {
        let base = table as usize;
        let region = map
            .region_containing(base)
            .map_err(|_| HookError::UnreadableVtable { address: base })?;
        if !region.readable || !region.contains(base, size_of::<*const ()>()) {
            return Err(HookError::UnreadableVtable { address: base });
        }
        // DEFERRED(vmthook): exact C++ vtable lengths are ABI- and class-specific.
        // Keep the observed null-terminator heuristic, but constrain it to one mapping
        // and a hard slot cap so an unfamiliar table fails closed.
        for index in 0..max_slots {
            let address =
                checked_pointer_address(base, index).map_err(|_| HookError::AddressOverflow {
                    what: "vtable entry",
                })?;
            if !region.contains(address, size_of::<*const ()>()) {
                return Err(HookError::TerminatorMappingBoundary { slot: index });
            }
            if unsafe { ptr::read(address as *const *const ()) }.is_null() {
                return Ok(index);
            }
        }
        Err(HookError::TerminatorLimit { max_slots })
    }

    unsafe fn get_table<M: MemoryMap>(
        instance: *const (),
        map: &M,
    ) -> HookResult<*const *const ()> {
        validate_pointer(instance as usize, "instance")?;
        ensure_readable(map, instance as usize, size_of::<*const *const ()>()).map_err(|_| {
            HookError::UnreadableObject {
                address: instance as usize,
            }
        })?;
        ensure_writable(map, instance as usize, size_of::<*const *const ()>()).map_err(|_| {
            HookError::UnwritableObject {
                address: instance as usize,
            }
        })?;
        let cell = unsafe { &*(instance as *const AtomicPtr<*const ()>) };
        let table = cell.load(Ordering::Acquire).cast_const();
        validate_pointer(table as usize, "vtable")?;
        Ok(table)
    }

    fn new(instance: *mut (), max_slots: usize) -> HookResult<Self> {
        let map = ProcessMemoryMap;
        let original_table = unsafe { Self::get_table(instance, &map)? };
        let table_len = unsafe { Self::count_funcs(original_table, max_slots, &map)? };
        for slot in 0..table_len {
            let address = checked_pointer_address(original_table as usize, slot).map_err(|_| {
                HookError::AddressOverflow {
                    what: "vtable entry",
                }
            })?;
            let original = unsafe { ptr::read(address as *const *const ()) };
            if original.is_null() {
                return Err(HookError::NullOriginal { slot });
            }
            ensure_executable(&map, original as usize).map_err(|_| {
                HookError::NonExecutableOriginal {
                    slot,
                    address: original as usize,
                }
            })?;
        }
        Ok(Self {
            original_table,
            instance,
            table_len,
            abi: VtableAbi::current()?,
            generations: Vec::new(),
            slots: HashMap::new(),
            restored: false,
        })
    }

    fn latest_table(&self) -> *const *const () {
        self.generations
            .last()
            .map_or(self.original_table, ClonedVtable::address_point)
    }

    fn original_function(&self, index: usize) -> *const () {
        // SAFETY: installation validates index against the discovered baseline.
        unsafe { *self.original_table.add(index) }
    }

    fn next_generation(&self) -> HookResult<ClonedVtable> {
        if let Some(current) = self.generations.last() {
            Ok(ClonedVtable {
                storage: current.storage.clone(),
                prefix_entries: current.prefix_entries,
            })
        } else {
            unsafe {
                ClonedVtable::copy_from(
                    self.original_table,
                    self.table_len,
                    self.abi,
                    &ProcessMemoryMap,
                )
            }
        }
    }

    fn compare_exchange_vptr(
        &self,
        expected: *const *const (),
        replacement: *const *const (),
    ) -> HookResult<()> {
        // DEFERRED(vmthook): this relies on aligned pointer atomicity when foreign
        // callers load the vptr non-atomically; Rust cannot formally model that race.
        let cell = unsafe { &*(self.instance as *const AtomicPtr<*const ()>) };
        match cell.compare_exchange(
            expected.cast_mut(),
            replacement.cast_mut(),
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => Ok(()),
            Err(actual) => Err(HookError::VptrConflict {
                expected: expected as usize,
                actual: actual as usize,
            }),
        }
    }
}

fn ensure_range(index: usize, max: usize) -> HookResult<()> {
    if index >= max {
        return Err(HookError::SlotOutOfRange {
            slot: index,
            len: max,
        });
    }

    Ok(())
}

fn validate_pointer(address: usize, description: &'static str) -> HookResult<()> {
    if address == 0 {
        return Err(HookError::NullPointer { what: description });
    }
    if address % align_of::<*const ()>() != 0 {
        return Err(HookError::MisalignedPointer {
            what: description,
            address,
        });
    }
    Ok(())
}

struct HookRecord {
    instance: Instance,
    slot: usize,
    name: Option<String>,
    context: Arc<dyn DispatchControl>,
}

struct Registry {
    instances: HashMap<Instance, Arc<Mutex<InstanceState>>>,
    by_id: HashMap<HookId, HookRecord>,
    by_slot: HashMap<(Instance, usize), HookId>,
}

fn registry() -> &'static Mutex<Registry> {
    // DEFERRED(vmthook): this registry is process-global only within one linked
    // copy of scan; coordinate it across dylibs if scan is linked more than once.
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        Mutex::new(Registry {
            instances: HashMap::new(),
            by_id: HashMap::new(),
            by_slot: HashMap::new(),
        })
    })
}

unsafe impl Send for InstanceState {}
unsafe impl Sync for InstanceState {}

/// A hooked function in an instance's vtable.
pub struct HookFunction {
    hook_id: HookId,
    context: Arc<dyn DispatchControl>,
}

impl HookFunction {
    /// Installs a pre-hook which runs before the captured original function.
    ///
    /// Dropping the returned handle does not remove or disable the hook. Hook
    /// code, contexts, and vtable generations are retained for process lifetime.
    /// The original is invoked exactly once, even if the pre-hook panics. A panic
    /// permanently disables the handler. Nested and concurrent entries bypass the
    /// handler without blocking and call the original directly.
    /// Panic containment requires an unwind panic strategy; native faults and
    /// foreign exceptions are not caught.
    ///
    /// Supported arguments and returns are pointers/references, `f32`, `f64`,
    /// `usize`, `isize`, and 8/16/32-bit integers. Arguments must be `Copy`.
    /// Aggregate, vector, variadic, hidden-sret, and custom ABI forms are not
    /// currently supported.
    ///
    ///
    /// # Safety
    ///
    /// `instance` must identify a live object with a compatible C++ vtable and
    /// remain alive while hooked. The original module must remain loaded. The
    /// callback's shared reference assumes foreign code does not mutate the object
    /// incompatibly while it runs.
    ///
    /// DEFERRED(vmthook): replace `&T` with an honest raw-pointer API before
    /// permitting concurrent or reentrant handlers.
    pub unsafe fn new_pre<R: 'static, T: 'static, Args: 'static>(
        instance: *mut T,
        index: usize,
        f: impl ThunkablePreHook<R, T, Args>,
    ) -> HookResult<Self> {
        unsafe { Self::new_pre_with_options(instance, index, f, HookOptions::default()) }
    }

    /// Installs a pre-hook with discovery and naming options.
    ///
    /// The first successful install fixes the instance's discovered table size;
    /// `max_vtable_slots` is ignored for later slots on that instance.
    ///
    /// # Safety
    /// `instance` must identify a live compatible C++ object for the process
    /// lifetime, its original module must remain loaded, and foreign mutation
    /// must not violate the shared callback reference's aliasing requirements.
    pub unsafe fn new_pre_with_options<R: 'static, T: 'static, Args: 'static>(
        instance: *mut T,
        index: usize,
        f: impl ThunkablePreHook<R, T, Args>,
        options: HookOptions,
    ) -> HookResult<Self> {
        let instance_key = Instance(instance.cast());
        let mut registry = registry().lock();
        if let Some(&existing) = registry.by_slot.get(&(instance_key, index)) {
            return Err(HookError::DuplicateHook {
                instance: instance as usize,
                slot: index,
                existing,
            });
        }

        let (instance_state, new_instance) = match registry.instances.get(&instance_key) {
            Some(state) => (Arc::clone(state), false),
            None => {
                let state = Arc::new(Mutex::new(InstanceState::new(
                    instance.cast(),
                    options.max_vtable_slots,
                )?));
                (state, true)
            }
        };
        let mut state = instance_state.lock();
        if state.restored {
            // DEFERRED(vmthook): support re-hooking a restored live object after
            // defining how foreign vptr changes establish a new baseline.
            return Err(HookError::InstanceRestored);
        }
        ensure_range(index, state.table_len)?;
        let expected = state.latest_table();
        let actual = unsafe { InstanceState::get_table(instance.cast(), &ProcessMemoryMap)? };
        if actual != expected {
            return Err(HookError::VptrConflict {
                expected: expected as usize,
                actual: actual as usize,
            });
        }
        let original = state.original_function(index);
        if original.is_null() {
            return Err(HookError::NullOriginal { slot: index });
        }
        ensure_executable(&ProcessMemoryMap, original as usize).map_err(|_| {
            HookError::NonExecutableOriginal {
                slot: index,
                address: original as usize,
            }
        })?;
        // Complete every fallible table read before emitting unreachable JIT code.
        let mut generation = state.next_generation()?;
        let hook_id = HookId::next();
        let context: Arc<dyn DispatchControl> =
            Arc::from(unsafe { f.into_dispatch(original, hook_id) });
        let trampoline = TrampolineStorage::global()
            .map_err(map_trampoline_error)?
            .make_trampoline(&*context, hook_id)
            .map_err(map_trampoline_error)?;
        *generation.function_entry_mut(index) = trampoline;
        let replacement = generation.address_point();
        state.generations.reserve(1);
        state.slots.reserve(1);
        if new_instance {
            registry.instances.reserve(1);
        }
        registry.by_slot.reserve(1);
        registry.by_id.reserve(1);
        state.compare_exchange_vptr(expected, replacement)?;
        state.generations.push(generation);
        state.slots.insert(index, hook_id);
        drop(state);

        if new_instance {
            registry
                .instances
                .insert(instance_key, Arc::clone(&instance_state));
        }
        registry.by_slot.insert((instance_key, index), hook_id);
        registry.by_id.insert(
            hook_id,
            HookRecord {
                instance: instance_key,
                slot: index,
                name: options.name,
                context: Arc::clone(&context),
            },
        );
        Ok(Self { hook_id, context })
    }

    /// Changes handler execution state. Returns `false` when a panicked hook
    /// rejects an attempt to re-enable it.
    pub fn set_enabled(&self, enabled: bool) -> bool {
        self.context.set_enabled(enabled)
    }

    pub fn diagnostics(&self) -> HookDiagnostics {
        self.context.diagnostics()
    }

    pub fn id(&self) -> HookId {
        self.hook_id
    }

    pub fn lookup(hook_id: HookId) -> Option<HookInfo> {
        let registry = registry().lock();
        registry
            .by_id
            .get(&hook_id)
            .map(|record| record.info(hook_id))
    }

    pub fn lookup_slot<T>(instance: *mut T, slot: usize) -> Option<HookInfo> {
        let registry = registry().lock();
        let id = *registry.by_slot.get(&(Instance(instance.cast()), slot))?;
        registry.by_id.get(&id).map(|record| record.info(id))
    }

    pub fn lookup_name(name: &str) -> Vec<HookInfo> {
        let registry = registry().lock();
        let mut hooks: Vec<_> = registry
            .by_id
            .iter()
            .filter(|(_, record)| record.name.as_deref() == Some(name))
            .map(|(&id, record)| record.info(id))
            .collect();
        hooks.sort_by_key(|info| info.hook_id);
        hooks
    }

    pub fn list_all() -> Vec<HookInfo> {
        let registry = registry().lock();
        let mut hooks: Vec<_> = registry
            .by_id
            .iter()
            .map(|(&id, record)| record.info(id))
            .collect();
        hooks.sort_by_key(|info| info.hook_id);
        hooks
    }

    pub fn set_enabled_by_id(hook_id: HookId, enabled: bool) -> HookResult<bool> {
        let registry = registry().lock();
        let record = registry
            .by_id
            .get(&hook_id)
            .ok_or(HookError::HookNotFound(hook_id))?;
        Ok(record.context.set_enabled(enabled))
    }

    pub fn list_for_instance<T>(instance: *mut T) -> Vec<HookInfo> {
        let key = Instance(instance.cast());
        let registry = registry().lock();
        let mut hooks: Vec<_> = registry
            .by_id
            .iter()
            .filter(|(_, record)| record.instance == key)
            .map(|(&id, record)| record.info(id))
            .collect();
        hooks.sort_by_key(|info| info.slot);
        hooks
    }
}

impl fmt::Debug for HookFunction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HookFunction")
            .field("hook_id", &self.hook_id)
            .finish()
    }
}

impl Drop for HookFunction {
    fn drop(&mut self) {
        // Intentionally empty: the strong registry owns the process-lifetime hook.
    }
}

impl HookRecord {
    fn info(&self, hook_id: HookId) -> HookInfo {
        HookInfo {
            hook_id,
            instance: self.instance.0 as usize,
            slot: self.slot,
            name: self.name.clone(),
            diagnostics: self.context.diagnostics(),
        }
    }
}

/// Retires all handlers and restores the original vptr if scan still owns it.
/// A vptr conflict leaves foreign state untouched and handlers disabled.
///
/// # Safety
/// The object must still be live and host calls must be quiescent for restoration.
pub unsafe fn restore_instance<T>(instance: *mut T) -> HookResult<()> {
    let key = Instance(instance.cast());
    let registry = registry().lock();
    let state = registry.instances.get(&key).ok_or(HookError::NullPointer {
        what: "managed instance",
    })?;
    let mut state = state.lock();
    if state.restored {
        return Err(HookError::InstanceRestored);
    }
    for id in state.slots.values() {
        if let Some(record) = registry.by_id.get(id) {
            record.context.retire(HookStatus::Disabled);
        }
    }
    let expected = state.latest_table();
    if let Err(error) = state.compare_exchange_vptr(expected, state.original_table) {
        return Err(error);
    }
    for id in state.slots.values() {
        if let Some(record) = registry.by_id.get(id) {
            record.context.retire(HookStatus::Restored);
        }
    }
    state.restored = true;
    // DEFERRED(vmthook): generations, trampolines, and contexts are never
    // reclaimed because callers may still be executing through an old vptr.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{bail, Result};
    use memory::MemoryRegion;
    use std::sync::atomic::AtomicU32;

    struct TableMap {
        region: MemoryRegion,
    }

    impl MemoryMap for TableMap {
        fn region_containing(&self, address: usize) -> Result<MemoryRegion> {
            if (self.region.start..self.region.end).contains(&address) {
                Ok(self.region)
            } else {
                bail!("mapping boundary reached")
            }
        }
    }

    fn map_for(entries: &[*const ()], mapped_entries: usize) -> TableMap {
        let start = entries.as_ptr() as usize;
        TableMap {
            region: MemoryRegion {
                start,
                end: start + mapped_entries * size_of::<*const ()>(),
                readable: true,
                writable: true,
                executable: false,
            },
        }
    }

    unsafe fn clone_layout(
        layout: &[*const ()],
        prefix_entries: usize,
        abi: VtableAbi,
    ) -> ClonedVtable {
        let map = map_for(layout, layout.len());
        let address_point = unsafe { layout.as_ptr().add(prefix_entries) };
        let table_len = layout.len() - prefix_entries - 1;
        unsafe { ClonedVtable::copy_from(address_point, table_len, abi, &map) }.unwrap()
    }

    #[test]
    fn discovers_null_terminated_table() {
        let entries = [1usize as *const (), 2usize as *const (), ptr::null()];
        let map = map_for(&entries, entries.len());

        let count = unsafe { InstanceState::count_funcs(entries.as_ptr(), 8, &map) }.unwrap();

        assert_eq!(count, 2);
    }

    #[test]
    fn rejects_table_without_terminator_before_cap() {
        let entries = [1usize as *const (), 2usize as *const ()];
        let map = map_for(&entries, entries.len());

        let error = unsafe { InstanceState::count_funcs(entries.as_ptr(), 2, &map) }.unwrap_err();

        assert_eq!(error, HookError::TerminatorLimit { max_slots: 2 });
    }

    #[test]
    fn rejects_mapping_boundary_before_terminator() {
        let entries = [1usize as *const (), 2usize as *const (), ptr::null()];
        let map = map_for(&entries, 2);

        let error = unsafe { InstanceState::count_funcs(entries.as_ptr(), 3, &map) }.unwrap_err();

        assert_eq!(error, HookError::TerminatorMappingBoundary { slot: 2 });
    }

    #[test]
    fn rejects_slot_at_or_beyond_table_length() {
        assert!(ensure_range(2, 2).is_err());
        assert!(ensure_range(3, 2).is_err());
        assert!(ensure_range(1, 2).is_ok());
    }

    #[test]
    fn rejects_null_and_misaligned_pointers() {
        assert_eq!(
            validate_pointer(0, "test"),
            Err(HookError::NullPointer { what: "test" })
        );
        assert_eq!(
            validate_pointer(1, "test"),
            Err(HookError::MisalignedPointer {
                what: "test",
                address: 1
            })
        );
    }

    #[test]
    fn rejects_unwritable_object_vptr_cell() {
        let table = [ptr::null::<()>()];
        let object_vptr = table.as_ptr();
        let address = &object_vptr as *const _ as usize;
        let map = TableMap {
            region: MemoryRegion {
                start: address,
                end: address + size_of::<*const *const ()>(),
                readable: true,
                writable: false,
                executable: false,
            },
        };

        let object = (&object_vptr as *const *const *const ()).cast::<()>();
        let error = unsafe { InstanceState::get_table(object, &map) }.unwrap_err();
        assert_eq!(error, HookError::UnwritableObject { address });
    }

    #[test]
    fn preserves_msvc_prefix_and_function_entries() {
        let layout = [
            0x101usize as *const (),
            0x201usize as *const (),
            0x301usize as *const (),
            ptr::null(),
        ];

        let clone = unsafe { clone_layout(&layout, 1, VtableAbi::Msvc64) };

        assert_eq!(&*clone.storage, &layout[..layout.len() - 1]);
        assert_eq!(clone.address_point(), unsafe {
            clone.storage.as_ptr().add(1)
        });
        assert_eq!(unsafe { *clone.address_point() }, layout[1]);
    }

    #[test]
    fn preserves_itanium_prefix_and_function_entries() {
        let layout = [
            (-16isize) as *const (),
            0x102usize as *const (),
            0x202usize as *const (),
            0x302usize as *const (),
            ptr::null(),
        ];

        let clone = unsafe { clone_layout(&layout, 2, VtableAbi::Itanium64) };

        assert_eq!(&*clone.storage, &layout[..layout.len() - 1]);
        assert_eq!(clone.address_point(), unsafe {
            clone.storage.as_ptr().add(2)
        });
        assert_eq!(unsafe { *clone.address_point() }, layout[2]);
    }

    #[test]
    fn replacement_vptr_targets_function_entry_zero() {
        let layout = [
            0x103usize as *const (),
            0x203usize as *const (),
            ptr::null(),
        ];
        let clone = unsafe { clone_layout(&layout, 1, VtableAbi::Msvc64) };
        let mut object_vptr = layout.as_ptr().wrapping_add(1);

        let cell = unsafe {
            &*((&mut object_vptr as *mut *const *const ()).cast::<AtomicPtr<*const ()>>())
        };
        cell.store(clone.address_point().cast_mut(), Ordering::Release);

        assert_eq!(object_vptr, clone.address_point());
        assert_eq!(unsafe { *object_vptr }, layout[1]);
    }

    #[test]
    fn rejects_unreadable_vtable_prefix() {
        let layout = [
            0x104usize as *const (),
            0x204usize as *const (),
            ptr::null(),
        ];
        let function_entries = &layout[1..];
        let map = map_for(function_entries, function_entries.len());

        let error =
            unsafe { ClonedVtable::copy_from(layout.as_ptr().add(1), 1, VtableAbi::Msvc64, &map) }
                .unwrap_err();

        assert!(matches!(error, HookError::InvalidPrefix { .. }));
    }

    #[repr(C)]
    struct HookTarget {
        vptr: *const *const (),
        calls: AtomicU32,
    }

    unsafe extern "C" fn original_zero(this: *mut HookTarget, value: u32) -> u32 {
        unsafe { &*this }.calls.fetch_add(1, Ordering::Relaxed);
        value + 10
    }

    unsafe extern "C" fn original_one(this: *mut HookTarget, value: u32) -> u32 {
        unsafe { &*this }.calls.fetch_add(1, Ordering::Relaxed);
        value + 20
    }

    fn leak_target() -> *mut HookTarget {
        let prefix = VtableAbi::current().unwrap().prefix_entries();
        let mut layout = vec![ptr::null(); prefix];
        layout.extend([
            original_zero as *const (),
            original_one as *const (),
            ptr::null(),
        ]);
        let layout = Box::leak(layout.into_boxed_slice());
        let vptr = unsafe { layout.as_ptr().add(prefix) };
        Box::into_raw(Box::new(HookTarget {
            vptr,
            calls: AtomicU32::new(0),
        }))
    }

    unsafe fn call_slot(target: *mut HookTarget, slot: usize, value: u32) -> u32 {
        let function: unsafe extern "C" fn(*mut HookTarget, u32) -> u32 =
            unsafe { std::mem::transmute(*(*target).vptr.add(slot)) };
        unsafe { function(target, value) }
    }

    #[test]
    fn multiple_slots_publish_immutable_generations() {
        let target = leak_target();
        let first =
            unsafe { HookFunction::new_pre::<u32, _, _>(target, 0, |_: &HookTarget, _: u32| {}) }
                .unwrap();
        let first_vptr = unsafe { (*target).vptr };
        let original_slot_one = unsafe { *first_vptr.add(1) };
        let second =
            unsafe { HookFunction::new_pre::<u32, _, _>(target, 1, |_: &HookTarget, _: u32| {}) }
                .unwrap();
        let second_vptr = unsafe { (*target).vptr };

        assert_ne!(first_vptr, second_vptr);
        assert_eq!(unsafe { *first_vptr.add(1) }, original_slot_one);
        assert_ne!(unsafe { *second_vptr.add(1) }, original_slot_one);
        assert_eq!(unsafe { call_slot(target, 0, 1) }, 11);
        assert_eq!(unsafe { call_slot(target, 1, 2) }, 22);
        assert_eq!(HookFunction::list_for_instance(target).len(), 2);
        assert_eq!(HookFunction::lookup(first.id()).unwrap().slot, 0);
        assert_eq!(
            HookFunction::lookup_slot(target, 1).unwrap().hook_id,
            second.id()
        );
    }

    #[test]
    fn duplicate_slot_is_rejected() {
        let target = leak_target();
        let _hook =
            unsafe { HookFunction::new_pre::<u32, _, _>(target, 0, |_: &HookTarget, _: u32| {}) }
                .unwrap();
        let error =
            unsafe { HookFunction::new_pre::<u32, _, _>(target, 0, |_: &HookTarget, _: u32| {}) }
                .unwrap_err();
        assert!(
            matches!(error, HookError::DuplicateHook { slot: 0, existing, .. } if existing == _hook.id())
        );
    }

    #[test]
    fn concurrent_duplicate_installation_has_one_winner() {
        let target = leak_target() as usize;
        let start = Arc::new(std::sync::Barrier::new(3));
        let mut threads = Vec::new();
        for _ in 0..2 {
            let start = Arc::clone(&start);
            threads.push(std::thread::spawn(move || {
                start.wait();
                unsafe {
                    HookFunction::new_pre::<u32, _, _>(
                        target as *mut HookTarget,
                        0,
                        |_: &HookTarget, _: u32| {},
                    )
                }
            }));
        }
        start.wait();
        let results: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(
            results
                .iter()
                .filter(|result| matches!(result, Err(HookError::DuplicateHook { .. })))
                .count(),
            1
        );
    }

    #[test]
    fn failed_first_install_leaves_no_managed_instance() {
        let target = leak_target();
        let original = unsafe { (*target).vptr };
        let error =
            unsafe { HookFunction::new_pre::<u32, _, _>(target, 2, |_: &HookTarget, _: u32| {}) }
                .unwrap_err();

        assert!(matches!(error, HookError::SlotOutOfRange { .. }));
        assert_eq!(unsafe { (*target).vptr }, original);
        assert!(HookFunction::list_for_instance(target).is_empty());
        assert!(matches!(
            unsafe { restore_instance(target) },
            Err(HookError::NullPointer {
                what: "managed instance"
            })
        ));
    }

    #[test]
    fn dropping_handle_and_install_thread_preserve_hook() {
        let target = leak_target();
        let target_address = target as usize;
        std::thread::spawn(move || {
            let target = target_address as *mut HookTarget;
            let hook = unsafe {
                HookFunction::new_pre::<u32, _, _>(target, 0, |this: &HookTarget, _: u32| {
                    this.calls.fetch_add(100, Ordering::Relaxed);
                })
            }
            .unwrap();
            drop(hook);
        })
        .join()
        .unwrap();

        assert_eq!(unsafe { call_slot(target, 0, 3) }, 13);
        assert_eq!(unsafe { &*target }.calls.load(Ordering::Relaxed), 101);
    }

    #[test]
    fn dispatch_does_not_lock_registry() {
        let target = leak_target();
        let _hook =
            unsafe { HookFunction::new_pre::<u32, _, _>(target, 0, |_: &HookTarget, _: u32| {}) }
                .unwrap();
        let _registry_guard = registry().lock();
        assert_eq!(unsafe { call_slot(target, 0, 4) }, 14);
    }

    #[test]
    fn enable_disable_bypasses_handler() {
        let target = leak_target();
        let hook = unsafe {
            HookFunction::new_pre::<u32, _, _>(target, 0, |this: &HookTarget, _: u32| {
                this.calls.fetch_add(100, Ordering::Relaxed);
            })
        }
        .unwrap();
        assert!(HookFunction::set_enabled_by_id(hook.id(), false).unwrap());
        assert_eq!(unsafe { call_slot(target, 0, 5) }, 15);
        assert_eq!(unsafe { &*target }.calls.load(Ordering::Relaxed), 1);
        assert!(HookFunction::set_enabled_by_id(hook.id(), true).unwrap());
        assert_eq!(unsafe { call_slot(target, 0, 6) }, 16);
        assert_eq!(unsafe { &*target }.calls.load(Ordering::Relaxed), 102);
    }

    #[test]
    fn restoration_requires_exact_latest_vptr() {
        let target = leak_target();
        let original = unsafe { (*target).vptr };
        let hook =
            unsafe { HookFunction::new_pre::<u32, _, _>(target, 0, |_: &HookTarget, _: u32| {}) }
                .unwrap();
        unsafe { restore_instance(target) }.unwrap();
        assert_eq!(unsafe { (*target).vptr }, original);
        assert!(!hook.diagnostics().enabled);
        assert_eq!(hook.diagnostics().last_status, HookStatus::Restored);
        assert!(!hook.set_enabled(true));

        let conflicted = leak_target();
        let conflict_hook = unsafe {
            HookFunction::new_pre::<u32, _, _>(conflicted, 0, |_: &HookTarget, _: u32| {})
        }
        .unwrap();
        let foreign = original;
        unsafe { (*conflicted).vptr = foreign };
        assert!(matches!(
            unsafe { restore_instance(conflicted) },
            Err(HookError::VptrConflict { .. })
        ));
        assert_eq!(unsafe { (*conflicted).vptr }, foreign);
        assert!(!conflict_hook.diagnostics().enabled);
        assert!(!conflict_hook.set_enabled(true));
    }

    #[test]
    fn registry_discovers_named_hooks() {
        let target = leak_target();
        let name = format!("named-hook-{}", target as usize);
        let hook = unsafe {
            HookFunction::new_pre_with_options::<u32, _, _>(
                target,
                0,
                |_: &HookTarget, _: u32| {},
                HookOptions {
                    name: Some(name.clone()),
                    ..HookOptions::default()
                },
            )
        }
        .unwrap();

        assert_eq!(HookFunction::lookup_name(&name)[0].hook_id, hook.id());
        assert!(HookFunction::list_all()
            .iter()
            .any(|info| info.hook_id == hook.id()));
        assert!(matches!(
            HookFunction::set_enabled_by_id(HookId(u64::MAX), false),
            Err(HookError::HookNotFound(_))
        ));
    }
}
