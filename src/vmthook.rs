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

use anyhow::Result;
use memory::{
    checked_pointer_address, ensure_executable, ensure_readable, MemoryMap, ProcessMemoryMap,
};
use parking_lot::Mutex;
use thunk::{DispatchControl, ThunkablePreHook, TrampolineStorage};

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
    InvalidPointer(String),
    InvalidVtable(String),
    SlotOutOfRange { slot: usize, len: usize },
    DuplicateHook { instance: usize, slot: usize },
    VptrConflict { expected: usize, actual: usize },
    InstanceRestored,
    UnsupportedTarget,
    Jit(String),
}

impl fmt::Display for HookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPointer(message) | Self::InvalidVtable(message) | Self::Jit(message) => {
                f.write_str(message)
            }
            Self::SlotOutOfRange { slot, len } => {
                write!(f, "vtable slot {slot} is outside table length {len}")
            }
            Self::DuplicateHook { instance, slot } => write!(
                f,
                "vtable slot {slot} on instance {instance:#x} is already hooked"
            ),
            Self::VptrConflict { expected, actual } => write!(
                f,
                "instance vptr changed: expected {expected:#x}, found {actual:#x}"
            ),
            Self::InstanceRestored => {
                f.write_str("restored hook instances cannot currently be hooked again")
            }
            Self::UnsupportedTarget => f.write_str("vtable hooking is unsupported on this target"),
        }
    }
}

impl Error for HookError {}

pub type HookResult<T> = std::result::Result<T, HookError>;

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
}

impl HookStatus {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Running,
            2 => Self::BusyBypass,
            3 => Self::Disabled,
            4 => Self::Panicked,
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
    ) -> Result<Self> {
        let prefix_entries = abi.prefix_entries();
        let mut entries = Vec::with_capacity(prefix_entries + table_len);
        let pointer_size = size_of::<*const ()>();

        for distance in (1..=prefix_entries).rev() {
            let offset = distance
                .checked_mul(pointer_size)
                .ok_or_else(|| anyhow::anyhow!("vtable prefix address arithmetic overflow"))?;
            let address = (original_table as usize)
                .checked_sub(offset)
                .ok_or_else(|| anyhow::anyhow!("vtable prefix address arithmetic overflow"))?;
            ensure_readable(map, address, pointer_size)?;
            entries.push(unsafe { ptr::read(address as *const *const ()) });
        }

        for index in 0..table_len {
            let address = checked_pointer_address(original_table as usize, index)?;
            ensure_readable(map, address, pointer_size)?;
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
            .map_err(|error| HookError::InvalidVtable(error.to_string()))?;
        if !region.readable || !region.contains(base, size_of::<*const ()>()) {
            return Err(HookError::InvalidVtable(
                "vtable does not begin in readable memory".into(),
            ));
        }
        // DEFERRED(vmthook): exact C++ vtable lengths are ABI- and class-specific.
        // Keep the observed null-terminator heuristic, but constrain it to one mapping
        // and a hard slot cap so an unfamiliar table fails closed.
        for index in 0..max_slots {
            let address = checked_pointer_address(base, index)
                .map_err(|error| HookError::InvalidVtable(error.to_string()))?;
            if !region.contains(address, size_of::<*const ()>()) {
                return Err(HookError::InvalidVtable(
                    "vtable mapping boundary reached before its null terminator".into(),
                ));
            }
            if unsafe { ptr::read(address as *const *const ()) }.is_null() {
                return Ok(index);
            }
        }
        Err(HookError::InvalidVtable(format!(
            "vtable has no null terminator within the {max_slots}-slot limit"
        )))
    }

    unsafe fn get_table<M: MemoryMap>(
        instance: *const (),
        map: &M,
    ) -> HookResult<*const *const ()> {
        validate_pointer(instance as usize, "instance")?;
        ensure_readable(map, instance as usize, size_of::<*const *const ()>())
            .map_err(|error| HookError::InvalidPointer(error.to_string()))?;
        let table = unsafe { ptr::read(instance as *const *const *const ()) };
        validate_pointer(table as usize, "vtable")?;
        Ok(table)
    }

    fn new(instance: *mut (), max_slots: usize) -> HookResult<Self> {
        let map = ProcessMemoryMap;
        let original_table = unsafe { Self::get_table(instance, &map)? };
        let table_len = unsafe { Self::count_funcs(original_table, max_slots, &map)? };
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
            .map_err(|error| HookError::InvalidVtable(error.to_string()))
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

fn validate_pointer(address: usize, description: &str) -> HookResult<()> {
    if address == 0 {
        return Err(HookError::InvalidPointer(format!(
            "{description} pointer is null"
        )));
    }
    if address % align_of::<*const ()>() != 0 {
        return Err(HookError::InvalidPointer(format!(
            "{description} pointer {address:#x} is not pointer-aligned"
        )));
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
    /// All arguments must be `Copy` and representable as
    /// [CraneLift basic types](https://docs.rs/cranelift-codegen/latest/cranelift_codegen/ir/types/index.html).
    /// If you need to do this for a custom type, then consider implementing [`crate::AsCraneliftAbi`] for it.
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

    pub unsafe fn new_pre_with_options<R: 'static, T: 'static, Args: 'static>(
        instance: *mut T,
        index: usize,
        f: impl ThunkablePreHook<R, T, Args>,
        options: HookOptions,
    ) -> HookResult<Self> {
        let instance_key = Instance(instance.cast());
        let mut registry = registry().lock();
        if registry.by_slot.contains_key(&(instance_key, index)) {
            return Err(HookError::DuplicateHook {
                instance: instance as usize,
                slot: index,
            });
        }

        let instance_state = match registry.instances.get(&instance_key) {
            Some(state) => Arc::clone(state),
            None => {
                let state = Arc::new(Mutex::new(InstanceState::new(
                    instance.cast(),
                    options.max_vtable_slots,
                )?));
                registry.instances.insert(instance_key, Arc::clone(&state));
                state
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
        ensure_executable(&ProcessMemoryMap, original as usize)
            .map_err(|error| HookError::InvalidVtable(error.to_string()))?;
        let hook_id = HookId::next();
        let context: Arc<dyn DispatchControl> =
            Arc::from(unsafe { f.into_dispatch(original, hook_id) });
        let trampoline = TrampolineStorage::global()
            .map_err(|error| HookError::Jit(error.to_string()))?
            .make_trampoline(&*context, hook_id)
            .map_err(|error| HookError::Jit(error.to_string()))?;
        let mut generation = state.next_generation()?;
        *generation.function_entry_mut(index) = trampoline;
        let replacement = generation.address_point();
        state.compare_exchange_vptr(expected, replacement)?;
        state.generations.push(generation);
        state.slots.insert(index, hook_id);

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

    pub fn set_enabled(&self, enabled: bool) {
        self.context.set_enabled(enabled);
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

/// Restores an instance's original vptr if scan still owns the latest vptr.
///
/// # Safety
/// The object must still be live and host calls must be quiescent for restoration.
pub unsafe fn restore_instance<T>(instance: *mut T) -> HookResult<()> {
    let key = Instance(instance.cast());
    let registry = registry().lock();
    let state = registry
        .instances
        .get(&key)
        .ok_or(HookError::InvalidPointer("instance is not managed".into()))?;
    let mut state = state.lock();
    for id in state.slots.values() {
        if let Some(record) = registry.by_id.get(id) {
            record.context.set_enabled(false);
        }
    }
    let expected = state.latest_table();
    state.compare_exchange_vptr(expected, state.original_table)?;
    state.restored = true;
    // DEFERRED(vmthook): generations, trampolines, and contexts are never
    // reclaimed because callers may still be executing through an old vptr.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::bail;
    use memory::MemoryRegion;

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

        assert!(error.to_string().contains("2-slot limit"));
    }

    #[test]
    fn rejects_mapping_boundary_before_terminator() {
        let entries = [1usize as *const (), 2usize as *const (), ptr::null()];
        let map = map_for(&entries, 2);

        let error = unsafe { InstanceState::count_funcs(entries.as_ptr(), 3, &map) }.unwrap_err();

        assert!(error.to_string().contains("mapping boundary"));
    }

    #[test]
    fn rejects_slot_at_or_beyond_table_length() {
        assert!(ensure_range(2, 2).is_err());
        assert!(ensure_range(3, 2).is_err());
        assert!(ensure_range(1, 2).is_ok());
    }

    #[test]
    fn rejects_null_and_misaligned_pointers() {
        assert!(validate_pointer(0, "test").is_err());
        assert!(validate_pointer(1, "test").is_err());
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

        assert!(error.to_string().contains("mapping"));
    }
}
