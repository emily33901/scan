pub mod thunk;

use std::{
    collections::HashMap,
    fmt,
    mem::{align_of, size_of},
    sync::{
        atomic::{AtomicPtr, AtomicU64, Ordering},
        Arc, OnceLock,
    },
};

#[cfg(all(
    feature = "fault-contained",
    any(target_os = "windows", target_os = "macos")
))]
use crate::memory::FaultContainedProcessMemory;
#[cfg(feature = "fault-contained")]
use crate::vtable::discover_fault_contained;
use crate::{
    function::{ExecutableAddress, ExecutablePolicy, FunctionError},
    memory::{Access, Address, MemoryMap, ProcessMemory, DEFAULT_MAX_REGIONS},
    vtable::{discover_direct, CxxAbi, DiscoveredVTable, VTableError, VTableSpec},
};
use parking_lot::Mutex;
use thiserror::Error;
use thunk::{
    DispatchControl, ThunkablePostHook, ThunkablePreHook, TrampolineError, TrampolineStorage,
};

static NEXT_HOOK_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HookOptions {
    pub name: Option<String>,
    pub max_vtable_slots: usize,
    pub discovery: HookDiscovery,
}

impl Default for HookOptions {
    fn default() -> Self {
        Self {
            name: None,
            max_vtable_slots: 4096,
            discovery: HookDiscovery::Direct,
        }
    }
}

/// Memory-copy strategy used for the first vtable discovery.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum HookDiscovery {
    /// Direct checked copies under the installer's unsafe stability contract.
    #[default]
    Direct,
    /// Operating-system-mediated reads that contain invalid source addresses.
    #[cfg(feature = "fault-contained")]
    FaultContained,
}

/// Operation being performed on the object's atomic vptr cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HookMemoryOperation {
    ReadVptr,
    WriteVptr,
}

#[derive(Debug, Error)]
pub enum HookError {
    #[error("{what} pointer is null")]
    NullPointer { what: &'static str },
    #[error("{what} pointer {address:#x} is not pointer-aligned")]
    MisalignedPointer { what: &'static str, address: usize },
    #[error("object vptr cell at {address:#x} is not readable")]
    UnreadableObject { address: usize },
    #[error("object vptr cell at {address:#x} is not writable")]
    UnwritableObject { address: usize },
    #[error("vtable memory at {address:#x} is not readable")]
    UnreadableVtable { address: usize },
    #[error("{what} address arithmetic overflow")]
    AddressOverflow { what: &'static str },
    #[error("vtable has no null terminator within the {max_slots}-slot limit")]
    TerminatorLimit { max_slots: usize },
    #[error("max_vtable_slots must be between 1 and {hard_limit}, got {requested}")]
    InvalidSlotLimit { requested: usize, hard_limit: usize },
    #[error("vtable mapping boundary reached at slot {slot} before its null terminator")]
    TerminatorMappingBoundary { slot: usize },
    #[error("vtable ABI prefix at {address:#x} is unreadable")]
    InvalidPrefix { address: usize },
    #[error("could not reserve owned storage for {requested} vtable words")]
    VtableAllocation { requested: usize },
    #[error("original function in slot {slot} is null")]
    NullOriginal { slot: usize },
    #[error("original function in slot {slot} at {address:#x} is not executable")]
    NonExecutableOriginal { slot: usize, address: usize },
    #[error("vtable slot {slot} is outside table length {len}")]
    SlotOutOfRange { slot: usize, len: usize },
    #[error(
        "vtable slot {slot} on instance {instance:#x} is already hooked by {}",
        existing.0
    )]
    DuplicateHook {
        instance: usize,
        slot: usize,
        existing: HookId,
    },
    #[error("instance vptr changed: expected {expected:#x}, found {actual:#x}")]
    VptrConflict { expected: usize, actual: usize },
    #[error("restored hook instances cannot currently be hooked again")]
    InstanceRestored,
    #[error("instance {instance:#x} is not managed")]
    InstanceNotManaged { instance: usize },
    #[error("hook {} was not found", .0.0)]
    HookNotFound(HookId),
    #[error("process hook identifier space is exhausted")]
    HookIdExhausted,
    #[error("vtable hooking is unsupported on this target")]
    UnsupportedTarget,
    #[error("failed to initialize trampoline JIT: {0}")]
    JitInitialization(String),
    #[error("failed to compile trampoline: {0}")]
    JitCompilation(String),
    #[error("failed to finalize trampoline: {0}")]
    JitFinalization(String),
    #[error("failed to validate the object vptr for {operation:?} at {address}")]
    Memory {
        operation: HookMemoryOperation,
        address: Address,
        #[source]
        source: crate::memory::MemoryError,
    },
    #[error("vtable discovery failed")]
    VTable(#[source] VTableError),
    #[error("original function in slot {slot} at {address} failed revalidation")]
    Function {
        slot: usize,
        address: Address,
        #[source]
        source: FunctionError,
    },
}

impl From<VTableError> for HookError {
    fn from(source: VTableError) -> Self {
        Self::VTable(source)
    }
}

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
    fn next() -> HookResult<Self> {
        NEXT_HOOK_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map(Self)
            .map_err(|_| HookError::HookIdExhausted)
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

fn current_abi() -> HookResult<CxxAbi> {
    CxxAbi::native().ok_or(HookError::UnsupportedTarget)
}

#[derive(Debug)]
struct ClonedVtable {
    storage: Box<[*const ()]>,
    prefix_entries: usize,
}

impl ClonedVtable {
    fn from_discovered(discovered: &DiscoveredVTable) -> Self {
        let prefix_entries = discovered.prefix().len();
        let mut storage = Vec::with_capacity(prefix_entries + discovered.entries().len());
        storage.extend(discovered.prefix().iter().map(|&word| word as *const ()));
        storage.extend(
            discovered
                .entries()
                .iter()
                .map(|entry| entry.address().get() as *const ()),
        );
        Self {
            storage: storage.into_boxed_slice(),
            prefix_entries,
        }
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
// SAFETY: `Instance` is only an opaque address used as a registry key. Moving or
// sharing the numeric identity does not grant access to the pointed-to object.
unsafe impl Send for Instance {}
// SAFETY: `Instance` is only compared and hashed; it is never dereferenced.
unsafe impl Sync for Instance {}

struct InstanceState {
    instance: *mut (),
    original_table: *const *const (),
    original_entries: Box<[ExecutableAddress]>,
    baseline: ClonedVtable,
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
    unsafe fn get_table<M: MemoryMap + ?Sized>(
        instance: *const (),
        map: &M,
    ) -> HookResult<*const *const ()> {
        validate_pointer(instance as usize, "instance")?;
        let address = Address::new(instance as usize);
        let range = address
            .checked_range(size_of::<*const *const ()>())
            .map_err(|source| HookError::Memory {
                operation: HookMemoryOperation::ReadVptr,
                address,
                source,
            })?;
        map.check_range(range, Access::READ, DEFAULT_MAX_REGIONS)
            .map_err(|source| HookError::Memory {
                operation: HookMemoryOperation::ReadVptr,
                address,
                source,
            })?;
        map.check_range(range, Access::WRITE, DEFAULT_MAX_REGIONS)
            .map_err(|source| HookError::Memory {
                operation: HookMemoryOperation::WriteVptr,
                address,
                source,
            })?;
        // SAFETY: the caller guarantees that `instance` denotes a live,
        // correctly aligned `AtomicPtr` vptr cell. The checks above establish
        // that its full storage is currently readable and writable.
        let cell = unsafe { &*(instance as *const AtomicPtr<*const ()>) };
        let table = cell.load(Ordering::Acquire).cast_const();
        validate_pointer(table as usize, "vtable")?;
        Ok(table)
    }

    unsafe fn new(
        instance: *mut (),
        requested_slot: usize,
        options: &HookOptions,
    ) -> HookResult<Self> {
        let memory = ProcessMemory::new();
        // Confirm up front that publication through the object's atomic vptr is
        // permitted. Discovery separately copies every foreign word into owned
        // storage before any hook state is published.
        // SAFETY: `new` inherits the caller's live, aligned, atomic-vptr-cell
        // contract and forwards the same instance address unchanged.
        let original_table = unsafe { Self::get_table(instance, &memory)? };
        let abi = current_abi()?;
        let spec = VTableSpec::new(
            abi,
            requested_slot,
            options.max_vtable_slots,
            ExecutablePolicy::default(),
        );
        let object = Address::new(instance as usize);
        let discovered = match options.discovery {
            // SAFETY: the caller guarantees that the vptr, ABI prefix, and
            // examined entries remain valid and stable for direct discovery.
            HookDiscovery::Direct => unsafe { discover_direct(&memory, object, spec) }?,
            #[cfg(all(
                feature = "fault-contained",
                any(target_os = "windows", target_os = "macos")
            ))]
            HookDiscovery::FaultContained => {
                discover_fault_contained(&FaultContainedProcessMemory::new(), object, spec)?
            }
            #[cfg(all(
                feature = "fault-contained",
                not(any(target_os = "windows", target_os = "macos"))
            ))]
            HookDiscovery::FaultContained => return Err(HookError::UnsupportedTarget),
        };
        if discovered.address_point().get() != original_table as usize {
            return Err(HookError::VptrConflict {
                expected: original_table as usize,
                actual: discovered.address_point().get(),
            });
        }
        let original_entries = discovered.entries().to_vec().into_boxed_slice();
        let baseline = ClonedVtable::from_discovered(&discovered);
        Ok(Self {
            original_table,
            instance,
            original_entries,
            baseline,
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
        self.original_entries[index].address().get() as *const ()
    }

    fn next_generation(&self) -> HookResult<ClonedVtable> {
        if let Some(current) = self.generations.last() {
            Ok(ClonedVtable {
                storage: current.storage.clone(),
                prefix_entries: current.prefix_entries,
            })
        } else {
            Ok(ClonedVtable {
                storage: self.baseline.storage.clone(),
                prefix_entries: self.baseline.prefix_entries,
            })
        }
    }

    fn compare_exchange_vptr(
        &self,
        expected: *const *const (),
        replacement: *const *const (),
    ) -> HookResult<()> {
        // DEFERRED(vmthook): this relies on aligned pointer atomicity when foreign
        // callers load the vptr non-atomically; Rust cannot formally model that race.
        // SAFETY: `InstanceState` is created only under the hook installation
        // contract, which keeps this live, aligned vptr cell valid for atomic
        // publication until restoration.
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

// SAFETY: raw addresses in `InstanceState` refer to process-lifetime hook state
// under the public installation contract. All mutation and foreign vptr access
// is serialized through the registry and this state's mutex.
unsafe impl Send for InstanceState {}
// SAFETY: shared access to mutable state is mutex-protected, while cloned vtable
// generations are immutable after publication and retained for process lifetime.
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
    /// `instance` must be non-null, pointer-aligned, and identify a live object
    /// whose first word is layout- and alignment-compatible with
    /// `AtomicPtr<*const ()>`. That vptr cell must stay mapped, readable,
    /// writable, and correctly synchronized through discovery and publication,
    /// and the object must remain alive while hooked. The original vtable ABI
    /// prefix and all examined entries must remain allocated, readable,
    /// initialized, and unchanged until the default direct discovery completes.
    /// The original module and every captured function must remain loaded and
    /// executable while the hook can dispatch. The callback's shared reference
    /// assumes foreign code does not mutate the object incompatibly while it
    /// runs. On Itanium targets, the class must not require vcall or vbase
    /// offsets before the copied offset-to-top and typeinfo words.
    ///
    /// DEFERRED(vmthook): replace `&T` with an honest raw-pointer API before
    /// permitting concurrent or reentrant handlers.
    /// DEFERRED(vmthook): support extended Itanium negative vtable entries after
    /// introducing a validated, caller-supplied prefix extent.
    /// During installation, host vptr access must either be quiescent or use
    /// atomic operations compatible with scan's atomic load and compare-exchange.
    pub unsafe fn new_pre<R: 'static, T: 'static, Args: 'static>(
        instance: *mut T,
        index: usize,
        f: impl ThunkablePreHook<R, T, Args>,
    ) -> HookResult<Self> {
        // SAFETY: this function imposes the same contract as the delegated
        // configurable constructor and forwards every argument unchanged.
        unsafe { Self::new_pre_with_options(instance, index, f, HookOptions::default()) }
    }

    /// Installs a post-transform hook which runs after the captured original.
    ///
    /// The original is invoked exactly once before the handler. The handler
    /// receives its captured return value and may replace it. Its return type
    /// must be `Copy` so a panicking handler can fall back to the original
    /// value. Dropping the returned handle does not remove or disable the hook.
    /// Nested and concurrent entries bypass the handler without blocking and
    /// return the captured original result directly. A handler panic permanently
    /// disables the handler and returns the original result.
    ///
    /// # Safety
    ///
    /// This has the same installation, object lifetime, vptr, ABI, module, and
    /// synchronization requirements as [`Self::new_pre`]. In addition, the
    /// receiver and any pointer or reference arguments supplied to the handler
    /// must remain valid after the original returns. The transformed return value
    /// must satisfy the hooked method's foreign ABI and semantic contract.
    pub unsafe fn new_post<R: Copy + 'static, T: 'static, Args: 'static>(
        instance: *mut T,
        index: usize,
        f: impl ThunkablePostHook<R, T, Args>,
    ) -> HookResult<Self> {
        // SAFETY: this function imposes the same contract as the delegated
        // configurable constructor and forwards every argument unchanged.
        unsafe { Self::new_post_with_options(instance, index, f, HookOptions::default()) }
    }

    /// Installs a pre-hook with discovery and naming options.
    ///
    /// The first successful install fixes the instance's discovered table size;
    /// `max_vtable_slots` is ignored for later slots on that instance.
    ///
    /// # Safety
    ///
    /// `instance` must be non-null, pointer-aligned, and point to a live object
    /// whose first word is layout- and alignment-compatible with
    /// `AtomicPtr<*const ()>`. The vptr cell must remain mapped, readable,
    /// writable, protection-stable, and synchronized through each checked load
    /// and compare-exchange. The object, original module, and captured functions
    /// must remain live for as long as dispatch is possible.
    /// Concurrent host vptr access must either be quiescent during installation
    /// or use atomic operations compatible with scan's load and compare-exchange.
    ///
    /// With [`HookDiscovery::Direct`], the vptr, selected ABI prefix, and every
    /// examined table word must additionally remain allocated, readable,
    /// initialized, synchronized, and unchanged until discovery has copied them.
    /// Fault-contained discovery removes the raw table-copy validity requirement,
    /// but not the atomic vptr-cell, ABI, function-lifetime, or callback aliasing
    /// requirements. Foreign mutation must not violate the callback's shared
    /// reference requirements.
    pub unsafe fn new_pre_with_options<R: 'static, T: 'static, Args: 'static>(
        instance: *mut T,
        index: usize,
        f: impl ThunkablePreHook<R, T, Args>,
        options: HookOptions,
    ) -> HookResult<Self> {
        let make_dispatch = |original, hook_id| {
            // SAFETY: common installation validated `original` as the
            // executable vtable entry for this advertised hook ABI.
            unsafe { f.into_dispatch(original, hook_id) }
        };
        // SAFETY: this public constructor supplies the object, vptr-cell,
        // table-lifetime, and ABI guarantees required by common installation.
        unsafe { Self::new_with_options(instance, index, options, make_dispatch) }
    }

    /// Installs a post-transform hook with discovery and naming options.
    ///
    /// The first successful install fixes the instance's discovered table size;
    /// `max_vtable_slots` is ignored for later slots on that instance. The
    /// original return is passed to the handler by value and the handler's return
    /// replaces it only when the handler completes successfully.
    ///
    /// # Safety
    ///
    /// This has the same installation requirements as
    /// [`Self::new_pre_with_options`]. The receiver, pointer/reference arguments,
    /// and return value also need to remain valid for the post-original handler;
    /// the handler's replacement return must satisfy the foreign method's ABI and
    /// semantic contract.
    pub unsafe fn new_post_with_options<R: Copy + 'static, T: 'static, Args: 'static>(
        instance: *mut T,
        index: usize,
        f: impl ThunkablePostHook<R, T, Args>,
        options: HookOptions,
    ) -> HookResult<Self> {
        let make_dispatch = |original, hook_id| {
            // SAFETY: common installation validated `original` as the
            // executable vtable entry for this advertised hook ABI.
            unsafe { f.into_dispatch(original, hook_id) }
        };
        // SAFETY: this public constructor supplies the object, vptr-cell,
        // table-lifetime, and ABI guarantees required by common installation.
        unsafe { Self::new_with_options(instance, index, options, make_dispatch) }
    }

    unsafe fn new_with_options<T: 'static>(
        instance: *mut T,
        index: usize,
        options: HookOptions,
        make_dispatch: impl FnOnce(*const (), HookId) -> Box<dyn DispatchControl>,
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
                // SAFETY: the public constructor delegates its object,
                // vptr-cell, table-lifetime, and discovery guarantees here.
                let state = Arc::new(Mutex::new(unsafe {
                    InstanceState::new(instance.cast(), index, &options)?
                }));
                (state, true)
            }
        };
        let mut state = instance_state.lock();
        if state.restored {
            // DEFERRED(vmthook): support re-hooking a restored live object after
            // defining how foreign vptr changes establish a new baseline.
            return Err(HookError::InstanceRestored);
        }
        ensure_range(index, state.original_entries.len())?;
        let expected = state.latest_table();
        let memory = ProcessMemory::new();
        // SAFETY: the public contract keeps the same object's atomic vptr cell
        // live, aligned, mapped, and synchronized throughout installation.
        let actual = unsafe { InstanceState::get_table(instance.cast(), &memory)? };
        if actual != expected {
            return Err(HookError::VptrConflict {
                expected: expected as usize,
                actual: actual as usize,
            });
        }
        // SAFETY: the installation contract keeps each captured original
        // function loaded and stable while hooks can dispatch through it.
        unsafe { state.original_entries[index].revalidate_direct(&memory) }.map_err(|source| {
            HookError::Function {
                slot: index,
                address: state.original_entries[index].address(),
                source,
            }
        })?;
        let original = state.original_function(index);
        if original.is_null() {
            return Err(HookError::NullOriginal { slot: index });
        }
        // Every foreign table read was completed when this instance was first
        // discovered. Later installs clone only owned immutable generations.
        let mut generation = state.next_generation()?;
        let hook_id = HookId::next()?;
        let context: Arc<dyn DispatchControl> = Arc::from(
            // SAFETY: discovery validated `original` as the executable entry for
            // this slot. The public constructor's generic bounds and ABI contract
            // govern the supplied dispatch-context factory.
            make_dispatch(original, hook_id),
        );
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
/// `instance` must still identify the same live, pointer-aligned object whose
/// first word is layout- and alignment-compatible with `AtomicPtr<*const ()>`.
/// The vptr cell must remain mapped, readable, writable, protection-stable, and
/// correctly synchronized through validation and compare-exchange. Host calls
/// must be quiescent for restoration; the object, original table, and captured
/// functions must remain live until no thread can execute an old generation.
pub unsafe fn restore_instance<T>(instance: *mut T) -> HookResult<()> {
    let key = Instance(instance.cast());
    let registry = registry().lock();
    let state = registry
        .instances
        .get(&key)
        .ok_or(HookError::InstanceNotManaged {
            instance: instance as usize,
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
    let memory = ProcessMemory::new();
    // SAFETY: the restoration contract keeps this same object's atomic vptr
    // cell live, aligned, mapped, and synchronized through restoration.
    let actual = unsafe { InstanceState::get_table(instance.cast(), &memory)? };
    if actual != expected {
        return Err(HookError::VptrConflict {
            expected: expected as usize,
            actual: actual as usize,
        });
    }
    state.compare_exchange_vptr(expected, state.original_table)?;
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
    use std::{ptr, sync::atomic::AtomicU32};

    #[test]
    fn rejects_slot_at_or_beyond_table_length() {
        assert!(ensure_range(2, 2).is_err());
        assert!(ensure_range(3, 2).is_err());
        assert!(ensure_range(1, 2).is_ok());
    }

    #[test]
    fn rejects_null_and_misaligned_pointers() {
        assert!(matches!(
            validate_pointer(0, "test"),
            Err(HookError::NullPointer { what: "test" })
        ));
        assert!(matches!(
            validate_pointer(1, "test"),
            Err(HookError::MisalignedPointer {
                what: "test",
                address: 1
            })
        ));
    }

    #[test]
    fn vptr_validation_preserves_the_memory_error_source() {
        let address = usize::MAX - (usize::MAX % align_of::<*const ()>());
        let memory = ProcessMemory::new();
        let error = unsafe { InstanceState::get_table(address as *const (), &memory) }.unwrap_err();

        assert!(matches!(
            error,
            HookError::Memory {
                operation: HookMemoryOperation::ReadVptr,
                source: crate::memory::MemoryError::AddressOverflow,
                ..
            }
        ));
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

    unsafe extern "C" fn replacement_one(this: *mut HookTarget, value: u32) -> u32 {
        unsafe { &*this }.calls.fetch_add(1, Ordering::Relaxed);
        value + 30
    }

    fn leak_target() -> *mut HookTarget {
        let prefix = current_abi().unwrap().prefix_words();
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

    fn leak_target_with_nonexecutable_second_slot() -> *mut HookTarget {
        let prefix = current_abi().unwrap().prefix_words();
        let mut layout = vec![ptr::null(); prefix];
        layout.extend([
            original_zero as *const (),
            ptr::without_provenance(1),
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
        unsafe { call_table_slot(target, (*target).vptr, slot, value) }
    }

    unsafe fn call_table_slot(
        target: *mut HookTarget,
        table: *const *const (),
        slot: usize,
        value: u32,
    ) -> u32 {
        let function: unsafe extern "C" fn(*mut HookTarget, u32) -> u32 =
            unsafe { std::mem::transmute(*table.add(slot)) };
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
        assert_eq!(unsafe { call_table_slot(target, first_vptr, 0, 3) }, 13);
        assert_eq!(unsafe { call_table_slot(target, first_vptr, 1, 4) }, 24);
        assert_eq!(HookFunction::list_for_instance(target).len(), 2);
        assert_eq!(HookFunction::lookup(first.id()).unwrap().slot, 0);
        assert_eq!(
            HookFunction::lookup_slot(target, 1).unwrap().hook_id,
            second.id()
        );
    }

    #[test]
    fn post_hook_transforms_captured_original_result() {
        let target = leak_target();
        let hook = unsafe {
            HookFunction::new_post::<u32, _, _>(
                target,
                0,
                |this: &HookTarget, value: u32, original_result: u32| {
                    assert_eq!(this.calls.load(Ordering::Relaxed), 1);
                    assert_eq!(value, 3);
                    assert_eq!(original_result, 13);
                    original_result + 100
                },
            )
        }
        .unwrap();

        assert_eq!(unsafe { call_slot(target, 0, 3) }, 113);
        assert_eq!(unsafe { &*target }.calls.load(Ordering::Relaxed), 1);
        assert!(hook.diagnostics().enabled);
        assert_eq!(hook.diagnostics().last_status, HookStatus::Ready);
    }

    #[test]
    fn later_installs_use_the_owned_discovery_snapshot() {
        let target = leak_target();
        let original_table = unsafe { (*target).vptr };
        let _first =
            unsafe { HookFunction::new_pre::<u32, _, _>(target, 0, |_: &HookTarget, _: u32| {}) }
                .unwrap();

        // The original table is foreign writable storage in this fixture. A
        // later install must not use this changed word as its captured original.
        unsafe {
            original_table
                .cast_mut()
                .add(1)
                .write(replacement_one as *const ())
        };
        let _second =
            unsafe { HookFunction::new_pre::<u32, _, _>(target, 1, |_: &HookTarget, _: u32| {}) }
                .unwrap();

        assert_eq!(unsafe { call_slot(target, 1, 2) }, 22);
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
    fn validates_every_original_entry_is_executable() {
        let target = leak_target_with_nonexecutable_second_slot();
        let error =
            unsafe { HookFunction::new_pre::<u32, _, _>(target, 0, |_: &HookTarget, _: u32| {}) }
                .unwrap_err();
        assert!(matches!(
            &error,
            HookError::VTable(VTableError::InvalidExecutable {
                slot: 1,
                address,
                ..
            }) if *address == Address::new(1)
        ));
        let source = std::error::Error::source(&error).unwrap();
        let discovery = source.downcast_ref::<VTableError>().unwrap();
        assert!(std::error::Error::source(discovery)
            .unwrap()
            .downcast_ref::<FunctionError>()
            .is_some());
    }

    #[cfg(feature = "fault-contained")]
    #[test]
    fn fault_contained_discovery_can_install_a_hook() {
        let target = leak_target();
        let hook = unsafe {
            HookFunction::new_pre_with_options::<u32, _, _>(
                target,
                0,
                |_: &HookTarget, _: u32| {},
                HookOptions {
                    discovery: HookDiscovery::FaultContained,
                    ..HookOptions::default()
                },
            )
        }
        .unwrap();

        assert_eq!(unsafe { call_slot(target, 0, 5) }, 15);
        assert!(hook.diagnostics().enabled);
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

        assert!(matches!(
            error,
            HookError::VTable(VTableError::SlotOutOfRange { .. })
        ));
        assert_eq!(unsafe { (*target).vptr }, original);
        assert!(HookFunction::list_for_instance(target).is_empty());
        assert!(matches!(
            unsafe { restore_instance(target) },
            Err(HookError::InstanceNotManaged { .. })
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
        for cycle in 0..3 {
            assert!(HookFunction::set_enabled_by_id(hook.id(), false).unwrap());
            assert_eq!(unsafe { call_slot(target, 0, cycle) }, cycle + 10);
            assert!(HookFunction::set_enabled_by_id(hook.id(), true).unwrap());
            assert_eq!(unsafe { call_slot(target, 0, cycle) }, cycle + 10);
        }
        assert_eq!(unsafe { &*target }.calls.load(Ordering::Relaxed), 306);
    }

    #[test]
    fn publication_conflict_leaves_foreign_vptr_unchanged() {
        let target = leak_target();
        let original = unsafe { (*target).vptr };
        let first =
            unsafe { HookFunction::new_pre::<u32, _, _>(target, 0, |_: &HookTarget, _: u32| {}) }
                .unwrap();
        let first_generation = unsafe { (*target).vptr };
        unsafe { (*target).vptr = original };

        let error =
            unsafe { HookFunction::new_pre::<u32, _, _>(target, 1, |_: &HookTarget, _: u32| {}) }
                .unwrap_err();

        assert!(matches!(error, HookError::VptrConflict { .. }));
        assert_eq!(unsafe { (*target).vptr }, original);
        assert_eq!(HookFunction::list_for_instance(target).len(), 1);
        assert_eq!(
            unsafe { call_table_slot(target, first_generation, 0, 7) },
            17
        );
        assert_eq!(HookFunction::lookup(first.id()).unwrap().slot, 0);
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
