mod memory;
pub mod thunk;

use std::{
    collections::HashMap,
    mem::{align_of, size_of},
    ptr,
    sync::{Arc, OnceLock, Weak},
};

use anyhow::{bail, Result};
use memory::{
    checked_pointer_address, ensure_executable, ensure_readable, MemoryMap, ProcessMemoryMap,
};
use parking_lot::{Mutex, MutexGuard};
use thunk::{ThunkableClosure, TrampolineStorage};

const DEFAULT_MAX_VTABLE_SLOTS: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VtableAbi {
    Msvc64,
    Itanium64,
}

impl VtableAbi {
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    fn current() -> Result<Self> {
        Ok(Self::Msvc64)
    }

    #[cfg(all(
        target_os = "macos",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    fn current() -> Result<Self> {
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
    fn current() -> Result<Self> {
        bail!("vtable hooking is unsupported on this target")
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

    fn len(&self) -> usize {
        self.storage.len() - self.prefix_entries
    }

    fn address_point(&self) -> *const *const () {
        // SAFETY: construction always reserves `prefix_entries` elements first.
        unsafe { self.storage.as_ptr().add(self.prefix_entries) }
    }

    fn function_entry_mut(&mut self, index: usize) -> &mut *const () {
        &mut self.storage[self.prefix_entries + index]
    }
}

thread_local! {
    static GLOBAL_TRAMPOLINE_STORAGE: TrampolineStorage = TrampolineStorage::new().unwrap();
}

#[repr(transparent)]
#[derive(PartialEq, Eq, PartialOrd, Hash)]
struct Instance(*mut ());
unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

#[derive(Debug)]
struct HookInstance {
    instance: *mut (),
    original_table: *const *const (),
    new_table: ClonedVtable,
    // NOTE(emily): These are not actually 0 arg functions
    closures: HashMap<usize, *const dyn Fn()>,
}

impl HookInstance {
    fn all_hooked_instances() -> MutexGuard<'static, HashMap<Instance, Weak<Mutex<HookInstance>>>> {
        static HOOKED_INSTANCES: OnceLock<Mutex<HashMap<Instance, Weak<Mutex<HookInstance>>>>> =
            OnceLock::new();

        let hooked_instances = HOOKED_INSTANCES.get_or_init(Default::default);

        let hooked_instances = hooked_instances.lock();
        hooked_instances
    }

    fn for_instance(instance: *mut ()) -> Result<Arc<Mutex<HookInstance>>> {
        let mut hooked_instances = HookInstance::all_hooked_instances();

        if let Some(hook) = hooked_instances
            .get(&Instance(instance))
            .and_then(|hook| hook.upgrade())
        {
            return Ok(hook);
        }

        let hook_instance = Arc::new(Mutex::new(HookInstance::new(instance)?));

        hooked_instances.insert(Instance(instance), Arc::downgrade(&hook_instance));

        Ok(hook_instance)
    }

    unsafe fn count_funcs<M: MemoryMap>(
        table: *const *const (),
        max_slots: usize,
        map: &M,
    ) -> Result<usize> {
        let base = table as usize;
        let region = map.region_containing(base)?;
        if !region.readable || !region.contains(base, size_of::<*const ()>()) {
            bail!("vtable does not begin in readable memory");
        }
        // DEFERRED(vmthook): exact C++ vtable lengths are ABI- and class-specific.
        // Keep the observed null-terminator heuristic, but constrain it to one mapping
        // and a hard slot cap so an unfamiliar table fails closed.
        for index in 0..max_slots {
            let address = checked_pointer_address(base, index)?;
            if !region.contains(address, size_of::<*const ()>()) {
                bail!("vtable mapping boundary reached before its null terminator");
            }
            if unsafe { ptr::read(address as *const *const ()) }.is_null() {
                return Ok(index);
            }
        }
        bail!("vtable has no null terminator within the {max_slots}-slot limit")
    }

    unsafe fn get_table<M: MemoryMap>(instance: *const (), map: &M) -> Result<*const *const ()> {
        validate_pointer(instance as usize, "instance")?;
        ensure_readable(map, instance as usize, size_of::<*const *const ()>())?;
        let table = unsafe { ptr::read(instance as *const *const *const ()) };
        validate_pointer(table as usize, "vtable")?;
        Ok(table)
    }

    #[allow(dead_code)]
    unsafe fn get_table_mut(instance: *mut ()) -> *mut *const () {
        *(instance as *const *mut *const ())
    }

    unsafe fn replace_table_pointer(instance: *mut (), new_table: *const *const ()) {
        let pointer_to_table = instance as *mut *const *const ();
        *pointer_to_table = new_table;
    }

    pub fn new(instance: *mut ()) -> Result<Self> {
        let map = ProcessMemoryMap;
        let original_table = unsafe { Self::get_table(instance, &map)? };

        let table_len =
            unsafe { Self::count_funcs(original_table, DEFAULT_MAX_VTABLE_SLOTS, &map)? };
        let new_table = unsafe {
            ClonedVtable::copy_from(original_table, table_len, VtableAbi::current()?, &map)?
        };

        unsafe { Self::replace_table_pointer(instance, new_table.address_point()) };

        Ok(Self {
            original_table,
            instance,
            new_table,
            closures: Default::default(),
        })
    }

    pub fn closure(&self, index: usize) -> *const dyn Fn() {
        self.closures[&index]
    }

    pub fn original_function(&self, index: usize) -> *const () {
        unsafe { *self.original_table.wrapping_add(index) }
    }

    pub fn hook_function(&mut self, index: usize, f: *const ()) -> Result<()> {
        ensure_range(index, self.new_table.len())?;
        let original = self.original_function(index);
        ensure_executable(&ProcessMemoryMap, original as usize)?;
        *self.new_table.function_entry_mut(index) = f;
        Ok(())
    }

    pub fn unhook_function(&mut self, index: usize) -> Result<()> {
        ensure_range(index, self.new_table.len())?;
        *self.new_table.function_entry_mut(index) =
            unsafe { *self.original_table.wrapping_add(index) };
        Ok(())
    }

    fn hook_function_with_closure<R: 'static, T: 'static, Args: 'static>(
        &mut self,
        index: usize,
        f: impl ThunkableClosure<R, T, Args>,
    ) -> Result<()> {
        let trampoline = GLOBAL_TRAMPOLINE_STORAGE.with(|thunk_storage| {
            let mut module = thunk_storage.module();
            f.make_trampoline(&mut module, index)
        })?;

        let closure = f.into_raw_closure();

        self.hook_function(index, trampoline)?;
        self.closures.insert(index, closure);

        Ok(())
    }

    fn unhook_function_with_closure(&mut self, index: usize) -> Result<()> {
        self.unhook_function(index)?;

        self.closures.remove(&index);

        Ok(())
    }
}

fn ensure_range(index: usize, max: usize) -> Result<()> {
    if index >= max {
        bail!("index {index} is outside the table len {max}");
    }

    Ok(())
}

fn validate_pointer(address: usize, description: &str) -> Result<()> {
    if address == 0 {
        bail!("{description} pointer is null");
    }
    if address % align_of::<*const ()>() != 0 {
        bail!("{description} pointer {address:#x} is not pointer-aligned");
    }
    Ok(())
}

impl Drop for HookInstance {
    fn drop(&mut self) {
        unsafe { Self::replace_table_pointer(self.instance, self.original_table) };
    }
}

unsafe impl Send for HookInstance {}
unsafe impl Sync for HookInstance {}

/// A hooked function in an instance's vtable.
#[derive(Debug)]
pub struct HookFunction {
    instance_hook: Arc<Mutex<HookInstance>>,
    index: usize,
}

impl HookFunction {
    /// Hook a function in an instances vtable, providing a closure that will be called instead of the
    /// original.
    /// The closure provided should take a `ctx`, `&mut this` and arguments of the function being called.
    ///
    /// When the [`HookFunction`] is dropped, the function will be un-hooked.
    ///
    ///
    /// All arguments of the function are expected to be able to be made into
    /// [CraneLift basic types](https://docs.rs/cranelift-codegen/latest/cranelift_codegen/ir/types/index.html).
    /// If you need to do this for a custom type, then consider implementing [`crate::AsCraneliftAbi`] for it.
    ///
    ///
    /// # Example
    ///
    /// ```rs
    ///  let instance_function_hook = HookFunction::new(
    ///      instance,
    ///      10,
    ///      |ctx, this, arg: usize| -> usize {
    ///         let arg = arg + 10
    ///         call_original(ctx, this, (arg,))
    ///     },
    /// )
    /// .unwrap();
    /// ```
    ///
    pub fn new<R: 'static, T: 'static, Args: 'static>(
        instance: *mut T,
        index: usize,
        f: impl ThunkableClosure<R, T, Args>,
    ) -> Result<Self> {
        let instance_hook = HookInstance::for_instance(instance as *mut ())?;

        instance_hook.lock().hook_function_with_closure(index, f)?;

        Ok(Self {
            index,
            instance_hook,
        })
    }
}

impl Drop for HookFunction {
    fn drop(&mut self) {
        self.instance_hook
            .lock()
            .unhook_function_with_closure(self.index)
            .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

        let count = unsafe { HookInstance::count_funcs(entries.as_ptr(), 8, &map) }.unwrap();

        assert_eq!(count, 2);
    }

    #[test]
    fn rejects_table_without_terminator_before_cap() {
        let entries = [1usize as *const (), 2usize as *const ()];
        let map = map_for(&entries, entries.len());

        let error = unsafe { HookInstance::count_funcs(entries.as_ptr(), 2, &map) }.unwrap_err();

        assert!(error.to_string().contains("2-slot limit"));
    }

    #[test]
    fn rejects_mapping_boundary_before_terminator() {
        let entries = [1usize as *const (), 2usize as *const (), ptr::null()];
        let map = map_for(&entries, 2);

        let error = unsafe { HookInstance::count_funcs(entries.as_ptr(), 3, &map) }.unwrap_err();

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

        unsafe {
            HookInstance::replace_table_pointer(
                (&mut object_vptr as *mut *const *const ()).cast(),
                clone.address_point(),
            );
        }

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
