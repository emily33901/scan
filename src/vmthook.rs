pub mod thunk;

use core::slice;
use std::{
    collections::HashMap,
    sync::{Arc, OnceLock, Weak},
};

use anyhow::{bail, Result};
use parking_lot::{Mutex, MutexGuard};
use thunk::{ThunkableClosure, TrampolineStorage};

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
    new_table: Box<[*const ()]>,
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

    fn for_instance(instance: *mut ()) -> Arc<Mutex<HookInstance>> {
        let mut hooked_instances = HookInstance::all_hooked_instances();

        if let Some(hook) = hooked_instances
            .get(&Instance(instance))
            .and_then(|hook| hook.upgrade())
        {
            return hook;
        }

        let hook_instance = Arc::new(Mutex::new(HookInstance::new(instance)));

        hooked_instances.insert(Instance(instance), Arc::downgrade(&hook_instance));

        hook_instance
    }

    unsafe fn count_funcs(table: *const *const ()) -> usize {
        for i in 0.. {
            if (*table.wrapping_add(i)).is_null() {
                return i;
            }
        }
        unreachable!("table must end at some point")
    }

    unsafe fn get_table(instance: *const ()) -> *const *const () {
        *(instance as *const *const *const ())
    }

    #[allow(dead_code)]
    unsafe fn get_table_mut(instance: *mut ()) -> *mut *const () {
        *(instance as *const *mut *const ())
    }

    unsafe fn replace_table_pointer(instance: *mut (), new_table: *const *const ()) {
        let pointer_to_table = instance as *mut *const *const ();
        *pointer_to_table = new_table;
    }

    pub fn new(instance: *mut ()) -> Self {
        let original_table = unsafe { Self::get_table(instance) };

        let table_len = unsafe { Self::count_funcs(original_table) };
        let new_table: Box<_> = unsafe { slice::from_raw_parts(original_table, table_len) }
            .to_vec()
            .into_boxed_slice();

        unsafe { Self::replace_table_pointer(instance, new_table.as_ptr()) };

        Self {
            original_table,
            instance,
            new_table,
            closures: Default::default(),
        }
    }

    pub fn closure(&self, index: usize) -> *const dyn Fn() {
        self.closures[&index]
    }

    pub fn original_function(&self, index: usize) -> *const () {
        unsafe { *self.original_table.wrapping_add(index) }
    }

    pub fn hook_function(&mut self, index: usize, f: *const ()) -> Result<()> {
        ensure_range(index, self.new_table.len())?;
        self.new_table[index] = f;
        Ok(())
    }

    pub fn unhook_function(&mut self, index: usize) -> Result<()> {
        ensure_range(index, self.new_table.len())?;
        self.new_table[index] = unsafe { *self.original_table.wrapping_add(index) };
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
    if index > max {
        bail!("index {index} is outside the table len {max}");
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

#[derive(Debug)]
pub struct HookFunction {
    instance_hook: Arc<Mutex<HookInstance>>,
    index: usize,
}

impl HookFunction {
    pub fn new<R: 'static, T: 'static, Args: 'static>(
        instance: *mut T,
        index: usize,
        f: impl ThunkableClosure<R, T, Args>,
    ) -> Result<Self> {
        let instance_hook = HookInstance::for_instance(instance as *mut ());

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
