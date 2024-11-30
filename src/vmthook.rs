use core::slice;
use std::{
    collections::HashMap,
    sync::{Arc, OnceLock, Weak},
};

use parking_lot::Mutex;

#[derive(Debug)]
struct HookInstance {
    instance: *mut (),
    original_table: *const *const (),
    new_table: Box<[*const ()]>,
}

impl HookInstance {
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
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .into_boxed_slice();

        unsafe { Self::replace_table_pointer(instance, new_table.as_ptr()) };

        Self {
            original_table,
            instance,
            new_table,
        }
    }

    pub fn original_function(&self, index: usize) -> *const () {
        unsafe { *self.original_table.wrapping_add(index) }
    }

    pub fn hook_function(&mut self, index: usize, f: *const ()) {
        assert!(index < self.new_table.len());
        self.new_table[index] = f;
    }

    pub fn unhook_function(&mut self, index: usize) {
        assert!(index < self.new_table.len());
        self.new_table[index] = unsafe { *self.original_table.wrapping_add(index) };
    }
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
    instance: *mut (),
    instance_hook: Arc<Mutex<HookInstance>>,
    index: usize,
    original_function: *const (),
}

impl HookFunction {
    fn get_or_make_instance_hook(instance: *mut ()) -> Arc<Mutex<HookInstance>> {
        #[repr(transparent)]
        #[derive(PartialEq, Eq, PartialOrd, Hash)]
        struct Instance(*mut ());
        unsafe impl Send for Instance {}
        unsafe impl Sync for Instance {}

        static HOOKED_INSTANCES: OnceLock<Mutex<HashMap<Instance, Weak<Mutex<HookInstance>>>>> =
            OnceLock::new();

        let hooked_instances = HOOKED_INSTANCES.get_or_init(|| Default::default());
        let mut hooked_instances = hooked_instances.lock();
        if let Some(weak_hook) = hooked_instances.get(&Instance(instance)) {
            if let Some(hook) = weak_hook.upgrade() {
                return hook;
            }
        }

        let hook_instance = Arc::new(Mutex::new(HookInstance::new(instance)));
        // No hook or doesn't exist anymore
        hooked_instances.insert(Instance(instance), Arc::downgrade(&hook_instance));

        hook_instance
    }

    pub fn new<T>(instance: *mut T, index: usize, f: *const ()) -> Self {
        let instance = instance as *mut ();
        let instance_hook = Self::get_or_make_instance_hook(instance);
        let original_function = {
            let mut instance_hook = instance_hook.lock();
            instance_hook.hook_function(index, f);
            instance_hook.original_function(index)
        };

        Self {
            index,
            instance_hook,
            original_function,
            instance,
        }
    }

    pub fn original(&self) -> *const () {
        self.original_function
    }
}

unsafe impl Send for HookFunction {}
unsafe impl Sync for HookFunction {}

impl Drop for HookFunction {
    fn drop(&mut self) {
        self.instance_hook.lock().unhook_function(self.index)
    }
}
