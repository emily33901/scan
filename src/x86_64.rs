use std::ffi::CStr;

/// Gets the virtual table at `offset` bytes from `instance`.
///
/// # Safety
/// * instance must be a valid pointer.
/// * `*(instance + offset)` must be a valid pointer.
///
pub unsafe fn virtual_table(instance: *const (), offset: usize) -> *const *const () {
    let table_pointer: *const *const *const () =
        std::mem::transmute((instance as *const u8).add(offset));
    unsafe { *table_pointer }
}

/// Gets the virtual function `index` at the vtable that is `offset` bytes from `instance`.
///
/// # Safety
/// * instance must be a valid pointer.
/// * `*(instance + offset)` must be a valid pointer.
///
pub unsafe fn virtual_function<T>(instance: *const T, offset: usize, index: usize) -> *const () {
    *(virtual_table(instance as *const (), offset).add(index))
}

#[repr(C)]
pub struct RTTICompleteObjectLocator {
    signature: u32,
    offset: u32,
    constructor_displacement_offset: u32,
    descriptor_offset: u32,
    class_descriptor_offset: u32,
    self_offset: u32,
}

impl RTTICompleteObjectLocator {
    pub fn type_descriptor(&self) -> &TypeDescriptor {
        let image_base = (self as *const _ as usize).saturating_sub(self.self_offset as usize);
        let descriptor_address = image_base + self.descriptor_offset as usize;

        unsafe { std::mem::transmute(descriptor_address) }
    }
}

#[repr(C)]
pub struct TypeDescriptor {
    vtable: *const *const (),
    spare: *const (),
    name_bytes: i8,
}

impl TypeDescriptor {
    pub fn name(&self) -> &CStr {
        unsafe { CStr::from_ptr(&self.name_bytes as *const _) }
    }
}

/// Get the type_info pointer from an instance, assuming the vtable is at 0 bytes from instance.
///
/// # Safety
/// * `instance` must be a valid pointer.
/// * `*(instance)` must be a valid vtable.
///
pub unsafe fn type_info<T>(instance: *const T) -> &'static RTTICompleteObjectLocator {
    let table = virtual_table(instance as *const (), 0);
    let type_info_ptr = table.offset(-1) as *const *const RTTICompleteObjectLocator;
    &(**type_info_ptr)
}
