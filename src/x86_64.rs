use std::ffi::CStr;

pub unsafe fn virtual_table(instance: *const (), offset: usize) -> *const *const () {
    let table_pointer: *const *const *const () =
        std::mem::transmute((instance as *const u8).offset(offset as isize));
    unsafe { *table_pointer }
}

pub fn virtual_function<T>(instance: *const T, offset: usize, index: usize) -> *const () {
    unsafe {
        std::mem::transmute(*(virtual_table(instance as *const (), offset).offset(index as isize)))
    }
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

pub fn type_info<T>(instance: *const T) -> &'static RTTICompleteObjectLocator {
    unsafe {
        let table = virtual_table(instance as *const (), 0);
        std::mem::transmute(*table.offset(-1))
    }
}
