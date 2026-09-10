//! Checked helpers for x86-64 virtual tables and Microsoft RTTI.

use std::mem::size_of;

use thiserror::Error;

use crate::memory::{
    Address, AddressRange, DirectMemoryAccess, DirectMemoryAccessExt, DirectMemoryReadExt,
    DirectMemoryScope, FaultContainedMemoryAccess, FaultContainedMemoryAccessExt,
    FaultContainedMemoryReadExt, MemoryError, ReadError, ReadLimits,
};
use crate::module::{ImageFormat, LoadedModule};

/// Copied layout of an MSVC x64 complete-object locator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct RttiCompleteObjectLocator {
    pub signature: u32,
    pub object_offset: u32,
    pub constructor_displacement_offset: u32,
    pub descriptor_offset: i32,
    pub class_descriptor_offset: i32,
    pub self_offset: i32,
}

// SAFETY: six 32-bit integer fields have no padding and every bit pattern is
// valid for the corresponding signed or unsigned integer.
unsafe impl bytemuck::Zeroable for RttiCompleteObjectLocator {}
unsafe impl bytemuck::Pod for RttiCompleteObjectLocator {}

/// Copied fixed-size prefix of an MSVC `TypeDescriptor`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct TypeDescriptorHeader {
    pub vtable: usize,
    pub spare: usize,
}

// SAFETY: the two-field C layout has no padding and every bit pattern is valid.
unsafe impl bytemuck::Zeroable for TypeDescriptorHeader {}
unsafe impl bytemuck::Pod for TypeDescriptorHeader {}

/// Owned result of resolving MSVC x64 RTTI from an object.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RttiTypeInfo {
    pub vtable: Address,
    pub locator_address: Address,
    pub locator: RttiCompleteObjectLocator,
    pub image_base: Address,
    pub descriptor_address: Address,
    pub descriptor: TypeDescriptorHeader,
    /// Mangled type name bytes, excluding the terminating NUL.
    pub name: Vec<u8>,
}

/// Caller-selected bounds for RTTI parsing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RttiLimits {
    pub max_name_len: usize,
    pub max_regions: usize,
}

impl RttiLimits {
    pub const fn new(max_name_len: usize, max_regions: usize) -> Self {
        Self {
            max_name_len,
            max_regions,
        }
    }
}

/// Failure while copying or validating x86-64 vtable and RTTI metadata.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RttiError {
    #[error(transparent)]
    Memory(#[from] MemoryError),
    #[error(transparent)]
    Read(#[from] ReadError),
    #[error("{field} contains a null address")]
    NullPointer { field: &'static str },
    #[error("unsupported MSVC RTTI locator signature {actual}; expected 1")]
    UnsupportedSignature { actual: u32 },
    #[error("MSVC RTTI requires a PE image, but the selected module is {actual:?}")]
    UnsupportedImageFormat { actual: ImageFormat },
    #[error("RTTI-derived image base {actual} does not match selected image base {expected}")]
    ImageBaseMismatch { expected: Address, actual: Address },
    #[error("{field} range {range:?} is outside selected image {image:?}")]
    OutsideImage {
        field: &'static str,
        range: AddressRange,
        image: AddressRange,
    },
}

fn require_pointer(address: Address, field: &'static str) -> Result<Address, RttiError> {
    if address.is_null() {
        Err(RttiError::NullPointer { field })
    } else {
        Ok(address)
    }
}

fn vtable_slot_address(vtable: Address, index: usize) -> Result<Address, RttiError> {
    let offset = index
        .checked_mul(size_of::<usize>())
        .ok_or(MemoryError::AddressOverflow)?;
    Ok(vtable.checked_add(offset)?)
}

fn resolve_vtable(
    instance: Address,
    offset: usize,
    mut read_address: impl FnMut(Address) -> Result<Address, MemoryError>,
) -> Result<Address, RttiError> {
    require_pointer(instance, "object")?;
    let pointer_address = instance.checked_add(offset)?;
    require_pointer(read_address(pointer_address)?, "object vtable")
}

fn resolve_virtual_function(
    instance: Address,
    offset: usize,
    index: usize,
    mut read_address: impl FnMut(Address) -> Result<Address, MemoryError>,
) -> Result<Address, RttiError> {
    let vtable = resolve_vtable(instance, offset, &mut read_address)?;
    let slot = vtable_slot_address(vtable, index)?;
    require_pointer(read_address(slot)?, "virtual function")
}

fn resolve_type_info(
    module: &LoadedModule,
    instance: Address,
    limits: RttiLimits,
    mut read_address: impl FnMut(Address) -> Result<Address, MemoryError>,
    mut read_locator: impl FnMut(Address) -> Result<RttiCompleteObjectLocator, MemoryError>,
    mut read_descriptor: impl FnMut(Address) -> Result<TypeDescriptorHeader, MemoryError>,
    mut read_name: impl FnMut(Address, usize, ReadLimits) -> Result<Vec<u8>, ReadError>,
) -> Result<RttiTypeInfo, RttiError> {
    if module.format != ImageFormat::Pe {
        return Err(RttiError::UnsupportedImageFormat {
            actual: module.format,
        });
    }
    let vtable = resolve_vtable(instance, 0, &mut read_address)?;
    let locator_pointer = vtable.checked_sub(size_of::<usize>())?;
    require_image_range(
        module,
        locator_pointer,
        size_of::<usize>(),
        "RTTI locator pointer",
    )?;
    let locator_address = require_pointer(read_address(locator_pointer)?, "RTTI locator")?;
    require_image_range(
        module,
        locator_address,
        size_of::<RttiCompleteObjectLocator>(),
        "RTTI locator",
    )?;
    let locator = read_locator(locator_address)?;
    if locator.signature != 1 {
        return Err(RttiError::UnsupportedSignature {
            actual: locator.signature,
        });
    }

    let image_base = locator_address.checked_add_signed(-i64::from(locator.self_offset))?;
    if image_base != module.base() {
        return Err(RttiError::ImageBaseMismatch {
            expected: module.base(),
            actual: image_base,
        });
    }
    let descriptor_address = image_base.checked_add_signed(i64::from(locator.descriptor_offset))?;
    require_image_range(
        module,
        descriptor_address,
        size_of::<TypeDescriptorHeader>(),
        "RTTI type descriptor",
    )?;
    let descriptor = read_descriptor(descriptor_address)?;
    let name_address = descriptor_address.checked_add(size_of::<TypeDescriptorHeader>())?;
    let bounded_name_len = contained_prefix_len(module, name_address, limits.max_name_len)?;
    let name_limits = ReadLimits::new(bounded_name_len, limits.max_regions);
    let name = read_name(name_address, bounded_name_len, name_limits)?;

    Ok(RttiTypeInfo {
        vtable,
        locator_address,
        locator,
        image_base,
        descriptor_address,
        descriptor,
        name,
    })
}

fn contained_prefix_len(
    module: &LoadedModule,
    address: Address,
    limit: usize,
) -> Result<usize, RttiError> {
    if limit == 0 {
        return Ok(0);
    }
    require_image_range(module, address, 1, "RTTI type name")?;

    let mut cursor = address;
    let mut length = 0usize;
    for _ in 0..module.mapped_ranges.len() {
        let Some(mapped) = module
            .mapped_ranges
            .iter()
            .copied()
            .find(|range| range.contains(cursor))
        else {
            break;
        };
        let available = mapped.end().get() - cursor.get();
        let count = available.min(limit - length);
        length += count;
        if length == limit {
            break;
        }
        cursor = mapped.end();
    }
    Ok(length)
}

fn require_image_range(
    module: &LoadedModule,
    address: Address,
    len: usize,
    field: &'static str,
) -> Result<AddressRange, RttiError> {
    let range = address.checked_range(len)?;
    if module.contains_range(range) {
        Ok(range)
    } else {
        Err(RttiError::OutsideImage {
            field,
            range,
            image: module.image,
        })
    }
}

/// Copies an object's vtable pointer through a direct backend.
///
/// # Safety
/// The direct-copy contract must hold for the pointer until copying completes.
pub unsafe fn read_virtual_table_direct<M: DirectMemoryAccess + ?Sized>(
    memory: &M,
    instance: Address,
    offset: usize,
) -> Result<Address, RttiError> {
    resolve_vtable(instance, offset, |address| {
        // SAFETY: delegated to this function's caller.
        unsafe { memory.read_address(address) }
    })
}

/// Copies an object's vtable pointer within an established direct scope.
pub fn read_virtual_table_scoped<M: DirectMemoryAccess + ?Sized>(
    scope: &DirectMemoryScope<'_, M>,
    instance: Address,
    offset: usize,
) -> Result<Address, RttiError> {
    resolve_vtable(instance, offset, |address| scope.read_address(address))
}

/// Copies an object's vtable pointer through a fault-contained backend.
pub fn try_read_virtual_table<M: FaultContainedMemoryAccess + ?Sized>(
    memory: &M,
    instance: Address,
    offset: usize,
) -> Result<Address, RttiError> {
    resolve_vtable(instance, offset, |address| memory.try_read_address(address))
}

/// Copies a virtual-function address through a direct backend.
///
/// # Safety
/// The direct-copy contract must hold for the object pointer and table slot.
pub unsafe fn read_virtual_function_address_direct<M: DirectMemoryAccess + ?Sized>(
    memory: &M,
    instance: Address,
    offset: usize,
    index: usize,
) -> Result<Address, RttiError> {
    resolve_virtual_function(instance, offset, index, |address| {
        // SAFETY: delegated to this function's caller.
        unsafe { memory.read_address(address) }
    })
}

/// Copies a virtual-function address within an established direct scope.
pub fn read_virtual_function_address_scoped<M: DirectMemoryAccess + ?Sized>(
    scope: &DirectMemoryScope<'_, M>,
    instance: Address,
    offset: usize,
    index: usize,
) -> Result<Address, RttiError> {
    resolve_virtual_function(instance, offset, index, |address| {
        scope.read_address(address)
    })
}

/// Copies a virtual-function address through a fault-contained backend.
pub fn try_read_virtual_function_address<M: FaultContainedMemoryAccess + ?Sized>(
    memory: &M,
    instance: Address,
    offset: usize,
    index: usize,
) -> Result<Address, RttiError> {
    resolve_virtual_function(instance, offset, index, |address| {
        memory.try_read_address(address)
    })
}

/// Resolves copied, bounded MSVC x64 RTTI through a direct backend.
///
/// # Safety
/// The direct-copy contract must hold for every visited RTTI range.
pub unsafe fn read_type_info_direct<M: DirectMemoryAccess + ?Sized>(
    memory: &M,
    module: &LoadedModule,
    instance: Address,
    limits: RttiLimits,
) -> Result<RttiTypeInfo, RttiError> {
    resolve_type_info(
        module,
        instance,
        limits,
        |address| unsafe { memory.read_address(address) },
        |address| unsafe { memory.read_value(address) },
        |address| unsafe { memory.read_value(address) },
        |address, max_len, read_limits| unsafe {
            memory.read_c_string_bytes(address, max_len, read_limits)
        },
    )
}

/// Resolves copied, bounded MSVC x64 RTTI in an established direct scope.
pub fn read_type_info_scoped<M: DirectMemoryAccess + ?Sized>(
    scope: &DirectMemoryScope<'_, M>,
    module: &LoadedModule,
    instance: Address,
    limits: RttiLimits,
) -> Result<RttiTypeInfo, RttiError> {
    resolve_type_info(
        module,
        instance,
        limits,
        |address| scope.read_address(address),
        |address| scope.read_value(address),
        |address| scope.read_value(address),
        |address, max_len, read_limits| scope.read_c_string_bytes(address, max_len, read_limits),
    )
}

/// Resolves copied, bounded MSVC x64 RTTI through a fault-contained backend.
pub fn try_read_type_info<M: FaultContainedMemoryAccess + ?Sized>(
    memory: &M,
    module: &LoadedModule,
    instance: Address,
    limits: RttiLimits,
) -> Result<RttiTypeInfo, RttiError> {
    resolve_type_info(
        module,
        instance,
        limits,
        |address| memory.try_read_address(address),
        |address| memory.try_read_value(address),
        |address| memory.try_read_value(address),
        |address, max_len, read_limits| {
            memory.try_read_c_string_bytes(address, max_len, read_limits)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::testing::FakeMemory;
    use crate::memory::{with_direct_memory_scope, Access, ModuleId, RegionKind};
    use crate::module::Section;
    use std::path::PathBuf;

    const BASE: usize = 0x1000;
    const INSTANCE: usize = 0x1010;
    const VTABLE: usize = 0x1108;
    const LOCATOR: usize = 0x1200;
    const DESCRIPTOR: usize = 0x1300;
    const LIMITS: RttiLimits = RttiLimits::new(64, 4);

    fn put<T: bytemuck::NoUninit>(image: &mut [u8], address: usize, value: &T) {
        let offset = address - BASE;
        let bytes = bytemuck::bytes_of(value);
        image[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    fn fixture_module() -> LoadedModule {
        LoadedModule {
            id: ModuleId(1),
            path: PathBuf::from("fixture.exe"),
            image: Address::new(BASE).checked_range(0x400).unwrap(),
            mapped_ranges: vec![Address::new(BASE).checked_range(0x400).unwrap()],
            format: ImageFormat::Pe,
            sections: Vec::<Section>::new(),
        }
    }

    fn fixture(name: &[u8], signature: u32) -> (FakeMemory, LoadedModule) {
        let mut image = vec![0_u8; 0x400];
        put(&mut image, INSTANCE, &VTABLE);
        put(&mut image, VTABLE - size_of::<usize>(), &LOCATOR);
        put(
            &mut image,
            LOCATOR,
            &RttiCompleteObjectLocator {
                signature,
                object_offset: 0,
                constructor_displacement_offset: 0,
                descriptor_offset: (DESCRIPTOR - BASE) as i32,
                class_descriptor_offset: 0x380,
                self_offset: (LOCATOR - BASE) as i32,
            },
        );
        put(
            &mut image,
            DESCRIPTOR,
            &TypeDescriptorHeader {
                vtable: 0xfeed,
                spare: 0,
            },
        );
        let name_offset = DESCRIPTOR - BASE + size_of::<TypeDescriptorHeader>();
        image[name_offset..name_offset + name.len()].copy_from_slice(name);

        let memory = FakeMemory::new()
            .with_region(
                Address::new(BASE),
                image,
                Access::READ,
                RegionKind::Image,
                None,
            )
            .unwrap();
        (memory, fixture_module())
    }

    #[test]
    fn scoped_and_fault_contained_rtti_are_owned_and_bounded() {
        assert_eq!(size_of::<RttiCompleteObjectLocator>(), 24);
        assert_eq!(size_of::<TypeDescriptorHeader>(), 2 * size_of::<usize>());
        let (memory, module) = fixture(b".?AVWidget@@\0", 1);
        // SAFETY: the fake backend's storage and mappings are stable.
        let scoped = unsafe {
            with_direct_memory_scope(&memory, |scope| {
                read_type_info_scoped(scope, &module, Address::new(INSTANCE), LIMITS)
            })
        }
        .unwrap();
        let contained =
            try_read_type_info(&memory, &module, Address::new(INSTANCE), LIMITS).unwrap();

        assert_eq!(scoped, contained);
        assert_eq!(scoped.image_base, Address::new(BASE));
        assert_eq!(scoped.descriptor_address, Address::new(DESCRIPTOR));
        assert_eq!(scoped.name, b".?AVWidget@@");
    }

    #[test]
    fn direct_vtable_helpers_copy_addresses_and_check_arithmetic() {
        let (memory, _) = fixture(b"X\0", 1);
        let vtable =
            unsafe { read_virtual_table_direct(&memory, Address::new(INSTANCE), 0).unwrap() };
        assert_eq!(vtable, Address::new(VTABLE));
        assert!(matches!(
            unsafe {
                read_virtual_function_address_direct(&memory, Address::new(INSTANCE), 0, usize::MAX)
            },
            Err(RttiError::Memory(MemoryError::AddressOverflow))
        ));
    }

    #[test]
    fn rejects_bad_signature_and_unterminated_name() {
        let (bad_signature, module) = fixture(b"X\0", 0);
        assert!(matches!(
            try_read_type_info(&bad_signature, &module, Address::new(INSTANCE), LIMITS),
            Err(RttiError::UnsupportedSignature { actual: 0 })
        ));

        let (unterminated, module) = fixture(b"abcdef", 1);
        assert!(matches!(
            try_read_type_info(
                &unterminated,
                &module,
                Address::new(INSTANCE),
                RttiLimits::new(6, 4),
            ),
            Err(RttiError::Read(ReadError::Memory(
                MemoryError::UnterminatedString { max_len: 6 }
            )))
        ));
    }

    #[test]
    fn rejects_null_vtable_and_invalid_image_relative_math() {
        let null = FakeMemory::new()
            .with_region(
                Address::new(INSTANCE),
                0usize.to_ne_bytes(),
                Access::READ,
                RegionKind::Private,
                None,
            )
            .unwrap();
        assert!(matches!(
            try_read_type_info(&null, &fixture_module(), Address::NULL, LIMITS),
            Err(RttiError::NullPointer { field: "object" })
        ));
        assert!(matches!(
            try_read_type_info(&null, &fixture_module(), Address::new(INSTANCE), LIMITS,),
            Err(RttiError::NullPointer {
                field: "object vtable"
            })
        ));

        let (mut invalid, module) = fixture(b"X\0", 1);
        let bad_locator = RttiCompleteObjectLocator {
            signature: 1,
            object_offset: 0,
            constructor_displacement_offset: 0,
            descriptor_offset: 0,
            class_descriptor_offset: 0,
            self_offset: i32::MAX,
        };
        let mut image = invalid.snapshot(Address::new(BASE), 0x400).unwrap();
        put(&mut image, LOCATOR, &bad_locator);
        invalid = FakeMemory::new()
            .with_region(
                Address::new(BASE),
                image,
                Access::READ,
                RegionKind::Image,
                None,
            )
            .unwrap();
        assert!(matches!(
            try_read_type_info(&invalid, &module, Address::new(INSTANCE), LIMITS),
            Err(RttiError::Memory(MemoryError::AddressOverflow))
        ));
    }
}
