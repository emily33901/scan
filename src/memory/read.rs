//! Bounded copied reads built on the process-memory backends.

use std::collections::TryReserveError;

use thiserror::Error;

use super::{
    Access, Address, DirectMemoryAccess, DirectMemoryScope, FaultContainedMemoryAccess,
    MemoryError, MemoryMap,
};

/// Caller-controlled bounds for an allocating memory read.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReadLimits {
    /// Largest owned buffer a read may allocate.
    pub max_allocation: usize,
    /// Largest number of mappings a read may traverse.
    pub max_regions: usize,
}

impl ReadLimits {
    pub const fn new(max_allocation: usize, max_regions: usize) -> Self {
        Self {
            max_allocation,
            max_regions,
        }
    }
}

/// Failure from a bounded read which may allocate owned storage.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ReadError {
    #[error(transparent)]
    Memory(#[from] MemoryError),
    #[error("failed to reserve {requested} bytes for a copied memory read")]
    Allocation {
        requested: usize,
        #[source]
        source: TryReserveError,
    },
}

fn validate_allocation(requested: usize, limits: ReadLimits) -> Result<(), MemoryError> {
    if requested == 0 {
        return Err(MemoryError::ZeroLength);
    }
    if requested > limits.max_allocation {
        return Err(MemoryError::LengthExceeded {
            requested,
            limit: limits.max_allocation,
        });
    }
    if limits.max_regions == 0 {
        return Err(MemoryError::RegionLimitExceeded { limit: 0 });
    }
    Ok(())
}

fn allocate_zeroed(len: usize) -> Result<Vec<u8>, ReadError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|source| ReadError::Allocation {
            requested: len,
            source,
        })?;
    bytes.resize(len, 0);
    Ok(bytes)
}

fn read_owned(
    map: &(impl MemoryMap + ?Sized),
    address: Address,
    len: usize,
    limits: ReadLimits,
    mut copy: impl FnMut(Address, &mut [u8]) -> Result<(), MemoryError>,
) -> Result<Vec<u8>, ReadError> {
    validate_allocation(len, limits)?;
    let range = address.checked_range(len)?;
    map.check_range(range, Access::READ, limits.max_regions)?;
    let mut bytes = allocate_zeroed(len)?;
    copy(address, &mut bytes)?;
    Ok(bytes)
}

fn read_c_string(
    map: &(impl MemoryMap + ?Sized),
    address: Address,
    max_len: usize,
    limits: ReadLimits,
    mut copy: impl FnMut(Address, &mut [u8]) -> Result<(), MemoryError>,
) -> Result<Vec<u8>, ReadError> {
    validate_allocation(max_len, limits)?;
    address.checked_range(max_len)?;

    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(max_len)
        .map_err(|source| ReadError::Allocation {
            requested: max_len,
            source,
        })?;

    let mut cursor = address;
    let mut remaining = max_len;
    for _ in 0..limits.max_regions {
        let region = map.region_containing(cursor)?;
        if !region.contains(cursor) || region.range.end() <= cursor {
            return Err(MemoryError::NonProgressingRegion { address: cursor }.into());
        }
        if !region.access.contains(Access::READ) {
            return Err(MemoryError::PermissionDenied {
                address: cursor,
                required: Access::READ,
                actual: region.access,
            }
            .into());
        }

        let available = region.range.end().get() - cursor.get();
        let chunk_len = available.min(remaining);
        let old_len = bytes.len();
        bytes.resize(old_len + chunk_len, 0);
        copy(cursor, &mut bytes[old_len..])?;

        if let Some(index) = bytes[old_len..].iter().position(|byte| *byte == 0) {
            bytes.truncate(old_len + index);
            return Ok(bytes);
        }

        remaining -= chunk_len;
        if remaining == 0 {
            return Err(MemoryError::UnterminatedString { max_len }.into());
        }
        cursor = region.range.end();
    }

    Err(MemoryError::RegionLimitExceeded {
        limit: limits.max_regions,
    }
    .into())
}

macro_rules! direct_native_methods {
    ($(($name:ident, $ty:ty)),+ $(,)?) => {
        $(
            #[doc = concat!("Copies a native-endian `", stringify!($ty), "` value.")]
            ///
            /// # Safety
            ///
            /// The caller must uphold [`DirectMemoryAccess::read_bytes_into`]
            /// for the complete value range.
            unsafe fn $name(&self, address: Address) -> Result<$ty, MemoryError> {
                // SAFETY: delegated to this method's caller.
                unsafe { super::DirectMemoryAccessExt::read_value::<$ty>(self, address) }
            }
        )+
    };
}

/// Allocating and scalar conveniences for a direct-copy backend.
pub trait DirectMemoryReadExt: DirectMemoryAccess {
    direct_native_methods!(
        (read_u8, u8),
        (read_u16, u16),
        (read_u32, u32),
        (read_u64, u64),
        (read_u128, u128),
        (read_usize, usize),
        (read_i8, i8),
        (read_i16, i16),
        (read_i32, i32),
        (read_i64, i64),
        (read_i128, i128),
        (read_isize, isize),
    );

    /// Copies a bounded range into owned storage.
    ///
    /// # Safety
    ///
    /// The caller must uphold [`DirectMemoryAccess::read_bytes_into`] for the
    /// complete range until the copy finishes.
    unsafe fn read_owned_bytes(
        &self,
        address: Address,
        len: usize,
        limits: ReadLimits,
    ) -> Result<Vec<u8>, ReadError> {
        read_owned(self, address, len, limits, |cursor, output| {
            // SAFETY: delegated to this method's caller.
            unsafe { self.read_bytes_into(cursor, output) }
        })
    }

    /// Copies a native-width pointer value without dereferencing it.
    ///
    /// # Safety
    ///
    /// The caller must uphold [`DirectMemoryAccess::read_bytes_into`] for a
    /// native `usize` at `address`.
    unsafe fn read_address(&self, address: Address) -> Result<Address, MemoryError> {
        // SAFETY: delegated to this method's caller.
        let value = unsafe { super::DirectMemoryAccessExt::read_value::<usize>(self, address)? };
        Ok(Address::new(value))
    }

    /// Copies a bounded, NUL-terminated byte string, excluding the NUL.
    ///
    /// # Safety
    ///
    /// The caller must uphold [`DirectMemoryAccess::read_bytes_into`] for each
    /// queried chunk until its copy finishes.
    unsafe fn read_c_string_bytes(
        &self,
        address: Address,
        max_len: usize,
        limits: ReadLimits,
    ) -> Result<Vec<u8>, ReadError> {
        read_c_string(self, address, max_len, limits, |cursor, output| {
            // SAFETY: delegated to this method's caller.
            unsafe { self.read_bytes_into(cursor, output) }
        })
    }
}

impl<M: DirectMemoryAccess + ?Sized> DirectMemoryReadExt for M {}

macro_rules! direct_native_reads {
    ($(($name:ident, $ty:ty)),+ $(,)?) => {
        impl<M: DirectMemoryAccess + ?Sized> DirectMemoryScope<'_, M> {
            $(
                #[doc = concat!("Copies a native-endian `", stringify!($ty), "` value.")]
                pub fn $name(&self, address: Address) -> Result<$ty, MemoryError> {
                    self.read_value::<$ty>(address)
                }
            )+
        }
    };
}

direct_native_reads!(
    (read_u8, u8),
    (read_u16, u16),
    (read_u32, u32),
    (read_u64, u64),
    (read_u128, u128),
    (read_usize, usize),
    (read_i8, i8),
    (read_i16, i16),
    (read_i32, i32),
    (read_i64, i64),
    (read_i128, i128),
    (read_isize, isize),
);

impl<M: DirectMemoryAccess + ?Sized> DirectMemoryScope<'_, M> {
    /// Copies a bounded range into owned storage.
    pub fn read_owned_bytes(
        &self,
        address: Address,
        len: usize,
        limits: ReadLimits,
    ) -> Result<Vec<u8>, ReadError> {
        read_owned(self.memory(), address, len, limits, |cursor, output| {
            self.read_bytes_into(cursor, output)
        })
    }

    /// Copies a native-width pointer value without dereferencing it.
    pub fn read_address(&self, address: Address) -> Result<Address, MemoryError> {
        self.read_usize(address).map(Address::new)
    }

    /// Copies a bounded, NUL-terminated byte string, excluding the NUL.
    pub fn read_c_string_bytes(
        &self,
        address: Address,
        max_len: usize,
        limits: ReadLimits,
    ) -> Result<Vec<u8>, ReadError> {
        read_c_string(self.memory(), address, max_len, limits, |cursor, output| {
            self.read_bytes_into(cursor, output)
        })
    }
}

macro_rules! fault_contained_native_methods {
    ($(($name:ident, $ty:ty)),+ $(,)?) => {
        $(
            #[doc = concat!("Copies a native-endian `", stringify!($ty), "` value.")]
            fn $name(&self, address: Address) -> Result<$ty, MemoryError> {
                super::FaultContainedMemoryAccessExt::try_read_value::<$ty>(self, address)
            }
        )+
    };
}

/// Allocating and scalar conveniences for a fault-contained backend.
pub trait FaultContainedMemoryReadExt: FaultContainedMemoryAccess {
    fault_contained_native_methods!(
        (try_read_u8, u8),
        (try_read_u16, u16),
        (try_read_u32, u32),
        (try_read_u64, u64),
        (try_read_u128, u128),
        (try_read_usize, usize),
        (try_read_i8, i8),
        (try_read_i16, i16),
        (try_read_i32, i32),
        (try_read_i64, i64),
        (try_read_i128, i128),
        (try_read_isize, isize),
    );

    fn try_read_owned_bytes(
        &self,
        address: Address,
        len: usize,
        limits: ReadLimits,
    ) -> Result<Vec<u8>, ReadError> {
        read_owned(self, address, len, limits, |cursor, output| {
            self.try_read_bytes_into(cursor, output)
        })
    }

    fn try_read_address(&self, address: Address) -> Result<Address, MemoryError> {
        super::FaultContainedMemoryAccessExt::try_read_value::<usize>(self, address)
            .map(Address::new)
    }

    fn try_read_c_string_bytes(
        &self,
        address: Address,
        max_len: usize,
        limits: ReadLimits,
    ) -> Result<Vec<u8>, ReadError> {
        read_c_string(self, address, max_len, limits, |cursor, output| {
            self.try_read_bytes_into(cursor, output)
        })
    }
}

impl<M: FaultContainedMemoryAccess + ?Sized> FaultContainedMemoryReadExt for M {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{AddressRange, MemoryRegion, RegionKind};

    struct SegmentedMemory {
        bytes: Vec<u8>,
        split: usize,
    }

    impl SegmentedMemory {
        fn new(bytes: &[u8], split: usize) -> Self {
            Self {
                bytes: bytes.to_vec(),
                split,
            }
        }
    }

    impl MemoryMap for SegmentedMemory {
        fn region_containing(&self, address: Address) -> Result<MemoryRegion, MemoryError> {
            let index = address.get();
            if index >= self.bytes.len() {
                return Err(MemoryError::Unmapped { address });
            }
            let (start, end) = if index < self.split {
                (0, self.split)
            } else {
                (self.split, self.bytes.len())
            };
            Ok(MemoryRegion {
                range: AddressRange::new(Address::new(start), end - start)?,
                access: Access::READ | Access::WRITE,
                kind: RegionKind::Private,
                module: None,
            })
        }
    }

    impl DirectMemoryAccess for SegmentedMemory {
        unsafe fn read_bytes_into(
            &self,
            address: Address,
            output: &mut [u8],
        ) -> Result<(), MemoryError> {
            let end = address.checked_add(output.len())?.get();
            let source = self
                .bytes
                .get(address.get()..end)
                .ok_or(MemoryError::PartialRead {
                    expected: output.len(),
                    actual: self.bytes.len().saturating_sub(address.get()),
                })?;
            output.copy_from_slice(source);
            Ok(())
        }

        unsafe fn write_bytes_from(
            &self,
            _address: Address,
            _bytes: &[u8],
        ) -> Result<(), MemoryError> {
            unreachable!("read tests do not write")
        }
    }

    unsafe impl FaultContainedMemoryAccess for SegmentedMemory {
        fn try_read_bytes_into(
            &self,
            address: Address,
            output: &mut [u8],
        ) -> Result<(), MemoryError> {
            // SAFETY: this fake implementation indexes owned storage.
            unsafe { self.read_bytes_into(address, output) }
        }

        unsafe fn try_write_bytes_from(
            &self,
            _address: Address,
            _bytes: &[u8],
        ) -> Result<(), MemoryError> {
            unreachable!("read tests do not write")
        }
    }

    const LIMITS: ReadLimits = ReadLimits::new(64, 4);

    #[test]
    fn reads_unaligned_native_scalar_and_null_address_value() {
        let mut bytes = vec![0xff];
        bytes.extend_from_slice(&0x1122_3344_u32.to_ne_bytes());
        bytes.extend_from_slice(&0usize.to_ne_bytes());
        let memory = SegmentedMemory::new(&bytes, bytes.len());

        // SAFETY: all fake storage remains allocated and readable.
        let scalar = unsafe {
            crate::memory::with_direct_memory_scope(&memory, |scope| {
                Ok::<_, MemoryError>((
                    scope.read_u32(Address::new(1))?,
                    scope.read_address(Address::new(5))?,
                ))
            })
        };
        let (scalar, pointer) = scalar.unwrap();
        assert_eq!(scalar, 0x1122_3344);
        assert_eq!(pointer, Address::NULL);
    }

    #[test]
    fn allocation_is_checked_before_copy() {
        let memory = SegmentedMemory::new(b"abcdef", 3);
        // SAFETY: fake storage is stable.
        let error = unsafe { memory.read_owned_bytes(Address::NULL, 6, ReadLimits::new(5, 2)) }
            .unwrap_err();
        assert!(matches!(
            error,
            ReadError::Memory(MemoryError::LengthExceeded {
                requested: 6,
                limit: 5
            })
        ));
    }

    #[test]
    fn c_string_can_end_at_or_cross_a_mapping_boundary() {
        let ending = SegmentedMemory::new(b"abc\0tail", 4);
        let crossing = SegmentedMemory::new(b"abcdef\0tail", 3);

        // SAFETY: fake storage is stable for both scoped operations.
        unsafe {
            crate::memory::with_direct_memory_scope(&ending, |scope| {
                assert_eq!(
                    scope.read_c_string_bytes(Address::NULL, 4, LIMITS).unwrap(),
                    b"abc"
                );
            });
            crate::memory::with_direct_memory_scope(&crossing, |scope| {
                assert_eq!(
                    scope.read_c_string_bytes(Address::NULL, 8, LIMITS).unwrap(),
                    b"abcdef"
                );
            });
        }
    }

    #[test]
    fn c_string_supports_empty_and_one_byte_values() {
        let memory = SegmentedMemory::new(b"\0x\0", 1);
        assert_eq!(
            memory
                .try_read_c_string_bytes(Address::NULL, 3, LIMITS)
                .unwrap(),
            b""
        );
        assert_eq!(
            memory
                .try_read_c_string_bytes(Address::new(1), 2, LIMITS)
                .unwrap(),
            b"x"
        );
    }

    #[test]
    fn c_string_reports_unterminated_and_region_limit() {
        let memory = SegmentedMemory::new(b"abcdef", 3);
        assert!(matches!(
            memory.try_read_c_string_bytes(Address::NULL, 6, LIMITS),
            Err(ReadError::Memory(MemoryError::UnterminatedString {
                max_len: 6
            }))
        ));
        assert!(matches!(
            memory.try_read_c_string_bytes(Address::NULL, 6, ReadLimits::new(6, 1)),
            Err(ReadError::Memory(MemoryError::RegionLimitExceeded {
                limit: 1
            }))
        ));
    }
}
