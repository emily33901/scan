//! Validated, non-borrowing views of foreign contiguous containers.

use std::mem::{align_of, size_of};
use std::ptr::NonNull;

use thiserror::Error;

use super::{
    Access, Address, AddressRange, DirectMemoryAccess, DirectMemoryAccessExt, DirectMemoryScope,
    FaultContainedMemoryAccess, FaultContainedMemoryAccessExt, ForeignMut, ForeignRef, MemoryError,
    MemoryMap, DEFAULT_MAX_REGIONS,
};

/// Accepted representations of an empty foreign vector.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EmptySpanPolicy {
    /// Accept either all-null metadata or equal, non-null begin/end pointers.
    #[default]
    AllowNullOrNonNull,
    /// Require every present pointer field to be null.
    RequireNull,
    /// Require non-null begin/end pointers and every present pointer non-null.
    RequireNonNull,
}

/// Describes the pointer fields and element contract of a foreign vector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ForeignVectorLayout {
    pub begin_offset: i64,
    pub end_offset: i64,
    pub capacity_offset: Option<i64>,
    pub element_size: usize,
    pub element_alignment: usize,
    pub max_elements: usize,
    pub empty_policy: EmptySpanPolicy,
}

/// A validated snapshot of foreign vector metadata.
///
/// This type deliberately does not borrow or confer reference validity on the
/// element storage. Reconstruct it to refresh the container metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ForeignSpan {
    begin: Address,
    end: Address,
    capacity: Option<Address>,
    len: usize,
    byte_len: usize,
    element_size: usize,
    element_alignment: usize,
    access: Access,
}

impl ForeignSpan {
    /// Copies and validates vector metadata using direct process access.
    ///
    /// # Safety
    ///
    /// The caller must uphold the direct backend's validity and synchronization
    /// requirements for every metadata read and for the element-range preflight.
    pub unsafe fn from_direct<M: DirectMemoryAccess + ?Sized>(
        memory: &M,
        vector: Address,
        layout: ForeignVectorLayout,
        access: Access,
    ) -> Result<Self, ForeignSpanError> {
        let begin = unsafe { read_pointer_direct(memory, vector, layout.begin_offset, "begin")? };
        let end = unsafe { read_pointer_direct(memory, vector, layout.end_offset, "end")? };
        let capacity = match layout.capacity_offset {
            Some(offset) => {
                Some(unsafe { read_pointer_direct(memory, vector, offset, "capacity")? })
            }
            None => None,
        };
        Self::validate(memory, begin, end, capacity, layout, access)
    }

    /// Copies and validates vector metadata under an established direct scope.
    pub fn from_scope<M: DirectMemoryAccess + ?Sized>(
        scope: &DirectMemoryScope<'_, M>,
        vector: Address,
        layout: ForeignVectorLayout,
        access: Access,
    ) -> Result<Self, ForeignSpanError> {
        let begin = read_pointer_scoped(scope, vector, layout.begin_offset, "begin")?;
        let end = read_pointer_scoped(scope, vector, layout.end_offset, "end")?;
        let capacity = match layout.capacity_offset {
            Some(offset) => Some(read_pointer_scoped(scope, vector, offset, "capacity")?),
            None => None,
        };
        Self::validate(scope.memory(), begin, end, capacity, layout, access)
    }

    /// Copies and validates vector metadata through a fault-contained backend.
    pub fn from_fault_contained<M: FaultContainedMemoryAccess + ?Sized>(
        memory: &M,
        vector: Address,
        layout: ForeignVectorLayout,
        access: Access,
    ) -> Result<Self, ForeignSpanError> {
        let begin = read_pointer_fault_contained(memory, vector, layout.begin_offset, "begin")?;
        let end = read_pointer_fault_contained(memory, vector, layout.end_offset, "end")?;
        let capacity = match layout.capacity_offset {
            Some(offset) => Some(read_pointer_fault_contained(
                memory, vector, offset, "capacity",
            )?),
            None => None,
        };
        Self::validate(memory, begin, end, capacity, layout, access)
    }

    fn validate<M: MemoryMap + ?Sized>(
        memory: &M,
        begin: Address,
        end: Address,
        capacity: Option<Address>,
        layout: ForeignVectorLayout,
        access: Access,
    ) -> Result<Self, ForeignSpanError> {
        validate_layout(layout)?;

        let pointers_are_all_null =
            begin.is_null() && end.is_null() && capacity.map(Address::is_null).unwrap_or(true);
        let any_pointer_is_null =
            begin.is_null() || end.is_null() || capacity.map(Address::is_null).unwrap_or(false);

        if any_pointer_is_null && !pointers_are_all_null {
            return Err(ForeignSpanError::MixedNullMetadata);
        }
        if pointers_are_all_null {
            if layout.empty_policy == EmptySpanPolicy::RequireNonNull {
                return Err(ForeignSpanError::EmptyPolicyViolation {
                    policy: layout.empty_policy,
                });
            }
            return Ok(Self::empty(begin, end, capacity, layout, access));
        }
        if layout.empty_policy == EmptySpanPolicy::RequireNull && begin == end {
            return Err(ForeignSpanError::EmptyPolicyViolation {
                policy: layout.empty_policy,
            });
        }

        let byte_len = end
            .get()
            .checked_sub(begin.get())
            .ok_or(ForeignSpanError::ReversedPointers { begin, end })?;
        if let Some(capacity) = capacity {
            if capacity < end {
                return Err(ForeignSpanError::CapacityBeforeEnd { end, capacity });
            }
            let capacity_bytes = capacity
                .get()
                .checked_sub(begin.get())
                .ok_or(ForeignSpanError::CapacityBeforeEnd { end, capacity })?;
            if capacity_bytes % layout.element_size != 0 {
                return Err(ForeignSpanError::IndivisibleCapacity {
                    bytes: capacity_bytes,
                    element_size: layout.element_size,
                });
            }
            let capacity_elements = capacity_bytes / layout.element_size;
            if capacity_elements > layout.max_elements {
                return Err(ForeignSpanError::ElementLimitExceeded {
                    elements: capacity_elements,
                    limit: layout.max_elements,
                });
            }
        }
        if byte_len % layout.element_size != 0 {
            return Err(ForeignSpanError::IndivisibleLength {
                bytes: byte_len,
                element_size: layout.element_size,
            });
        }
        begin
            .require_alignment(layout.element_alignment)
            .map_err(ForeignSpanError::Memory)?;
        end.require_alignment(layout.element_alignment)
            .map_err(ForeignSpanError::Memory)?;
        if let Some(capacity) = capacity {
            capacity
                .require_alignment(layout.element_alignment)
                .map_err(ForeignSpanError::Memory)?;
        }
        if byte_len > isize::MAX as usize {
            return Err(ForeignSpanError::ByteLengthExceeded {
                bytes: byte_len,
                limit: isize::MAX as usize,
            });
        }
        let len = byte_len / layout.element_size;
        if len > layout.max_elements {
            return Err(ForeignSpanError::ElementLimitExceeded {
                elements: len,
                limit: layout.max_elements,
            });
        }

        if byte_len != 0 {
            let range = begin
                .checked_range(byte_len)
                .map_err(ForeignSpanError::Memory)?;
            memory
                .check_range(range, access, DEFAULT_MAX_REGIONS)
                .map_err(ForeignSpanError::Memory)?;
        }

        Ok(Self {
            begin,
            end,
            capacity,
            len,
            byte_len,
            element_size: layout.element_size,
            element_alignment: layout.element_alignment,
            access,
        })
    }

    const fn empty(
        begin: Address,
        end: Address,
        capacity: Option<Address>,
        layout: ForeignVectorLayout,
        access: Access,
    ) -> Self {
        Self {
            begin,
            end,
            capacity,
            len: 0,
            byte_len: 0,
            element_size: layout.element_size,
            element_alignment: layout.element_alignment,
            access,
        }
    }

    pub const fn len(self) -> usize {
        self.len
    }

    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    pub const fn byte_len(self) -> usize {
        self.byte_len
    }

    pub const fn access(self) -> Access {
        self.access
    }

    pub fn range(self) -> Option<AddressRange> {
        if self.is_empty() {
            None
        } else {
            AddressRange::new(self.begin, self.byte_len).ok()
        }
    }

    /// Copies one element using direct process access.
    ///
    /// # Safety
    ///
    /// The caller must uphold the direct backend contract for the element.
    pub unsafe fn read_direct<T: bytemuck::AnyBitPattern, M: DirectMemoryAccess + ?Sized>(
        self,
        memory: &M,
        index: usize,
    ) -> Result<T, ForeignSpanError> {
        let address = self.element_address::<T>(index, Access::READ)?;
        unsafe { memory.read_value(address) }.map_err(ForeignSpanError::Memory)
    }

    /// Copies one element under an established direct scope.
    pub fn read_scoped<T: bytemuck::AnyBitPattern, M: DirectMemoryAccess + ?Sized>(
        self,
        scope: &DirectMemoryScope<'_, M>,
        index: usize,
    ) -> Result<T, ForeignSpanError> {
        let address = self.element_address::<T>(index, Access::READ)?;
        scope.read_value(address).map_err(ForeignSpanError::Memory)
    }

    /// Copies one element through a fault-contained backend.
    pub fn read_fault_contained<
        T: bytemuck::AnyBitPattern,
        M: FaultContainedMemoryAccess + ?Sized,
    >(
        self,
        memory: &M,
        index: usize,
    ) -> Result<T, ForeignSpanError> {
        let address = self.element_address::<T>(index, Access::READ)?;
        memory
            .try_read_value(address)
            .map_err(ForeignSpanError::Memory)
    }

    /// Copies one element into foreign storage using direct access.
    ///
    /// # Safety
    ///
    /// Replacing the element bytes must preserve host program invariants.
    pub unsafe fn write_direct<T: bytemuck::NoUninit, M: DirectMemoryAccess + ?Sized>(
        self,
        memory: &M,
        index: usize,
        value: &T,
    ) -> Result<(), ForeignSpanError> {
        let address = self.element_address::<T>(index, Access::WRITE)?;
        unsafe { memory.write_value(address, value) }.map_err(ForeignSpanError::Memory)
    }

    /// Copies one element into foreign storage under a direct scope.
    ///
    /// # Safety
    ///
    /// Replacing the element bytes must preserve host program invariants.
    pub unsafe fn write_scoped<T: bytemuck::NoUninit, M: DirectMemoryAccess + ?Sized>(
        self,
        scope: &mut DirectMemoryScope<'_, M>,
        index: usize,
        value: &T,
    ) -> Result<(), ForeignSpanError> {
        let address = self.element_address::<T>(index, Access::WRITE)?;
        let bytes = bytemuck::bytes_of(value);
        unsafe { scope.write_bytes_from(address, bytes) }.map_err(ForeignSpanError::Memory)
    }

    /// Copies one element through a fault-contained backend.
    ///
    /// # Safety
    ///
    /// Replacing the element bytes must preserve host program invariants.
    pub unsafe fn write_fault_contained<
        T: bytemuck::NoUninit,
        M: FaultContainedMemoryAccess + ?Sized,
    >(
        self,
        memory: &M,
        index: usize,
        value: &T,
    ) -> Result<(), ForeignSpanError> {
        let address = self.element_address::<T>(index, Access::WRITE)?;
        unsafe { memory.try_write_value(address, value) }.map_err(ForeignSpanError::Memory)
    }

    /// Performs one exact direct transfer for the entire span.
    ///
    /// # Safety
    ///
    /// Replacing all element bytes must preserve host program invariants.
    pub unsafe fn write_all_direct<M: DirectMemoryAccess + ?Sized>(
        self,
        memory: &M,
        bytes: &[u8],
    ) -> Result<(), ForeignSpanError> {
        self.check_bulk_write(bytes)?;
        if bytes.is_empty() {
            return Ok(());
        }
        unsafe { memory.write_bytes_from(self.begin, bytes) }.map_err(ForeignSpanError::Memory)
    }

    /// Performs one exact scoped transfer for the entire span.
    ///
    /// # Safety
    ///
    /// Replacing all element bytes must preserve host program invariants.
    pub unsafe fn write_all_scoped<M: DirectMemoryAccess + ?Sized>(
        self,
        scope: &mut DirectMemoryScope<'_, M>,
        bytes: &[u8],
    ) -> Result<(), ForeignSpanError> {
        self.check_bulk_write(bytes)?;
        if bytes.is_empty() {
            return Ok(());
        }
        unsafe { scope.write_bytes_from(self.begin, bytes) }.map_err(ForeignSpanError::Memory)
    }

    /// Performs one exact fault-contained transfer for the entire span.
    ///
    /// # Safety
    ///
    /// Replacing all element bytes must preserve host program invariants.
    pub unsafe fn write_all_fault_contained<M: FaultContainedMemoryAccess + ?Sized>(
        self,
        memory: &M,
        bytes: &[u8],
    ) -> Result<(), ForeignSpanError> {
        self.check_bulk_write(bytes)?;
        if bytes.is_empty() {
            return Ok(());
        }
        unsafe { memory.try_write_bytes_from(self.begin, bytes) }.map_err(ForeignSpanError::Memory)
    }

    /// Converts this snapshot into a shared slice capability.
    ///
    /// # Safety
    ///
    /// The element range must be one live allocation containing initialized,
    /// valid `T` values for `'scope`. It must remain immovable and readable,
    /// and ordinary shared-reference aliasing and synchronization rules apply.
    pub unsafe fn foreign_slice<'a, 'scope, T, M: DirectMemoryAccess + ?Sized>(
        self,
        scope: &'a DirectMemoryScope<'scope, M>,
    ) -> Result<ForeignRef<'a, [T]>, ForeignSpanError> {
        self.check_type::<T>()?;
        self.check_access(Access::READ)?;
        if let Some(range) = self.range() {
            scope
                .check_range(range, Access::READ, DEFAULT_MAX_REGIONS)
                .map_err(ForeignSpanError::Memory)?;
        }
        let pointer = self.slice_pointer::<T>();
        // SAFETY: the caller establishes allocation, initialization, lifetime,
        // and aliasing validity. Empty slices use a non-null aligned pointer.
        let slice = unsafe { std::slice::from_raw_parts(pointer.as_ptr(), self.len) };
        Ok(ForeignRef::from(slice))
    }

    /// Converts this snapshot into an exclusive slice capability.
    ///
    /// # Safety
    ///
    /// The element range must be one live allocation containing initialized,
    /// valid `T` values for `'scope`, with exclusive access for that lifetime.
    /// Ordinary mutable-reference move, drop, and replacement operations must
    /// also be valid for the host container.
    pub unsafe fn foreign_slice_mut<'a, 'scope, T, M: DirectMemoryAccess + ?Sized>(
        self,
        scope: &'a mut DirectMemoryScope<'scope, M>,
    ) -> Result<ForeignMut<'a, [T]>, ForeignSpanError> {
        self.check_type::<T>()?;
        self.check_access(Access::READ | Access::WRITE)?;
        if let Some(range) = self.range() {
            scope
                .check_range(range, Access::READ | Access::WRITE, DEFAULT_MAX_REGIONS)
                .map_err(ForeignSpanError::Memory)?;
        }
        let pointer = self.slice_pointer::<T>();
        // SAFETY: the caller establishes one allocation and exclusive access;
        // empty slices use a non-null aligned pointer.
        let slice = unsafe { std::slice::from_raw_parts_mut(pointer.as_ptr(), self.len) };
        Ok(ForeignMut::from(slice))
    }

    fn slice_pointer<T>(self) -> NonNull<T> {
        if self.is_empty() {
            NonNull::dangling()
        } else {
            // Validation rejects null for non-empty spans.
            unsafe { NonNull::new_unchecked(self.begin.get() as *mut T) }
        }
    }

    fn element_address<T>(
        self,
        index: usize,
        required: Access,
    ) -> Result<Address, ForeignSpanError> {
        self.check_type::<T>()?;
        self.check_access(required)?;
        if index >= self.len {
            return Err(ForeignSpanError::IndexOutOfBounds {
                index,
                len: self.len,
            });
        }
        let offset = index
            .checked_mul(self.element_size)
            .ok_or(ForeignSpanError::AddressOverflow)?;
        self.begin
            .checked_add(offset)
            .map_err(ForeignSpanError::Memory)
    }

    fn check_type<T>(self) -> Result<(), ForeignSpanError> {
        let actual_size = size_of::<T>();
        if actual_size == 0 || actual_size != self.element_size {
            return Err(ForeignSpanError::ElementSizeMismatch {
                declared: self.element_size,
                actual: actual_size,
            });
        }
        let actual_alignment = align_of::<T>();
        if self.element_alignment < actual_alignment
            || self.element_alignment % actual_alignment != 0
        {
            return Err(ForeignSpanError::ElementAlignmentMismatch {
                declared: self.element_alignment,
                required: actual_alignment,
            });
        }
        Ok(())
    }

    fn check_access(self, required: Access) -> Result<(), ForeignSpanError> {
        if self.access.contains(required) {
            Ok(())
        } else {
            Err(ForeignSpanError::AccessNotValidated {
                required,
                validated: self.access,
            })
        }
    }

    fn check_bulk_write(self, bytes: &[u8]) -> Result<(), ForeignSpanError> {
        if !self.access.contains(Access::WRITE) {
            return Err(ForeignSpanError::AccessNotValidated {
                required: Access::WRITE,
                validated: self.access,
            });
        }
        if bytes.len() != self.byte_len {
            return Err(ForeignSpanError::BulkLengthMismatch {
                expected: self.byte_len,
                actual: bytes.len(),
            });
        }
        if bytes.is_empty() {
            return Ok(());
        }
        Ok(())
    }
}

fn validate_layout(layout: ForeignVectorLayout) -> Result<(), ForeignSpanError> {
    if layout.element_size == 0 {
        return Err(ForeignSpanError::ZeroElementSize);
    }
    if layout.element_alignment == 0 || !layout.element_alignment.is_power_of_two() {
        return Err(ForeignSpanError::InvalidElementAlignment {
            alignment: layout.element_alignment,
        });
    }
    Ok(())
}

fn metadata_address(
    vector: Address,
    offset: i64,
    field: &'static str,
) -> Result<Address, ForeignSpanError> {
    vector
        .checked_add_signed(offset)
        .map_err(|source| ForeignSpanError::Metadata { field, source })
}

unsafe fn read_pointer_direct<M: DirectMemoryAccess + ?Sized>(
    memory: &M,
    vector: Address,
    offset: i64,
    field: &'static str,
) -> Result<Address, ForeignSpanError> {
    let address = metadata_address(vector, offset, field)?;
    unsafe { memory.read_value::<usize>(address) }
        .map(Address::new)
        .map_err(|source| ForeignSpanError::Metadata { field, source })
}

fn read_pointer_scoped<M: DirectMemoryAccess + ?Sized>(
    scope: &DirectMemoryScope<'_, M>,
    vector: Address,
    offset: i64,
    field: &'static str,
) -> Result<Address, ForeignSpanError> {
    let address = metadata_address(vector, offset, field)?;
    scope
        .read_value::<usize>(address)
        .map(Address::new)
        .map_err(|source| ForeignSpanError::Metadata { field, source })
}

fn read_pointer_fault_contained<M: FaultContainedMemoryAccess + ?Sized>(
    memory: &M,
    vector: Address,
    offset: i64,
    field: &'static str,
) -> Result<Address, ForeignSpanError> {
    let address = metadata_address(vector, offset, field)?;
    memory
        .try_read_value::<usize>(address)
        .map(Address::new)
        .map_err(|source| ForeignSpanError::Metadata { field, source })
}

/// Errors produced while validating or accessing a foreign span.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[non_exhaustive]
pub enum ForeignSpanError {
    #[error("failed to read foreign vector {field} metadata")]
    Metadata {
        field: &'static str,
        #[source]
        source: MemoryError,
    },
    #[error("foreign span memory validation failed")]
    Memory(#[source] MemoryError),
    #[error("foreign vector element size must be nonzero")]
    ZeroElementSize,
    #[error("foreign vector element alignment {alignment} is not a nonzero power of two")]
    InvalidElementAlignment { alignment: usize },
    #[error("foreign vector metadata mixes null and non-null pointers")]
    MixedNullMetadata,
    #[error("empty vector metadata violates {policy:?}")]
    EmptyPolicyViolation { policy: EmptySpanPolicy },
    #[error("foreign vector end {end} precedes begin {begin}")]
    ReversedPointers { begin: Address, end: Address },
    #[error("foreign vector capacity {capacity} precedes end {end}")]
    CapacityBeforeEnd { end: Address, capacity: Address },
    #[error("foreign vector length {bytes} is not divisible by element size {element_size}")]
    IndivisibleLength { bytes: usize, element_size: usize },
    #[error("foreign vector capacity {bytes} is not divisible by element size {element_size}")]
    IndivisibleCapacity { bytes: usize, element_size: usize },
    #[error("foreign span contains {elements} elements, exceeding limit {limit}")]
    ElementLimitExceeded { elements: usize, limit: usize },
    #[error("foreign span byte length {bytes} exceeds reference limit {limit}")]
    ByteLengthExceeded { bytes: usize, limit: usize },
    #[error("typed access size {actual} does not match declared element size {declared}")]
    ElementSizeMismatch { declared: usize, actual: usize },
    #[error("declared alignment {declared} does not satisfy typed alignment {required}")]
    ElementAlignmentMismatch { declared: usize, required: usize },
    #[error("element index {index} is outside span length {len}")]
    IndexOutOfBounds { index: usize, len: usize },
    #[error("span was validated for {validated:?}, not required access {required:?}")]
    AccessNotValidated { required: Access, validated: Access },
    #[error("bulk write length {actual} does not equal span byte length {expected}")]
    BulkLengthMismatch { expected: usize, actual: usize },
    #[error("element address arithmetic overflowed")]
    AddressOverflow,
}

/// Concise alias for [`ForeignSpanError`].
pub type SpanError = ForeignSpanError;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{MemoryRegion, RegionKind};

    struct StableMemory(Vec<MemoryRegion>);

    impl MemoryMap for StableMemory {
        fn region_containing(&self, address: Address) -> Result<MemoryRegion, MemoryError> {
            self.0
                .iter()
                .copied()
                .find(|region| region.contains(address))
                .ok_or(MemoryError::Unmapped { address })
        }
    }

    impl DirectMemoryAccess for StableMemory {
        unsafe fn read_bytes_into(
            &self,
            address: Address,
            output: &mut [u8],
        ) -> Result<(), MemoryError> {
            self.check_range(
                address.checked_range(output.len())?,
                Access::READ,
                DEFAULT_MAX_REGIONS,
            )?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    address.get() as *const u8,
                    output.as_mut_ptr(),
                    output.len(),
                )
            };
            Ok(())
        }

        unsafe fn write_bytes_from(
            &self,
            address: Address,
            bytes: &[u8],
        ) -> Result<(), MemoryError> {
            self.check_range(
                address.checked_range(bytes.len())?,
                Access::WRITE,
                DEFAULT_MAX_REGIONS,
            )?;
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), address.get() as *mut u8, bytes.len())
            };
            Ok(())
        }
    }

    unsafe impl FaultContainedMemoryAccess for StableMemory {
        fn try_read_bytes_into(
            &self,
            address: Address,
            output: &mut [u8],
        ) -> Result<(), MemoryError> {
            unsafe { self.read_bytes_into(address, output) }
        }

        unsafe fn try_write_bytes_from(
            &self,
            address: Address,
            bytes: &[u8],
        ) -> Result<(), MemoryError> {
            unsafe { self.write_bytes_from(address, bytes) }
        }
    }

    fn region<T>(value: &T, access: Access) -> MemoryRegion {
        let start = Address::new(value as *const T as usize);
        MemoryRegion {
            range: start.checked_range(size_of::<T>()).unwrap(),
            access,
            kind: RegionKind::Private,
            module: None,
        }
    }

    fn layout() -> ForeignVectorLayout {
        ForeignVectorLayout {
            begin_offset: 0,
            end_offset: size_of::<usize>() as i64,
            capacity_offset: Some((2 * size_of::<usize>()) as i64),
            element_size: size_of::<u32>(),
            element_alignment: align_of::<u32>(),
            max_elements: 4,
            empty_policy: EmptySpanPolicy::AllowNullOrNonNull,
        }
    }

    #[test]
    fn scoped_construction_and_indexed_reads_are_bounded() {
        let elements = [10_u32, 20, 30];
        let begin = elements.as_ptr() as usize;
        let metadata = [begin, begin + size_of::<u32>() * elements.len(), begin + 16];
        let layout = layout();
        let memory = StableMemory(vec![
            region(&metadata, Access::READ),
            region(&elements, Access::READ),
        ]);
        let vector = Address::new(metadata.as_ptr() as usize);
        unsafe {
            super::super::with_direct_memory_scope(&memory, |scope| {
                let span = ForeignSpan::from_scope(scope, vector, layout, Access::READ).unwrap();
                assert_eq!(span.len(), 3);
                assert_eq!(span.read_scoped::<u32, _>(scope, 2).unwrap(), 30);
                assert!(matches!(
                    span.read_scoped::<u32, _>(scope, 3),
                    Err(ForeignSpanError::IndexOutOfBounds { .. })
                ));
                assert!(matches!(
                    span.read_scoped::<u16, _>(scope, 0),
                    Err(ForeignSpanError::ElementSizeMismatch { .. })
                ));
            })
        }
    }

    #[test]
    fn empty_policy_and_null_combinations_are_explicit() {
        let metadata = [0_usize; 3];
        let memory = StableMemory(vec![region(&metadata, Access::READ)]);
        let vector = Address::new(metadata.as_ptr() as usize);
        let mut layout = layout();
        unsafe {
            super::super::with_direct_memory_scope(&memory, |scope| {
                let span = ForeignSpan::from_scope(scope, vector, layout, Access::READ).unwrap();
                assert!(span.is_empty());
                layout.empty_policy = EmptySpanPolicy::RequireNonNull;
                assert!(matches!(
                    ForeignSpan::from_scope(scope, vector, layout, Access::READ),
                    Err(ForeignSpanError::EmptyPolicyViolation { .. })
                ));
            })
        }

        let mixed = [0_usize, 4, 4];
        let mixed_memory = StableMemory(vec![region(&mixed, Access::READ)]);
        let mixed_vector = Address::new(mixed.as_ptr() as usize);
        layout.empty_policy = EmptySpanPolicy::AllowNullOrNonNull;
        let error =
            unsafe { ForeignSpan::from_direct(&mixed_memory, mixed_vector, layout, Access::READ) };
        assert!(matches!(error, Err(ForeignSpanError::MixedNullMetadata)));
    }

    #[test]
    fn ordering_divisibility_and_limits_are_rejected() {
        let elements = [1_u32, 2];
        let start = elements.as_ptr() as usize;
        let mut layout = layout();
        let cases = [
            ([start + 4, start, start + 8], "reversed"),
            ([start, start + 3, start + 8], "indivisible"),
            ([start, start + 8, start + 4], "capacity"),
        ];
        for (metadata, expected) in cases {
            let memory = StableMemory(vec![
                region(&metadata, Access::READ),
                region(&elements, Access::READ),
            ]);
            let result = unsafe {
                ForeignSpan::from_direct(
                    &memory,
                    Address::new(metadata.as_ptr() as usize),
                    layout,
                    Access::READ,
                )
            };
            match expected {
                "reversed" => assert!(matches!(
                    result,
                    Err(ForeignSpanError::ReversedPointers { .. })
                )),
                "indivisible" => assert!(matches!(
                    result,
                    Err(ForeignSpanError::IndivisibleLength { .. })
                )),
                _ => assert!(matches!(
                    result,
                    Err(ForeignSpanError::CapacityBeforeEnd { .. })
                )),
            }
        }

        layout.max_elements = 1;
        let metadata = [start, start + 8, start + 8];
        let memory = StableMemory(vec![
            region(&metadata, Access::READ),
            region(&elements, Access::READ),
        ]);
        let result = unsafe {
            ForeignSpan::from_direct(
                &memory,
                Address::new(metadata.as_ptr() as usize),
                layout,
                Access::READ,
            )
        };
        assert!(matches!(
            result,
            Err(ForeignSpanError::ElementLimitExceeded { .. })
        ));
    }

    #[test]
    fn write_access_and_exact_bulk_length_are_enforced() {
        let mut elements = [10_u32, 20, 30];
        let begin = elements.as_mut_ptr() as usize;
        let metadata = [begin, begin + size_of::<u32>() * elements.len(), begin + 16];
        let layout = layout();
        let memory = StableMemory(vec![
            region(&metadata, Access::READ),
            region(&elements, Access::READ | Access::WRITE),
        ]);
        let span = unsafe {
            ForeignSpan::from_direct(
                &memory,
                Address::new(metadata.as_ptr() as usize),
                layout,
                Access::READ | Access::WRITE,
            )
            .unwrap()
        };
        assert!(matches!(
            unsafe { span.write_all_direct(&memory, &[0; 4]) },
            Err(ForeignSpanError::BulkLengthMismatch { .. })
        ));
        unsafe { span.write_direct(&memory, 1, &77_u32).unwrap() };
        assert_eq!(elements[1], 77);
    }

    #[test]
    fn unsafe_slice_capabilities_cover_stable_single_allocation() {
        let mut elements = [10_u32, 20, 30];
        let begin = elements.as_mut_ptr() as usize;
        let metadata = [begin, begin + size_of::<u32>() * elements.len(), begin + 16];
        let layout = layout();
        let memory = StableMemory(vec![
            region(&metadata, Access::READ),
            region(&elements, Access::READ | Access::WRITE),
        ]);
        let vector = Address::new(metadata.as_ptr() as usize);
        unsafe {
            super::super::with_direct_memory_scope(&memory, |scope| {
                let span =
                    ForeignSpan::from_scope(scope, vector, layout, Access::READ | Access::WRITE)
                        .unwrap();
                let mut slice = span.foreign_slice_mut::<u32, _>(scope).unwrap();
                slice[0] = 99;
            })
        }
        assert_eq!(elements[0], 99);
    }
}
