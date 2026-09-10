//! Bounded, allocation-free pointer-chain evaluation.

use std::mem::{align_of, size_of};

use thiserror::Error;

use super::{
    Access, Address, AddressRange, DirectMemoryAccess, DirectMemoryAccessExt, DirectMemoryScope,
    FaultContainedMemoryAccess, FaultContainedMemoryAccessExt, MemoryError, MemoryMap, ModuleId,
};

/// Limits applied to one pointer-chain evaluation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChainLimits {
    pub max_steps: usize,
    pub max_regions: usize,
    pub max_read_bytes: usize,
}

impl Default for ChainLimits {
    fn default() -> Self {
        Self {
            max_steps: 64,
            max_regions: 128,
            max_read_bytes: 4 * 1024,
        }
    }
}

/// An operation in a pointer chain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChainStep {
    Add(i64),
    ReadPointer,
    ReadU8,
    ReadU16,
    ReadU32,
    ReadU64,
    ReadUsize,
    Assert(Assertion),
}

/// An ordered assertion over the current chain value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Assertion {
    NonNull,
    Aligned {
        alignment: usize,
    },
    Mapped {
        len: usize,
        access: Access,
    },
    InModule {
        module: ModuleId,
        len: usize,
    },
    Executable {
        len: usize,
        policy: ChainExecutablePolicy,
    },
    UnsignedRange {
        min: u64,
        max: u64,
    },
}

/// Additional membership required by an executable assertion.
///
/// Keeping this policy in the chain layer avoids a
/// `memory -> function -> memory` module dependency for memory-only users.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ChainExecutablePolicy {
    /// Accept any mapping that has execute permission.
    #[default]
    Any,
    /// Require every asserted byte to belong to this module.
    Module(ModuleId),
    /// Require module membership and containment in this executable section.
    ExecutableSection {
        module: ModuleId,
        range: AddressRange,
    },
}

/// The current value of a pointer-chain evaluation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChainValue {
    Address(Address),
    Unsigned(u64),
}

impl ChainValue {
    pub const fn kind(self) -> ChainValueKind {
        match self {
            Self::Address(_) => ChainValueKind::Address,
            Self::Unsigned(_) => ChainValueKind::Unsigned,
        }
    }

    pub const fn as_address(self) -> Option<Address> {
        match self {
            Self::Address(address) => Some(address),
            Self::Unsigned(_) => None,
        }
    }

    pub const fn as_unsigned(self) -> Option<u64> {
        match self {
            Self::Unsigned(value) => Some(value),
            Self::Address(_) => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChainValueKind {
    Address,
    Unsigned,
}

/// A borrowed pointer-chain plan. Evaluating it performs no allocation.
#[derive(Clone, Copy, Debug)]
pub struct PointerChain<'steps> {
    steps: &'steps [ChainStep],
    limits: ChainLimits,
}

impl<'steps> PointerChain<'steps> {
    pub const fn new(steps: &'steps [ChainStep]) -> Self {
        Self {
            steps,
            limits: ChainLimits {
                max_steps: 64,
                max_regions: 128,
                max_read_bytes: 4 * 1024,
            },
        }
    }

    pub const fn with_limits(steps: &'steps [ChainStep], limits: ChainLimits) -> Self {
        Self { steps, limits }
    }

    pub const fn steps(&self) -> &'steps [ChainStep] {
        self.steps
    }

    pub const fn limits(&self) -> ChainLimits {
        self.limits
    }

    /// Evaluates using direct process-local reads.
    ///
    /// # Safety
    ///
    /// Every range read by the plan must satisfy the safety contract of
    /// [`DirectMemoryAccess::read_bytes_into`] for the duration of the read.
    pub unsafe fn evaluate_direct<M: DirectMemoryAccess + ?Sized>(
        &self,
        memory: &M,
        start: Address,
    ) -> Result<ChainValue, ChainError> {
        // SAFETY: the caller accepts the direct backend's read contract.
        unsafe {
            self.evaluate_with(memory, start, |address, width| match width {
                ReadWidth::U8 => memory.read_value::<u8>(address).map(u64::from),
                ReadWidth::U16 => memory.read_value::<u16>(address).map(u64::from),
                ReadWidth::U32 => memory.read_value::<u32>(address).map(u64::from),
                ReadWidth::U64 => memory.read_value::<u64>(address),
                ReadWidth::Usize => memory
                    .read_value::<usize>(address)
                    .map(|value| value as u64),
            })
        }
    }

    /// Evaluates through a branded direct-memory scope.
    pub fn evaluate_scoped<M: DirectMemoryAccess + ?Sized>(
        &self,
        scope: &DirectMemoryScope<'_, M>,
        start: Address,
    ) -> Result<ChainValue, ChainError> {
        self.evaluate_with(scope.memory(), start, |address, width| match width {
            ReadWidth::U8 => scope.read_value::<u8>(address).map(u64::from),
            ReadWidth::U16 => scope.read_value::<u16>(address).map(u64::from),
            ReadWidth::U32 => scope.read_value::<u32>(address).map(u64::from),
            ReadWidth::U64 => scope.read_value::<u64>(address),
            ReadWidth::Usize => scope.read_value::<usize>(address).map(|value| value as u64),
        })
    }

    /// Evaluates through a backend that contains invalid-address faults.
    pub fn evaluate_fault_contained<M: FaultContainedMemoryAccess + ?Sized>(
        &self,
        memory: &M,
        start: Address,
    ) -> Result<ChainValue, ChainError> {
        self.evaluate_with(memory, start, |address, width| match width {
            ReadWidth::U8 => memory.try_read_value::<u8>(address).map(u64::from),
            ReadWidth::U16 => memory.try_read_value::<u16>(address).map(u64::from),
            ReadWidth::U32 => memory.try_read_value::<u32>(address).map(u64::from),
            ReadWidth::U64 => memory.try_read_value::<u64>(address),
            ReadWidth::Usize => memory
                .try_read_value::<usize>(address)
                .map(|value| value as u64),
        })
    }

    fn evaluate_with<M, F>(
        &self,
        memory: &M,
        start: Address,
        mut read: F,
    ) -> Result<ChainValue, ChainError>
    where
        M: MemoryMap + ?Sized,
        F: FnMut(Address, ReadWidth) -> Result<u64, MemoryError>,
    {
        if self.steps.len() > self.limits.max_steps {
            return Err(ChainError::StepLimitExceeded {
                actual: self.steps.len(),
                limit: self.limits.max_steps,
            });
        }

        let mut value = ChainValue::Address(start);
        let mut bytes_read = 0usize;
        for (step_index, step) in self.steps.iter().copied().enumerate() {
            value = match step {
                ChainStep::Add(offset) => {
                    let address = require_address(value, step_index, ChainOperation::Add)?;
                    let derived = address.checked_add_signed(offset).map_err(|source| {
                        ChainError::Memory {
                            step: step_index,
                            address,
                            source,
                        }
                    })?;
                    ChainValue::Address(derived)
                }
                ChainStep::ReadPointer => {
                    let address = require_address(value, step_index, ChainOperation::Read)?;
                    let raw = read_checked(
                        memory,
                        &mut read,
                        address,
                        ReadWidth::Usize,
                        step_index,
                        &mut bytes_read,
                        self.limits,
                    )?;
                    let pointer = usize::try_from(raw).map_err(|_| ChainError::InvalidPointer {
                        step: step_index,
                        value: raw,
                    })?;
                    ChainValue::Address(Address::new(pointer))
                }
                ChainStep::ReadU8 => self.read_unsigned(
                    memory,
                    &mut read,
                    value,
                    ReadWidth::U8,
                    step_index,
                    &mut bytes_read,
                )?,
                ChainStep::ReadU16 => self.read_unsigned(
                    memory,
                    &mut read,
                    value,
                    ReadWidth::U16,
                    step_index,
                    &mut bytes_read,
                )?,
                ChainStep::ReadU32 => self.read_unsigned(
                    memory,
                    &mut read,
                    value,
                    ReadWidth::U32,
                    step_index,
                    &mut bytes_read,
                )?,
                ChainStep::ReadU64 => self.read_unsigned(
                    memory,
                    &mut read,
                    value,
                    ReadWidth::U64,
                    step_index,
                    &mut bytes_read,
                )?,
                ChainStep::ReadUsize => self.read_unsigned(
                    memory,
                    &mut read,
                    value,
                    ReadWidth::Usize,
                    step_index,
                    &mut bytes_read,
                )?,
                ChainStep::Assert(assertion) => {
                    assert_value(memory, value, assertion, step_index, self.limits)?;
                    value
                }
            };
        }
        Ok(value)
    }

    fn read_unsigned<M, F>(
        &self,
        memory: &M,
        read: &mut F,
        value: ChainValue,
        width: ReadWidth,
        step: usize,
        bytes_read: &mut usize,
    ) -> Result<ChainValue, ChainError>
    where
        M: MemoryMap + ?Sized,
        F: FnMut(Address, ReadWidth) -> Result<u64, MemoryError>,
    {
        let address = require_address(value, step, ChainOperation::Read)?;
        read_checked(memory, read, address, width, step, bytes_read, self.limits)
            .map(ChainValue::Unsigned)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReadWidth {
    U8,
    U16,
    U32,
    U64,
    Usize,
}

impl ReadWidth {
    const fn bytes(self) -> usize {
        match self {
            Self::U8 => size_of::<u8>(),
            Self::U16 => size_of::<u16>(),
            Self::U32 => size_of::<u32>(),
            Self::U64 => size_of::<u64>(),
            Self::Usize => size_of::<usize>(),
        }
    }

    const fn alignment(self) -> usize {
        match self {
            Self::U8 => align_of::<u8>(),
            Self::U16 => align_of::<u16>(),
            Self::U32 => align_of::<u32>(),
            Self::U64 => align_of::<u64>(),
            Self::Usize => align_of::<usize>(),
        }
    }
}

fn read_checked<M, F>(
    memory: &M,
    read: &mut F,
    address: Address,
    width: ReadWidth,
    step: usize,
    bytes_read: &mut usize,
    limits: ChainLimits,
) -> Result<u64, ChainError>
where
    M: MemoryMap + ?Sized,
    F: FnMut(Address, ReadWidth) -> Result<u64, MemoryError>,
{
    address
        .require_non_null()
        .map_err(|source| ChainError::Memory {
            step,
            address,
            source,
        })?;
    address
        .require_alignment(width.alignment())
        .map_err(|source| ChainError::Memory {
            step,
            address,
            source,
        })?;
    let requested = bytes_read
        .checked_add(width.bytes())
        .ok_or(ChainError::ReadLimitExceeded {
            step,
            requested: usize::MAX,
            limit: limits.max_read_bytes,
        })?;
    if requested > limits.max_read_bytes {
        return Err(ChainError::ReadLimitExceeded {
            step,
            requested,
            limit: limits.max_read_bytes,
        });
    }
    let range = address
        .checked_range(width.bytes())
        .map_err(|source| ChainError::Memory {
            step,
            address,
            source,
        })?;
    memory
        .check_range(range, Access::READ, limits.max_regions)
        .map_err(|source| ChainError::Memory {
            step,
            address,
            source,
        })?;
    let result = read(address, width).map_err(|source| ChainError::Memory {
        step,
        address,
        source,
    })?;
    *bytes_read = requested;
    Ok(result)
}

fn require_address(
    value: ChainValue,
    step: usize,
    operation: ChainOperation,
) -> Result<Address, ChainError> {
    match value {
        ChainValue::Address(address) => Ok(address),
        ChainValue::Unsigned(_) => Err(ChainError::TypeMismatch {
            step,
            operation,
            expected: ChainValueKind::Address,
            actual: ChainValueKind::Unsigned,
        }),
    }
}

fn assert_value<M: MemoryMap + ?Sized>(
    memory: &M,
    value: ChainValue,
    assertion: Assertion,
    step: usize,
    limits: ChainLimits,
) -> Result<(), ChainError> {
    match assertion {
        Assertion::NonNull => {
            let address = require_address(value, step, ChainOperation::Assert)?;
            address
                .require_non_null()
                .map_err(|source| ChainError::Memory {
                    step,
                    address,
                    source,
                })?;
        }
        Assertion::Aligned { alignment } => {
            let address = require_address(value, step, ChainOperation::Assert)?;
            address
                .require_alignment(alignment)
                .map_err(|source| ChainError::Memory {
                    step,
                    address,
                    source,
                })?;
        }
        Assertion::Mapped { len, access } => {
            let address = require_address(value, step, ChainOperation::Assert)?;
            let range = address
                .checked_range(len)
                .map_err(|source| ChainError::Memory {
                    step,
                    address,
                    source,
                })?;
            memory
                .check_range(range, access, limits.max_regions)
                .map_err(|source| ChainError::Memory {
                    step,
                    address,
                    source,
                })?;
        }
        Assertion::InModule { module, len } => {
            let address = require_address(value, step, ChainOperation::Assert)?;
            let range = address
                .checked_range(len)
                .map_err(|source| ChainError::Memory {
                    step,
                    address,
                    source,
                })?;
            check_module(memory, range, module, limits.max_regions, step)?;
        }
        Assertion::Executable { len, policy } => {
            let address = require_address(value, step, ChainOperation::Assert)?;
            let range = address
                .checked_range(len)
                .map_err(|source| ChainError::Memory {
                    step,
                    address,
                    source,
                })?;
            memory
                .check_range(range, Access::EXECUTE, limits.max_regions)
                .map_err(|source| ChainError::Memory {
                    step,
                    address,
                    source,
                })?;
            match policy {
                ChainExecutablePolicy::Any => {}
                ChainExecutablePolicy::Module(module) => {
                    check_module(memory, range, module, limits.max_regions, step)?;
                }
                ChainExecutablePolicy::ExecutableSection {
                    module,
                    range: section,
                } => {
                    if !section.contains_range(range) {
                        return Err(ChainError::ExecutableSectionMismatch {
                            step,
                            requested: range,
                            section,
                        });
                    }
                    check_module(memory, range, module, limits.max_regions, step)?;
                }
            }
        }
        Assertion::UnsignedRange { min, max } => {
            let actual = match value {
                ChainValue::Unsigned(value) => value,
                ChainValue::Address(_) => {
                    return Err(ChainError::TypeMismatch {
                        step,
                        operation: ChainOperation::Assert,
                        expected: ChainValueKind::Unsigned,
                        actual: ChainValueKind::Address,
                    });
                }
            };
            if min > max {
                return Err(ChainError::InvalidIntegerRange { step, min, max });
            }
            if !(min..=max).contains(&actual) {
                return Err(ChainError::IntegerOutOfRange {
                    step,
                    value: actual,
                    min,
                    max,
                });
            }
        }
    }
    Ok(())
}

fn check_module<M: MemoryMap + ?Sized>(
    memory: &M,
    range: AddressRange,
    expected: ModuleId,
    max_regions: usize,
    step: usize,
) -> Result<(), ChainError> {
    if max_regions == 0 {
        return Err(ChainError::Memory {
            step,
            address: range.start(),
            source: MemoryError::RegionLimitExceeded { limit: 0 },
        });
    }
    let mut cursor = range.start();
    for _ in 0..max_regions {
        let region = memory
            .region_containing(cursor)
            .map_err(|source| ChainError::Memory {
                step,
                address: cursor,
                source,
            })?;
        if !region.contains(cursor) || region.range.end() <= cursor {
            return Err(ChainError::Memory {
                step,
                address: cursor,
                source: MemoryError::NonProgressingRegion { address: cursor },
            });
        }
        if region.module != Some(expected) {
            return Err(ChainError::ModuleMismatch {
                step,
                address: cursor,
                expected,
                actual: region.module,
            });
        }
        if region.range.end() >= range.end() {
            return Ok(());
        }
        cursor = region.range.end();
    }
    Err(ChainError::Memory {
        step,
        address: cursor,
        source: MemoryError::RegionLimitExceeded { limit: max_regions },
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChainOperation {
    Add,
    Read,
    Assert,
}

/// A bounded pointer-chain failure with the responsible step preserved.
#[derive(Debug, Error, Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum ChainError {
    #[error("pointer chain has {actual} steps, exceeding limit {limit}")]
    StepLimitExceeded { actual: usize, limit: usize },
    #[error("pointer chain step {step} would read {requested} bytes, exceeding limit {limit}")]
    ReadLimitExceeded {
        step: usize,
        requested: usize,
        limit: usize,
    },
    #[error("pointer chain step {step} cannot perform {operation:?} on {actual:?}")]
    TypeMismatch {
        step: usize,
        operation: ChainOperation,
        expected: ChainValueKind,
        actual: ChainValueKind,
    },
    #[error("pointer chain step {step} at {address} failed")]
    Memory {
        step: usize,
        address: Address,
        #[source]
        source: MemoryError,
    },
    #[error("pointer chain step {step} produced pointer {value:#x} outside usize")]
    InvalidPointer { step: usize, value: u64 },
    #[error(
        "pointer chain step {step} expected module {expected:?} at {address}, found {actual:?}"
    )]
    ModuleMismatch {
        step: usize,
        address: Address,
        expected: ModuleId,
        actual: Option<ModuleId>,
    },
    #[error(
        "pointer chain step {step} executable range {requested:?} is outside section {section:?}"
    )]
    ExecutableSectionMismatch {
        step: usize,
        requested: AddressRange,
        section: AddressRange,
    },
    #[error("pointer chain step {step} has invalid integer range {min}..={max}")]
    InvalidIntegerRange { step: usize, min: u64, max: u64 },
    #[error("pointer chain step {step} value {value} is outside {min}..={max}")]
    IntegerOutOfRange {
        step: usize,
        value: u64,
        min: u64,
        max: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::testing::{FakeMemory, FakeOperation, MutationTiming};
    use crate::memory::{MemoryRegion, RegionKind, DEFAULT_MAX_REGIONS};

    struct TestMemory {
        regions: Vec<MemoryRegion>,
    }

    impl MemoryMap for TestMemory {
        fn region_containing(&self, address: Address) -> Result<MemoryRegion, MemoryError> {
            self.regions
                .iter()
                .copied()
                .find(|region| region.contains(address))
                .ok_or(MemoryError::Unmapped { address })
        }
    }

    impl DirectMemoryAccess for TestMemory {
        unsafe fn read_bytes_into(
            &self,
            address: Address,
            output: &mut [u8],
        ) -> Result<(), MemoryError> {
            let range = address.checked_range(output.len())?;
            self.check_range(range, Access::READ, DEFAULT_MAX_REGIONS)?;
            // SAFETY: tests register live Rust allocations covering all reads.
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
            _address: Address,
            _bytes: &[u8],
        ) -> Result<(), MemoryError> {
            unreachable!()
        }
    }

    fn region<T: ?Sized>(value: &T) -> MemoryRegion {
        MemoryRegion {
            range: Address::new(value as *const T as *const () as usize)
                .checked_range(size_of_val(value))
                .unwrap(),
            access: Access::READ,
            kind: RegionKind::Private,
            module: None,
        }
    }

    fn pointer_bytes(address: usize) -> Vec<u8> {
        address.to_ne_bytes().to_vec()
    }

    fn read_count(memory: &FakeMemory) -> usize {
        memory
            .access_log()
            .iter()
            .filter(|entry| entry.operation == FakeOperation::Read)
            .count()
    }

    #[test]
    fn evaluates_two_level_pointer_chain() {
        let target = 99_u64;
        let mut object = [0usize; 8];
        object[4] = &target as *const u64 as usize;
        let root = object.as_ptr() as usize;
        let memory = TestMemory {
            regions: vec![region(&root), region(&object)],
        };
        let steps = [
            ChainStep::ReadPointer,
            ChainStep::Add(32),
            ChainStep::ReadPointer,
            ChainStep::Assert(Assertion::NonNull),
        ];
        let chain = PointerChain::new(&steps);
        let result = unsafe {
            super::super::with_direct_memory_scope(&memory, |scope| {
                chain.evaluate_scoped(scope, Address::new(&root as *const usize as usize))
            })
        };
        assert_eq!(
            result,
            Ok(ChainValue::Address(Address::new(
                &target as *const u64 as usize
            )))
        );
    }

    #[test]
    fn reads_unsigned_value_and_checks_range() {
        let scalar = 17_u32;
        let memory = TestMemory {
            regions: vec![region(&scalar)],
        };
        let steps = [
            ChainStep::ReadU32,
            ChainStep::Assert(Assertion::UnsignedRange { min: 10, max: 20 }),
        ];
        let result = unsafe {
            PointerChain::new(&steps)
                .evaluate_direct(&memory, Address::new(&scalar as *const u32 as usize))
        };
        assert_eq!(result, Ok(ChainValue::Unsigned(17)));
    }

    #[test]
    fn stops_before_reading_past_limits() {
        let scalar = 17_u64;
        let memory = TestMemory {
            regions: vec![region(&scalar)],
        };
        let steps = [ChainStep::ReadU64];
        let limits = ChainLimits {
            max_steps: 1,
            max_regions: 1,
            max_read_bytes: 4,
        };
        let result = unsafe {
            PointerChain::with_limits(&steps, limits)
                .evaluate_direct(&memory, Address::new(&scalar as *const u64 as usize))
        };
        assert_eq!(
            result,
            Err(ChainError::ReadLimitExceeded {
                step: 0,
                requested: 8,
                limit: 4,
            })
        );
    }

    #[test]
    fn rejects_steps_after_scalar_without_accessing_memory() {
        let scalar = 17_u8;
        let memory = TestMemory {
            regions: vec![region(&scalar)],
        };
        let steps = [ChainStep::ReadU8, ChainStep::ReadU8];
        let result = unsafe {
            PointerChain::new(&steps)
                .evaluate_direct(&memory, Address::new(&scalar as *const u8 as usize))
        };
        assert!(matches!(
            result,
            Err(ChainError::TypeMismatch {
                step: 1,
                expected: ChainValueKind::Address,
                actual: ChainValueKind::Unsigned,
                ..
            })
        ));
    }

    #[test]
    fn rejects_null_unreadable_misaligned_and_undersized_targets() {
        let unreadable_root = FakeMemory::new();
        unreadable_root
            .map(
                Address::new(0x8000),
                pointer_bytes(0x9000),
                Access::empty(),
                RegionKind::Private,
                None,
            )
            .unwrap();
        let dereference = [ChainStep::ReadPointer, ChainStep::ReadPointer];
        let result = PointerChain::new(&dereference)
            .evaluate_fault_contained(&unreadable_root, Address::new(0x8000));
        assert!(matches!(
            result,
            Err(ChainError::Memory {
                step: 0,
                source: MemoryError::PermissionDenied { .. },
                ..
            })
        ));
        assert_eq!(read_count(&unreadable_root), 0);

        let null = FakeMemory::new();
        null.map(
            Address::new(0x1000),
            pointer_bytes(0),
            Access::READ,
            RegionKind::Private,
            None,
        )
        .unwrap();
        let result =
            PointerChain::new(&dereference).evaluate_fault_contained(&null, Address::new(0x1000));
        assert!(matches!(
            result,
            Err(ChainError::Memory {
                step: 1,
                source: MemoryError::NullAddress,
                ..
            })
        ));
        assert_eq!(read_count(&null), 1);

        let unreadable = FakeMemory::new();
        unreadable
            .map(
                Address::new(0x1000),
                pointer_bytes(0x2000),
                Access::READ,
                RegionKind::Private,
                None,
            )
            .unwrap();
        unreadable
            .map(
                Address::new(0x2000),
                vec![0; size_of::<usize>()],
                Access::empty(),
                RegionKind::Private,
                None,
            )
            .unwrap();
        let result = PointerChain::new(&dereference)
            .evaluate_fault_contained(&unreadable, Address::new(0x1000));
        assert!(matches!(
            result,
            Err(ChainError::Memory {
                step: 1,
                source: MemoryError::PermissionDenied { .. },
                ..
            })
        ));
        assert_eq!(read_count(&unreadable), 1);

        let misaligned = FakeMemory::new();
        misaligned
            .map(
                Address::new(0x1000),
                pointer_bytes(0x2001),
                Access::READ,
                RegionKind::Private,
                None,
            )
            .unwrap();
        let result = PointerChain::new(&dereference)
            .evaluate_fault_contained(&misaligned, Address::new(0x1000));
        assert!(matches!(
            result,
            Err(ChainError::Memory {
                step: 1,
                source: MemoryError::Misaligned { .. },
                ..
            })
        ));
        assert_eq!(read_count(&misaligned), 1);

        let undersized = FakeMemory::new();
        undersized
            .map(
                Address::new(0x1000),
                pointer_bytes(0x3000),
                Access::READ,
                RegionKind::Private,
                None,
            )
            .unwrap();
        undersized
            .map(
                Address::new(0x3000),
                [1, 2, 3, 4],
                Access::READ,
                RegionKind::Private,
                None,
            )
            .unwrap();
        let read_u64 = [ChainStep::ReadPointer, ChainStep::ReadU64];
        let result = PointerChain::new(&read_u64)
            .evaluate_fault_contained(&undersized, Address::new(0x1000));
        assert!(matches!(
            result,
            Err(ChainError::Memory {
                step: 1,
                source: MemoryError::Unmapped { .. },
                ..
            })
        ));
        assert_eq!(read_count(&undersized), 1);
    }

    #[test]
    fn signed_overflow_is_terminal_before_memory_access() {
        let memory = FakeMemory::new();
        let steps = [ChainStep::Add(1), ChainStep::ReadU8];
        let result =
            PointerChain::new(&steps).evaluate_fault_contained(&memory, Address::new(usize::MAX));
        assert!(matches!(
            result,
            Err(ChainError::Memory {
                step: 0,
                source: MemoryError::AddressOverflow,
                ..
            })
        ));
        assert!(memory.access_log().is_empty());
    }

    #[test]
    fn scripted_mutation_is_observed_as_a_validated_read() {
        let memory = FakeMemory::new();
        memory
            .map(
                Address::new(0x1000),
                pointer_bytes(0x2000),
                Access::READ,
                RegionKind::Private,
                None,
            )
            .unwrap();
        memory
            .map(
                Address::new(0x2000),
                pointer_bytes(0xdead),
                Access::READ,
                RegionKind::Private,
                None,
            )
            .unwrap();
        memory
            .mutate_on(
                FakeOperation::Read,
                2,
                MutationTiming::Before,
                Address::new(0x2000),
                pointer_bytes(0x3000),
            )
            .unwrap();
        let steps = [ChainStep::ReadPointer, ChainStep::ReadPointer];
        assert_eq!(
            PointerChain::new(&steps).evaluate_fault_contained(&memory, Address::new(0x1000)),
            Ok(ChainValue::Address(Address::new(0x3000)))
        );
        assert_eq!(read_count(&memory), 2);
    }

    #[test]
    fn executable_policies_and_module_assertions_are_enforced() {
        let memory = FakeMemory::new();
        let module = ModuleId(7);
        memory
            .map(
                Address::new(0x4000),
                [0; 16],
                Access::READ | Access::EXECUTE,
                RegionKind::Image,
                Some(module),
            )
            .unwrap();
        let section = AddressRange::new(Address::new(0x4000), 8).unwrap();

        for policy in [
            ChainExecutablePolicy::Any,
            ChainExecutablePolicy::Module(module),
            ChainExecutablePolicy::ExecutableSection {
                module,
                range: section,
            },
        ] {
            let steps = [ChainStep::Assert(Assertion::Executable { len: 4, policy })];
            assert_eq!(
                PointerChain::new(&steps).evaluate_fault_contained(&memory, Address::new(0x4000)),
                Ok(ChainValue::Address(Address::new(0x4000)))
            );
        }

        let wrong_module = [ChainStep::Assert(Assertion::Executable {
            len: 4,
            policy: ChainExecutablePolicy::Module(ModuleId(8)),
        })];
        assert!(matches!(
            PointerChain::new(&wrong_module)
                .evaluate_fault_contained(&memory, Address::new(0x4000)),
            Err(ChainError::ModuleMismatch { step: 0, .. })
        ));

        let outside_section = [ChainStep::Assert(Assertion::Executable {
            len: 4,
            policy: ChainExecutablePolicy::ExecutableSection {
                module,
                range: section,
            },
        })];
        assert!(matches!(
            PointerChain::new(&outside_section)
                .evaluate_fault_contained(&memory, Address::new(0x4008)),
            Err(ChainError::ExecutableSectionMismatch { step: 0, .. })
        ));

        let module_check = [ChainStep::Assert(Assertion::InModule { module, len: 8 })];
        assert!(PointerChain::new(&module_check)
            .evaluate_fault_contained(&memory, Address::new(0x4000))
            .is_ok());
    }

    #[test]
    fn terminal_type_error_prevents_all_later_reads() {
        let memory = FakeMemory::new();
        memory
            .map(
                Address::new(0x5000),
                [17],
                Access::READ,
                RegionKind::Private,
                None,
            )
            .unwrap();
        let steps = [ChainStep::ReadU8, ChainStep::ReadU8, ChainStep::ReadU8];
        assert!(matches!(
            PointerChain::new(&steps).evaluate_fault_contained(&memory, Address::new(0x5000)),
            Err(ChainError::TypeMismatch { step: 1, .. })
        ));
        assert_eq!(read_count(&memory), 1);

        let scripted = FakeMemory::new();
        scripted
            .map(
                Address::new(0x6000),
                pointer_bytes(0x7000),
                Access::READ,
                RegionKind::Private,
                None,
            )
            .unwrap();
        scripted
            .map(
                Address::new(0x7000),
                pointer_bytes(0x8000),
                Access::READ,
                RegionKind::Private,
                None,
            )
            .unwrap();
        scripted
            .fail_on(
                FakeOperation::Read,
                2,
                MemoryError::InvalidValue {
                    context: "scripted chain failure",
                },
            )
            .unwrap();
        let pointer_reads = [
            ChainStep::ReadPointer,
            ChainStep::ReadPointer,
            ChainStep::ReadPointer,
        ];
        assert!(matches!(
            PointerChain::new(&pointer_reads)
                .evaluate_fault_contained(&scripted, Address::new(0x6000)),
            Err(ChainError::Memory {
                step: 1,
                source: MemoryError::InvalidValue {
                    context: "scripted chain failure"
                },
                ..
            })
        ));
        assert_eq!(read_count(&scripted), 2);
    }
}
