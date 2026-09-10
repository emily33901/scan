//! Bounded, backend-independent address-resolution plans.

use std::fmt;
use std::mem::size_of;

use thiserror::Error;

use crate::memory::{
    Access, Address, DirectMemoryAccess, DirectMemoryScope, FaultContainedMemoryAccess, MemoryError,
};

mod aarch64;
pub mod x86_64;

#[derive(Clone, Copy, Debug)]
pub struct MethodPlan<'a> {
    steps: &'a [MethodStep],
}

impl<'a> MethodPlan<'a> {
    pub const fn new(steps: &'a [MethodStep]) -> Self {
        Self { steps }
    }

    pub const fn steps(self) -> &'a [MethodStep] {
        self.steps
    }

    /// # Safety
    ///
    /// Every range read, including ranges derived by earlier steps, must stay
    /// live, mapped, readable, initialized, and synchronized for the call.
    pub unsafe fn evaluate_direct<M: DirectMemoryAccess + ?Sized>(
        self,
        memory: &M,
        start: Address,
        limits: MethodLimits,
    ) -> Result<MethodValue, MethodError> {
        let mut reader = DirectReader { memory };
        evaluate(self.steps, &mut reader, start, limits)
    }

    pub fn evaluate_scoped<M: DirectMemoryAccess + ?Sized>(
        self,
        scope: &DirectMemoryScope<'_, M>,
        start: Address,
        limits: MethodLimits,
    ) -> Result<MethodValue, MethodError> {
        let mut reader = ScopedReader { scope };
        evaluate(self.steps, &mut reader, start, limits)
    }

    pub fn evaluate_fault_contained<M: FaultContainedMemoryAccess + ?Sized>(
        self,
        memory: &M,
        start: Address,
        limits: MethodLimits,
    ) -> Result<MethodValue, MethodError> {
        let mut reader = FaultContainedReader { memory };
        evaluate(self.steps, &mut reader, start, limits)
    }
}

impl<'a> From<&'a [MethodStep]> for MethodPlan<'a> {
    fn from(steps: &'a [MethodStep]) -> Self {
        Self::new(steps)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum MethodStep {
    Add(i64),
    ReadPointer {
        offset: i64,
    },
    ReadUnsigned {
        offset: i64,
        width: IntegerWidth,
    },
    UnsignedToAddress,
    X86ResolveRelative {
        displacement_offset: usize,
        width: SignedWidth,
    },
    X86FollowDirectBranch,
    Arm64ReadImmediate,
    Arm64ResolveImmediateRelative,
    Arm64ResolvePageOffset {
        instruction_offset: usize,
    },
    Arm64FollowDirectBranch,
}

impl MethodStep {
    pub const fn operation(self) -> MethodOperation {
        match self {
            Self::Add(_) => MethodOperation::Add,
            Self::ReadPointer { .. } => MethodOperation::ReadPointer,
            Self::ReadUnsigned { .. } => MethodOperation::ReadUnsigned,
            Self::UnsignedToAddress => MethodOperation::UnsignedToAddress,
            Self::X86ResolveRelative { .. } => MethodOperation::X86ResolveRelative,
            Self::X86FollowDirectBranch => MethodOperation::X86FollowDirectBranch,
            Self::Arm64ReadImmediate => MethodOperation::Arm64ReadImmediate,
            Self::Arm64ResolveImmediateRelative => MethodOperation::Arm64ResolveImmediateRelative,
            Self::Arm64ResolvePageOffset { .. } => MethodOperation::Arm64ResolvePageOffset,
            Self::Arm64FollowDirectBranch => MethodOperation::Arm64FollowDirectBranch,
        }
    }

    const fn required_architecture(self) -> Option<Architecture> {
        match self {
            Self::X86ResolveRelative { .. } | Self::X86FollowDirectBranch => {
                Some(Architecture::X86_64)
            }
            Self::Arm64ReadImmediate
            | Self::Arm64ResolveImmediateRelative
            | Self::Arm64ResolvePageOffset { .. }
            | Self::Arm64FollowDirectBranch => Some(Architecture::Aarch64),
            Self::Add(_)
            | Self::ReadPointer { .. }
            | Self::ReadUnsigned { .. }
            | Self::UnsignedToAddress => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Architecture {
    X86_64,
    Aarch64,
    Other,
}

impl Architecture {
    #[cfg(target_arch = "x86_64")]
    const fn current() -> Self {
        Self::X86_64
    }

    #[cfg(target_arch = "aarch64")]
    const fn current() -> Self {
        Self::Aarch64
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    const fn current() -> Self {
        Self::Other
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntegerWidth {
    U8,
    U16,
    U32,
    U64,
    Usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignedWidth {
    I32,
    I64,
}

impl IntegerWidth {
    const fn bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 => 4,
            Self::U64 => 8,
            Self::Usize => size_of::<usize>(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MethodValue {
    Address(Address),
    Signed(i64),
    Unsigned(u64),
}

impl MethodValue {
    pub fn into_address(self) -> Result<Address, MethodValueKind> {
        match self {
            Self::Address(address) => Ok(address),
            Self::Signed(_) => Err(MethodValueKind::Signed),
            Self::Unsigned(_) => Err(MethodValueKind::Unsigned),
        }
    }

    pub const fn kind(self) -> MethodValueKind {
        match self {
            Self::Address(_) => MethodValueKind::Address,
            Self::Signed(_) => MethodValueKind::Signed,
            Self::Unsigned(_) => MethodValueKind::Unsigned,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MethodValueKind {
    Address,
    Signed,
    Unsigned,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MethodLimits {
    pub max_steps: usize,
    pub max_instruction_bytes: usize,
    pub max_regions_per_read: usize,
}

impl Default for MethodLimits {
    fn default() -> Self {
        Self {
            max_steps: 128,
            max_instruction_bytes: 4_096,
            max_regions_per_read: 1_024,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MethodOperation {
    Plan,
    Add,
    ReadPointer,
    ReadUnsigned,
    UnsignedToAddress,
    X86ResolveRelative,
    X86FollowDirectBranch,
    Arm64ReadImmediate,
    Arm64ResolveImmediateRelative,
    Arm64ResolvePageOffset,
    Arm64FollowDirectBranch,
}

impl fmt::Display for MethodOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Debug, Error)]
#[error("method step {step_index} ({operation}) failed: {cause}")]
pub struct MethodError {
    pub step_index: usize,
    pub operation: MethodOperation,
    #[source]
    pub cause: MethodErrorCause,
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MethodErrorCause {
    #[error("plan has {actual} steps, exceeding limit {limit}")]
    StepLimitExceeded { actual: usize, limit: usize },
    #[error("instruction reads used {actual} bytes, exceeding limit {limit}")]
    InstructionByteLimitExceeded { actual: usize, limit: usize },
    #[error("operation requires an address, but current value is {actual:?}")]
    AddressRequired { actual: MethodValueKind },
    #[error("address calculation overflowed")]
    AddressOverflow,
    #[error("operation requires {required:?}, but current architecture is {actual:?}")]
    UnsupportedArchitecture {
        required: Architecture,
        actual: Architecture,
    },
    #[error("memory access failed")]
    Memory(#[source] MemoryError),
    #[error("instruction decoding failed")]
    Instruction(#[source] InstructionError),
}

#[derive(Debug, Error, Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum InstructionError {
    #[error("instruction is truncated: expected {expected} bytes, got {actual}")]
    Truncated { expected: usize, actual: usize },
    #[error("unsupported instruction opcode {opcode:#010x}")]
    UnsupportedOpcode { opcode: u32 },
    #[error("ADRP destination x{adrp_destination} does not match pair base x{pair_base}")]
    RegisterMismatch { adrp_destination: u8, pair_base: u8 },
    #[error("register x{register} is invalid as {role}")]
    InvalidRegister { register: u8, role: &'static str },
    #[error("instruction address calculation overflowed")]
    AddressOverflow,
}

trait MemoryReader {
    fn check_read(
        &self,
        address: Address,
        len: usize,
        max_regions: usize,
    ) -> Result<(), MemoryError>;
    fn read_bytes(&mut self, address: Address, output: &mut [u8]) -> Result<(), MemoryError>;
}

struct DirectReader<'a, M: ?Sized> {
    memory: &'a M,
}

impl<M: DirectMemoryAccess + ?Sized> MemoryReader for DirectReader<'_, M> {
    fn check_read(
        &self,
        address: Address,
        len: usize,
        max_regions: usize,
    ) -> Result<(), MemoryError> {
        self.memory
            .check_range(address.checked_range(len)?, Access::READ, max_regions)
    }

    fn read_bytes(&mut self, address: Address, output: &mut [u8]) -> Result<(), MemoryError> {
        // SAFETY: only evaluate_direct constructs this adapter and delegates its
        // contract to the caller.
        unsafe { self.memory.read_bytes_into(address, output) }
    }
}

struct ScopedReader<'a, 'scope, M: ?Sized> {
    scope: &'a DirectMemoryScope<'scope, M>,
}

impl<M: DirectMemoryAccess + ?Sized> MemoryReader for ScopedReader<'_, '_, M> {
    fn check_read(
        &self,
        address: Address,
        len: usize,
        max_regions: usize,
    ) -> Result<(), MemoryError> {
        self.scope
            .check_range(address.checked_range(len)?, Access::READ, max_regions)
    }

    fn read_bytes(&mut self, address: Address, output: &mut [u8]) -> Result<(), MemoryError> {
        self.scope.read_bytes_into(address, output)
    }
}

struct FaultContainedReader<'a, M: ?Sized> {
    memory: &'a M,
}

impl<M: FaultContainedMemoryAccess + ?Sized> MemoryReader for FaultContainedReader<'_, M> {
    fn check_read(
        &self,
        address: Address,
        len: usize,
        max_regions: usize,
    ) -> Result<(), MemoryError> {
        self.memory
            .check_range(address.checked_range(len)?, Access::READ, max_regions)
    }

    fn read_bytes(&mut self, address: Address, output: &mut [u8]) -> Result<(), MemoryError> {
        self.memory.try_read_bytes_into(address, output)
    }
}

struct Evaluation<'a, R: ?Sized> {
    reader: &'a mut R,
    limits: MethodLimits,
    instruction_bytes: usize,
}

impl<R: MemoryReader + ?Sized> Evaluation<'_, R> {
    fn read(&mut self, address: Address, output: &mut [u8]) -> Result<(), MethodErrorCause> {
        self.reader
            .check_read(address, output.len(), self.limits.max_regions_per_read)
            .map_err(MethodErrorCause::Memory)?;
        self.reader
            .read_bytes(address, output)
            .map_err(MethodErrorCause::Memory)
    }

    fn read_instruction(
        &mut self,
        address: Address,
        output: &mut [u8],
    ) -> Result<(), MethodErrorCause> {
        let used = self.instruction_bytes.checked_add(output.len()).ok_or(
            MethodErrorCause::InstructionByteLimitExceeded {
                actual: usize::MAX,
                limit: self.limits.max_instruction_bytes,
            },
        )?;
        if used > self.limits.max_instruction_bytes {
            return Err(MethodErrorCause::InstructionByteLimitExceeded {
                actual: used,
                limit: self.limits.max_instruction_bytes,
            });
        }
        self.read(address, output)?;
        self.instruction_bytes = used;
        Ok(())
    }

    fn read_u32_instruction(&mut self, address: Address) -> Result<u32, MethodErrorCause> {
        let mut bytes = [0_u8; 4];
        self.read_instruction(address, &mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_pointer(&mut self, address: Address) -> Result<Address, MethodErrorCause> {
        let mut bytes = [0_u8; size_of::<usize>()];
        self.read(address, &mut bytes)?;
        Address::new(usize::from_ne_bytes(bytes))
            .require_non_null()
            .map_err(MethodErrorCause::Memory)
    }

    fn read_unsigned(
        &mut self,
        address: Address,
        width: IntegerWidth,
    ) -> Result<u64, MethodErrorCause> {
        let mut bytes = [0_u8; 8];
        self.read(address, &mut bytes[..width.bytes()])?;
        Ok(match width {
            IntegerWidth::U8 => u64::from(bytes[0]),
            IntegerWidth::U16 => u64::from(u16::from_ne_bytes([bytes[0], bytes[1]])),
            IntegerWidth::U32 => {
                u64::from(u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            }
            IntegerWidth::U64 => u64::from_ne_bytes(bytes),
            IntegerWidth::Usize => {
                let mut native = [0_u8; size_of::<usize>()];
                native.copy_from_slice(&bytes[..size_of::<usize>()]);
                usize::from_ne_bytes(native) as u64
            }
        })
    }

    fn read_x86_branch(&mut self, address: Address) -> Result<([u8; 6], usize), MethodErrorCause> {
        let mut bytes = [0_u8; 6];
        self.read_instruction(address, &mut bytes[..1])?;
        let known = if bytes[0] == 0x0f {
            self.read_instruction(checked_add(address, 1)?, &mut bytes[1..2])?;
            2
        } else {
            1
        };
        let length = x86_64::direct_branch_length(&bytes[..known])?;
        self.read_instruction(checked_add(address, known)?, &mut bytes[known..length])?;
        Ok((bytes, length))
    }
}

fn evaluate<R: MemoryReader + ?Sized>(
    steps: &[MethodStep],
    reader: &mut R,
    start: Address,
    limits: MethodLimits,
) -> Result<MethodValue, MethodError> {
    if steps.len() > limits.max_steps {
        return Err(MethodError {
            step_index: limits.max_steps,
            operation: MethodOperation::Plan,
            cause: MethodErrorCause::StepLimitExceeded {
                actual: steps.len(),
                limit: limits.max_steps,
            },
        });
    }
    let mut evaluation = Evaluation {
        reader,
        limits,
        instruction_bytes: 0,
    };
    let mut value = MethodValue::Address(start);
    for (step_index, step) in steps.iter().copied().enumerate() {
        let operation = step.operation();
        value = evaluate_step(&mut evaluation, value, step).map_err(|cause| MethodError {
            step_index,
            operation,
            cause,
        })?;
    }
    Ok(value)
}

fn evaluate_step<R: MemoryReader + ?Sized>(
    evaluation: &mut Evaluation<'_, R>,
    value: MethodValue,
    step: MethodStep,
) -> Result<MethodValue, MethodErrorCause> {
    if let Some(required) = step.required_architecture() {
        let actual = Architecture::current();
        if actual != required {
            return Err(MethodErrorCause::UnsupportedArchitecture { required, actual });
        }
    }

    let address = match step {
        MethodStep::UnsignedToAddress => {
            return match value {
                MethodValue::Unsigned(value) => usize::try_from(value)
                    .map(Address::new)
                    .map(MethodValue::Address)
                    .map_err(|_| MethodErrorCause::AddressOverflow),
                other => Err(MethodErrorCause::AddressRequired {
                    actual: other.kind(),
                }),
            }
        }
        _ => value
            .into_address()
            .map_err(|actual| MethodErrorCause::AddressRequired { actual })?,
    };
    match step {
        MethodStep::Add(offset) => Ok(MethodValue::Address(checked_add_signed(address, offset)?)),
        MethodStep::ReadPointer { offset } => Ok(MethodValue::Address(
            evaluation.read_pointer(checked_add_signed(address, offset)?)?,
        )),
        MethodStep::ReadUnsigned { offset, width } => Ok(MethodValue::Unsigned(
            evaluation.read_unsigned(checked_add_signed(address, offset)?, width)?,
        )),
        MethodStep::X86ResolveRelative {
            displacement_offset,
            width,
        } => {
            let displacement_address = checked_add(address, displacement_offset)?;
            let (displacement, encoded_width) = match width {
                SignedWidth::I32 => {
                    let mut bytes = [0_u8; 4];
                    evaluation.read_instruction(displacement_address, &mut bytes)?;
                    (i64::from(i32::from_le_bytes(bytes)), 4)
                }
                SignedWidth::I64 => {
                    let mut bytes = [0_u8; 8];
                    evaluation.read_instruction(displacement_address, &mut bytes)?;
                    (i64::from_le_bytes(bytes), 8)
                }
            };
            Ok(MethodValue::Address(x86_64::relative_target(
                address,
                displacement_offset
                    .checked_add(encoded_width)
                    .ok_or(MethodErrorCause::AddressOverflow)?,
                displacement,
            )?))
        }
        MethodStep::X86FollowDirectBranch => {
            let (bytes, length) = evaluation.read_x86_branch(address)?;
            Ok(MethodValue::Address(x86_64::direct_branch_target(
                address,
                &bytes[..length],
            )?))
        }
        MethodStep::Arm64ReadImmediate => Ok(MethodValue::Signed(aarch64::instruction_immediate(
            evaluation.read_u32_instruction(address)?,
        )?)),
        MethodStep::Arm64ResolveImmediateRelative => {
            let immediate =
                aarch64::instruction_immediate(evaluation.read_u32_instruction(address)?)?;
            Ok(MethodValue::Address(checked_add_signed(
                address, immediate,
            )?))
        }
        MethodStep::Arm64ResolvePageOffset { instruction_offset } => {
            let first_address = checked_add(address, instruction_offset)?;
            let first = evaluation.read_u32_instruction(first_address)?;
            let second = evaluation.read_u32_instruction(checked_add(first_address, 4)?)?;
            Ok(MethodValue::Address(aarch64::page_pair_target(
                first_address,
                first,
                second,
            )?))
        }
        MethodStep::Arm64FollowDirectBranch => Ok(MethodValue::Address(
            aarch64::direct_branch_target(address, evaluation.read_u32_instruction(address)?)?,
        )),
        MethodStep::UnsignedToAddress => Err(MethodErrorCause::AddressRequired {
            actual: value.kind(),
        }),
    }
}

fn checked_add(address: Address, offset: usize) -> Result<Address, MethodErrorCause> {
    address
        .checked_add(offset)
        .map_err(|_| MethodErrorCause::AddressOverflow)
}

fn checked_add_signed(address: Address, offset: i64) -> Result<Address, MethodErrorCause> {
    address
        .checked_add_signed(offset)
        .map_err(|_| MethodErrorCause::AddressOverflow)
}

impl From<InstructionError> for MethodErrorCause {
    fn from(source: InstructionError) -> Self {
        Self::Instruction(source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::testing::FakeMemory;
    use crate::memory::{MemoryRegion, RegionKind};

    const BASE: Address = Address::new(0x1000);

    fn memory(bytes: &[u8]) -> FakeMemory {
        let memory = FakeMemory::new();
        memory
            .map_region(
                MemoryRegion {
                    range: BASE.checked_range(bytes.len()).unwrap(),
                    access: Access::READ,
                    kind: RegionKind::Private,
                    module: None,
                },
                bytes,
            )
            .unwrap();
        memory
    }

    fn evaluate_fault(bytes: &[u8], steps: &[MethodStep]) -> Result<MethodValue, MethodError> {
        MethodPlan::new(steps).evaluate_fault_contained(
            &memory(bytes),
            BASE,
            MethodLimits::default(),
        )
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn resolves_signed_x86_displacement() {
        assert_eq!(
            evaluate_fault(
                &(-8_i32).to_le_bytes(),
                &[MethodStep::X86ResolveRelative {
                    displacement_offset: 0,
                    width: SignedWidth::I32,
                }]
            )
            .unwrap(),
            MethodValue::Address(Address::new(0x0ffc))
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn follows_x86_branch_without_overreading() {
        assert_eq!(
            evaluate_fault(&[0xeb, 0xfe], &[MethodStep::X86FollowDirectBranch]).unwrap(),
            MethodValue::Address(BASE)
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn arm_page_pair_applies_the_plan_offset() {
        let adrp_x3 = 0x9000_0003_u32;
        let add_x3_x3_20 = 0x9100_5063_u32;
        let mut bytes = vec![0; 12];
        bytes[4..8].copy_from_slice(&adrp_x3.to_le_bytes());
        bytes[8..12].copy_from_slice(&add_x3_x3_20.to_le_bytes());
        assert_eq!(
            evaluate_fault(
                &bytes,
                &[MethodStep::Arm64ResolvePageOffset {
                    instruction_offset: 4,
                }],
            )
            .unwrap(),
            MethodValue::Address(Address::new(0x1014))
        );
    }

    #[test]
    fn reads_pointers_and_owned_scalars() {
        assert_eq!(
            evaluate_fault(
                &0x1234_usize.to_le_bytes(),
                &[MethodStep::ReadPointer { offset: 0 }]
            )
            .unwrap(),
            MethodValue::Address(Address::new(0x1234))
        );
        assert_eq!(
            evaluate_fault(
                &[0x34, 0x12],
                &[MethodStep::ReadUnsigned {
                    offset: 0,
                    width: IntegerWidth::U16
                }]
            )
            .unwrap(),
            MethodValue::Unsigned(0x1234)
        );

        let null = evaluate_fault(
            &[0; size_of::<usize>()],
            &[MethodStep::ReadPointer { offset: 0 }],
        )
        .unwrap_err();
        assert!(matches!(
            null.cause,
            MethodErrorCause::Memory(MemoryError::NullAddress)
        ));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn preserves_typed_step_context() {
        let error = evaluate_fault(&[0x90], &[MethodStep::X86FollowDirectBranch]).unwrap_err();
        assert_eq!(error.step_index, 0);
        assert_eq!(error.operation, MethodOperation::X86FollowDirectBranch);
        assert!(matches!(
            error.cause,
            MethodErrorCause::Instruction(InstructionError::UnsupportedOpcode { .. })
        ));
    }

    #[test]
    fn enforces_limits_and_value_kinds() {
        let limits = MethodLimits {
            max_steps: 1,
            ..MethodLimits::default()
        };
        let error = MethodPlan::new(&[MethodStep::Add(1), MethodStep::Add(1)])
            .evaluate_fault_contained(&memory(&[0]), BASE, limits)
            .unwrap_err();
        assert!(matches!(
            error.cause,
            MethodErrorCause::StepLimitExceeded { .. }
        ));

        #[cfg(target_arch = "x86_64")]
        {
            let limits = MethodLimits {
                max_instruction_bytes: 1,
                ..MethodLimits::default()
            };
            let error = MethodPlan::new(&[MethodStep::X86ResolveRelative {
                displacement_offset: 0,
                width: SignedWidth::I32,
            }])
            .evaluate_fault_contained(&memory(&[0; 4]), BASE, limits)
            .unwrap_err();
            assert!(matches!(
                error.cause,
                MethodErrorCause::InstructionByteLimitExceeded { .. }
            ));
        }

        #[cfg(target_arch = "aarch64")]
        {
            let limits = MethodLimits {
                max_instruction_bytes: 1,
                ..MethodLimits::default()
            };
            let error = MethodPlan::new(&[MethodStep::Arm64ReadImmediate])
                .evaluate_fault_contained(&memory(&[0; 4]), BASE, limits)
                .unwrap_err();
            assert!(matches!(
                error.cause,
                MethodErrorCause::InstructionByteLimitExceeded { .. }
            ));
        }

        let error = evaluate_fault(
            &[1],
            &[
                MethodStep::ReadUnsigned {
                    offset: 0,
                    width: IntegerWidth::U8,
                },
                MethodStep::Add(1),
            ],
        )
        .unwrap_err();
        assert!(matches!(
            error.cause,
            MethodErrorCause::AddressRequired {
                actual: MethodValueKind::Unsigned
            }
        ));
    }

    #[test]
    fn memory_failures_remain_typed() {
        let error = evaluate_fault(&[0], &[MethodStep::ReadPointer { offset: 0 }]).unwrap_err();
        assert!(matches!(error.cause, MethodErrorCause::Memory(_)));
    }

    #[test]
    fn rejects_steps_for_a_different_architecture() {
        #[cfg(target_arch = "x86_64")]
        let step = MethodStep::Arm64ReadImmediate;
        #[cfg(target_arch = "aarch64")]
        let step = MethodStep::X86FollowDirectBranch;
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let step = MethodStep::X86FollowDirectBranch;

        let error = evaluate_fault(&[0; 8], &[step]).unwrap_err();
        assert_eq!(error.step_index, 0);
        assert!(matches!(
            error.cause,
            MethodErrorCause::UnsupportedArchitecture { .. }
        ));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn read_range_honors_region_limit() {
        let memory = FakeMemory::new();
        for (offset, bytes) in [(0, [0xfc, 0xff]), (2, [0xff, 0xff])] {
            memory
                .map_region(
                    MemoryRegion {
                        range: BASE.checked_add(offset).unwrap().checked_range(2).unwrap(),
                        access: Access::READ,
                        kind: RegionKind::Private,
                        module: None,
                    },
                    bytes,
                )
                .unwrap();
        }
        let limits = MethodLimits {
            max_regions_per_read: 1,
            ..MethodLimits::default()
        };
        let error = MethodPlan::new(&[MethodStep::X86ResolveRelative {
            displacement_offset: 0,
            width: SignedWidth::I32,
        }])
        .evaluate_fault_contained(&memory, BASE, limits)
        .unwrap_err();
        assert!(matches!(
            error.cause,
            MethodErrorCause::Memory(MemoryError::RegionLimitExceeded { limit: 1 })
        ));
    }
}
