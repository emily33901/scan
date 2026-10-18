//! Live-memory adapters for portable address-resolution plans.

use std::mem::size_of;

use thiserror::Error;

use crate::memory::{
    Access, Address, DirectMemoryAccess, DirectMemoryScope, FaultContainedMemoryAccess, MemoryError,
};

pub use scan_plan::{
    ByteOrder, EvaluationError as PlanEvaluationError,
    EvaluationErrorCause as PlanEvaluationErrorCause, InstructionError,
    IntegerWidth as PlanIntegerWidth, Limits as PlanLimits, Operation as MethodOperation,
    Operation as PlanOperation, Plan, PlanMemory, PointerWidth, ReadPurpose, SignedIntegerWidth,
    SignedWidth, Step as PlanStep, TargetArchitecture, TargetContext, Value as PlanValue,
    ValueKind as MethodValueKind, ValueKind as PlanValueKind, SCHEMA_VERSION,
};

pub mod x86_64;

/// Adapts checked, direct access to [`PlanMemory`].
///
/// # Safety
///
/// For every address that the evaluated plan can derive, the complete range
/// must remain mapped, readable, initialized, and synchronized until the plan
/// finishes evaluating.
pub struct DirectPlanMemory<'a, M: ?Sized> {
    memory: &'a M,
    max_regions_per_read: usize,
}

impl<'a, M: DirectMemoryAccess + ?Sized> DirectPlanMemory<'a, M> {
    /// Creates a portable-plan adapter over direct process memory.
    ///
    /// # Safety
    ///
    /// The caller must uphold the type-level safety contract for every read
    /// requested through this adapter.
    pub const unsafe fn new(memory: &'a M, max_regions_per_read: usize) -> Self {
        Self {
            memory,
            max_regions_per_read,
        }
    }
}

impl<M: DirectMemoryAccess + ?Sized> PlanMemory for DirectPlanMemory<'_, M> {
    type Error = MemoryError;

    fn read_exact(
        &self,
        target_address: u64,
        _purpose: ReadPurpose,
        destination: &mut [u8],
    ) -> Result<(), Self::Error> {
        let address = checked_address(target_address)?;
        self.memory.check_range(
            address.checked_range(destination.len())?,
            Access::READ,
            self.max_regions_per_read,
        )?;
        // SAFETY: construction of this adapter delegates the direct-read
        // contract to its caller, and the complete range was preflighted.
        unsafe { self.memory.read_bytes_into(address, destination) }
    }
}

/// Adapts a branded direct-memory scope to [`PlanMemory`].
pub struct ScopedPlanMemory<'a, 'scope, M: ?Sized> {
    scope: &'a DirectMemoryScope<'scope, M>,
    max_regions_per_read: usize,
}

impl<'a, 'scope, M: DirectMemoryAccess + ?Sized> ScopedPlanMemory<'a, 'scope, M> {
    pub const fn new(scope: &'a DirectMemoryScope<'scope, M>, max_regions_per_read: usize) -> Self {
        Self {
            scope,
            max_regions_per_read,
        }
    }
}

impl<M: DirectMemoryAccess + ?Sized> PlanMemory for ScopedPlanMemory<'_, '_, M> {
    type Error = MemoryError;

    fn read_exact(
        &self,
        target_address: u64,
        _purpose: ReadPurpose,
        destination: &mut [u8],
    ) -> Result<(), Self::Error> {
        let address = checked_address(target_address)?;
        self.scope.check_range(
            address.checked_range(destination.len())?,
            Access::READ,
            self.max_regions_per_read,
        )?;
        self.scope.read_bytes_into(address, destination)
    }
}

/// Adapts fault-contained memory access to [`PlanMemory`].
pub struct FaultContainedPlanMemory<'a, M: ?Sized> {
    memory: &'a M,
    max_regions_per_read: usize,
}

impl<'a, M: FaultContainedMemoryAccess + ?Sized> FaultContainedPlanMemory<'a, M> {
    pub const fn new(memory: &'a M, max_regions_per_read: usize) -> Self {
        Self {
            memory,
            max_regions_per_read,
        }
    }
}

impl<M: FaultContainedMemoryAccess + ?Sized> PlanMemory for FaultContainedPlanMemory<'_, M> {
    type Error = MemoryError;

    fn read_exact(
        &self,
        target_address: u64,
        _purpose: ReadPurpose,
        destination: &mut [u8],
    ) -> Result<(), Self::Error> {
        let address = checked_address(target_address)?;
        self.memory.check_range(
            address.checked_range(destination.len())?,
            Access::READ,
            self.max_regions_per_read,
        )?;
        self.memory.try_read_bytes_into(address, destination)
    }
}

fn checked_address(address: u64) -> Result<Address, MemoryError> {
    usize::try_from(address)
        .map(Address::new)
        .map_err(|_| MemoryError::AddressOverflow)
}

/// Compatibility view of a method plan using scan's original step syntax.
///
/// Evaluation is performed as one [`scan_plan::Plan`], so all limits and
/// instruction accounting apply across the complete plan.
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
        // SAFETY: this method has the same contract as the adapter constructor.
        let memory = unsafe { DirectPlanMemory::new(memory, limits.max_regions_per_read) };
        self.evaluate(&memory, start, limits)
    }

    pub fn evaluate_scoped<M: DirectMemoryAccess + ?Sized>(
        self,
        scope: &DirectMemoryScope<'_, M>,
        start: Address,
        limits: MethodLimits,
    ) -> Result<MethodValue, MethodError> {
        self.evaluate(
            &ScopedPlanMemory::new(scope, limits.max_regions_per_read),
            start,
            limits,
        )
    }

    pub fn evaluate_fault_contained<M: FaultContainedMemoryAccess + ?Sized>(
        self,
        memory: &M,
        start: Address,
        limits: MethodLimits,
    ) -> Result<MethodValue, MethodError> {
        self.evaluate(
            &FaultContainedPlanMemory::new(memory, limits.max_regions_per_read),
            start,
            limits,
        )
    }

    fn evaluate<M: PlanMemory<Error = MemoryError> + ?Sized>(
        self,
        memory: &M,
        start: Address,
        limits: MethodLimits,
    ) -> Result<MethodValue, MethodError> {
        if self.steps.len() > limits.max_steps {
            return Err(MethodError {
                step_index: limits.max_steps,
                operation: MethodOperation::Plan,
                cause: MethodErrorCause::StepLimitExceeded {
                    actual: self.steps.len(),
                    limit: limits.max_steps,
                },
            });
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        validate_architectures(self.steps)?;
        let mut steps = Vec::with_capacity(self.steps.len());
        for (step_index, step) in self.steps.iter().copied().enumerate() {
            steps.push(step.to_plan_step().map_err(|cause| MethodError {
                step_index,
                operation: step.operation(),
                cause,
            })?);
        }
        let plan = Plan::new(inferred_result_kind(self.steps), steps);
        let value = plan
            .evaluate(
                memory,
                start.get() as u64,
                native_target_context(),
                limits.into_plan_limits(),
            )
            .map_err(MethodError::from_plan)?;
        MethodValue::from_plan(value).ok_or(MethodError {
            step_index: self.steps.len(),
            operation: MethodOperation::Plan,
            cause: MethodErrorCause::AddressOverflow,
        })
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
    ReadSigned {
        offset: i64,
        width: SignedIntegerWidth,
    },
    UnsignedToAddress,
    X86ResolveRelative {
        displacement_offset: usize,
        width: SignedWidth,
    },
    X86ResolvePcRelative {
        displacement_offset: usize,
        instruction_length: usize,
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
            Self::ReadSigned { .. } => MethodOperation::ReadSigned,
            Self::UnsignedToAddress => MethodOperation::UnsignedToAddress,
            Self::X86ResolveRelative { .. } => MethodOperation::X86ResolveRelative,
            Self::X86ResolvePcRelative { .. } => MethodOperation::X86ResolvePcRelative,
            Self::X86FollowDirectBranch => MethodOperation::X86FollowDirectBranch,
            Self::Arm64ReadImmediate => MethodOperation::Arm64ReadImmediate,
            Self::Arm64ResolveImmediateRelative => MethodOperation::Arm64ResolveImmediateRelative,
            Self::Arm64ResolvePageOffset { .. } => MethodOperation::Arm64ResolvePageOffset,
            Self::Arm64FollowDirectBranch => MethodOperation::Arm64FollowDirectBranch,
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    const fn required_architecture(self) -> Option<Architecture> {
        match self {
            Self::X86ResolveRelative { .. }
            | Self::X86ResolvePcRelative { .. }
            | Self::X86FollowDirectBranch => Some(Architecture::X86_64),
            Self::Arm64ReadImmediate
            | Self::Arm64ResolveImmediateRelative
            | Self::Arm64ResolvePageOffset { .. }
            | Self::Arm64FollowDirectBranch => Some(Architecture::Aarch64),
            _ => None,
        }
    }

    fn to_plan_step(self) -> Result<PlanStep, MethodErrorCause> {
        Ok(match self {
            Self::Add(offset) => PlanStep::Add { offset },
            Self::ReadPointer { offset } => PlanStep::ReadPointer {
                offset,
                width: native_pointer_width(),
            },
            Self::ReadUnsigned { offset, width } => PlanStep::ReadUnsigned {
                offset,
                width: width.to_plan_width(),
            },
            Self::ReadSigned { offset, width } => PlanStep::ReadSigned { offset, width },
            Self::X86ResolvePcRelative {
                displacement_offset,
                instruction_length,
            } => PlanStep::X86ResolvePcRelative {
                displacement_offset: u64::try_from(displacement_offset)
                    .map_err(|_| MethodErrorCause::AddressOverflow)?,
                instruction_length: u64::try_from(instruction_length)
                    .map_err(|_| MethodErrorCause::AddressOverflow)?,
            },
            Self::UnsignedToAddress => PlanStep::UnsignedToAddress,
            Self::X86ResolveRelative {
                displacement_offset,
                width,
            } => PlanStep::X86ResolveRelative {
                displacement_offset: u64::try_from(displacement_offset)
                    .map_err(|_| MethodErrorCause::AddressOverflow)?,
                width,
            },
            Self::X86FollowDirectBranch => PlanStep::X86FollowDirectBranch,
            Self::Arm64ReadImmediate => PlanStep::Arm64ReadImmediate,
            Self::Arm64ResolveImmediateRelative => PlanStep::Arm64ResolveImmediateRelative,
            Self::Arm64ResolvePageOffset { instruction_offset } => {
                PlanStep::Arm64ResolvePageOffset {
                    instruction_offset: u64::try_from(instruction_offset)
                        .map_err(|_| MethodErrorCause::AddressOverflow)?,
                }
            }
            Self::Arm64FollowDirectBranch => PlanStep::Arm64FollowDirectBranch,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Architecture {
    X86_64,
    Aarch64,
    Other,
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
impl Architecture {
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

impl IntegerWidth {
    const fn to_plan_width(self) -> PlanIntegerWidth {
        match self {
            Self::U8 => PlanIntegerWidth::U8,
            Self::U16 => PlanIntegerWidth::U16,
            Self::U32 => PlanIntegerWidth::U32,
            Self::U64 => PlanIntegerWidth::U64,
            Self::Usize if size_of::<usize>() == 4 => PlanIntegerWidth::U32,
            Self::Usize => PlanIntegerWidth::U64,
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

    fn from_plan(value: PlanValue) -> Option<Self> {
        match value {
            PlanValue::Address(address) => checked_address(address).ok().map(Self::Address),
            PlanValue::Signed(value) => Some(Self::Signed(value)),
            PlanValue::Unsigned(value) => Some(Self::Unsigned(value)),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MethodLimits {
    pub max_steps: usize,
    pub max_instruction_bytes: usize,
    pub max_regions_per_read: usize,
}

impl MethodLimits {
    const fn into_plan_limits(self) -> PlanLimits {
        PlanLimits {
            max_steps: self.max_steps,
            max_instruction_bytes: self.max_instruction_bytes,
            max_reads: usize::MAX,
            max_read_bytes: usize::MAX,
        }
    }
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

#[derive(Debug, Error)]
#[error("method step {step_index} ({operation}) failed: {cause}")]
pub struct MethodError {
    pub step_index: usize,
    pub operation: MethodOperation,
    #[source]
    pub cause: MethodErrorCause,
}

impl MethodError {
    fn from_plan(error: PlanEvaluationError<MemoryError>) -> Self {
        Self {
            step_index: error.step_index,
            operation: error.operation,
            cause: MethodErrorCause::from_plan(error.cause),
        }
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MethodErrorCause {
    #[error("plan has {actual} steps, exceeding limit {limit}")]
    StepLimitExceeded { actual: usize, limit: usize },
    #[error("instruction reads used {actual} bytes, exceeding limit {limit}")]
    InstructionByteLimitExceeded { actual: usize, limit: usize },
    #[error("memory reads used {actual} operations, exceeding limit {limit}")]
    ReadLimitExceeded { actual: usize, limit: usize },
    #[error("memory reads used {actual} bytes, exceeding limit {limit}")]
    ReadByteLimitExceeded { actual: usize, limit: usize },
    #[error("operation requires an address, but current value is {actual:?}")]
    AddressRequired { actual: MethodValueKind },
    #[error("plan declares {expected:?}, but produced {actual:?}")]
    ResultKindMismatch {
        expected: MethodValueKind,
        actual: MethodValueKind,
    },
    #[error("address calculation overflowed")]
    AddressOverflow,
    #[error("unsupported target context: {architecture:?}/{pointer_width:?}/{byte_order:?}")]
    UnsupportedTargetContext {
        architecture: TargetArchitecture,
        pointer_width: PointerWidth,
        byte_order: ByteOrder,
    },
    #[error("pointer width {width:?} does not match target width {target:?}")]
    PointerWidthMismatch {
        width: PointerWidth,
        target: PointerWidth,
    },
    #[error("operation requires {required:?}, but current architecture is {actual:?}")]
    UnsupportedArchitecture {
        required: Architecture,
        actual: Architecture,
    },
    #[error("memory access failed")]
    Memory(#[source] MemoryError),
    #[error("instruction decoding failed")]
    Instruction(#[source] InstructionError),
    #[error("portable plan evaluation failed")]
    Plan(#[source] PlanEvaluationErrorCause<MemoryError>),
}

impl MethodErrorCause {
    fn from_plan(cause: PlanEvaluationErrorCause<MemoryError>) -> Self {
        match cause {
            PlanEvaluationErrorCause::StepLimitExceeded { actual, limit } => {
                Self::StepLimitExceeded { actual, limit }
            }
            PlanEvaluationErrorCause::InstructionByteLimitExceeded { actual, limit } => {
                Self::InstructionByteLimitExceeded { actual, limit }
            }
            PlanEvaluationErrorCause::ReadLimitExceeded { actual, limit } => {
                Self::ReadLimitExceeded { actual, limit }
            }
            PlanEvaluationErrorCause::ReadByteLimitExceeded { actual, limit } => {
                Self::ReadByteLimitExceeded { actual, limit }
            }
            PlanEvaluationErrorCause::AddressRequired { actual } => {
                Self::AddressRequired { actual }
            }
            PlanEvaluationErrorCause::ResultKindMismatch { expected, actual } => {
                Self::ResultKindMismatch { expected, actual }
            }
            PlanEvaluationErrorCause::AddressOverflow => Self::AddressOverflow,
            PlanEvaluationErrorCause::UnsupportedTargetContext {
                architecture,
                pointer_width,
                byte_order,
            } => Self::UnsupportedTargetContext {
                architecture,
                pointer_width,
                byte_order,
            },
            PlanEvaluationErrorCause::NullPointer => Self::Memory(MemoryError::NullAddress),
            PlanEvaluationErrorCause::PointerWidthMismatch { width, target } => {
                Self::PointerWidthMismatch { width, target }
            }
            PlanEvaluationErrorCause::UnsupportedArchitecture { required, actual } => {
                Self::UnsupportedArchitecture {
                    required: architecture_from_plan(required),
                    actual: architecture_from_plan(actual),
                }
            }
            PlanEvaluationErrorCause::Memory(error) => Self::Memory(error),
            PlanEvaluationErrorCause::Instruction(error) => Self::Instruction(error),
            other => Self::Plan(other),
        }
    }
}

impl From<InstructionError> for MethodErrorCause {
    fn from(source: InstructionError) -> Self {
        Self::Instruction(source)
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn validate_architectures(steps: &[MethodStep]) -> Result<(), MethodError> {
    let actual = Architecture::current();
    for (step_index, step) in steps.iter().copied().enumerate() {
        if let Some(required) = step.required_architecture() {
            if actual != required {
                return Err(MethodError {
                    step_index,
                    operation: step.operation(),
                    cause: MethodErrorCause::UnsupportedArchitecture { required, actual },
                });
            }
        }
    }
    Ok(())
}

fn inferred_result_kind(steps: &[MethodStep]) -> PlanValueKind {
    steps
        .last()
        .map_or(PlanValueKind::Address, |step| match step {
            MethodStep::ReadUnsigned { .. } => PlanValueKind::Unsigned,
            MethodStep::Arm64ReadImmediate | MethodStep::ReadSigned { .. } => PlanValueKind::Signed,
            _ => PlanValueKind::Address,
        })
}

const fn native_pointer_width() -> PointerWidth {
    if size_of::<usize>() == 4 {
        PointerWidth::U32
    } else {
        PointerWidth::U64
    }
}

const fn native_target_context() -> TargetContext {
    TargetContext {
        #[cfg(target_arch = "aarch64")]
        architecture: TargetArchitecture::Aarch64,
        #[cfg(not(target_arch = "aarch64"))]
        architecture: TargetArchitecture::X86_64,
        pointer_width: native_pointer_width(),
        #[cfg(target_endian = "little")]
        byte_order: ByteOrder::Little,
        #[cfg(target_endian = "big")]
        byte_order: ByteOrder::Big,
    }
}

const fn architecture_from_plan(architecture: TargetArchitecture) -> Architecture {
    match architecture {
        TargetArchitecture::X86_64 => Architecture::X86_64,
        TargetArchitecture::Aarch64 => Architecture::Aarch64,
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

    #[test]
    fn canonical_plan_uses_fault_contained_adapter() {
        let plan = Plan::new(
            PlanValueKind::Unsigned,
            vec![PlanStep::ReadUnsigned {
                offset: 0,
                width: PlanIntegerWidth::U16,
            }],
        );
        assert_eq!(
            plan.evaluate(
                &FaultContainedPlanMemory::new(&memory(&[0x34, 0x12]), 1),
                BASE.get() as u64,
                native_target_context(),
                PlanLimits::default(),
            )
            .unwrap(),
            PlanValue::Unsigned(0x1234)
        );
    }

    #[test]
    fn canonical_plan_uses_direct_and_scoped_adapters() {
        use crate::memory::with_direct_memory_scope;

        let memory = memory(&[0x34, 0x12]);
        let plan = Plan::new(
            PlanValueKind::Unsigned,
            vec![PlanStep::ReadUnsigned {
                offset: 0,
                width: PlanIntegerWidth::U16,
            }],
        );
        let evaluate = |memory: &dyn PlanMemory<Error = MemoryError>| {
            plan.evaluate(
                memory,
                BASE.get() as u64,
                native_target_context(),
                PlanLimits::default(),
            )
            .unwrap()
        };

        // SAFETY: FakeMemory owns stable initialized storage for the evaluation.
        let direct = unsafe { DirectPlanMemory::new(&memory, 1) };
        assert_eq!(evaluate(&direct), PlanValue::Unsigned(0x1234));

        // SAFETY: FakeMemory owns stable initialized storage for the scope.
        let scoped = unsafe {
            with_direct_memory_scope(&memory, |scope| evaluate(&ScopedPlanMemory::new(scope, 1)))
        };
        assert_eq!(scoped, PlanValue::Unsigned(0x1234));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn new_steps_agree_across_adapters_and_method_plans() {
        use crate::memory::with_direct_memory_scope;

        let bytes = [0x83, 0x3d, 0xf0, 0xff, 0xff, 0xff, 0x01];
        let memory = memory(&bytes);
        for (step, result, expected) in [
            (
                MethodStep::ReadSigned {
                    offset: 2,
                    width: SignedIntegerWidth::I32,
                },
                PlanValueKind::Signed,
                PlanValue::Signed(-16),
            ),
            (
                MethodStep::X86ResolvePcRelative {
                    displacement_offset: 2,
                    instruction_length: 7,
                },
                PlanValueKind::Address,
                PlanValue::Address(BASE.get() as u64 - 9),
            ),
        ] {
            let plan = Plan::new(result, vec![step.to_plan_step().unwrap()]);
            let evaluate = |memory: &dyn PlanMemory<Error = MemoryError>| {
                plan.evaluate(
                    memory,
                    BASE.get() as u64,
                    native_target_context(),
                    PlanLimits::default(),
                )
                .unwrap()
            };
            assert_eq!(
                evaluate(&FaultContainedPlanMemory::new(&memory, 1)),
                expected
            );
            // SAFETY: FakeMemory owns stable initialized storage for all reads.
            assert_eq!(
                evaluate(&unsafe { DirectPlanMemory::new(&memory, 1) }),
                expected
            );
            // SAFETY: FakeMemory owns stable initialized storage for the scope.
            let scoped = unsafe {
                with_direct_memory_scope(&memory, |scope| {
                    evaluate(&ScopedPlanMemory::new(scope, 1))
                })
            };
            assert_eq!(scoped, expected);
            assert_eq!(
                evaluate_fault(&bytes, &[step]).unwrap(),
                MethodValue::from_plan(expected).unwrap()
            );
        }
    }
}
