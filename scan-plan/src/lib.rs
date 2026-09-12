//! Portable, bounded evaluation of typed binary address-resolution plans.

mod aarch64;
pub mod x86_64;

use std::error::Error as StdError;
use std::fmt;

use thiserror::Error;

/// The only schema version this crate writes or accepts.
pub const SCHEMA_VERSION: u32 = 1;

/// Applies a recipe's signed match offset to a raw target address.
pub fn apply_match_offset(raw_match: u64, match_offset: i64) -> Result<u64, MatchOffsetError> {
    raw_match
        .checked_add_signed(match_offset)
        .ok_or(MatchOffsetError {
            raw_match,
            match_offset,
        })
}

/// The raw match and signed offset do not identify a target address.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[error("match offset {match_offset} overflows raw match address {raw_match:#x}")]
#[non_exhaustive]
pub struct MatchOffsetError {
    pub raw_match: u64,
    pub match_offset: i64,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TargetArchitecture {
    X86_64,
    Aarch64,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PointerWidth {
    U32,
    U64,
}

impl PointerWidth {
    const fn bytes(self) -> usize {
        match self {
            Self::U32 => 4,
            Self::U64 => 8,
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ByteOrder {
    Little,
    Big,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TargetContext {
    pub architecture: TargetArchitecture,
    pub pointer_width: PointerWidth,
    pub byte_order: ByteOrder,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueKind {
    Address,
    Signed,
    Unsigned,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Value {
    Address(u64),
    Signed(i64),
    Unsigned(u64),
}

impl Value {
    pub const fn kind(self) -> ValueKind {
        match self {
            Self::Address(_) => ValueKind::Address,
            Self::Signed(_) => ValueKind::Signed,
            Self::Unsigned(_) => ValueKind::Unsigned,
        }
    }

    pub fn into_address(self) -> Result<u64, ValueKind> {
        match self {
            Self::Address(value) => Ok(value),
            other => Err(other.kind()),
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntegerWidth {
    U8,
    U16,
    U32,
    U64,
}

impl IntegerWidth {
    const fn bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 => 4,
            Self::U64 => 8,
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignedWidth {
    I32,
    I64,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)
)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Step {
    Add {
        offset: i64,
    },
    ReadPointer {
        offset: i64,
        width: PointerWidth,
    },
    ReadUnsigned {
        offset: i64,
        width: IntegerWidth,
    },
    UnsignedToAddress,
    X86ResolveRelative {
        displacement_offset: u64,
        width: SignedWidth,
    },
    X86FollowDirectBranch,
    Arm64ReadImmediate,
    Arm64ResolveImmediateRelative,
    Arm64ResolvePageOffset {
        instruction_offset: u64,
    },
    Arm64FollowDirectBranch,
}

impl Step {
    pub const fn operation(self) -> Operation {
        match self {
            Self::Add { .. } => Operation::Add,
            Self::ReadPointer { .. } => Operation::ReadPointer,
            Self::ReadUnsigned { .. } => Operation::ReadUnsigned,
            Self::UnsignedToAddress => Operation::UnsignedToAddress,
            Self::X86ResolveRelative { .. } => Operation::X86ResolveRelative,
            Self::X86FollowDirectBranch => Operation::X86FollowDirectBranch,
            Self::Arm64ReadImmediate => Operation::Arm64ReadImmediate,
            Self::Arm64ResolveImmediateRelative => Operation::Arm64ResolveImmediateRelative,
            Self::Arm64ResolvePageOffset { .. } => Operation::Arm64ResolvePageOffset,
            Self::Arm64FollowDirectBranch => Operation::Arm64FollowDirectBranch,
        }
    }

    const fn required_architecture(self) -> Option<TargetArchitecture> {
        match self {
            Self::X86ResolveRelative { .. } | Self::X86FollowDirectBranch => {
                Some(TargetArchitecture::X86_64)
            }
            Self::Arm64ReadImmediate
            | Self::Arm64ResolveImmediateRelative
            | Self::Arm64ResolvePageOffset { .. }
            | Self::Arm64FollowDirectBranch => Some(TargetArchitecture::Aarch64),
            _ => None,
        }
    }
}

/// A versioned plan with an explicitly declared terminal value type.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Plan {
    schema_version: u32,
    pub result: ValueKind,
    pub steps: Vec<Step>,
}

impl Plan {
    pub fn new(result: ValueKind, steps: Vec<Step>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            result,
            steps,
        }
    }

    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    pub fn evaluate<M: PlanMemory + ?Sized>(
        &self,
        memory: &M,
        start: u64,
        context: TargetContext,
        limits: Limits,
    ) -> Result<Value, EvaluationError<M::Error>> {
        evaluate(self, memory, start, context, limits)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Plan {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct WirePlan {
            schema_version: u32,
            result: ValueKind,
            steps: Vec<Step>,
        }

        let wire = <WirePlan as serde::Deserialize>::deserialize(deserializer)?;
        if wire.schema_version != SCHEMA_VERSION {
            return Err(serde::de::Error::custom(format_args!(
                "unsupported schema version {}; expected {SCHEMA_VERSION}",
                wire.schema_version
            )));
        }
        Ok(Self::new(wire.result, wire.steps))
    }
}

/// Describes why the evaluator is requesting target bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ReadPurpose {
    Instruction,
    Pointer,
    Unsigned,
}

/// Checked access to bytes in the target's virtual address space.
pub trait PlanMemory {
    type Error: StdError + 'static;

    fn read_exact(
        &self,
        target_address: u64,
        purpose: ReadPurpose,
        destination: &mut [u8],
    ) -> Result<(), Self::Error>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Limits {
    pub max_steps: usize,
    pub max_instruction_bytes: usize,
    pub max_reads: usize,
    pub max_read_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_steps: 128,
            max_instruction_bytes: 4_096,
            max_reads: 1_024,
            max_read_bytes: 64 * 1_024,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Operation {
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

impl fmt::Display for Operation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

#[derive(Debug, Error)]
#[error("plan step {step_index} ({operation}) failed: {cause}")]
pub struct EvaluationError<E: StdError + 'static> {
    pub step_index: usize,
    pub operation: Operation,
    #[source]
    pub cause: EvaluationErrorCause<E>,
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum EvaluationErrorCause<E: StdError + 'static> {
    #[error("plan has {actual} steps, exceeding limit {limit}")]
    StepLimitExceeded { actual: usize, limit: usize },
    #[error("instruction reads used {actual} bytes, exceeding limit {limit}")]
    InstructionByteLimitExceeded { actual: usize, limit: usize },
    #[error("memory reads used {actual} operations, exceeding limit {limit}")]
    ReadLimitExceeded { actual: usize, limit: usize },
    #[error("memory reads used {actual} bytes, exceeding limit {limit}")]
    ReadByteLimitExceeded { actual: usize, limit: usize },
    #[error("operation requires an address, but current value is {actual:?}")]
    AddressRequired { actual: ValueKind },
    #[error("plan declares {expected:?}, but produced {actual:?}")]
    ResultKindMismatch {
        expected: ValueKind,
        actual: ValueKind,
    },
    #[error("address calculation overflowed")]
    AddressOverflow,
    #[error("unsupported target context: {architecture:?}/{pointer_width:?}/{byte_order:?}")]
    UnsupportedTargetContext {
        architecture: TargetArchitecture,
        pointer_width: PointerWidth,
        byte_order: ByteOrder,
    },
    #[error("null pointer read")]
    NullPointer,
    #[error("pointer width {width:?} does not match target width {target:?}")]
    PointerWidthMismatch {
        width: PointerWidth,
        target: PointerWidth,
    },
    #[error("operation requires {required:?}, but target architecture is {actual:?}")]
    UnsupportedArchitecture {
        required: TargetArchitecture,
        actual: TargetArchitecture,
    },
    #[error("memory access failed")]
    Memory(#[source] E),
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

impl<E: StdError + 'static> From<InstructionError> for EvaluationErrorCause<E> {
    fn from(source: InstructionError) -> Self {
        Self::Instruction(source)
    }
}

struct Evaluation<'a, M: ?Sized> {
    memory: &'a M,
    context: TargetContext,
    limits: Limits,
    reads: usize,
    read_bytes: usize,
    instruction_bytes: usize,
}

impl<M: PlanMemory + ?Sized> Evaluation<'_, M> {
    fn read(
        &mut self,
        address: u64,
        purpose: ReadPurpose,
        output: &mut [u8],
    ) -> Result<(), EvaluationErrorCause<M::Error>> {
        if let Some(last_offset) = output.len().checked_sub(1) {
            let last_offset =
                u64::try_from(last_offset).map_err(|_| EvaluationErrorCause::AddressOverflow)?;
            address
                .checked_add(last_offset)
                .ok_or(EvaluationErrorCause::AddressOverflow)?;
        }
        let reads = self
            .reads
            .checked_add(1)
            .ok_or(EvaluationErrorCause::ReadLimitExceeded {
                actual: usize::MAX,
                limit: self.limits.max_reads,
            })?;
        if reads > self.limits.max_reads {
            return Err(EvaluationErrorCause::ReadLimitExceeded {
                actual: reads,
                limit: self.limits.max_reads,
            });
        }
        let read_bytes = self.read_bytes.checked_add(output.len()).ok_or(
            EvaluationErrorCause::ReadByteLimitExceeded {
                actual: usize::MAX,
                limit: self.limits.max_read_bytes,
            },
        )?;
        if read_bytes > self.limits.max_read_bytes {
            return Err(EvaluationErrorCause::ReadByteLimitExceeded {
                actual: read_bytes,
                limit: self.limits.max_read_bytes,
            });
        }
        self.memory
            .read_exact(address, purpose, output)
            .map_err(EvaluationErrorCause::Memory)?;
        self.reads = reads;
        self.read_bytes = read_bytes;
        Ok(())
    }

    fn read_instruction(
        &mut self,
        address: u64,
        output: &mut [u8],
    ) -> Result<(), EvaluationErrorCause<M::Error>> {
        let used = self.instruction_bytes.checked_add(output.len()).ok_or(
            EvaluationErrorCause::InstructionByteLimitExceeded {
                actual: usize::MAX,
                limit: self.limits.max_instruction_bytes,
            },
        )?;
        if used > self.limits.max_instruction_bytes {
            return Err(EvaluationErrorCause::InstructionByteLimitExceeded {
                actual: used,
                limit: self.limits.max_instruction_bytes,
            });
        }
        self.read(address, ReadPurpose::Instruction, output)?;
        self.instruction_bytes = used;
        Ok(())
    }

    fn read_u32_instruction(
        &mut self,
        address: u64,
    ) -> Result<u32, EvaluationErrorCause<M::Error>> {
        let mut bytes = [0; 4];
        self.read_instruction(address, &mut bytes)?;
        Ok(match self.context.byte_order {
            ByteOrder::Little => u32::from_le_bytes(bytes),
            ByteOrder::Big => u32::from_be_bytes(bytes),
        })
    }

    fn read_unsigned(
        &mut self,
        address: u64,
        width: IntegerWidth,
    ) -> Result<u64, EvaluationErrorCause<M::Error>> {
        let mut bytes = [0; 8];
        let len = width.bytes();
        self.read(address, ReadPurpose::Unsigned, &mut bytes[..len])?;
        Ok(decode_unsigned(&bytes[..len], self.context.byte_order))
    }

    fn read_pointer(
        &mut self,
        address: u64,
        width: PointerWidth,
    ) -> Result<u64, EvaluationErrorCause<M::Error>> {
        if width != self.context.pointer_width {
            return Err(EvaluationErrorCause::PointerWidthMismatch {
                width,
                target: self.context.pointer_width,
            });
        }
        let mut bytes = [0; 8];
        let len = width.bytes();
        self.read(address, ReadPurpose::Pointer, &mut bytes[..len])?;
        let pointer = decode_unsigned(&bytes[..len], self.context.byte_order);
        if pointer == 0 {
            return Err(EvaluationErrorCause::NullPointer);
        }
        Ok(pointer)
    }

    fn read_x86_branch(
        &mut self,
        address: u64,
    ) -> Result<([u8; 6], usize), EvaluationErrorCause<M::Error>> {
        let mut bytes = [0; 6];
        self.read_instruction(address, &mut bytes[..1])?;
        let known = if bytes[0] == 0x0f {
            self.read_instruction(checked_add(address, 1)?, &mut bytes[1..2])?;
            2
        } else {
            1
        };
        let length = x86_64::direct_branch_length(&bytes[..known])?;
        self.read_instruction(
            checked_add(address, known as u64)?,
            &mut bytes[known..length],
        )?;
        Ok((bytes, length))
    }
}

pub fn evaluate<M: PlanMemory + ?Sized>(
    plan: &Plan,
    memory: &M,
    start: u64,
    context: TargetContext,
    limits: Limits,
) -> Result<Value, EvaluationError<M::Error>> {
    if plan.steps.len() > limits.max_steps {
        return Err(EvaluationError {
            step_index: limits.max_steps,
            operation: Operation::Plan,
            cause: EvaluationErrorCause::StepLimitExceeded {
                actual: plan.steps.len(),
                limit: limits.max_steps,
            },
        });
    }
    let mut evaluation = Evaluation {
        memory,
        context,
        limits,
        reads: 0,
        read_bytes: 0,
        instruction_bytes: 0,
    };
    let mut value = Value::Address(start);
    for (step_index, step) in plan.steps.iter().copied().enumerate() {
        value = evaluate_step(&mut evaluation, value, step).map_err(|cause| EvaluationError {
            step_index,
            operation: step.operation(),
            cause,
        })?;
    }
    if value.kind() != plan.result {
        return Err(EvaluationError {
            step_index: plan.steps.len(),
            operation: Operation::Plan,
            cause: EvaluationErrorCause::ResultKindMismatch {
                expected: plan.result,
                actual: value.kind(),
            },
        });
    }
    Ok(value)
}

fn evaluate_step<M: PlanMemory + ?Sized>(
    evaluation: &mut Evaluation<'_, M>,
    value: Value,
    step: Step,
) -> Result<Value, EvaluationErrorCause<M::Error>> {
    if let Some(required) = step.required_architecture() {
        if evaluation.context.architecture != required {
            return Err(EvaluationErrorCause::UnsupportedArchitecture {
                required,
                actual: evaluation.context.architecture,
            });
        }
    }
    if let Step::UnsignedToAddress = step {
        return match value {
            Value::Unsigned(value) => Ok(Value::Address(value)),
            other => Err(EvaluationErrorCause::AddressRequired {
                actual: other.kind(),
            }),
        };
    }
    let address = value
        .into_address()
        .map_err(|actual| EvaluationErrorCause::AddressRequired { actual })?;
    match step {
        Step::Add { offset } => Ok(Value::Address(checked_add_signed(address, offset)?)),
        Step::ReadPointer { offset, width } => Ok(Value::Address(
            evaluation.read_pointer(checked_add_signed(address, offset)?, width)?,
        )),
        Step::ReadUnsigned { offset, width } => Ok(Value::Unsigned(
            evaluation.read_unsigned(checked_add_signed(address, offset)?, width)?,
        )),
        Step::X86ResolveRelative {
            displacement_offset,
            width,
        } => {
            let displacement_address = checked_add(address, displacement_offset)?;
            let (displacement, encoded_width) = match width {
                SignedWidth::I32 => {
                    let mut bytes = [0; 4];
                    evaluation.read_instruction(displacement_address, &mut bytes)?;
                    (i64::from(i32::from_le_bytes(bytes)), 4)
                }
                SignedWidth::I64 => {
                    let mut bytes = [0; 8];
                    evaluation.read_instruction(displacement_address, &mut bytes)?;
                    (i64::from_le_bytes(bytes), 8)
                }
            };
            let end_offset = displacement_offset
                .checked_add(encoded_width)
                .ok_or(EvaluationErrorCause::AddressOverflow)?;
            Ok(Value::Address(x86_64::relative_target(
                address,
                end_offset,
                displacement,
            )?))
        }
        Step::X86FollowDirectBranch => {
            let (bytes, length) = evaluation.read_x86_branch(address)?;
            Ok(Value::Address(x86_64::direct_branch_target(
                address,
                &bytes[..length],
            )?))
        }
        Step::Arm64ReadImmediate => Ok(Value::Signed(aarch64::instruction_immediate(
            evaluation.read_u32_instruction(address)?,
        )?)),
        Step::Arm64ResolveImmediateRelative => {
            let immediate =
                aarch64::instruction_immediate(evaluation.read_u32_instruction(address)?)?;
            Ok(Value::Address(checked_add_signed(address, immediate)?))
        }
        Step::Arm64ResolvePageOffset { instruction_offset } => {
            let first_address = checked_add(address, instruction_offset)?;
            let first = evaluation.read_u32_instruction(first_address)?;
            let second = evaluation.read_u32_instruction(checked_add(first_address, 4)?)?;
            Ok(Value::Address(aarch64::page_pair_target(
                first_address,
                first,
                second,
            )?))
        }
        Step::Arm64FollowDirectBranch => Ok(Value::Address(aarch64::direct_branch_target(
            address,
            evaluation.read_u32_instruction(address)?,
        )?)),
        Step::UnsignedToAddress => unreachable!("handled above"),
    }
}

fn checked_add<E: StdError + 'static>(
    address: u64,
    offset: u64,
) -> Result<u64, EvaluationErrorCause<E>> {
    address
        .checked_add(offset)
        .ok_or(EvaluationErrorCause::AddressOverflow)
}

fn checked_add_signed<E: StdError + 'static>(
    address: u64,
    offset: i64,
) -> Result<u64, EvaluationErrorCause<E>> {
    if offset >= 0 {
        address.checked_add(offset as u64)
    } else {
        address.checked_sub(offset.unsigned_abs())
    }
    .ok_or(EvaluationErrorCause::AddressOverflow)
}

fn decode_unsigned(bytes: &[u8], order: ByteOrder) -> u64 {
    match order {
        ByteOrder::Little => bytes.iter().enumerate().fold(0, |value, (index, byte)| {
            value | (u64::from(*byte) << (index * 8))
        }),
        ByteOrder::Big => bytes
            .iter()
            .fold(0, |value, byte| (value << 8) | u64::from(*byte)),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    const BASE: u64 = 0x1000;

    #[derive(Clone, Copy, Debug, Error)]
    #[error("address is outside test memory")]
    struct TestMemoryError;

    struct TestMemory(Vec<u8>);

    impl PlanMemory for TestMemory {
        type Error = TestMemoryError;

        fn read_exact(
            &self,
            target_address: u64,
            _purpose: ReadPurpose,
            destination: &mut [u8],
        ) -> Result<(), Self::Error> {
            let offset = target_address
                .checked_sub(BASE)
                .and_then(|offset| usize::try_from(offset).ok())
                .ok_or(TestMemoryError)?;
            let end = offset
                .checked_add(destination.len())
                .ok_or(TestMemoryError)?;
            destination.copy_from_slice(self.0.get(offset..end).ok_or(TestMemoryError)?);
            Ok(())
        }
    }

    struct TrackingMemory {
        inner: TestMemory,
        purposes: RefCell<Vec<ReadPurpose>>,
    }

    impl PlanMemory for TrackingMemory {
        type Error = TestMemoryError;

        fn read_exact(
            &self,
            target_address: u64,
            purpose: ReadPurpose,
            destination: &mut [u8],
        ) -> Result<(), Self::Error> {
            self.inner
                .read_exact(target_address, purpose, destination)?;
            self.purposes.borrow_mut().push(purpose);
            Ok(())
        }
    }

    fn context(architecture: TargetArchitecture) -> TargetContext {
        TargetContext {
            architecture,
            pointer_width: PointerWidth::U64,
            byte_order: ByteOrder::Little,
        }
    }

    fn run(
        bytes: &[u8],
        architecture: TargetArchitecture,
        result: ValueKind,
        steps: Vec<Step>,
    ) -> Result<Value, EvaluationError<TestMemoryError>> {
        Plan::new(result, steps).evaluate(
            &TestMemory(bytes.to_vec()),
            BASE,
            context(architecture),
            Limits::default(),
        )
    }

    #[test]
    fn empty_plan_returns_start_address() {
        assert_eq!(
            run(&[], TargetArchitecture::X86_64, ValueKind::Address, vec![]).unwrap(),
            Value::Address(BASE)
        );
    }

    #[test]
    fn applies_signed_match_offsets_with_checked_target_arithmetic() {
        assert_eq!(apply_match_offset(BASE, 7).unwrap(), BASE + 7);
        assert_eq!(apply_match_offset(BASE, -7).unwrap(), BASE - 7);

        let underflow = apply_match_offset(0, -1).unwrap_err();
        assert_eq!(underflow.raw_match, 0);
        assert_eq!(underflow.match_offset, -1);

        let overflow = apply_match_offset(u64::MAX, 1).unwrap_err();
        assert_eq!(overflow.raw_match, u64::MAX);
        assert_eq!(overflow.match_offset, 1);
        assert!(apply_match_offset(0, i64::MIN).is_err());
    }

    #[test]
    fn evaluates_typed_reads_and_explicit_conversion() {
        let pointer = 0x1234_u64.to_le_bytes();
        assert_eq!(
            run(
                &pointer,
                TargetArchitecture::X86_64,
                ValueKind::Address,
                vec![Step::ReadPointer {
                    offset: 0,
                    width: PointerWidth::U64,
                }],
            )
            .unwrap(),
            Value::Address(0x1234)
        );
        assert_eq!(
            run(
                &[0x34, 0x12],
                TargetArchitecture::X86_64,
                ValueKind::Address,
                vec![
                    Step::ReadUnsigned {
                        offset: 0,
                        width: IntegerWidth::U16,
                    },
                    Step::UnsignedToAddress,
                ],
            )
            .unwrap(),
            Value::Address(0x1234)
        );
    }

    #[test]
    fn identifies_each_memory_read_purpose() {
        let mut bytes = vec![0; 13];
        bytes[4..12].copy_from_slice(&(BASE + 12).to_le_bytes());
        bytes[12] = 7;
        let memory = TrackingMemory {
            inner: TestMemory(bytes),
            purposes: RefCell::new(Vec::new()),
        };
        let value = Plan::new(
            ValueKind::Unsigned,
            vec![
                Step::X86ResolveRelative {
                    displacement_offset: 0,
                    width: SignedWidth::I32,
                },
                Step::ReadPointer {
                    offset: 0,
                    width: PointerWidth::U64,
                },
                Step::ReadUnsigned {
                    offset: 0,
                    width: IntegerWidth::U8,
                },
            ],
        )
        .evaluate(
            &memory,
            BASE,
            context(TargetArchitecture::X86_64),
            Limits::default(),
        )
        .unwrap();

        assert_eq!(value, Value::Unsigned(7));
        assert_eq!(
            *memory.purposes.borrow(),
            [
                ReadPurpose::Instruction,
                ReadPurpose::Pointer,
                ReadPurpose::Unsigned,
            ]
        );
    }

    #[test]
    fn resolves_cross_architecture_instructions() {
        assert_eq!(
            run(
                &(-8_i32).to_le_bytes(),
                TargetArchitecture::X86_64,
                ValueKind::Address,
                vec![Step::X86ResolveRelative {
                    displacement_offset: 0,
                    width: SignedWidth::I32,
                }],
            )
            .unwrap(),
            Value::Address(0x0ffc)
        );
        assert_eq!(
            run(
                &0x97ff_ffff_u32.to_le_bytes(),
                TargetArchitecture::Aarch64,
                ValueKind::Address,
                vec![Step::Arm64FollowDirectBranch],
            )
            .unwrap(),
            Value::Address(0x0ffc)
        );
    }

    #[test]
    fn reports_architecture_kind_pointer_and_null_errors() {
        let architecture = run(
            &[0; 4],
            TargetArchitecture::X86_64,
            ValueKind::Signed,
            vec![Step::Arm64ReadImmediate],
        )
        .unwrap_err();
        assert!(matches!(
            architecture.cause,
            EvaluationErrorCause::UnsupportedArchitecture { .. }
        ));

        let kind = run(
            &[1],
            TargetArchitecture::X86_64,
            ValueKind::Address,
            vec![Step::ReadUnsigned {
                offset: 0,
                width: IntegerWidth::U8,
            }],
        )
        .unwrap_err();
        assert!(matches!(
            kind.cause,
            EvaluationErrorCause::ResultKindMismatch { .. }
        ));

        let null = run(
            &[0; 8],
            TargetArchitecture::X86_64,
            ValueKind::Address,
            vec![Step::ReadPointer {
                offset: 0,
                width: PointerWidth::U64,
            }],
        )
        .unwrap_err();
        assert!(matches!(null.cause, EvaluationErrorCause::NullPointer));

        let width = Plan::new(
            ValueKind::Address,
            vec![Step::ReadPointer {
                offset: 0,
                width: PointerWidth::U32,
            }],
        )
        .evaluate(
            &TestMemory(vec![1; 8]),
            BASE,
            context(TargetArchitecture::X86_64),
            Limits::default(),
        )
        .unwrap_err();
        assert!(matches!(
            width.cause,
            EvaluationErrorCause::PointerWidthMismatch { .. }
        ));

        let range = Plan::new(
            ValueKind::Unsigned,
            vec![Step::ReadUnsigned {
                offset: 0,
                width: IntegerWidth::U16,
            }],
        )
        .evaluate(
            &TestMemory(vec![]),
            u64::MAX,
            context(TargetArchitecture::X86_64),
            Limits::default(),
        )
        .unwrap_err();
        assert!(matches!(range.cause, EvaluationErrorCause::AddressOverflow));
    }

    #[test]
    fn enforces_whole_plan_limits() {
        let plan = Plan::new(
            ValueKind::Address,
            vec![Step::Add { offset: 1 }, Step::Add { offset: 1 }],
        );
        let error = plan
            .evaluate(
                &TestMemory(vec![]),
                BASE,
                context(TargetArchitecture::X86_64),
                Limits {
                    max_steps: 1,
                    ..Limits::default()
                },
            )
            .unwrap_err();
        assert!(matches!(
            error.cause,
            EvaluationErrorCause::StepLimitExceeded { .. }
        ));

        let instruction = run_with_limits(
            &[0; 4],
            vec![Step::X86ResolveRelative {
                displacement_offset: 0,
                width: SignedWidth::I32,
            }],
            Limits {
                max_instruction_bytes: 3,
                ..Limits::default()
            },
        );
        assert!(matches!(
            instruction.unwrap_err().cause,
            EvaluationErrorCause::InstructionByteLimitExceeded { .. }
        ));

        let reads = run_with_limits(
            &[1],
            vec![Step::ReadUnsigned {
                offset: 0,
                width: IntegerWidth::U8,
            }],
            Limits {
                max_reads: 0,
                ..Limits::default()
            },
        );
        assert!(matches!(
            reads.unwrap_err().cause,
            EvaluationErrorCause::ReadLimitExceeded { .. }
        ));
    }

    fn run_with_limits(
        bytes: &[u8],
        steps: Vec<Step>,
        limits: Limits,
    ) -> Result<Value, EvaluationError<TestMemoryError>> {
        Plan::new(ValueKind::Unsigned, steps).evaluate(
            &TestMemory(bytes.to_vec()),
            BASE,
            context(TargetArchitecture::X86_64),
            limits,
        )
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_is_versioned_snake_case_and_strict() {
        let json = r#"{
            "schema_version": 1,
            "result": "address",
            "steps": [{"action":"x86_resolve_relative","displacement_offset":1,"width":"i32"}]
        }"#;
        let plan: Plan = serde_json::from_str(json).unwrap();
        assert_eq!(plan.schema_version(), SCHEMA_VERSION);
        assert_eq!(
            serde_json::to_value(&plan).unwrap()["steps"][0]["action"],
            "x86_resolve_relative"
        );

        for invalid in [
            json.replace("\"schema_version\": 1", "\"schema_version\": 2"),
            json.replace(
                "\"result\": \"address\"",
                "\"result\": \"address\", \"extra\": true",
            ),
            json.replace("\"width\":\"i32\"", "\"width\":\"i32\", \"extra\": true"),
            json.replace("x86_resolve_relative", "unknown_action"),
        ] {
            assert!(serde_json::from_str::<Plan>(&invalid).is_err(), "{invalid}");
        }
    }
}
