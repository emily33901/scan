#![cfg(feature = "serde")]

use std::fmt;

use scan_plan::{
    ByteOrder, EvaluationErrorCause, Limits, Plan, PlanMemory, PointerWidth, TargetArchitecture,
    TargetContext, Value,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Corpus {
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    context: TargetContext,
    raw_match: u64,
    #[serde(default)]
    match_offset: i64,
    #[serde(default)]
    memory: Vec<MemoryRange>,
    plan: Plan,
    #[serde(default)]
    limits: FixtureLimits,
    expected: Expected,
}

#[derive(Debug, Deserialize)]
struct MemoryRange {
    address: u64,
    bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
struct FixtureLimits {
    max_steps: Option<usize>,
    max_instruction_bytes: Option<usize>,
    max_reads: Option<usize>,
    max_read_bytes: Option<usize>,
}

impl FixtureLimits {
    fn resolve(self) -> Limits {
        let defaults = Limits::default();
        Limits {
            max_steps: self.max_steps.unwrap_or(defaults.max_steps),
            max_instruction_bytes: self
                .max_instruction_bytes
                .unwrap_or(defaults.max_instruction_bytes),
            max_reads: self.max_reads.unwrap_or(defaults.max_reads),
            max_read_bytes: self.max_read_bytes.unwrap_or(defaults.max_read_bytes),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
enum Expected {
    Value {
        kind: String,
        value: serde_json::Value,
    },
    Error {
        step: usize,
        cause: String,
    },
    MatchOffsetError,
}

#[derive(Clone, Copy, Debug)]
struct FixtureMemoryError;

impl fmt::Display for FixtureMemoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("address is outside fixture memory")
    }
}

impl std::error::Error for FixtureMemoryError {}

impl PlanMemory for Case {
    type Error = FixtureMemoryError;

    fn read_exact(&self, address: u64, destination: &mut [u8]) -> Result<(), Self::Error> {
        for range in &self.memory {
            let Some(offset) = address.checked_sub(range.address) else {
                continue;
            };
            let Ok(offset) = usize::try_from(offset) else {
                continue;
            };
            let Some(end) = offset.checked_add(destination.len()) else {
                continue;
            };
            if let Some(bytes) = range.bytes.get(offset..end) {
                destination.copy_from_slice(bytes);
                return Ok(());
            }
        }
        Err(FixtureMemoryError)
    }
}

fn checked_match_offset(address: u64, offset: i64) -> Option<u64> {
    if offset >= 0 {
        address.checked_add(offset as u64)
    } else {
        address.checked_sub(offset.unsigned_abs())
    }
}

fn cause_name(cause: &EvaluationErrorCause<FixtureMemoryError>) -> &'static str {
    match cause {
        EvaluationErrorCause::StepLimitExceeded { .. } => "step_limit_exceeded",
        EvaluationErrorCause::InstructionByteLimitExceeded { .. } => {
            "instruction_byte_limit_exceeded"
        }
        EvaluationErrorCause::ReadLimitExceeded { .. } => "read_limit_exceeded",
        EvaluationErrorCause::ReadByteLimitExceeded { .. } => "read_byte_limit_exceeded",
        EvaluationErrorCause::AddressRequired { .. } => "address_required",
        EvaluationErrorCause::ResultKindMismatch { .. } => "result_kind_mismatch",
        EvaluationErrorCause::AddressOverflow => "address_overflow",
        EvaluationErrorCause::UnsupportedTargetContext { .. } => "unsupported_target_context",
        EvaluationErrorCause::NullPointer => "null_pointer",
        EvaluationErrorCause::PointerWidthMismatch { .. } => "pointer_width_mismatch",
        EvaluationErrorCause::UnsupportedArchitecture { .. } => "unsupported_architecture",
        EvaluationErrorCause::Memory(_) => "memory",
        EvaluationErrorCause::Instruction(_) => "instruction",
        _ => "unknown",
    }
}

fn expected_value(kind: &str, value: &serde_json::Value) -> Value {
    match kind {
        "address" => Value::Address(value.as_u64().expect("address fixture value")),
        "signed" => Value::Signed(value.as_i64().expect("signed fixture value")),
        "unsigned" => Value::Unsigned(value.as_u64().expect("unsigned fixture value")),
        other => panic!("unknown fixture value kind {other}"),
    }
}

#[test]
fn canonical_golden_cases() {
    let corpus: Corpus = serde_json::from_str(include_str!("fixtures/canonical.json"))
        .expect("canonical fixture corpus must deserialize");

    for case in corpus.cases {
        let start = checked_match_offset(case.raw_match, case.match_offset);
        if matches!(case.expected, Expected::MatchOffsetError) {
            assert!(
                start.is_none(),
                "{}: match offset should overflow",
                case.name
            );
            continue;
        }
        let start =
            start.unwrap_or_else(|| panic!("{}: unexpected match offset overflow", case.name));
        let actual = case
            .plan
            .evaluate(&case, start, case.context, case.limits.resolve());
        match &case.expected {
            Expected::Value { kind, value } => assert_eq!(
                actual.unwrap_or_else(|error| panic!("{}: {error}", case.name)),
                expected_value(kind, value),
                "{}",
                case.name
            ),
            Expected::Error { step, cause } => {
                let error = actual.unwrap_err();
                assert_eq!(error.step_index, *step, "{}", case.name);
                assert_eq!(cause_name(&error.cause), cause, "{}", case.name);
            }
            Expected::MatchOffsetError => unreachable!(),
        }
    }
}

#[test]
fn supported_contexts_are_explicit() {
    let plan = Plan::new(scan_plan::ValueKind::Address, vec![]);
    let memory = Case {
        name: "context".into(),
        context: TargetContext {
            architecture: TargetArchitecture::X86_64,
            pointer_width: PointerWidth::U64,
            byte_order: ByteOrder::Little,
        },
        raw_match: 0,
        match_offset: 0,
        memory: vec![],
        plan: plan.clone(),
        limits: FixtureLimits::default(),
        expected: Expected::Value {
            kind: "address".into(),
            value: 0.into(),
        },
    };

    for context in [
        TargetContext {
            architecture: TargetArchitecture::X86_64,
            pointer_width: PointerWidth::U32,
            byte_order: ByteOrder::Little,
        },
        TargetContext {
            architecture: TargetArchitecture::Aarch64,
            pointer_width: PointerWidth::U64,
            byte_order: ByteOrder::Big,
        },
    ] {
        let error = plan
            .evaluate(&memory, 0, context, Limits::default())
            .unwrap_err();
        assert!(matches!(
            error.cause,
            EvaluationErrorCause::UnsupportedTargetContext { .. }
        ));
    }
}
