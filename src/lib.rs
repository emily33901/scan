//! Checked access to memory and executable code in the current process.
//!
//! Mapping information is a point-in-time observation. It can reject invalid
//! ranges, but cannot by itself prove that foreign storage remains live or is a
//! valid Rust value. See [`memory`] for the direct and fault-contained access
//! contracts.

#![deny(unsafe_op_in_unsafe_fn)]
#![cfg_attr(not(test), deny(clippy::undocumented_unsafe_blocks))]

pub mod function;
pub mod memory;
pub mod method;
pub mod module;
pub mod vtable;

/// The canonical, portable plan model and evaluator.
pub use scan_plan;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

mod vmthook;

pub use vmthook::thunk::{AsCraneliftAbi, ThunkablePostHook, ThunkablePreHook};
pub use vmthook::{
    restore_instance, HookDiagnostics, HookDiscovery, HookError, HookFunction, HookId, HookInfo,
    HookMemoryOperation, HookOptions, HookResult, HookStatus,
};
