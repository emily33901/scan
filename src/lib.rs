//! Checked access to memory and executable code in the current process.
//!
//! Mapping information is a point-in-time observation. It can reject invalid
//! ranges, but cannot by itself prove that foreign storage remains live or is a
//! valid Rust value. See [`memory`] for the direct and fault-contained access
//! contracts.

pub mod memory;
pub mod method;

#[cfg(target_os = "windows")]
mod windows;

#[cfg(target_os = "windows")]
pub use windows::Module;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_os = "macos")]
mod macos;

#[cfg(target_os = "macos")]
pub use macos::Module;

mod vmthook;

pub use vmthook::thunk::{AsCraneliftAbi, ThunkablePreHook};
pub use vmthook::{
    restore_instance, HookDiagnostics, HookError, HookFunction, HookId, HookInfo, HookOptions,
    HookResult, HookStatus,
};
