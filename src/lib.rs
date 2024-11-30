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

use anyhow::Result;

pub use vmthook::HookFunction;
