#[cfg(target_os = "macos")]
pub mod macos;

#[cfg(target_os = "windows")]
pub mod windows;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(tag = "action")]
pub enum Action {
    Add { offset: isize },
    ResolveRelative { offset: usize },
    Dereference {},
    ResolvePageAndOffsetAddress { offset: usize },
    ImmediateFromInstructionAtAddress {},
    ResolveImmediateRelativeAddress {},
    ResolvePageOffsetRelativeAddress {},
}

pub fn execute_plan(mut address: usize, actions: &Vec<Action>) -> Result<usize> {
    for action in actions {
        match action {
            &Action::Add { offset } => {
                address = address
                    .checked_add_signed(offset)
                    .ok_or(anyhow!("failed checked add"))?;
            }

            #[cfg(target_arch = "x86_64")]
            &Action::ResolveRelative { offset } => {
                address = x86_64::resolve_relative_address(address, offset);
            }
            #[cfg(target_arch = "x86_64")]
            &Action::Dereference {} => address = unsafe { *(address as *const *const ()) as usize },

            #[cfg(target_arch = "aarch64")]
            &Action::ResolvePageAndOffsetAddress { offset } => {
                address = macos::aarch64::resolve_page_and_offset_load_at_address(address)?;
            }
            #[cfg(target_arch = "aarch64")]
            &Action::ImmediateFromInstructionAtAddress {} => {
                address = macos::aarch64::immediate_from_instruction_at_address(address)? as usize;
            }
            #[cfg(target_arch = "aarch64")]
            &Action::ResolveImmediateRelativeAddress {} => {
                address = macos::aarch64::resolve_relative_address(
                    address,
                    macos::aarch64::immediate_from_instruction_at_address(address)
                        .context("resolve relative address")?,
                );
            }
            #[cfg(target_arch = "aarch64")]
            &Action::ResolvePageOffsetRelativeAddress {} => {
                address = macos::aarch64::resolve_page_and_offset_load_at_address(address)?;
                // let page = resolve_page_aligned_relative_address(
                //     address,
                //     immediate_from_instruction_at_address(address)
                //         .context("resolve relative address page")?,
                // );

                // let offset = immediate_from_instruction_at_address(address + 4)
                //     .context("resolve relative address offset")?;

                // address = (page
                //     .checked_add_signed(offset)
                //     .ok_or(anyhow!("failed checked add")));
            }

            // #[cfg(target_arch = "aarch64")]
            // &Action::ResolvePageAlignedRelativeAddress {} => {}
            _ => unimplemented!(),
        }
    }

    Ok(address)
}
