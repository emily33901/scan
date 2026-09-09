#[cfg(target_os = "macos")]
pub mod macos;

#[cfg(target_os = "windows")]
pub mod windows;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

use std::collections::HashMap;

// NOTE(emily): Context used on arm64
#[allow(unused_imports)]
use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

#[cfg(any(target_arch = "aarch64", test))]
mod aarch64;

pub type CustomActionFn<'a> = Box<dyn Fn(usize) -> Result<usize> + 'a>;
pub type CustomActions<'a> = HashMap<String, CustomActionFn<'a>>;

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "action")]
pub enum Action {
    Add {
        offset: isize,
    },
    ResolveRelative {
        offset: usize,
    },
    Dereference {},
    ResolvePageAndOffsetAddress {
        offset: usize,
    },
    ImmediateFromInstructionAtAddress {},
    ResolveImmediateRelativeAddress {},
    ResolvePageOffsetRelativeAddress {},
    /// Follows an encoded direct branch or call at the current address.
    ///
    /// This is deliberately separate from `ResolveRelative`: callers do not
    /// need to hard-code the displacement field's offset or width.
    FollowDirectBranch {},
    Custom {
        name: String,
    },
}

pub fn execute_plan(
    mut address: usize,
    actions: &Vec<Action>,
    custom_actions: Option<&CustomActions>,
) -> Result<usize> {
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
            &Action::FollowDirectBranch {} => {
                address = x86_64::follow_direct_branch_at_address(address)?;
            }
            #[cfg(target_arch = "x86_64")]
            &Action::Dereference {} => address = unsafe { *(address as *const *const ()) as usize },

            #[cfg(target_arch = "aarch64")]
            &Action::ResolvePageAndOffsetAddress { offset } => {
                address = macos::aarch64::resolve_page_and_offset_load_at_address(
                    aarch64::page_pair_instruction_address(address, offset),
                )?;
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
            }
            #[cfg(target_arch = "aarch64")]
            &Action::FollowDirectBranch {} => {
                address = macos::aarch64::follow_direct_branch_at_address(address)?;
            }

            Action::Custom { name } => {
                let Some(custom_action) = custom_actions
                    .and_then(|actions| actions.iter().find_map(|(k, v)| (k == name).then_some(v)))
                else {
                    bail!("Expected custom function {name} to exist");
                };

                address = custom_action(address)?;
            }

            unknown_action => {
                bail!(
                    "action {:?} is not implemented for your platform",
                    unknown_action
                );
            }
        }
    }

    Ok(address)
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    use super::Action;

    #[derive(Serialize, Deserialize)]
    struct ScanSignature {
        plan: Vec<Action>,
    }

    #[test]
    fn follow_direct_branch_round_trips_through_toml_plan() {
        let text = r#"
[[plan]]
action = "FollowDirectBranch"
"#;
        let signature: ScanSignature = toml::from_str(text).unwrap();
        assert!(matches!(
            signature.plan.as_slice(),
            [Action::FollowDirectBranch {}]
        ));

        let serialized = toml::to_string(&signature).unwrap();
        assert!(serialized.contains("[[plan]]"));
        assert!(serialized.contains("action = \"FollowDirectBranch\""));

        let round_trip: ScanSignature = toml::from_str(&serialized).unwrap();
        assert!(matches!(
            round_trip.plan.as_slice(),
            [Action::FollowDirectBranch {}]
        ));
    }
}
