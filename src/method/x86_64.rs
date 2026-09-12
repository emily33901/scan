//! Compatibility helpers for x86-64 relative-control-flow decoding.

use crate::memory::Address;

use super::InstructionError;

pub fn direct_branch_target(
    instruction_address: Address,
    bytes: &[u8],
) -> Result<Address, InstructionError> {
    let address = scan_plan::x86_64::direct_branch_target(instruction_address.get() as u64, bytes)?;
    usize::try_from(address)
        .map(Address::new)
        .map_err(|_| InstructionError::AddressOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follows_signed_direct_branches() {
        assert_eq!(
            direct_branch_target(Address::new(0x1000), &[0xe8, 0x20, 0, 0, 0]).unwrap(),
            Address::new(0x1025)
        );
        assert_eq!(
            direct_branch_target(Address::new(0x1000), &[0xeb, 0xf0]).unwrap(),
            Address::new(0x0ff2)
        );
        assert_eq!(
            direct_branch_target(Address::new(0x1000), &[0x0f, 0x85, 0xf6, 0xff, 0xff, 0xff])
                .unwrap(),
            Address::new(0x0ffc)
        );
    }

    #[test]
    fn rejects_unknown_and_truncated_instructions() {
        assert!(matches!(
            direct_branch_target(Address::new(0x1000), &[0x90]),
            Err(InstructionError::UnsupportedOpcode { .. })
        ));
        assert!(matches!(
            direct_branch_target(Address::new(0x1000), &[0xe8, 0, 0]),
            Err(InstructionError::Truncated { .. })
        ));
    }
}
