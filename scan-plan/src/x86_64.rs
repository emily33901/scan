//! Pure x86-64 relative-control-flow decoding.

use crate::InstructionError;

/// Resolves a supported direct branch from its instruction bytes.
pub fn direct_branch_target(
    instruction_address: u64,
    bytes: &[u8],
) -> Result<u64, InstructionError> {
    let opcode = *bytes.first().ok_or(InstructionError::Truncated {
        expected: 1,
        actual: 0,
    })?;
    match opcode {
        0xe8 | 0xe9 => relative_target(instruction_address, 5, i64::from(read_i32(bytes, 1)?)),
        0xeb | 0x70..=0x7f | 0xe0..=0xe3 => {
            relative_target(instruction_address, 2, i64::from(read_i8(bytes, 1)?))
        }
        0x0f if matches!(bytes.get(1), Some(0x80..=0x8f)) => {
            relative_target(instruction_address, 6, i64::from(read_i32(bytes, 2)?))
        }
        _ => Err(InstructionError::UnsupportedOpcode {
            opcode: u32::from(opcode),
        }),
    }
}

pub(crate) fn direct_branch_length(bytes: &[u8]) -> Result<usize, InstructionError> {
    let opcode = *bytes.first().ok_or(InstructionError::Truncated {
        expected: 1,
        actual: 0,
    })?;
    match opcode {
        0xe8 | 0xe9 => Ok(5),
        0xeb | 0x70..=0x7f | 0xe0..=0xe3 => Ok(2),
        0x0f => match bytes.get(1) {
            Some(0x80..=0x8f) => Ok(6),
            Some(opcode) => Err(InstructionError::UnsupportedOpcode {
                opcode: (u32::from(0x0f_u8) << 8) | u32::from(*opcode),
            }),
            None => Err(InstructionError::Truncated {
                expected: 2,
                actual: 1,
            }),
        },
        _ => Err(InstructionError::UnsupportedOpcode {
            opcode: u32::from(opcode),
        }),
    }
}

pub(crate) fn relative_target(
    address: u64,
    instruction_size: u64,
    displacement: i64,
) -> Result<u64, InstructionError> {
    address
        .checked_add(instruction_size)
        .and_then(|next| next.checked_add_signed(displacement))
        .ok_or(InstructionError::AddressOverflow)
}

fn read_i8(bytes: &[u8], offset: usize) -> Result<i8, InstructionError> {
    bytes
        .get(offset)
        .copied()
        .map(|byte| byte as i8)
        .ok_or(InstructionError::Truncated {
            expected: offset + 1,
            actual: bytes.len(),
        })
}

fn read_i32(bytes: &[u8], offset: usize) -> Result<i32, InstructionError> {
    let end = offset
        .checked_add(4)
        .ok_or(InstructionError::AddressOverflow)?;
    let raw = bytes.get(offset..end).ok_or(InstructionError::Truncated {
        expected: end,
        actual: bytes.len(),
    })?;
    let mut copied = [0_u8; 4];
    copied.copy_from_slice(raw);
    Ok(i32::from_le_bytes(copied))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follows_signed_direct_branches() {
        assert_eq!(
            direct_branch_target(0x1000, &[0xe8, 0x20, 0, 0, 0]).unwrap(),
            0x1025
        );
        assert_eq!(direct_branch_target(0x1000, &[0xeb, 0xf0]).unwrap(), 0x0ff2);
        assert_eq!(
            direct_branch_target(0x1000, &[0x0f, 0x85, 0xf6, 0xff, 0xff, 0xff]).unwrap(),
            0x0ffc
        );
    }

    #[test]
    fn reports_direct_branch_lengths() {
        assert_eq!(direct_branch_length(&[0xe8]).unwrap(), 5);
        assert_eq!(direct_branch_length(&[0xeb]).unwrap(), 2);
        assert_eq!(direct_branch_length(&[0x0f, 0x84]).unwrap(), 6);
    }

    #[test]
    fn rejects_unknown_truncated_and_overflowing_instructions() {
        assert!(matches!(
            direct_branch_target(0x1000, &[0x90]),
            Err(InstructionError::UnsupportedOpcode { .. })
        ));
        assert!(matches!(
            direct_branch_target(0x1000, &[0xe8, 0, 0]),
            Err(InstructionError::Truncated { .. })
        ));
        assert!(matches!(
            direct_branch_length(&[0x0f]),
            Err(InstructionError::Truncated { .. })
        ));
        assert_eq!(
            relative_target(u64::MAX, 1, 0),
            Err(InstructionError::AddressOverflow)
        );
        assert_eq!(
            relative_target(0, 0, -1),
            Err(InstructionError::AddressOverflow)
        );
    }
}
