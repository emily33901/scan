//! Pure AArch64 instruction decoding.

use crate::InstructionError;

const ADRP_MASK: u32 = 0x9f00_0000;
const ADRP_OPCODE: u32 = 0x9000_0000;
const ADD_X_IMMEDIATE_MASK: u32 = 0xff00_0000;
const ADD_X_IMMEDIATE_OPCODE: u32 = 0x9100_0000;
const LDR_X_UNSIGNED_MASK: u32 = 0xffc0_0000;
const LDR_X_UNSIGNED_OPCODE: u32 = 0xf940_0000;
const BRANCH_MASK: u32 = 0x7c00_0000;
const BRANCH_OPCODE: u32 = 0x1400_0000;

pub(crate) fn instruction_immediate(instruction: u32) -> Result<i64, InstructionError> {
    if instruction & LDR_X_UNSIGNED_MASK == LDR_X_UNSIGNED_OPCODE {
        return Ok(i64::from((instruction >> 10) & 0x0fff) * 8);
    }
    if instruction & BRANCH_MASK == BRANCH_OPCODE {
        return Ok(sign_extend(u64::from(instruction & 0x03ff_ffff), 26) * 4);
    }
    if instruction & ADRP_MASK == ADRP_OPCODE {
        return Ok(adrp_page_displacement(instruction));
    }
    if instruction & ADD_X_IMMEDIATE_MASK == ADD_X_IMMEDIATE_OPCODE {
        return Ok(i64::from(add_immediate(instruction)));
    }
    Err(InstructionError::UnsupportedOpcode {
        opcode: instruction,
    })
}

pub(crate) fn direct_branch_target(
    address: u64,
    instruction: u32,
) -> Result<u64, InstructionError> {
    if instruction & BRANCH_MASK != BRANCH_OPCODE {
        return Err(InstructionError::UnsupportedOpcode {
            opcode: instruction,
        });
    }
    address
        .checked_add_signed(instruction_immediate(instruction)?)
        .ok_or(InstructionError::AddressOverflow)
}

pub(crate) fn page_pair_target(
    instruction_address: u64,
    adrp: u32,
    pair: u32,
) -> Result<u64, InstructionError> {
    if adrp & ADRP_MASK != ADRP_OPCODE {
        return Err(InstructionError::UnsupportedOpcode { opcode: adrp });
    }
    let adrp_destination = (adrp & 0x1f) as u8;
    if adrp_destination == 31 {
        return Err(InstructionError::InvalidRegister {
            register: adrp_destination,
            role: "ADRP destination",
        });
    }
    let (pair_base, offset) = if pair & ADD_X_IMMEDIATE_MASK == ADD_X_IMMEDIATE_OPCODE {
        (((pair >> 5) & 0x1f) as u8, u64::from(add_immediate(pair)))
    } else if pair & LDR_X_UNSIGNED_MASK == LDR_X_UNSIGNED_OPCODE {
        (
            ((pair >> 5) & 0x1f) as u8,
            u64::from((pair >> 10) & 0x0fff) * 8,
        )
    } else {
        return Err(InstructionError::UnsupportedOpcode { opcode: pair });
    };
    if adrp_destination != pair_base {
        return Err(InstructionError::RegisterMismatch {
            adrp_destination,
            pair_base,
        });
    }

    (instruction_address & !0xfff)
        .checked_add_signed(adrp_page_displacement(adrp))
        .and_then(|page| page.checked_add(offset))
        .ok_or(InstructionError::AddressOverflow)
}

fn adrp_page_displacement(instruction: u32) -> i64 {
    let immediate =
        u64::from(((instruction >> 5) & 0x7ffff) << 2) | u64::from((instruction >> 29) & 0x3);
    sign_extend(immediate, 21) * 4096
}

fn add_immediate(instruction: u32) -> u32 {
    let immediate = (instruction >> 10) & 0x0fff;
    if instruction & (1 << 22) != 0 {
        immediate << 12
    } else {
        immediate
    }
}

fn sign_extend(value: u64, width: u32) -> i64 {
    let shift = 64 - width;
    ((value << shift) as i64) >> shift
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adrp(rd: u8, pages: i32) -> u32 {
        let immediate = (pages as u32) & 0x1f_ffff;
        ADRP_OPCODE
            | ((immediate & 0x3) << 29)
            | (((immediate >> 2) & 0x7ffff) << 5)
            | u32::from(rd)
    }

    fn add(rd: u8, rn: u8, immediate: u16, shift: bool) -> u32 {
        ADD_X_IMMEDIATE_OPCODE
            | (u32::from(shift) << 22)
            | (u32::from(immediate) << 10)
            | (u32::from(rn) << 5)
            | u32::from(rd)
    }

    fn ldr(rt: u8, rn: u8, immediate: u16) -> u32 {
        LDR_X_UNSIGNED_OPCODE | (u32::from(immediate) << 10) | (u32::from(rn) << 5) | u32::from(rt)
    }

    #[test]
    fn follows_signed_b_and_bl() {
        assert_eq!(direct_branch_target(0x1000, 0x1400_0002).unwrap(), 0x1008);
        assert_eq!(direct_branch_target(0x1000, 0x97ff_ffff).unwrap(), 0x0ffc);
    }

    #[test]
    fn decodes_supported_immediates() {
        assert_eq!(instruction_immediate(ldr(0, 3, 7)).unwrap(), 56);
        assert_eq!(instruction_immediate(adrp(3, -2)).unwrap(), -8192);
        assert_eq!(instruction_immediate(add(3, 3, 2, true)).unwrap(), 8192);
    }

    #[test]
    fn page_pair_honors_unsigned_scaled_ldr_offset() {
        assert_eq!(
            page_pair_target(0x1ffc, adrp(3, 1), ldr(0, 3, 0x0fff)).unwrap(),
            0x9ff8
        );
    }

    #[test]
    fn page_pair_supports_shifted_add_and_signed_pages() {
        assert_eq!(
            page_pair_target(0x2000, adrp(7, -1), add(7, 7, 2, true)).unwrap(),
            0x3000
        );
    }

    #[test]
    fn page_pair_rejects_bad_opcodes_and_registers() {
        assert!(matches!(
            page_pair_target(0x1000, 0xd503_201f, add(0, 0, 1, false)),
            Err(InstructionError::UnsupportedOpcode { .. })
        ));
        assert!(matches!(
            page_pair_target(0x1000, adrp(1, 0), ldr(0, 2, 1)),
            Err(InstructionError::RegisterMismatch { .. })
        ));
        assert!(matches!(
            page_pair_target(0x1000, adrp(31, 0), ldr(0, 31, 1)),
            Err(InstructionError::InvalidRegister { .. })
        ));
    }

    #[test]
    fn rejects_page_and_branch_overflow() {
        assert_eq!(
            page_pair_target(u64::MAX, adrp(0, 1), add(0, 0, 0, false)),
            Err(InstructionError::AddressOverflow)
        );
        assert_eq!(
            direct_branch_target(u64::MAX, 0x1400_0001),
            Err(InstructionError::AddressOverflow)
        );
        assert_eq!(
            direct_branch_target(0, 0x17ff_ffff),
            Err(InstructionError::AddressOverflow)
        );
    }
}
