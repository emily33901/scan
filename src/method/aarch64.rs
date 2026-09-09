use anyhow::{anyhow, bail, Result};

/// Resolves an ARM64 ADRP page displacement without reading process memory.
pub(crate) fn page_aligned_relative_target(
    address: usize,
    page_displacement: isize,
) -> Result<usize> {
    let displacement = page_displacement
        .checked_shl(12)
        .ok_or_else(|| anyhow!("ADRP page displacement overflow"))?;
    (address & !0xfff)
        .checked_add_signed(displacement)
        .ok_or_else(|| anyhow!("ADRP target address overflow"))
}

/// Keeps the legacy scan-plan contract for page-pair resolution.
///
/// `ResolvePageAndOffsetAddress.offset` has always been serialized, but the
/// runtime resolves the ADRP/ADD or ADRP/LDR pair at the current address.
pub(crate) const fn page_pair_instruction_address(address: usize, _offset: usize) -> usize {
    address
}

/// Resolves an ARM64 `B` or `BL` instruction from its encoded word.
pub(crate) fn direct_branch_target(address: usize, instruction: u32) -> Result<usize> {
    match instruction & 0xfc00_0000 {
        0x1400_0000 | 0x9400_0000 => {}
        _ => bail!("instruction is not an ARM64 B or BL"),
    }

    let imm26 = (instruction & 0x03ff_ffff) as i32;
    let displacement = ((imm26 << 6) >> 4) as isize;
    address
        .checked_add_signed(displacement)
        .ok_or_else(|| anyhow!("direct branch target address overflow"))
}

#[cfg(test)]
mod tests {
    use super::{
        direct_branch_target, page_aligned_relative_target, page_pair_instruction_address,
    };

    #[test]
    fn adrp_uses_the_current_instruction_page() {
        assert_eq!(page_aligned_relative_target(0x1FFC, 1).unwrap(), 0x2000);
        assert_eq!(page_aligned_relative_target(0x2000, -1).unwrap(), 0x1000);
    }

    #[test]
    fn page_pair_actions_ignore_the_serialized_offset() {
        assert_eq!(page_pair_instruction_address(0x1234, 4), 0x1234);
    }

    #[test]
    fn follows_b_and_bl_with_signed_offsets() {
        assert_eq!(direct_branch_target(0x1000, 0x1400_0002).unwrap(), 0x1008);
        assert_eq!(direct_branch_target(0x1000, 0x97FF_FFFF).unwrap(), 0x0FFC);
    }

    #[test]
    fn rejects_non_branch_instructions() {
        assert!(direct_branch_target(0x1000, 0xD503_201F).is_err());
    }
}
