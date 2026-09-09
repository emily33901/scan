use anyhow::{anyhow, bail, Result};

pub fn resolve_relative_address(addr: usize, offset: usize) -> usize {
    unsafe {
        let inside = ((addr + offset) as *const i32).read_unaligned() as isize;
        relative_target(addr, offset + 4, inside).expect("relative target overflow")
    }
}

/// Resolve the target of a direct x86-64 branch from its instruction bytes.
///
/// The supported encodings are `call rel32`, `jmp rel8/rel32`, short
/// conditional branches, loop-family branches, and near conditional branches.
/// Keeping the decoder slice-based makes its semantics testable without reading
/// process memory.
pub(crate) fn direct_branch_target(instruction_address: usize, bytes: &[u8]) -> Result<usize> {
    let Some(&opcode) = bytes.first() else {
        bail!("direct branch instruction is empty");
    };

    match opcode {
        0xE8 | 0xE9 => relative_target(
            instruction_address,
            5,
            read_i32(bytes, 1, "rel32 branch")? as isize,
        ),
        0xEB | 0x70..=0x7F | 0xE0..=0xE3 => relative_target(
            instruction_address,
            2,
            read_i8(bytes, 1, "rel8 branch")? as isize,
        ),
        0x0F if matches!(bytes.get(1), Some(0x80..=0x8F)) => relative_target(
            instruction_address,
            6,
            read_i32(bytes, 2, "near conditional branch")? as isize,
        ),
        _ => bail!("unsupported direct branch opcode {opcode:#04X}"),
    }
}

pub(crate) fn follow_direct_branch_at_address(address: usize) -> Result<usize> {
    // Callers must provide a mapped, executable instruction address, just as for
    // the existing ResolveRelative action. Read only the encoded instruction,
    // rather than unconditionally reading six bytes at a page boundary.
    let first = unsafe { (address as *const u8).read() };
    let prefix = if first == 0x0F {
        [first, unsafe { (address as *const u8).add(1).read() }]
    } else {
        [first, 0]
    };
    let length = direct_branch_length(&prefix)?;
    let bytes = unsafe { std::slice::from_raw_parts(address as *const u8, length) };
    direct_branch_target(address, bytes)
}

fn direct_branch_length(bytes: &[u8]) -> Result<usize> {
    let Some(&opcode) = bytes.first() else {
        bail!("direct branch instruction is empty");
    };

    match opcode {
        0xE8 | 0xE9 => Ok(5),
        0xEB | 0x70..=0x7F | 0xE0..=0xE3 => Ok(2),
        0x0F if matches!(bytes.get(1), Some(0x80..=0x8F)) => Ok(6),
        0x0F => bail!("unsupported direct branch opcode 0x0F"),
        _ => bail!("unsupported direct branch opcode {opcode:#04X}"),
    }
}

fn relative_target(address: usize, instruction_size: usize, displacement: isize) -> Result<usize> {
    let next_instruction = address
        .checked_add(instruction_size)
        .ok_or_else(|| anyhow!("direct branch instruction address overflow"))?;
    next_instruction
        .checked_add_signed(displacement)
        .ok_or_else(|| anyhow!("direct branch target overflow"))
}

fn read_i8(bytes: &[u8], offset: usize, name: &str) -> Result<i8> {
    bytes
        .get(offset)
        .copied()
        .map(|byte| byte as i8)
        .ok_or_else(|| anyhow!("{name} is truncated"))
}

fn read_i32(bytes: &[u8], offset: usize, name: &str) -> Result<i32> {
    let bytes = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| anyhow!("{name} is truncated"))?;
    Ok(i32::from_le_bytes(
        bytes.try_into().expect("slice length is checked"),
    ))
}

#[cfg(test)]
mod tests {
    use super::direct_branch_target;

    #[test]
    fn follows_call_rel32() {
        assert_eq!(
            direct_branch_target(0x1000, &[0xE8, 0x20, 0, 0, 0]).unwrap(),
            0x1025
        );
    }

    #[test]
    fn follows_negative_short_jump() {
        assert_eq!(direct_branch_target(0x1000, &[0xEB, 0xF0]).unwrap(), 0x0FF2);
    }

    #[test]
    fn follows_near_conditional_branch() {
        assert_eq!(
            direct_branch_target(0x1000, &[0x0F, 0x85, 0xF6, 0xFF, 0xFF, 0xFF]).unwrap(),
            0x0FFC
        );
    }

    #[test]
    fn rejects_non_branch_and_truncated_encodings() {
        assert!(direct_branch_target(0x1000, &[0x90]).is_err());
        assert!(direct_branch_target(0x1000, &[0xE8, 0, 0]).is_err());
    }
}
