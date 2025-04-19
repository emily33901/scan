use anyhow::{anyhow, Context, Result};

use disarm64::decoder::{
    Operation, ADDSUB_IMM, BL_ADDR_PCREL26, BRANCH_IMM, LDST_POS, MOVEWIDE, PCRELADDR,
};

pub(crate) fn get_immediate(op: Operation) -> Result<isize> {
    match op {
        Operation::LDST_POS(LDST_POS::LDR_Rt_ADDR_UIMM12(bitfield)) => {
            let imm12 = bitfield.imm12();
            const x: isize = 64 - 52;
            Ok((imm12 as isize & x) << 3)
        }
        Operation::BRANCH_IMM(BRANCH_IMM::BL_ADDR_PCREL26(bitfield)) => {
            let imm26 = bitfield.imm26();
            Ok((((imm26 as isize) << 38) >> 38) * 4)
        }
        Operation::PCRELADDR(PCRELADDR::ADRP_Rd_ADDR_ADRP(bitfield)) => {
            assert!(bitfield.immhi() < (1 << 19), "immhi must be a 19-bit value");
            assert!(bitfield.immlo() < (1 << 2), "immlo must be a 2-bit value");
            let combined_imm = ((bitfield.immhi() << 2) | bitfield.immlo()) as usize;
            let sign_extended_imm = ((combined_imm as isize) << 43) >> 43;
            Ok(sign_extended_imm)
        }
        Operation::ADDSUB_IMM(ADDSUB_IMM::ADD_Rd_SP_Rn_SP_AIMM(bitfield)) => {
            Ok((bitfield.imm12() << (bitfield.shift() * 12)) as isize)
        }
        x => Err(anyhow!("get_immediate has no idea what to do with {:?}", x)),
    }
}

pub(crate) fn immediate_from_instruction_at_address(addr: usize) -> Result<isize> {
    let raw_instruction = unsafe { *(addr as *const u32) };
    let instruction = disarm64::decoder::decode(raw_instruction).ok_or(anyhow!(
        "unable to decode instruction {:8X}",
        raw_instruction
    ))?;
    Ok(get_immediate(instruction.operation)?)
}

pub(crate) fn resolve_relative_address(address: usize, offset: isize) -> usize {
    address.checked_add_signed(offset).unwrap()
}

pub(crate) fn resolve_page_aligned_relative_address(address: usize, offset: isize) -> usize {
    // https://developer.arm.com/documentation/ddi0602/2024-03/Base-Instructions/ADRP--Form-PC-relative-address-to-4KB-page-?lang=en
    ((address + 4) & !0xfff)
        .checked_add_signed(offset << 12)
        .unwrap()
}

pub fn resolve_page_and_offset_load_at_address(address: usize) -> Result<usize> {
    let page = resolve_page_aligned_relative_address(
        address,
        immediate_from_instruction_at_address(address).context("page")?,
    );

    let offset = immediate_from_instruction_at_address(address + 4).context("offset")?;

    Ok(page.checked_add_signed(offset).unwrap())
}
