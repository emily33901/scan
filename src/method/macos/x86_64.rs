#[cfg(target_arch = "x86_64")]
pub fn resolve_relative_address(addr: usize, offset: usize) -> usize {
    unsafe {
        let inside = ((addr + offset) as *const i32).read_unaligned() as isize;
        let new_addr = (addr as *const u8).offset(inside) as usize;

        new_addr + (offset + 4)
    }
}
