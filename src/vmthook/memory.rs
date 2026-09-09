use anyhow::{anyhow, bail, Result};
use std::mem::size_of;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MemoryRegion {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) readable: bool,
    pub(crate) executable: bool,
}

impl MemoryRegion {
    pub(crate) fn contains(self, address: usize, size: usize) -> bool {
        address >= self.start && address.checked_add(size).is_some_and(|end| end <= self.end)
    }
}

pub(crate) trait MemoryMap {
    fn region_containing(&self, address: usize) -> Result<MemoryRegion>;
}

pub(crate) struct ProcessMemoryMap;

pub(crate) fn checked_pointer_address(base: usize, index: usize) -> Result<usize> {
    let offset = index
        .checked_mul(size_of::<*const ()>())
        .ok_or_else(|| anyhow!("vtable address arithmetic overflow"))?;
    base.checked_add(offset)
        .ok_or_else(|| anyhow!("vtable address arithmetic overflow"))
}

pub(crate) fn ensure_readable<M: MemoryMap>(map: &M, address: usize, size: usize) -> Result<()> {
    let region = map.region_containing(address)?;
    if !region.readable || !region.contains(address, size) {
        bail!("memory range {address:#x}..+{size} is not readable");
    }
    Ok(())
}

pub(crate) fn ensure_executable<M: MemoryMap>(map: &M, address: usize) -> Result<()> {
    let region = map.region_containing(address)?;
    if !region.executable || !region.contains(address, 1) {
        bail!("address {address:#x} is not executable");
    }
    Ok(())
}

#[cfg(target_os = "windows")]
impl MemoryMap for ProcessMemoryMap {
    fn region_containing(&self, address: usize) -> Result<MemoryRegion> {
        use std::mem::{size_of, zeroed};
        use windows::Win32::System::Memory::{
            VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE, PAGE_EXECUTE_READ,
            PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_NOACCESS,
            PAGE_READONLY, PAGE_READWRITE, PAGE_WRITECOPY,
        };

        let mut info: MEMORY_BASIC_INFORMATION = unsafe { zeroed() };
        let written = unsafe {
            VirtualQuery(
                Some(address as *const _),
                &mut info,
                size_of::<MEMORY_BASIC_INFORMATION>(),
            )
        };
        if written == 0 {
            bail!("VirtualQuery failed for address {address:#x}");
        }

        let start = info.BaseAddress as usize;
        let end = start
            .checked_add(info.RegionSize)
            .ok_or_else(|| anyhow!("memory region end overflow"))?;
        if !(start..end).contains(&address) {
            bail!("VirtualQuery returned a non-containing region for {address:#x}");
        }

        let protection = info.Protect;
        let accessible = info.State == MEM_COMMIT
            && !protection.contains(PAGE_GUARD)
            && !protection.contains(PAGE_NOACCESS);
        let executable = accessible
            && [
                PAGE_EXECUTE,
                PAGE_EXECUTE_READ,
                PAGE_EXECUTE_READWRITE,
                PAGE_EXECUTE_WRITECOPY,
            ]
            .iter()
            .any(|flag| protection.contains(*flag));
        let readable = accessible
            && [
                PAGE_READONLY,
                PAGE_READWRITE,
                PAGE_WRITECOPY,
                PAGE_EXECUTE_READ,
                PAGE_EXECUTE_READWRITE,
                PAGE_EXECUTE_WRITECOPY,
            ]
            .iter()
            .any(|flag| protection.contains(*flag));

        Ok(MemoryRegion {
            start,
            end,
            readable,
            executable,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedMap(MemoryRegion);

    impl MemoryMap for FixedMap {
        fn region_containing(&self, address: usize) -> Result<MemoryRegion> {
            if (self.0.start..self.0.end).contains(&address) {
                Ok(self.0)
            } else {
                bail!("address is outside the test mapping")
            }
        }
    }

    #[test]
    fn readable_range_must_not_cross_mapping_boundary() {
        let map = FixedMap(MemoryRegion {
            start: 0x1000,
            end: 0x1008,
            readable: true,
            executable: false,
        });

        assert!(ensure_readable(&map, 0x1004, 8).is_err());
    }

    #[test]
    fn pointer_address_arithmetic_is_checked() {
        assert!(checked_pointer_address(usize::MAX - 3, 1).is_err());
        assert!(checked_pointer_address(0, usize::MAX).is_err());
    }

    #[test]
    fn executable_check_rejects_read_only_mapping() {
        let map = FixedMap(MemoryRegion {
            start: 0x1000,
            end: 0x2000,
            readable: true,
            executable: false,
        });

        assert!(ensure_executable(&map, 0x1000).is_err());
    }
}

#[cfg(target_os = "macos")]
impl MemoryMap for ProcessMemoryMap {
    fn region_containing(&self, requested: usize) -> Result<MemoryRegion> {
        use mach2::{
            kern_return::KERN_SUCCESS,
            traps::mach_task_self,
            vm::mach_vm_region,
            vm_prot::{VM_PROT_EXECUTE, VM_PROT_READ},
            vm_region::{vm_region_basic_info_64, vm_region_info_t, VM_REGION_BASIC_INFO_64},
        };
        use std::mem::zeroed;

        let mut address = requested as u64;
        let mut size = 0;
        let mut object_name = 0;
        let mut info: vm_region_basic_info_64 = unsafe { zeroed() };
        let mut info_count = vm_region_basic_info_64::count();
        let result = unsafe {
            mach_vm_region(
                mach_task_self(),
                &mut address,
                &mut size,
                VM_REGION_BASIC_INFO_64,
                (&mut info as *mut _) as vm_region_info_t,
                &mut info_count,
                &mut object_name,
            )
        };
        if result != KERN_SUCCESS {
            bail!("mach_vm_region failed for address {requested:#x}: {result}");
        }

        let start =
            usize::try_from(address).map_err(|_| anyhow!("memory region start overflow"))?;
        let size = usize::try_from(size).map_err(|_| anyhow!("memory region size overflow"))?;
        let end = start
            .checked_add(size)
            .ok_or_else(|| anyhow!("memory region end overflow"))?;
        if !(start..end).contains(&requested) {
            bail!("mach_vm_region returned a non-containing region for {requested:#x}");
        }

        Ok(MemoryRegion {
            start,
            end,
            readable: info.protection & VM_PROT_READ != 0,
            executable: info.protection & VM_PROT_EXECUTE != 0,
        })
    }
}

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
impl MemoryMap for ProcessMemoryMap {
    fn region_containing(&self, _address: usize) -> Result<MemoryRegion> {
        bail!("mapped-memory validation is unsupported on this platform")
    }
}
