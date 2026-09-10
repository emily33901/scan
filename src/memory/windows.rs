//! Windows mapping queries and current-process byte-copy backends.

use std::ffi::c_void;
use std::mem::{size_of, zeroed};
use std::ptr;

use windows::Win32::System::Memory::{
    VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_IMAGE, MEM_MAPPED, MEM_PRIVATE,
    PAGE_EXECUTE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, PAGE_GUARD,
    PAGE_NOACCESS, PAGE_PROTECTION_FLAGS, PAGE_READONLY, PAGE_READWRITE, PAGE_WRITECOPY,
};

use super::{
    Access, Address, AddressRange, DirectMemoryAccess, MemoryError, MemoryMap, MemoryRegion,
    PlatformOperation, ProcessMemory, RegionKind, DEFAULT_MAX_REGIONS,
};

impl MemoryMap for ProcessMemory {
    fn region_containing(&self, address: Address) -> Result<MemoryRegion, MemoryError> {
        query_region(address)
    }
}

impl DirectMemoryAccess for ProcessMemory {
    unsafe fn read_bytes_into(
        &self,
        address: Address,
        output: &mut [u8],
    ) -> Result<(), MemoryError> {
        let range = address.checked_range(output.len())?;
        self.check_range(range, Access::READ, DEFAULT_MAX_REGIONS)?;
        // SAFETY: the trait caller keeps the validated source range alive and
        // readable for the copy. `ptr::copy` deliberately permits overlap.
        unsafe {
            ptr::copy(
                address.get() as *const u8,
                output.as_mut_ptr(),
                output.len(),
            )
        };
        Ok(())
    }

    unsafe fn write_bytes_from(&self, address: Address, bytes: &[u8]) -> Result<(), MemoryError> {
        let range = address.checked_range(bytes.len())?;
        self.check_range(range, Access::WRITE, DEFAULT_MAX_REGIONS)?;
        // SAFETY: the trait caller keeps the validated destination range alive
        // and writable and is responsible for the foreign object's invariants.
        unsafe { ptr::copy(bytes.as_ptr(), address.get() as *mut u8, bytes.len()) };
        Ok(())
    }
}

fn query_region(address: Address) -> Result<MemoryRegion, MemoryError> {
    // SAFETY: `MEMORY_BASIC_INFORMATION` is a plain FFI output structure for
    // which the all-zero bit pattern is valid before `VirtualQuery` fills it.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { zeroed() };
    // SAFETY: `info` is valid output storage of the size supplied. VirtualQuery
    // inspects the numeric address without dereferencing it in Rust.
    let written = unsafe {
        VirtualQuery(
            Some(address.get() as *const c_void),
            &mut info,
            size_of::<MEMORY_BASIC_INFORMATION>(),
        )
    };
    if written == 0 {
        return Err(last_platform_error(PlatformOperation::Query));
    }

    let start = Address::new(info.BaseAddress as usize);
    let range = AddressRange::new(start, info.RegionSize)?;
    if !range.contains(address) {
        return Err(MemoryError::NonProgressingRegion { address });
    }
    if info.State != MEM_COMMIT {
        return Err(MemoryError::Unmapped { address });
    }

    Ok(MemoryRegion {
        range,
        access: access_from_protection(info.Protect),
        kind: region_kind(info.Type),
        module: None,
    })
}

fn access_from_protection(protection: PAGE_PROTECTION_FLAGS) -> Access {
    if protection.contains(PAGE_GUARD) || protection.contains(PAGE_NOACCESS) {
        return Access::empty();
    }

    // PAGE_GUARD, PAGE_NOCACHE, PAGE_WRITECOMBINE, and CFG flags are modifiers;
    // the base protection occupies the low byte.
    let base = protection.0 & 0xff;
    if base == PAGE_READONLY.0 {
        Access::READ
    } else if base == PAGE_READWRITE.0 || base == PAGE_WRITECOPY.0 {
        Access::READ | Access::WRITE
    } else if base == PAGE_EXECUTE.0 {
        Access::EXECUTE
    } else if base == PAGE_EXECUTE_READ.0 {
        Access::READ | Access::EXECUTE
    } else if base == PAGE_EXECUTE_READWRITE.0 || base == PAGE_EXECUTE_WRITECOPY.0 {
        Access::READ | Access::WRITE | Access::EXECUTE
    } else {
        Access::empty()
    }
}

fn region_kind(kind: windows::Win32::System::Memory::PAGE_TYPE) -> RegionKind {
    if kind == MEM_PRIVATE {
        RegionKind::Private
    } else if kind == MEM_IMAGE {
        RegionKind::Image
    } else if kind == MEM_MAPPED {
        RegionKind::Mapped
    } else {
        RegionKind::Unknown
    }
}

fn last_platform_error(operation: PlatformOperation) -> MemoryError {
    let code = std::io::Error::last_os_error()
        .raw_os_error()
        .map_or(-1_i64, i64::from);
    MemoryError::Platform { operation, code }
}

#[cfg(feature = "fault-contained")]
#[derive(Clone, Copy, Debug, Default)]
pub struct FaultContainedProcessMemory {
    _private: (),
}

#[cfg(feature = "fault-contained")]
impl FaultContainedProcessMemory {
    pub const fn new() -> Self {
        Self { _private: () }
    }
}

#[cfg(feature = "fault-contained")]
impl MemoryMap for FaultContainedProcessMemory {
    fn region_containing(&self, address: Address) -> Result<MemoryRegion, MemoryError> {
        query_region(address)
    }
}

#[cfg(feature = "fault-contained")]
// SAFETY: every trait operation delegates the potentially faulting transfer to
// ReadProcessMemory or WriteProcessMemory after validating the entire range.
unsafe impl super::FaultContainedMemoryAccess for FaultContainedProcessMemory {
    fn try_read_bytes_into(&self, address: Address, output: &mut [u8]) -> Result<(), MemoryError> {
        use windows::Win32::System::Diagnostics::Debug::ReadProcessMemory;
        use windows::Win32::System::Threading::GetCurrentProcess;

        super::reject_read_overlap(address, output)?;
        let range = address.checked_range(output.len())?;
        self.check_range(range, Access::READ, DEFAULT_MAX_REGIONS)?;
        let mut transferred = 0_usize;
        // SAFETY: ReadProcessMemory contains invalid-address faults and writes
        // at most `output.len()` bytes into valid caller-owned storage.
        let result = unsafe {
            ReadProcessMemory(
                GetCurrentProcess(),
                address.get() as *const c_void,
                output.as_mut_ptr().cast::<c_void>(),
                output.len(),
                Some(&mut transferred),
            )
        };
        if transferred != 0 && transferred != output.len() {
            return Err(MemoryError::PartialRead {
                expected: output.len(),
                actual: transferred,
            });
        }
        result.map_err(|error| MemoryError::Platform {
            operation: PlatformOperation::Read,
            code: i64::from(error.code().0),
        })?;
        if transferred != output.len() {
            return Err(MemoryError::PartialRead {
                expected: output.len(),
                actual: transferred,
            });
        }
        Ok(())
    }

    unsafe fn try_write_bytes_from(
        &self,
        address: Address,
        bytes: &[u8],
    ) -> Result<(), MemoryError> {
        use windows::Win32::System::Diagnostics::Debug::WriteProcessMemory;
        use windows::Win32::System::Threading::GetCurrentProcess;

        let range = address.checked_range(bytes.len())?;
        self.check_range(range, Access::WRITE, DEFAULT_MAX_REGIONS)?;
        let mut transferred = 0_usize;
        // SAFETY: WriteProcessMemory contains invalid-address faults. The trait
        // caller is responsible for preserving destination object invariants.
        let result = unsafe {
            WriteProcessMemory(
                GetCurrentProcess(),
                address.get() as *const c_void,
                bytes.as_ptr().cast::<c_void>(),
                bytes.len(),
                Some(&mut transferred),
            )
        };
        if transferred != 0 && transferred != bytes.len() {
            return Err(MemoryError::PartialWrite {
                expected: bytes.len(),
                actual: transferred,
            });
        }
        result.map_err(|error| MemoryError::Platform {
            operation: PlatformOperation::Write,
            code: i64::from(error.code().0),
        })?;
        if transferred != bytes.len() {
            return Err(MemoryError::PartialWrite {
                expected: bytes.len(),
                actual: transferred,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protection_modifiers_do_not_change_the_base_access() {
        let modified = PAGE_EXECUTE_READWRITE | windows::Win32::System::Memory::PAGE_NOCACHE;
        assert_eq!(
            access_from_protection(modified),
            Access::READ | Access::WRITE | Access::EXECUTE
        );
        assert_eq!(
            access_from_protection(PAGE_READWRITE | PAGE_GUARD),
            Access::empty()
        );
    }

    #[test]
    fn process_map_finds_a_stable_stack_allocation() {
        let value = 17_u64;
        let address = Address::new(&value as *const u64 as usize);
        let region = ProcessMemory::new().region_containing(address).unwrap();
        assert!(region.range.contains(address));
        assert!(region.access.contains(Access::READ));
    }

    #[test]
    fn direct_backend_copies_a_stable_value() {
        let value = 0x0102_0304_u32;
        let address = Address::new(&value as *const u32 as usize);
        let mut output = [0_u8; size_of::<u32>()];
        // SAFETY: `value` remains live and readable throughout the copy.
        unsafe {
            ProcessMemory::new()
                .read_bytes_into(address, &mut output)
                .unwrap()
        };
        assert_eq!(u32::from_ne_bytes(output), value);
    }

    #[test]
    fn direct_backend_writes_a_stable_value() {
        let mut value = 3_u32;
        let address = Address::new(&mut value as *mut u32 as usize);
        // SAFETY: `value` remains live and exclusively accessible throughout
        // the write, and every byte pattern is valid for `u32`.
        unsafe {
            ProcessMemory::new()
                .write_bytes_from(address, &9_u32.to_ne_bytes())
                .unwrap()
        };
        assert_eq!(value, 9);
    }

    #[cfg(feature = "fault-contained")]
    #[test]
    fn fault_contained_backend_copies_and_rejects_bad_addresses() {
        use super::super::FaultContainedMemoryAccess;

        let value = 23_u32;
        let address = Address::new(&value as *const u32 as usize);
        let memory = FaultContainedProcessMemory::new();
        let mut output = [0_u8; size_of::<u32>()];
        memory.try_read_bytes_into(address, &mut output).unwrap();
        assert_eq!(u32::from_ne_bytes(output), value);

        assert!(memory
            .try_read_bytes_into(Address::new(1), &mut output)
            .is_err());

        let overlap = Address::new(output.as_ptr() as usize);
        assert!(matches!(
            memory.try_read_bytes_into(overlap, &mut output),
            Err(MemoryError::OverlappingRanges { .. })
        ));
    }
}
