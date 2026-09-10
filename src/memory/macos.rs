//! macOS implementation of current-process memory access.

use std::mem::MaybeUninit;

use mach2::kern_return::KERN_SUCCESS;
use mach2::mach_port::mach_port_deallocate;
use mach2::port::{mach_port_t, MACH_PORT_NULL};
use mach2::traps::mach_task_self;
use mach2::vm::{mach_vm_read_overwrite, mach_vm_region, mach_vm_write};
use mach2::vm_prot::{VM_PROT_EXECUTE, VM_PROT_READ, VM_PROT_WRITE};
use mach2::vm_region::{vm_region_basic_info_64, VM_REGION_BASIC_INFO_64};
use mach2::vm_types::{mach_vm_address_t, mach_vm_size_t, vm_offset_t};

use super::{
    Access, Address, AddressRange, DirectMemoryAccess, MemoryError, MemoryMap, MemoryRegion,
    PlatformOperation, ProcessMemory, RegionKind, DEFAULT_MAX_REGIONS,
};

/// Releases an object-name send right returned by `mach_vm_region`.
struct ObjectNamePort {
    task: mach_port_t,
    name: mach_port_t,
}

impl Drop for ObjectNamePort {
    fn drop(&mut self) {
        if self.name != MACH_PORT_NULL {
            // SAFETY: `name` is the object-name send right returned for this
            // query and `task` is the current task's IPC namespace.
            let _ = unsafe { mach_port_deallocate(self.task, self.name) };
        }
    }
}

impl MemoryMap for ProcessMemory {
    fn region_containing(&self, address: Address) -> Result<MemoryRegion, MemoryError> {
        let requested =
            mach_vm_address_t::try_from(address.get()).map_err(|_| MemoryError::InvalidValue {
                context: "Mach query address",
            })?;
        let mut region_address = requested;
        let mut region_size: mach_vm_size_t = 0;
        let mut info = MaybeUninit::<vm_region_basic_info_64>::uninit();
        let mut info_count = vm_region_basic_info_64::count();
        let mut object_name = MACH_PORT_NULL;

        // SAFETY: all output pointers refer to initialized storage of the types
        // and sizes required by `VM_REGION_BASIC_INFO_64`.
        let task = unsafe { mach_task_self() };
        let result = unsafe {
            mach_vm_region(
                task,
                &mut region_address,
                &mut region_size,
                VM_REGION_BASIC_INFO_64,
                info.as_mut_ptr().cast::<libc::c_int>(),
                &mut info_count,
                &mut object_name,
            )
        };
        let _object_name = ObjectNamePort {
            task,
            name: object_name,
        };

        if result != KERN_SUCCESS {
            return Err(MemoryError::Platform {
                operation: PlatformOperation::Query,
                code: i64::from(result),
            });
        }
        if info_count < vm_region_basic_info_64::count() {
            return Err(MemoryError::InvalidValue {
                context: "Mach region info count",
            });
        }

        // SAFETY: a successful call with the expected count initialized the
        // complete basic-info structure.
        let info = unsafe { info.assume_init() };
        let start = usize::try_from(region_address)
            .map(Address::new)
            .map_err(|_| MemoryError::InvalidValue {
                context: "Mach region start",
            })?;
        let len = usize::try_from(region_size).map_err(|_| MemoryError::InvalidValue {
            context: "Mach region length",
        })?;
        let range = AddressRange::new(start, len)?;

        // `mach_vm_region` may return the next mapping after the requested
        // address, so explicitly reject holes.
        if !range.contains(address) {
            return Err(MemoryError::Unmapped { address });
        }

        let mut access = Access::empty();
        if info.protection & VM_PROT_READ != 0 {
            access |= Access::READ;
        }
        if info.protection & VM_PROT_WRITE != 0 {
            access |= Access::WRITE;
        }
        if info.protection & VM_PROT_EXECUTE != 0 {
            access |= Access::EXECUTE;
        }

        Ok(MemoryRegion {
            range,
            access,
            kind: if info.shared != 0 {
                RegionKind::Shared
            } else {
                RegionKind::Private
            },
            module: None,
        })
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
        // SAFETY: the trait caller keeps the validated source range live and
        // readable through this overlap-tolerant byte copy.
        unsafe {
            std::ptr::copy(
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
        // SAFETY: the trait caller keeps the validated destination range live
        // and writable and guarantees the semantic validity of the write.
        unsafe { std::ptr::copy(bytes.as_ptr(), address.get() as *mut u8, bytes.len()) };
        Ok(())
    }
}

/// Fault-contained current-process access backed by Mach VM operations.
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
        ProcessMemory::new().region_containing(address)
    }
}

#[cfg(feature = "fault-contained")]
unsafe impl super::FaultContainedMemoryAccess for FaultContainedProcessMemory {
    fn try_read_bytes_into(&self, address: Address, output: &mut [u8]) -> Result<(), MemoryError> {
        let range = address.checked_range(output.len())?;
        self.check_range(range, Access::READ, DEFAULT_MAX_REGIONS)?;
        let source =
            mach_vm_address_t::try_from(address.get()).map_err(|_| MemoryError::InvalidValue {
                context: "Mach read address",
            })?;
        let destination =
            mach_vm_address_t::try_from(output.as_mut_ptr() as usize).map_err(|_| {
                MemoryError::InvalidValue {
                    context: "Mach read destination",
                }
            })?;
        let size =
            mach_vm_size_t::try_from(output.len()).map_err(|_| MemoryError::LengthExceeded {
                requested: output.len(),
                limit: mach_vm_size_t::MAX as usize,
            })?;
        let mut transferred = 0;

        // SAFETY: Mach contains invalid-source faults and writes at most `size`
        // bytes into the caller-provided output allocation.
        let result = unsafe {
            mach_vm_read_overwrite(
                mach_task_self(),
                source,
                size,
                destination,
                &mut transferred,
            )
        };
        if result != KERN_SUCCESS {
            return Err(MemoryError::Platform {
                operation: PlatformOperation::Read,
                code: i64::from(result),
            });
        }
        if transferred != size {
            return Err(MemoryError::PartialRead {
                expected: output.len(),
                actual: usize::try_from(transferred).unwrap_or(usize::MAX),
            });
        }
        Ok(())
    }

    unsafe fn try_write_bytes_from(
        &self,
        address: Address,
        bytes: &[u8],
    ) -> Result<(), MemoryError> {
        let range = address.checked_range(bytes.len())?;
        self.check_range(range, Access::WRITE, DEFAULT_MAX_REGIONS)?;
        let destination =
            mach_vm_address_t::try_from(address.get()).map_err(|_| MemoryError::InvalidValue {
                context: "Mach write address",
            })?;
        let count = u32::try_from(bytes.len()).map_err(|_| MemoryError::LengthExceeded {
            requested: bytes.len(),
            limit: u32::MAX as usize,
        })?;

        // SAFETY: the byte slice is live for the call, and Mach contains
        // invalid-destination faults. Semantic write validity is the caller's
        // responsibility under the trait contract.
        let result = unsafe {
            mach_vm_write(
                mach_task_self(),
                destination,
                bytes.as_ptr() as vm_offset_t,
                count,
            )
        };
        if result != KERN_SUCCESS {
            return Err(MemoryError::Platform {
                operation: PlatformOperation::Write,
                code: i64::from(result),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_and_reads_owned_memory() {
        let value = 0x1122_3344_5566_7788_u64;
        let address = Address::new(&value as *const u64 as usize);
        let memory = ProcessMemory::new();
        let region = memory.region_containing(address).unwrap();
        assert!(region.range.contains(address));
        assert!(region.access.contains(Access::READ));

        let mut copied = [0_u8; std::mem::size_of::<u64>()];
        // SAFETY: `value` remains live and immutable throughout the copy.
        unsafe { memory.read_bytes_into(address, &mut copied).unwrap() };
        assert_eq!(u64::from_ne_bytes(copied), value);
    }

    #[cfg(feature = "fault-contained")]
    #[test]
    fn fault_contained_reads_owned_memory() {
        use super::super::FaultContainedMemoryAccess;

        let value = 0xaabb_ccdd_u32;
        let address = Address::new(&value as *const u32 as usize);
        let memory = FaultContainedProcessMemory::new();
        let mut copied = [0_u8; std::mem::size_of::<u32>()];
        memory.try_read_bytes_into(address, &mut copied).unwrap();
        assert_eq!(u32::from_ne_bytes(copied), value);
    }
}
