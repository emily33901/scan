#![cfg(any(target_os = "windows", target_os = "macos"))]

use scan::memory::{
    Access, Address, DirectMemoryAccess, MemoryError, MemoryMap, ProcessMemory, DEFAULT_MAX_REGIONS,
};

#[cfg(feature = "fault-contained")]
use scan::memory::{FaultContainedMemoryAccess, FaultContainedProcessMemory};

#[cfg(target_os = "windows")]
mod platform {
    use std::ffi::c_void;
    use std::mem::MaybeUninit;

    use windows::Win32::System::Memory::{
        VirtualAlloc, VirtualFree, VirtualProtect, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE,
        PAGE_GUARD, PAGE_NOACCESS, PAGE_PROTECTION_FLAGS, PAGE_READONLY, PAGE_READWRITE,
    };
    use windows::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};

    use super::Address;

    pub struct ScratchPages {
        base: *mut c_void,
        pub page_size: usize,
        pages: usize,
    }

    impl ScratchPages {
        pub fn new(pages: usize) -> Self {
            let mut info = MaybeUninit::<SYSTEM_INFO>::uninit();
            // SAFETY: `info` is correctly sized writable output storage.
            unsafe { GetSystemInfo(info.as_mut_ptr()) };
            // SAFETY: GetSystemInfo initializes the complete structure.
            let page_size = unsafe { info.assume_init() }.dwPageSize as usize;
            let len = page_size.checked_mul(pages).expect("scratch mapping size");
            // SAFETY: requests a fresh private mapping with no address hint.
            let base = unsafe { VirtualAlloc(None, len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
            assert!(
                !base.is_null(),
                "VirtualAlloc failed: {}",
                std::io::Error::last_os_error()
            );
            Self {
                base,
                page_size,
                pages,
            }
        }

        pub fn address(&self, offset: usize) -> Address {
            assert!(offset < self.page_size * self.pages);
            Address::new(self.base as usize + offset)
        }

        pub fn protect(&self, page: usize, protection: PAGE_PROTECTION_FLAGS) {
            assert!(page < self.pages);
            let address = (self.base as usize + page * self.page_size) as *const c_void;
            let mut old = PAGE_PROTECTION_FLAGS::default();
            // SAFETY: the requested page is contained by this live allocation.
            unsafe { VirtualProtect(address, self.page_size, protection, &mut old) }
                .expect("VirtualProtect failed");
        }

        pub fn make_read_only(&self, page: usize) {
            self.protect(page, PAGE_READONLY);
        }

        pub fn make_inaccessible(&self, page: usize) {
            self.protect(page, PAGE_NOACCESS);
        }

        pub fn make_guarded(&self, page: usize) {
            self.protect(page, PAGE_READWRITE | PAGE_GUARD);
        }

        pub fn release(mut self) -> Address {
            let address = Address::new(self.base as usize);
            // SAFETY: this is the allocation base and MEM_RELEASE requires size zero.
            unsafe { VirtualFree(self.base, 0, MEM_RELEASE) }.expect("VirtualFree failed");
            self.base = std::ptr::null_mut();
            address
        }
    }

    impl Drop for ScratchPages {
        fn drop(&mut self) {
            if !self.base.is_null() {
                // SAFETY: this is the allocation base and is released exactly once.
                let _ = unsafe { VirtualFree(self.base, 0, MEM_RELEASE) };
            }
        }
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use std::ffi::c_void;

    use super::Address;

    pub struct ScratchPages {
        base: *mut c_void,
        pub page_size: usize,
        pages: usize,
    }

    impl ScratchPages {
        pub fn new(pages: usize) -> Self {
            // SAFETY: sysconf has no pointer arguments or memory side effects.
            let raw_page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            assert!(raw_page_size > 0, "sysconf(_SC_PAGESIZE) failed");
            let page_size = raw_page_size as usize;
            let len = page_size.checked_mul(pages).expect("scratch mapping size");
            // SAFETY: requests a fresh anonymous private mapping with no address hint.
            let base = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANON,
                    -1,
                    0,
                )
            };
            assert_ne!(
                base,
                libc::MAP_FAILED,
                "mmap failed: {}",
                std::io::Error::last_os_error()
            );
            Self {
                base,
                page_size,
                pages,
            }
        }

        pub fn address(&self, offset: usize) -> Address {
            assert!(offset < self.page_size * self.pages);
            Address::new(self.base as usize + offset)
        }

        fn protect(&self, page: usize, protection: libc::c_int) {
            assert!(page < self.pages);
            let address = (self.base as usize + page * self.page_size) as *mut c_void;
            // SAFETY: the requested page is contained by this live allocation.
            let result = unsafe { libc::mprotect(address, self.page_size, protection) };
            assert_eq!(
                result,
                0,
                "mprotect failed: {}",
                std::io::Error::last_os_error()
            );
        }

        pub fn make_read_only(&self, page: usize) {
            self.protect(page, libc::PROT_READ);
        }

        pub fn make_inaccessible(&self, page: usize) {
            self.protect(page, libc::PROT_NONE);
        }

        pub fn release(mut self) -> Address {
            let address = Address::new(self.base as usize);
            // SAFETY: this exactly covers the live mapping.
            let result = unsafe { libc::munmap(self.base, self.page_size * self.pages) };
            assert_eq!(
                result,
                0,
                "munmap failed: {}",
                std::io::Error::last_os_error()
            );
            self.base = std::ptr::null_mut();
            address
        }
    }

    impl Drop for ScratchPages {
        fn drop(&mut self) {
            if !self.base.is_null() {
                // SAFETY: this exactly covers the mapping and runs at most once.
                let _ = unsafe { libc::munmap(self.base, self.page_size * self.pages) };
            }
        }
    }
}

use platform::ScratchPages;

#[test]
fn stable_readable_writable_page_supports_checked_copies() {
    let pages = ScratchPages::new(1);
    let address = pages.address(0);
    let memory = ProcessMemory::new();
    let value = 0x1122_3344_5566_7788_u64;

    // SAFETY: the scratch allocation remains live, writable, and exclusively
    // controlled by this test for both exact copies.
    unsafe { memory.write_bytes_from(address, &value.to_ne_bytes()) }.unwrap();
    let mut copied = [0_u8; size_of::<u64>()];
    // SAFETY: the initialized scratch bytes remain live and readable.
    unsafe { memory.read_bytes_into(address, &mut copied) }.unwrap();

    assert_eq!(u64::from_ne_bytes(copied), value);
    let region = memory.region_containing(address).unwrap();
    assert!(region.access.contains(Access::READ | Access::WRITE));
}

#[test]
fn read_only_page_rejects_write_permission() {
    let pages = ScratchPages::new(1);
    pages.make_read_only(0);
    let address = pages.address(0);
    let region = ProcessMemory::new().region_containing(address).unwrap();

    assert!(region.access.contains(Access::READ));
    assert!(!region.access.contains(Access::WRITE));
    assert!(matches!(
        ProcessMemory::new().check_range(
            address.checked_range(1).unwrap(),
            Access::WRITE,
            DEFAULT_MAX_REGIONS,
        ),
        Err(MemoryError::PermissionDenied { .. })
    ));
}

#[inline(never)]
fn executable_probe() {}

#[test]
fn executable_image_reports_execute_permission() {
    let address = Address::new(executable_probe as *const () as usize);
    let region = ProcessMemory::new().region_containing(address).unwrap();

    assert!(region.range.contains(address));
    assert!(region.access.contains(Access::EXECUTE));
}

#[test]
fn inaccessible_page_and_cross_boundary_range_are_rejected() {
    let pages = ScratchPages::new(2);
    pages.make_inaccessible(1);
    let inaccessible = pages.address(pages.page_size);
    let crossing = pages.address(pages.page_size - 1);
    let memory = ProcessMemory::new();

    let inaccessible_region = memory.region_containing(inaccessible).unwrap();
    assert!(inaccessible_region.access.is_empty());
    assert!(matches!(
        memory.check_range(
            crossing.checked_range(2).unwrap(),
            Access::READ,
            DEFAULT_MAX_REGIONS,
        ),
        Err(MemoryError::PermissionDenied { address, .. }) if address == inaccessible
    ));
}

#[test]
fn released_page_is_reported_unmapped() {
    #[cfg(target_os = "macos")]
    {
        const ISOLATED: &str = "SCAN_ISOLATED_UNMAPPED_PAGE_TEST";
        if std::env::var_os(ISOLATED).is_none() {
            // After munmap, sibling tests and test-harness threads can reuse
            // the address for mappings or stack guards. Query in a separate
            // process running only this test, with no sibling test threads.
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "released_page_is_reported_unmapped",
                    "--test-threads=1",
                    "--nocapture",
                ])
                .env(ISOLATED, "1")
                .output()
                .expect("start isolated unmapped-page test");
            assert!(
                output.status.success(),
                "isolated unmapped-page test failed ({}):\n{}\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
            return;
        }
    }

    let address = ScratchPages::new(1).release();
    let result = ProcessMemory::new().region_containing(address);
    assert!(
        matches!(result, Err(MemoryError::Unmapped { address: rejected }) if rejected == address),
        "released address {address:?}: {result:?}"
    );
}

#[cfg(target_os = "windows")]
#[test]
fn guarded_page_reports_no_access_without_touching_it() {
    let pages = ScratchPages::new(1);
    pages.make_guarded(0);
    let address = pages.address(0);
    let region = ProcessMemory::new().region_containing(address).unwrap();

    assert!(region.access.is_empty());
    assert!(matches!(
        ProcessMemory::new().check_range(
            address.checked_range(1).unwrap(),
            Access::READ,
            DEFAULT_MAX_REGIONS,
        ),
        Err(MemoryError::PermissionDenied { .. })
    ));
}

#[cfg(feature = "fault-contained")]
#[test]
fn fault_contained_read_rejects_invalid_and_inaccessible_addresses() {
    let pages = ScratchPages::new(1);
    pages.make_inaccessible(0);
    let memory = FaultContainedProcessMemory::new();
    let mut output = [0_u8; 8];

    assert!(memory
        .try_read_bytes_into(Address::new(1), &mut output)
        .is_err());
    assert!(memory
        .try_read_bytes_into(pages.address(0), &mut output)
        .is_err());
}
