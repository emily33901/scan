use std::ops::Range;
use windows::{
    core::HSTRING,
    Win32::{
        Foundation::{FreeLibrary, HMODULE},
        System::{
            Diagnostics::Debug::IMAGE_NT_HEADERS64, LibraryLoader::LoadLibraryW,
            SystemServices::IMAGE_DOS_HEADER,
        },
    },
};

pub struct Module {
    address: usize,
    ll: libloading::Library,
}

use anyhow::Result;

impl Module {
    pub fn code_section_address_range(&self) -> Range<usize> {
        let (start, size) = self.code_range();
        start..start + size
    }

    fn code_range(&self) -> (usize, usize) {
        unsafe {
            let dos_header = (self.address as *const IMAGE_DOS_HEADER).as_ref().unwrap();
            let nt_header = ((self.address + dos_header.e_lfanew as usize)
                as *const IMAGE_NT_HEADERS64)
                .as_ref()
                .unwrap();
            (
                self.address + nt_header.OptionalHeader.BaseOfCode as usize,
                nt_header.OptionalHeader.SizeOfCode as usize,
            )
        }
    }

    fn code_slice(&self) -> &[u8] {
        let (start, size) = self.code_range();
        unsafe { std::slice::from_raw_parts(start as *const u8, size) }
    }

    fn scan_slice(&self, s: &[u8], pattern: &str, offset: usize) -> Result<Option<usize>> {
        let result = patternscan::scan_first_match(std::io::Cursor::new(s), pattern)?
            .map(|addr| self.code_range().0 + addr + offset);

        Ok(result)
    }

    pub fn scan(&self, pattern: &str, offset: usize) -> Result<Option<usize>> {
        let code_slice = self.code_slice();
        self.scan_slice(code_slice, pattern, offset)
    }

    pub fn new(name: &str) -> Result<Module> {
        let module_handle = unsafe { LoadLibraryW(&HSTRING::from(name))? };

        Ok(Self {
            address: module_handle.0 as usize,
            ll: unsafe { libloading::Library::new(name) }.unwrap(),
        })
    }

    pub fn export<F>(&self, name: &[u8]) -> Result<libloading::Symbol<F>> {
        Ok(unsafe { self.ll.get(name) }?)
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        let _ = unsafe { FreeLibrary(HMODULE(self.address as isize)) };
    }
}
