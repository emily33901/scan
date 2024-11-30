use anyhow::Context;
use libc::dlopen;
use mach2::dyld::_dyld_get_image_header;
use mach2::dyld::_dyld_get_image_name;
use mach2::dyld::_dyld_get_image_vmaddr_slide;
use mach2::dyld::_dyld_image_count;
use object::read::macho::MachHeader;
use object::read::macho::Segment;
use object::LittleEndian;
use std::ffi::CStr;

pub struct Module {
    handle: usize,
    code_range: (usize, usize),
    ll: libloading::Library,
}

impl Module {
    fn code_slice(&self) -> &[u8] {
        let (start, size) = self.code_range;
        unsafe { std::slice::from_raw_parts(start as *const u8, size) }
    }

    fn scan_slice(&self, s: &[u8], pattern: &str, offset: usize) -> Result<Option<usize>> {
        let result = patternscan::scan_first_match(std::io::Cursor::new(s), pattern)?
            .map(|addr| self.code_range.0 + addr + offset);

        Ok(result)
    }

    pub fn scan(&self, pattern: &str, offset: usize) -> Result<Option<usize>> {
        let code_slice = self.code_slice();
        self.scan_slice(code_slice, pattern, offset)
    }

    pub fn new(name: &str) -> Result<Module> {
        let file = std::ffi::CString::new(name).unwrap();
        let handle = unsafe { dlopen(file.as_ptr(), libc::RTLD_LAZY) };

        let ll = unsafe { libloading::Library::new(name)? };

        // Find the image we want in this list
        let code_range = find_code_range_for_image(&ll, name)?;

        Ok(Self {
            handle: handle as usize,
            code_range,
            ll,
        })
    }

    pub fn export<F>(&self, name: &[u8]) -> Result<libloading::Symbol<F>> {
        Ok(unsafe { self.ll.get(name) }?)
    }
}

fn find_code_range_for_image(ll: &libloading::Library, name: &str) -> Result<(usize, usize)> {
    let (mach_header, slide) = (|| {
        let image_count = unsafe { _dyld_image_count() };
        for i in 0..image_count {
            let image_name = unsafe { _dyld_get_image_name(i) };
            let image_name = unsafe { CStr::from_ptr(image_name) };

            if let Some(filename) = std::path::Path::new(&image_name.to_string_lossy().to_string())
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
            {
                if &filename == name {
                    // Found this libs index
                    return Ok(unsafe {
                        (_dyld_get_image_header(i), _dyld_get_image_vmaddr_slide(i))
                    });
                }
            }
        }
        Err(anyhow::anyhow!("unable to find image for code-range"))
    })()?;

    let header_symbol = mach_header as *const u8;

    // let header_address_ptr = header_address as *const u8;
    // Invent a slice in order to read header
    let slice = unsafe { std::slice::from_raw_parts(header_symbol, 0x10000) };

    let header = object::macho::MachHeader64::<object::LittleEndian>::parse(slice, 0)
        .expect("Failed to parse Mach-O header");

    let mut load_commands = header
        .load_commands(object::LittleEndian, slice, 0)
        .expect("Failed to get load commands");

    while let Some(command) = load_commands.next()? {
        if let Some((segment, slice)) = command.segment_64()? {
            if segment.name() == b"__TEXT" {
                let address = segment.vmaddr(LittleEndian) as usize;
                let size = segment.vmsize(LittleEndian) as usize;

                return Ok((
                    address
                        .checked_add_signed(slide)
                        .context("Failed to add slide to segment vmaddr")?,
                    size as usize,
                ));
            }
        }
    }

    unreachable!()
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            libc::dlclose(self.handle as *mut std::ffi::c_void);
        }
    }
}

use anyhow::Result;

#[cfg(target_arch = "aarch64")]
pub fn resolve_relative_address(addr: usize, offset: usize) -> usize {
    todo!("Not implemented for arm64 macos")
}
