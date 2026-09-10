#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
#[test]
fn required_rosetta_job_is_actually_translated() {
    if std::env::var_os("SCAN_REQUIRE_ROSETTA").is_none() {
        return;
    }

    let mut translated = 0_i32;
    let mut size = std::mem::size_of_val(&translated);
    // SAFETY: the name is NUL-terminated and both output pointers describe a
    // live `i32` allocation of the supplied size. This reads kernel metadata.
    let result = unsafe {
        libc::sysctlbyname(
            c"sysctl.proc_translated".as_ptr(),
            (&mut translated as *mut i32).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    assert_eq!(result, 0, "sysctl.proc_translated query failed");
    assert_eq!(translated, 1, "x86_64 test binary is not under Rosetta");
}
