/// On MacOS we are using disarm64 to decode arm64 instructions and then get immediates from those.
/// disarm64 takes an incredibly long time to build by default as it builds decision trees for the entire
/// instruction set, which we don't care about, we maybe use 4 instructions, so we can cut down
/// quite significantly on the build time if we just use those.
/// This function will do the work of getting the git repo, and running disarm64_gen to build a decoder
/// which can then be `include!`ed in scan.

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() -> anyhow::Result<()> {
    use std::{path::PathBuf, process::Command};

    use anyhow::ensure;

    const DISARM64_REVISION: &str = "4a5744dddcf7377172f7e33bd4483f7e31688171";

    let output_dir: PathBuf = std::env::var("OUT_DIR")?.into();
    std::env::set_current_dir(&output_dir)?;

    let output_decoder_path = output_dir.join("decoder.rs");
    let output_mod_rs_path = output_dir.join("mod.rs");

    if !std::fs::exists("disarm64")? {
        let mut command = Command::new("git");
        command.args([
            "clone",
            "--no-checkout",
            "https://github.com/kromych/disarm64",
        ]);

        ensure!(command.spawn()?.wait()?.success());
    }

    let current_revision = Command::new("git")
        .args(["-C", "disarm64", "rev-parse", "HEAD"])
        .output()?;
    ensure!(current_revision.status.success());
    if String::from_utf8(current_revision.stdout)?.trim() != DISARM64_REVISION {
        let mut command = Command::new("git");
        command.args([
            "-C",
            "disarm64",
            "fetch",
            "--depth",
            "1",
            "origin",
            DISARM64_REVISION,
        ]);
        ensure!(command.spawn()?.wait()?.success());

        let mut command = Command::new("git");
        command.args(["-C", "disarm64", "checkout", "--detach", "FETCH_HEAD"]);
        ensure!(command.spawn()?.wait()?.success());
    }

    std::env::set_current_dir(output_dir.join("disarm64"))?;

    let mut command = Command::new(env!("CARGO"));
    command.args([
        "run",
        "--bin",
        "disarm64_gen",
        "--release",
        "--",
        "aarch64.json",
        "-c",
        "ldst_pos,branch_imm,pcreladdr,addsub_imm",
        "-r",
        output_decoder_path.to_str().unwrap(),
    ]);

    ensure!(command.spawn()?.wait()?.success());

    let mut command = Command::new("rustfmt");
    command.args([output_decoder_path.to_str().unwrap()]);

    ensure!(command.spawn()?.wait()?.success());

    // Write a mod.rs
    std::fs::write(
        output_dir.join("mod.rs"),
        "pub mod decoder; pub use decoder::*;",
    )?;

    eprintln!("mod.rs is at {}", output_mod_rs_path.display());

    println!(
        "cargo::rustc-env=DECODER_MOD={}",
        output_mod_rs_path.display()
    );

    Ok(())
}

#[cfg(target_os = "windows")]
fn main() {}
