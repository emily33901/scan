// On macOS we use disarm64 to decode arm64 instructions and extract their immediates.
// Generating its complete instruction decoder is extremely slow, so the build script
// generates a decoder containing only the four instruction classes scan needs.

type BuildResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn ensure_success(success: bool, action: &'static str) -> BuildResult {
    if success {
        Ok(())
    } else {
        Err(std::io::Error::other(action).into())
    }
}

fn generate_arm64_decoder() -> BuildResult {
    use std::{path::PathBuf, process::Command};

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

        ensure_success(
            command.spawn()?.wait()?.success(),
            "failed to clone disarm64",
        )?;
    }

    let current_revision = Command::new("git")
        .args(["-C", "disarm64", "rev-parse", "HEAD"])
        .output()?;
    ensure_success(
        current_revision.status.success(),
        "failed to inspect disarm64 revision",
    )?;
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
        ensure_success(
            command.spawn()?.wait()?.success(),
            "failed to fetch disarm64",
        )?;

        let mut command = Command::new("git");
        command.args(["-C", "disarm64", "checkout", "--detach", "FETCH_HEAD"]);
        ensure_success(
            command.spawn()?.wait()?.success(),
            "failed to check out disarm64",
        )?;
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

    ensure_success(
        command.spawn()?.wait()?.success(),
        "failed to generate arm64 decoder",
    )?;

    let mut command = Command::new("rustfmt");
    command.args([output_decoder_path.to_str().unwrap()]);

    ensure_success(
        command.spawn()?.wait()?.success(),
        "failed to format arm64 decoder",
    )?;

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

fn main() -> BuildResult {
    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_OS");
    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS")?;
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH")?;

    if target_os == "macos" && target_arch == "aarch64" {
        generate_arm64_decoder()?;
    }

    Ok(())
}
