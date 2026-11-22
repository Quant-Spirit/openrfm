// Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
use clap::{Parser, Subcommand};
use rustler_archive::pre;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Convert a raw database (Parquet tables plus task metadata) into an archive.
    Pre(pre::Cli),
}

/// Input (`RELBENCH_DATA_ROOT`) and output (`RELBENCH_PRE_ROOT`) roots must be set explicitly.
fn env_root(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| {
        eprintln!("error: environment variable {name} is not set");
        std::process::exit(2)
    })
}

fn main() {
    let Command::Pre(cli) = Cli::parse().command;
    let (data_root, pre_root) = (env_root("RELBENCH_DATA_ROOT"), env_root("RELBENCH_PRE_ROOT"));
    pyo3::Python::initialize();
    pre::run(&cli, &data_root, &pre_root);
}
