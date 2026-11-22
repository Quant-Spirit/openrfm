use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use worldgen::forecast::{self as v6, V6Archetype, V6Stage, V6Variant};
use worldgen::{
    corpus_scale_coverage, generate_db, print_corpus_scale_coverage, write_generated_db,
    GenOptions, GeneratedDb,
};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The label axis. Only the temporal-kernel axis is part of this release.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum AxisArg {
    TemporalKernel,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum StageArg {
    S1,
    S2,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ArchetypeArg {
    /// Dense own-history forecast.
    A1,
    /// Peer / hub collaborative through a heavy-tailed hub.
    A4,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum VariantArg {
    /// A1 with long own histories; the label reads the own event count over a long window.
    LongHistory,
    /// One row per entity; the label is a low-rank nonlinear map of the own-row features.
    FRank,
}

#[derive(Debug, Parser)]
#[command(version, about = "Synthetic relational database generator")]
struct Cli {
    #[arg(long)]
    seed_start: u64,
    #[arg(long)]
    num_dbs: usize,
    #[arg(long, value_enum)]
    axis: AxisArg,
    #[arg(long)]
    output_root: PathBuf,
    #[arg(long, default_value_t = 3)]
    min_tables: usize,
    #[arg(long, default_value_t = 15)]
    max_tables: usize,
    #[arg(long, default_value_t = 5)]
    min_cols: usize,
    #[arg(long, default_value_t = 20)]
    max_cols: usize,
    #[arg(long, default_value_t = 500)]
    min_entity_rows: usize,
    #[arg(long, default_value_t = 1000)]
    max_entity_rows: usize,
    #[arg(long, default_value_t = 2000)]
    min_activity_rows: usize,
    #[arg(long, default_value_t = 5000)]
    max_activity_rows: usize,
    /// Publish `xagg_*` exact-aggregate sidecar columns (count/sum/mean/max of a
    /// child table over 7d/30d/all windows, pushed onto the parent row).
    #[arg(long, default_value_t = false)]
    exact_aggregates: bool,
    /// Publish `nuisance_text_{i}` high-cardinality text feature columns that are
    /// independent of the labels.
    #[arg(long, default_value_t = false)]
    nuisance_text: bool,
    /// Forecasting-world stage.
    #[arg(long, value_enum, requires = "v6_kind")]
    v6_stage: Option<StageArg>,
    /// Archetype of every slot.
    #[arg(long, value_enum, requires = "v6_stage", group = "v6_kind")]
    v6_archetype: Option<ArchetypeArg>,
    /// Variant of every slot: its own slot numbering, seed range and archetype.
    #[arg(long, value_enum, requires = "v6_stage", group = "v6_kind")]
    v6_variant: Option<VariantArg>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.num_dbs == 0 {
        bail!("--num-dbs must be positive");
    }
    let AxisArg::TemporalKernel = cli.axis;
    let mut options = GenOptions {
        exact_aggregates: cli.exact_aggregates,
        nuisance_text: cli.nuisance_text,
        ..GenOptions::default()
    };
    options.schema.num_tables_range = ordered(cli.min_tables, cli.max_tables);
    options.schema.num_cols_range = ordered(cli.min_cols, cli.max_cols);
    options.schema.rows_entity_range = ordered(cli.min_entity_rows, cli.max_entity_rows);
    options.schema.rows_activity_range = ordered(cli.min_activity_rows, cli.max_activity_rows);
    options.validate_corpus_scale()?;

    let output_root = expanduser(cli.output_root.clone());
    std::fs::create_dir_all(&output_root)
        .with_context(|| format!("failed to create {}", output_root.display()))?;
    let started = Instant::now();
    let seeds: Vec<u64> = (0..cli.num_dbs)
        .map(|i| cli.seed_start + i as u64)
        .collect();
    let records = seeds
        .par_iter()
        .map(|seed| match cli.v6_stage {
            Some(stage) => {
                let stage = match stage {
                    StageArg::S1 => V6Stage::S1,
                    StageArg::S2 => V6Stage::S2,
                };
                let kind = match (cli.v6_archetype, cli.v6_variant) {
                    (_, Some(VariantArg::LongHistory)) => Kind::Variant(V6Variant::LongHistory),
                    (_, Some(VariantArg::FRank)) => Kind::Variant(V6Variant::FRank),
                    (Some(ArchetypeArg::A1), None) => Kind::Archetype(V6Archetype::A1),
                    (Some(ArchetypeArg::A4), None) => Kind::Archetype(V6Archetype::A4),
                    (None, None) => unreachable!("clap requires an archetype or a variant"),
                };
                generate_v6(stage, kind, *seed as usize, &options, &output_root)
            }
            None => generate_plain(*seed, &options, &output_root),
        })
        .collect::<Result<Vec<_>>>()?;
    let manifest = json!({
        "schema_version": 1,
        "backend": "rust",
        "generator": "worldgen",
        "axis": "TemporalKernel",
        "seed_start": cli.seed_start,
        "num_dbs": cli.num_dbs,
        "output_root": output_root.display().to_string(),
        "records": records,
        "elapsed_sec": started.elapsed().as_secs_f64(),
    });
    std::fs::write(
        output_root.join("worldgen_manifest.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    eprintln!(
        "generated {} DBs under {} in {:.2}s",
        cli.num_dbs,
        output_root.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn generate_plain(seed: u64, options: &GenOptions, output_root: &Path) -> Result<Value> {
    let t0 = Instant::now();
    let db = generate_db(seed, options)?;
    write_generated_db(output_root, &db)?;
    let mut record = json!({
        "seed": seed,
        "db_name": db.name,
        "status": "generated",
        "output_dir": output_root.join(&db.name).display().to_string(),
        "table_count": db.tables.len(),
        "program_label_count": db.program_metadata.len(),
        "elapsed_sec": t0.elapsed().as_secs_f64(),
    });
    attach_corpus_scale_receipt(&mut record, &db, options)?;
    Ok(record)
}

/// Retries a slot with up to `V6_MAX_ATTEMPTS` generation seeds; a slot whose
/// every seed is refused fails the run.
#[derive(Clone, Copy)]
enum Kind {
    Archetype(V6Archetype),
    Variant(V6Variant),
}

fn generate_v6(
    stage: V6Stage,
    kind: Kind,
    slot: usize,
    options: &GenOptions,
    output_root: &Path,
) -> Result<Value> {
    let t0 = Instant::now();
    let archetype = match kind {
        Kind::Variant(variant) => variant.archetype(),
        Kind::Archetype(archetype) => archetype,
    };
    let mut refusals = Vec::new();
    for attempt in 0..v6::V6_MAX_ATTEMPTS {
        let (gen_seed, world) = match kind {
            Kind::Variant(variant) => {
                let gen_seed = v6::variant_slot_seed(variant, stage, slot, attempt);
                let mut world = v6::sample_variant_world(gen_seed, stage, variant);
                v6::assign_variant_slot(&mut world, slot);
                (gen_seed, world)
            }
            Kind::Archetype(archetype) => {
                let gen_seed = v6::slot_seed(stage, slot, attempt);
                (gen_seed, v6::sample_world(gen_seed, stage, archetype))
            }
        };
        let mut world_options = options.clone();
        v6::apply_world(&world, &mut world_options);
        match v6::generate_world(gen_seed, &world_options, slot) {
            Ok(db) => {
                write_generated_db(output_root, &db)?;
                let mut record = json!({
                    "seed": gen_seed,
                    "v6_slot": slot,
                    "v6_stage": stage.as_str(),
                    "v6_archetype": archetype.as_str(),
                    "v6_attempt": attempt,
                    "v6_refusals": refusals,
                    "v6_world": world,
                    "db_name": db.name,
                    "status": "generated",
                    "output_dir": output_root.join(&db.name).display().to_string(),
                    "table_count": db.tables.len(),
                    "program_label_count": db.program_metadata.len(),
                    "elapsed_sec": t0.elapsed().as_secs_f64(),
                });
                if let Kind::Variant(variant) = kind {
                    record["v6_variant"] = json!(variant.as_str());
                }
                attach_corpus_scale_receipt(&mut record, &db, &world_options)?;
                std::fs::write(
                    output_root.join(&db.name).join("v6_world.json"),
                    serde_json::to_string_pretty(&record)? + "\n",
                )?;
                return Ok(record);
            }
            Err(error) => refusals.push(format!(
                "{gen_seed}: {}",
                format!("{error:#}").chars().take(200).collect::<String>()
            )),
        }
    }
    bail!(
        "v6 slot {slot}: {} generation seeds refused: {refusals:?}",
        v6::V6_MAX_ATTEMPTS
    );
}

fn attach_corpus_scale_receipt(
    record: &mut Value,
    db: &GeneratedDb,
    options: &GenOptions,
) -> Result<()> {
    if !options.corpus_scale_enabled() {
        return Ok(());
    }
    let report = corpus_scale_coverage(db, None);
    print_corpus_scale_coverage(&db.name, &report);
    record
        .as_object_mut()
        .context("generation record is not an object")?
        .insert("corpus_scale".to_string(), serde_json::to_value(report)?);
    Ok(())
}

fn ordered(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn expanduser(path: PathBuf) -> PathBuf {
    let home = std::env::var_os("HOME");
    match (path.to_str(), home) {
        (Some("~"), Some(home)) => PathBuf::from(home),
        (Some(p), Some(home)) if p.starts_with("~/") => PathBuf::from(home).join(&p[2..]),
        _ => path,
    }
}
