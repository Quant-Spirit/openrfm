// Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
use crate::archive_v1::TableType;
use crate::nodes::build_graph;
use crate::outputs::{write_col_stats, write_column_index, write_table_info, write_v1};
use crate::stats::{fit_column_stats, training_cutoff};
use crate::tables::read_tables;
use crate::text_output::{encode_and_write_text_embeddings, encoder_file_tag, write_standard_text_outputs};
use crate::time_cutoffs;
use crate::vocabulary::build_class_vocabularies;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use polars::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::hash::{BuildHasherDefault, DefaultHasher};
use std::path::Path;
use std::time::Instant;

pub(crate) const PBAR_TEMPLATE: &str = "{percent}% {bar} {decimal_bytes}/{decimal_total_bytes} [{elapsed_precise}<{eta_precise}, {decimal_bytes_per_sec}]";

#[derive(Debug, Clone)]
pub(crate) struct ColStat {
    pub(crate) mean: f64,
    pub(crate) std: f64,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Table {
    pub table_name: String,
    pub df: DataFrame,
    pub col_stats: Vec<ColStat>,
    pub pcol_name: Option<String>,
    pub fcol_name_to_ptable_name: HashMap<String, String>,
    pub tcol_name: Option<String>,
    pub node_idx_offset: i32,
    pub discrete_target_num_classes: HashMap<String, i32>,
}

pub(crate) type TableMap = HashMap<(String, TableType), Table, BuildHasherDefault<DefaultHasher>>;

#[derive(Parser)]
pub struct Cli {
    /// Archive format.
    #[arg(long, default_value = "v2", value_parser = ["v1", "v2"])]
    pub archive_format: String,
    #[arg(default_value = "rel-f1")]
    pub db_name: String,
    #[arg(long, default_value = "all-MiniLM-L12-v2")]
    pub text_encoder: String,
    #[arg(long, default_value_t = 8192)]
    pub text_encoder_batch: usize,
    #[arg(long, default_value_t = 256)]
    pub text_encoder_max_length: usize,
    #[arg(long, default_value = "auto")]
    pub text_encoder_device: String,
    /// Explicit opt-in to CPU text encoding.
    #[arg(long, default_value_t = false)]
    pub allow_cpu: bool,
    /// Fail on a shared cache miss instead of starting a per-database model.
    #[arg(long, default_value_t = false)]
    pub text_embedding_cache_only: bool,
    /// JSON map from database name to its validation start date.
    #[arg(long)]
    pub time_cutoffs_file: Option<String>,
    /// Keep cell datetimes in fp32 in a v2 archive (the backbone training corpus); bf16 otherwise.
    #[arg(long, default_value_t = false)]
    pub fp32_datetimes: bool,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct TaskMetadata {
    pub task_type: String,
    pub entity_table: String,
    pub target_col: String,
}

pub(crate) type TaskMetadataMap = HashMap<String, HashMap<String, TaskMetadata>>;

fn parse_task_metadata_value(val: serde_json::Value) -> Option<TaskMetadataMap> {
    let serde_json::Value::Object(databases) = val else {
        return None;
    };
    let mut result: TaskMetadataMap = HashMap::new();
    for (db_name, tasks_val) in databases {
        if db_name == "schema_version" || db_name == "tasks" {
            continue;
        }
        let serde_json::Value::Object(tasks) = tasks_val else {
            continue;
        };
        let mut db_tasks: HashMap<String, TaskMetadata> = HashMap::new();
        for (task_name, task_info) in tasks {
            let serde_json::Value::Object(info) = task_info else {
                continue;
            };
            let task_type = info
                .get("task_type")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let entity_table = info
                .get("entity_table")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let target_col = info
                .get("target_col")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            db_tasks.insert(
                task_name,
                TaskMetadata {
                    task_type,
                    entity_table,
                    target_col,
                },
            );
        }
        result.insert(db_name, db_tasks);
    }
    (!result.is_empty()).then_some(result)
}

/// The task declarations in `<dataset>/task_metadata.json`, if present.
fn load_task_metadata(dataset_path: &str) -> Option<TaskMetadataMap> {
    let path = Path::new(dataset_path).join("task_metadata.json");
    if !path.exists() {
        return None;
    }
    let parsed = fs::read_to_string(&path)
        .map_err(|err| err.to_string())
        .and_then(|contents| serde_json::from_str(&contents).map_err(|err| err.to_string()));
    match parsed.map(parse_task_metadata_value) {
        Ok(Some(result)) => {
            eprintln!("Loaded task metadata from {}", path.display());
            Some(result)
        }
        Ok(None) => {
            eprintln!("Ignoring unsupported task metadata shape at {}", path.display());
            None
        }
        Err(err) => {
            eprintln!("Failed to read task metadata from {}: {}", path.display(), err);
            None
        }
    }
}

/// Copies the database's top-level JSON files (task declarations, generator sidecars) next to the archive, where the
/// v2 writer binds them; the converter's own outputs of the same name are kept.
fn copy_sidecars(dataset_path: &str, pre_path: &str) -> std::io::Result<()> {
    for entry in fs::read_dir(dataset_path)? {
        let path = entry?.path();
        let dest = Path::new(pre_path).join(path.file_name().unwrap());
        if path.is_file() && path.extension().is_some_and(|e| e == "json") && !dest.exists() {
            fs::copy(&path, dest)?;
        }
    }
    Ok(())
}

/// Converts `<data_root>/<db>` into the archive `<pre_root>/<db>`.
pub fn run(cli: &Cli, data_root: &str, pre_root: &str) {
    let time_cutoffs_result = time_cutoffs::load_time_cutoffs(&cli.time_cutoffs_file)
        .unwrap_or_else(|e| panic!("failed to load time cutoffs file: {e:#}"));
    let val_time_ns = time_cutoffs_result.cutoffs.get(&cli.db_name).cloned();
    let is_no_temporal_split = time_cutoffs_result.no_temporal_split.contains(&cli.db_name);
    let is_known_database = val_time_ns.is_some() || is_no_temporal_split;
    if let (false, Some(path)) = (is_known_database, &cli.time_cutoffs_file) {
        eprintln!(
            "WARNING: No time cutoff found for database '{}' in cutoffs file '{}'.",
            cli.db_name, path
        );
        eprintln!("This may result in data leakage. All data will be treated as training data.");
    } else if is_no_temporal_split {
        eprintln!(
            "INFO: Database '{}' uses non-temporal split (no time cutoff configured).",
            cli.db_name
        );
        eprintln!(
            "All data will be used for statistics computation (no temporal data leakage concern)."
        );
    }
    let dataset_path = format!("{}/{}", data_root, cli.db_name);

    println!("reading tables...");
    let tic = Instant::now();
    let task_metadata = load_task_metadata(&dataset_path);
    let class_vocabularies = build_class_vocabularies(&dataset_path, &task_metadata, &cli.db_name);
    let (mut table_map, num_rows, num_cells) =
        read_tables(&cli.db_name, &dataset_path, &task_metadata, &class_vocabularies);
    println!("done in {:?}.", tic.elapsed());

    let val_time_ns = training_cutoff(&table_map, val_time_ns);
    println!("computing column stats (Pass 1: Fit on Train Split)...");
    let tic = Instant::now();
    fit_column_stats(&mut table_map, val_time_ns);
    println!("done in {:?}.", tic.elapsed());

    println!("making node vector...");
    let tic = Instant::now();
    let pbar = ProgressBar::new(num_cells as u64)
        .with_style(ProgressStyle::default_bar().template(PBAR_TEMPLATE).unwrap());
    let graph = build_graph(&table_map, num_rows, &pbar);
    pbar.finish();
    println!("done in {:?}.", tic.elapsed());

    let pre_path = format!("{}/{}", pre_root, cli.db_name);
    fs::create_dir_all(Path::new(&pre_path)).unwrap();
    println!("writing out text...");
    let tic = Instant::now();
    // Text ids are assigned while nodes are built; encode once the vocabulary is final so
    // embedding row i belongs to text id i.
    write_standard_text_outputs(&pre_path, graph.text_to_idx, |pre_path, text_vec| {
        encode_and_write_text_embeddings(cli, pre_path, text_vec)
    })
    .expect("text encoding failed");
    println!("done in {:?}.", tic.elapsed());
    println!("writing out column index...");
    write_column_index(&pre_path, graph.column_index);
    println!("writing out table info...");
    write_table_info(&pre_path, &table_map);
    println!("writing out column statistics...");
    write_col_stats(&pre_path, &table_map, &cli.db_name, val_time_ns, is_no_temporal_split);

    let tic = Instant::now();
    if cli.archive_format == "v2" {
        copy_sidecars(&dataset_path, &pre_path).expect("copying the database's JSON sidecars failed");
        crate::archive_v2::write(
            Path::new(&pre_path),
            &graph.nodes,
            &graph.p2f,
            &cli.text_encoder,
            &encoder_file_tag(&cli.text_encoder),
            cli.fp32_datetimes,
        )
        .expect("direct v2 archive write failed");
        return;
    }
    write_v1(&pre_path, graph.nodes, &graph.p2f);
    println!("done in {:?}.", tic.elapsed());
}
