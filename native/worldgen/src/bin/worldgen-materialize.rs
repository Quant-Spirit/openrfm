use anyhow::{bail, Context, Result};
use clap::Parser;
use worldgen::{
    add_manifest_target, base_value_digest, expected_target_names, generate_db_skeleton,
    integrity_checks, write_generated_db, GenOptions, GeneratedDb, IntegrityCheck, MixtureManifest,
    MixtureRecord, RecordRefused, SelectionContract,
};
use rayon::prelude::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Families this binary materializes.
const FAMILIES: [&str; 3] = [
    "tabicl_v2_like_discrete",
    "support_required_multiclass",
    "relational_operator_discrete",
];

#[derive(Debug, Parser)]
#[command(about = "Materialize the planned single-table and relational discrete families")]
struct Cli {
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    expected_manifest_sha256: String,
    #[arg(long)]
    output_root: PathBuf,
    #[arg(long, value_parser = ["train"])]
    split: Option<String>,
    #[arg(long, value_parser = FAMILIES)]
    family: Option<String>,
    /// Materialize only the first N selected records (0 = all).
    #[arg(long, default_value_t = 0)]
    limit: usize,
    #[arg(long)]
    workers: Option<usize>,
    /// Record a record that fails a shortcut gate as excluded instead of failing the run.
    #[arg(long)]
    publish_gated_subset: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let raw: Value = serde_json::from_slice(
        &fs::read(&cli.manifest)
            .with_context(|| format!("failed reading {}", cli.manifest.display()))?,
    )?;
    let declared = raw
        .get("manifest_sha256")
        .and_then(Value::as_str)
        .context("manifest is missing manifest_sha256")?
        .to_string();
    let observed = manifest_content_sha256(raw.clone())?;
    if declared != observed || observed != cli.expected_manifest_sha256 {
        bail!(
            "manifest hash mismatch: declared={declared} observed={observed} expected={}",
            cli.expected_manifest_sha256
        );
    }
    let manifest: MixtureManifest = serde_json::from_value(raw)?;
    manifest.validate()?;
    if let Some(workers) = cli.workers {
        if workers == 0 {
            bail!("--workers must be positive");
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build_global()
            .ok();
    }
    let mut selected = manifest
        .records
        .iter()
        .filter(|record| {
            cli.split
                .as_ref()
                .is_none_or(|split| &record.split == split)
        })
        .filter(|record| {
            cli.family
                .as_ref()
                .is_none_or(|family| &record.family == family)
        })
        .collect::<Vec<_>>();
    if cli.limit > 0 {
        selected.truncate(cli.limit);
    }
    if selected.is_empty() {
        bail!("record selection is empty");
    }
    if let Some(record) = selected
        .iter()
        .find(|r| !FAMILIES.contains(&r.family.as_str()))
    {
        bail!(
            "{}: family {} is not supported; select --family",
            record.db_name,
            record.family
        );
    }
    let output_root = &cli.output_root;
    fs::create_dir_all(output_root)?;
    let outcomes = selected
        .par_iter()
        .map(|record| {
            let built = materialize(record, output_root, &observed);
            classify_outcome(record, built, cli.publish_gated_subset)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut records = Vec::new();
    let mut excluded = Vec::new();
    for outcome in outcomes {
        match outcome {
            RecordOutcome::Published(row) => records.push(row),
            RecordOutcome::Excluded(row) => excluded.push(row),
        }
    }
    excluded.sort_by_key(|row| {
        row.get("db_name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    });
    let audit = json!({
        "schema_version": 3,
        "manifest_sha256": observed,
        "intent_manifest_schema_version": manifest.schema_version,
        "selected_split": cli.split,
        "selected_count": records.len(),
        "planned_count": selected.len(),
        "published_count": records.len(),
        "excluded_count": excluded.len(),
        "publish_gated_subset": cli.publish_gated_subset,
        "excluded": excluded,
        "hash_semantics": {
            "artifact_container_sha256": "sha256 of each written file's bytes; parquet bytes are \
                not reproducible across runs, so use it only to detect later file changes",
            "content_sha256": "sha256 over the per-column value digests of every column; \
                a deterministic function of the generation seed",
        },
        "records": records,
    });
    write_json_atomic(&output_root.join("materialization_audit.json"), &audit)?;
    eprintln!(
        "materialized {} of {} planned DBs under {}; {} excluded",
        audit["published_count"],
        audit["planned_count"],
        output_root.display(),
        audit["excluded_count"],
    );
    Ok(())
}

/// Builds one record: skeleton, target, checks, then the database and its sidecars.
fn materialize(record: &MixtureRecord, output_root: &Path, manifest_sha256: &str) -> Result<Value> {
    let mut db = generate_db_skeleton(record.generation_seed, &GenOptions::default())?;
    db.name = record.db_name.clone();
    let base_columns: BTreeMap<String, Vec<String>> = db
        .tables
        .iter()
        .map(|table| {
            (
                table.name.clone(),
                table.column_names().map(str::to_string).collect(),
            )
        })
        .collect();
    let base_value_sha256 = base_column_hashes(&db, &base_columns)?;
    add_manifest_target(&mut db, record, SelectionContract::Gated)?;
    if base_column_hashes(&db, &base_columns)? != base_value_sha256 {
        bail!(
            "{} target generation mutated base column values",
            record.db_name
        );
    }
    let expected_names = expected_target_names(record)?;
    let materialized_names = db
        .program_metadata
        .iter()
        .map(|metadata| metadata.target_col.clone())
        .collect::<Vec<_>>();
    if materialized_names != expected_names {
        bail!(
            "{} materialized {materialized_names:?} but the manifest declares {expected_names:?}",
            record.db_name
        );
    }
    let failed = db.failed_shortcut_gates();
    if !failed.is_empty() {
        return Err(RecordRefused::shortcut_gate(format!(
            "{} failed {} shortcut gate(s):\n{}",
            record.db_name,
            failed.len(),
            failed
                .iter()
                .map(|verdict| verdict.failure_message())
                .collect::<Vec<_>>()
                .join("\n"),
        )));
    }
    let checks = integrity_checks(
        &db,
        record,
        &base_columns,
        expected_names.len(),
        manifest_sha256,
    );
    if let Some(check) = checks.iter().find(|check| !check.passed) {
        bail!(
            "{} failed integrity check {}: {}",
            record.db_name,
            check.name,
            check.evidence
        );
    }
    let db_dir = output_root.join(&db.name);
    write_generated_db(output_root, &db)?;
    fs::write(
        db_dir.join("family_manifest.json"),
        serde_json::to_string_pretty(&worldgen::manifest_for_record(&db, record)?)? + "\n",
    )?;
    let target_value_sha256 = published_target_hashes(&db, &expected_names)?;
    write_integrity_sidecar(
        &db_dir,
        &expected_names,
        &base_columns,
        &base_value_sha256,
        &target_value_sha256,
        &checks,
    )?;
    write_gate_certificate(&db_dir, record, &db)?;
    let mismatches = worldgen::verify_base_value_digests(&db_dir, &base_value_sha256)?;
    if !mismatches.is_empty() {
        bail!(
            "{} does not round-trip its immutable base values:\n{}",
            record.db_name,
            mismatches
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }
    let mut row = serde_json::to_value(record)?;
    let fields = row
        .as_object_mut()
        .context("a MixtureRecord must serialize to a JSON object")?;
    fields.insert("selected_generation_seed".to_string(), json!(db.seed));
    fields.insert("opaque_target_names".to_string(), json!(materialized_names));
    fields.insert("v4_geometry_report".to_string(), Value::Null);
    fields.insert("status".to_string(), json!("materialized"));
    fields.insert("content_sha256".to_string(), json!(content_sha256(&db)?));
    fields.insert(
        "artifact_container_sha256".to_string(),
        json!(artifact_hashes(&db_dir)?),
    );
    Ok(row)
}

enum RecordOutcome {
    Published(Value),
    Excluded(Value),
}

fn manifest_content_sha256(mut payload: Value) -> Result<String> {
    payload
        .as_object_mut()
        .context("manifest root must be an object")?
        .remove("manifest_sha256");
    let canonical = serde_json::to_vec(&canonicalize(payload))?;
    Ok(format!("{:x}", Sha256::digest(canonical)))
}

fn canonicalize(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(key, value)| (key, canonicalize(value)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.into_iter().map(canonicalize).collect()),
        value => value,
    }
}

fn write_integrity_sidecar(
    db_dir: &Path,
    targets: &[String],
    base_columns: &BTreeMap<String, Vec<String>>,
    base_value_sha256: &BTreeMap<String, BTreeMap<String, String>>,
    published_target_value_sha256: &BTreeMap<String, BTreeMap<String, String>>,
    checks: &[IntegrityCheck],
) -> Result<()> {
    let rendered: BTreeMap<&str, Value> = checks
        .iter()
        .map(|check| {
            (
                check.name,
                json!({ "passed": check.passed, "evidence": check.evidence }),
            )
        })
        .collect();
    write_json_atomic(
        &db_dir.join("generator_integrity.json"),
        &json!({
            "schema_version": 2,
            "kind": "manifest_db",
            "opaque_target": targets.first(),
            "opaque_targets": targets,
            "immutable_base_columns": base_columns,
            "immutable_base_value_sha256": base_value_sha256,
            "published_target_value_sha256": published_target_value_sha256,
            "checks": rendered,
        }),
    )
}

/// Write the shortcut-gate certificate next to the data.
fn write_gate_certificate(db_dir: &Path, record: &MixtureRecord, db: &GeneratedDb) -> Result<()> {
    write_json_atomic(
        &db_dir.join("shortcut_gates.json"),
        &json!({
            // 3 adds `published_column_index`.
            "schema_version": 3,
            "kind": "shortcut_gate_certificate",
            "family": record.family,
            "db_name": record.db_name,
            "target_k": record.target_k,
            "ceiling": worldgen::SHORTCUT_CEILING,
            "gated": !db.shortcut_gates.is_empty(),
            "gate_sidecars": gate_sidecar_scope(),
            "determinism": gate_determinism(db),
            "published_column_index": worldgen::published_column_index(db),
            "verdicts": db.shortcut_gates,
        }),
    )
}

/// How far the verdicts here sit from their own thresholds, against a bound on the only
/// nondeterminism in the measurement.
fn gate_determinism(db: &GeneratedDb) -> Value {
    let mut min_margin = f64::INFINITY;
    let mut jitter_bound = 0.0f64;
    for verdict in &db.shortcut_gates {
        let threshold = verdict.threshold_ratio * verdict.budget_nats;
        let margin = if verdict.gate.is_shortcut() {
            threshold - verdict.measured_nats
        } else {
            verdict.measured_nats - threshold
        };
        min_margin = min_margin.min(margin);
        // Four ULP at the larger of the two magnitudes in play. Generous: the
        // observed spread is one to two ULP of `measured_nats` alone.
        let scale = verdict.measured_nats.abs().max(verdict.budget_nats.abs());
        jitter_bound = jitter_bound.max(4.0 * f64::EPSILON * scale);
    }
    if db.shortcut_gates.is_empty() {
        return json!({
            "verdicts": 0,
            "note": "no verdict to characterize; this database carries no gate measurement",
        });
    }
    json!({
        "verdicts": db.shortcut_gates.len(),
        "min_threshold_margin_nats": min_margin,
        "summation_jitter_bound_nats": jitter_bound,
        "margin_in_jitter_bounds": if jitter_bound > 0.0 { min_margin / jitter_bound } else { f64::INFINITY },
        "note": "measured_nats is a rayon-reduced plug-in estimate, so the same seed \
                 reproduces it only to within one or two ULP. min_threshold_margin_nats is \
                 the closest any verdict here sits to its own threshold; compare it with \
                 summation_jitter_bound_nats (four ULP at the magnitude in play) before \
                 reading a re-measurement difference as a changed verdict.",
    })
}

fn base_column_hashes(
    db: &GeneratedDb,
    base_columns: &BTreeMap<String, Vec<String>>,
) -> Result<BTreeMap<String, BTreeMap<String, String>>> {
    base_columns
        .iter()
        .map(|(table_name, column_names)| -> Result<_> {
            let table = db.table(table_name).with_context(|| {
                format!("base table {table_name:?} disappeared during target generation")
            })?;
            let columns = column_names
                .iter()
                .map(|name| -> Result<_> {
                    let values = table.column(name).with_context(|| {
                        format!(
                            "base column {table_name}.{name} disappeared during target generation"
                        )
                    })?;
                    // The digest itself lives in `worldgen::parquet` so the
                    // writer here and the verifier that reads the artifact back
                    // cannot drift into hashing different things.
                    Ok((name.clone(), base_value_digest(values)))
                })
                .collect::<Result<_>>()?;
            Ok((table_name.clone(), columns))
        })
        .collect()
}

/// The digest of every PUBLISHED TARGET column, keyed table -> column.
fn published_target_hashes(
    db: &GeneratedDb,
    targets: &[String],
) -> Result<BTreeMap<String, BTreeMap<String, String>>> {
    let mut out: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    for target in targets {
        let mut found = false;
        for table in &db.tables {
            if let Some(values) = table.column(target) {
                out.entry(table.name.clone())
                    .or_default()
                    .insert(target.clone(), base_value_digest(values));
                found = true;
            }
        }
        if !found {
            bail!("published target {target:?} is not a column of any table");
        }
    }
    Ok(out)
}

/// A reproducible content identity for one materialized record.
fn content_sha256(db: &GeneratedDb) -> Result<String> {
    let mut digests: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    for table in &db.tables {
        let names: Vec<String> = table.column_names().map(str::to_string).collect();
        let mut columns: BTreeMap<String, String> = BTreeMap::new();
        for name in names {
            let values = table.column(&name).with_context(|| {
                format!(
                    "column {}.{name} vanished while hashing content",
                    table.name
                )
            })?;
            columns.insert(name, base_value_digest(values));
        }
        digests.insert(table.name.clone(), columns);
    }
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&digests)?)
    ))
}

fn artifact_hashes(db_dir: &Path) -> Result<BTreeMap<String, String>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(db_dir)? {
        let path = entry?.path();
        if path.is_file() {
            paths.push(path);
        } else if path.file_name().and_then(|name| name.to_str()) == Some("db") {
            paths.extend(
                fs::read_dir(path)?
                    .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                    .filter(|path| path.is_file()),
            );
        }
    }
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            let relative = path.strip_prefix(db_dir)?.to_string_lossy().to_string();
            let digest = Sha256::digest(fs::read(&path)?);
            Ok((relative, format!("{digest:x}")))
        })
        .collect()
}

fn write_json_atomic(path: &Path, payload: &Value) -> Result<()> {
    let temporary = path.with_file_name(format!(
        ".{}.tmp.{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("json"),
        std::process::id()
    ));
    fs::write(&temporary, serde_json::to_vec_pretty(payload)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

/// Split one record's result into "published" or "refused", or propagate it.
fn classify_outcome(
    record: &MixtureRecord,
    built: Result<Value>,
    publish_gated_subset: bool,
) -> Result<RecordOutcome> {
    match built {
        Ok(row) => Ok(RecordOutcome::Published(row)),
        Err(error) if publish_gated_subset => match error.downcast::<RecordRefused>() {
            Ok(refusal) => Ok(RecordOutcome::Excluded(json!({
                "db_name": record.db_name,
                "family": record.family,
                "split": record.split,
                "generation_seed": record.generation_seed,
                "reason": refusal.reason,
                "detail": refusal.detail,
                "evidence": refusal.evidence,
            }))),
            Err(other) => Err(other),
        },
        Err(error) => Err(error),
    }
}

/// Which gate sidecars this database is answerable for.
fn gate_sidecar_scope() -> BTreeMap<&'static str, String> {
    let mut scope = BTreeMap::new();
    scope.insert(
        "shortcut_gates.json",
        "required: written by this binary".to_string(),
    );
    scope.insert(
        "mean_shaped_filter.json",
        "not_applicable: the mean-shaped filter fits child-column means against a regression \
         target; these families publish permuted class ids, for which it is undefined. The \
         shortcut-gate set in this file replaces it."
            .to_string(),
    );
    scope
}
