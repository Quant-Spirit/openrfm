use crate::labels::LabelMetadata;
use crate::scm::{ColumnData, GeneratedDb, TableData};
use anyhow::{anyhow, bail, Context, Result};
use arrow_array::{
    ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray,
    TimestampNanosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;

pub fn write_generated_db(root: &Path, db: &GeneratedDb) -> Result<()> {
    let db_dir = root.join(&db.name);
    let parquet_dir = db_dir.join("db");
    fs::create_dir_all(&parquet_dir)
        .with_context(|| format!("failed to create {}", parquet_dir.display()))?;
    if crate::sj::statistics(db).is_some() {
        for entry in fs::read_dir(&parquet_dir)? {
            let path = entry?.path();
            if path.extension().is_some_and(|ext| ext == "parquet") {
                anyhow::ensure!(
                    db.tables
                        .iter()
                        .any(|t| path.file_stem().is_some_and(|stem| stem == t.name.as_str())),
                    "stale S-J table {}; use a fresh output directory",
                    path.display()
                );
            }
        }
    }
    for table in &db.tables {
        write_table_parquet(&parquet_dir.join(format!("{}.parquet", table.name)), table)?;
    }
    write_sidecars(&db_dir, db)?;
    Ok(())
}

pub fn write_table_parquet(path: &Path, table: &TableData) -> Result<()> {
    let mut fields = Vec::with_capacity(table.columns.len());
    let mut arrays = Vec::with_capacity(table.columns.len());
    for (name, data) in &table.columns {
        let (field, array) = arrow_column(name, data);
        fields.push(field);
        arrays.push(array);
    }
    let mut metadata = HashMap::new();
    metadata.insert(
        "pkey_col".to_string(),
        serde_json::to_string(&Some(table.pkey_col.clone()))?,
    );
    metadata.insert(
        "fkey_col_to_pkey_table".to_string(),
        serde_json::to_string(&table.fkey_col_to_pkey_table)?,
    );
    metadata.insert(
        "time_col".to_string(),
        serde_json::to_string(&table.time_col)?,
    );
    metadata.insert(
        "hsbm_hierarchy".to_string(),
        serde_json::to_string(&table.hsbm_hierarchy)?,
    );
    let kv_metadata = metadata
        .iter()
        .map(|(key, value)| KeyValue {
            key: key.clone(),
            value: Some(value.clone()),
        })
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new_with_metadata(fields, metadata));
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let props = WriterProperties::builder()
        .set_key_value_metadata(Some(kv_metadata))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn arrow_column(name: &str, data: &ColumnData) -> (Field, ArrayRef) {
    match data {
        ColumnData::I64(values) => (
            Field::new(name, DataType::Int64, false),
            Arc::new(Int64Array::from(values.clone())) as ArrayRef,
        ),
        ColumnData::F64(values) => (
            Field::new(name, DataType::Float64, true),
            Arc::new(Float64Array::from(values.clone())) as ArrayRef,
        ),
        ColumnData::Bool(values) => (
            Field::new(name, DataType::Boolean, true),
            Arc::new(BooleanArray::from(values.clone())) as ArrayRef,
        ),
        ColumnData::Utf8(values) => (
            Field::new(name, DataType::Utf8, true),
            Arc::new(StringArray::from(values.clone())) as ArrayRef,
        ),
        ColumnData::TimestampNs(values) => (
            Field::new(name, DataType::Timestamp(TimeUnit::Nanosecond, None), false),
            Arc::new(TimestampNanosecondArray::from(values.clone())) as ArrayRef,
        ),
    }
}

pub fn write_sidecars(db_dir: &Path, db: &GeneratedDb) -> Result<()> {
    // Validate and render before touching any sidecar so malformed metadata cannot replace
    // program_info.json while retaining stale task metadata.
    let task_metadata = (!db.program_metadata.is_empty() || !db.exact_aggregates.is_empty())
        .then(|| task_metadata_json(db))
        .transpose()?;

    fs::create_dir_all(db_dir)?;
    write_json_atomic(
        &db_dir.join("family_manifest.json"),
        &crate::taxonomy::manifest(db)?,
    )?;
    if let Some(stats) = crate::sj::statistics(db) {
        write_json_atomic(
            &db_dir.join("sj_metadata.json"),
            &serde_json::to_value(stats)?,
        )?;
    }
    if let Some(receipt) = &db.condrel_receipt {
        write_json_atomic(
            &db_dir.join(crate::sj_families::CONDREL_RECEIPT_FILE),
            &serde_json::to_value(receipt)?,
        )?;
    } else {
        let _ = fs::remove_file(db_dir.join(crate::sj_families::CONDREL_RECEIPT_FILE));
    }
    if let Some(manifest) = &db.xrep_manifest {
        write_json_atomic(
            &db_dir.join(crate::sj_families::XREP_MANIFEST_FILE),
            &serde_json::to_value(manifest)?,
        )?;
    } else {
        let _ = fs::remove_file(db_dir.join(crate::sj_families::XREP_MANIFEST_FILE));
    }
    let mut table_info = serde_json::Map::new();
    let mut node_idx_offset = 0usize;
    for table in db.tables.iter().filter(|table| table.num_rows() > 0) {
        table_info.insert(
            format!("{}:Db", table.name),
            json!({
                "num_nodes": table.num_rows(),
                "node_idx_offset": node_idx_offset,
            }),
        );
        node_idx_offset += table.num_rows();
    }

    let mut column_index = serde_json::Map::new();
    let mut idx = 0usize;
    for table in &db.tables {
        for (col_name, _) in &table.columns {
            column_index.insert(format!("{col_name} of {}", table.name), json!(idx));
            idx += 1;
        }
    }

    write_json_atomic(
        &db_dir.join("table_info.json"),
        &serde_json::Value::Object(table_info),
    )?;
    write_json_atomic(
        &db_dir.join("column_index.json"),
        &serde_json::Value::Object(column_index),
    )?;
    if let Some(task_metadata) = task_metadata {
        let program_info = if db
            .program_metadata
            .iter()
            .any(|program| program.family.starts_with("p2_v4_"))
        {
            let mut public_programs = db.program_metadata.clone();
            for program in &mut public_programs {
                if program.family.starts_with("p2_v4_") {
                    program.params.remove("dependency_certificate");
                }
            }
            serde_json::to_value(public_programs)?
        } else {
            serde_json::to_value(&db.program_metadata)?
        };
        write_json_atomic(&db_dir.join("program_info.json"), &program_info)?;
        write_json_atomic(&db_dir.join("task_metadata.json"), &task_metadata)?;
    } else {
        let _ = fs::remove_file(db_dir.join("program_info.json"));
        let _ = fs::remove_file(db_dir.join("task_metadata.json"));
    }
    if db.mean_shaped_filter.is_empty() {
        let _ = fs::remove_file(db_dir.join("mean_shaped_filter.json"));
    } else {
        write_json_atomic(
            &db_dir.join("mean_shaped_filter.json"),
            &serde_json::to_value(&db.mean_shaped_filter)?,
        )?;
    }
    write_gate_certificate(db_dir, db)?;
    write_integrity_anchor(db_dir, db)?;
    Ok(())
}

/// Text a column's index entry carries INSTEAD of a verdict list, when the
/// generator recorded no verdict that is about it.
pub const UNCLAIMED_TARGET_DECLARATION: &str =
    "no gate verdict on this database is about this column: the family that published it \
     either runs no gate or recorded none under a name this column can be resolved from. This \
     is a stated absence, not a missing entry -- an absent entry would be a defect in the index";

/// The `published column name -> verdicts about it` index.
pub fn published_column_index(db: &GeneratedDb) -> serde_json::Value {
    // The published-target set, taken from the generator's own registry rather than from
    // `program_metadata`.
    let mut targets: BTreeMap<String, String> = BTreeMap::new();
    for table in &db.tables {
        for (name, _) in &table.columns {
            if crate::scm::FeatureContract::is_published_target(name) {
                targets.insert(name.clone(), table.name.clone());
            }
        }
    }
    for metadata in &db.program_metadata {
        targets.insert(metadata.target_col.clone(), metadata.target_table.clone());
    }

    let mut columns = serde_json::Map::new();
    let mut resolutions = 0usize;
    let mut unclaimed_columns = 0usize;
    let mut reached: HashSet<usize> = HashSet::new();
    for (column, table) in &targets {
        let mut by_name: Vec<usize> = db
            .shortcut_gates
            .iter()
            .enumerate()
            .filter(|(_, verdict)| verdict.target == *column && verdict.target_table == *table)
            .map(|(index, _)| index)
            .collect();
        let declared: Vec<usize> = db
            .gate_verdict_claims
            .get(column)
            .cloned()
            .unwrap_or_default();
        let mut all = by_name.clone();
        all.extend(declared.iter().copied());
        all.sort_unstable();
        all.dedup();
        by_name.retain(|index| !declared.contains(index));
        resolutions += all.len();
        reached.extend(all.iter().copied());
        let mut entry = serde_json::Map::new();
        entry.insert("target_table".to_string(), json!(table));
        entry.insert("verdicts".to_string(), json!(all));
        entry.insert("resolved_by_name".to_string(), json!(by_name));
        entry.insert("resolved_by_claim".to_string(), json!(declared));
        if all.is_empty() {
            unclaimed_columns += 1;
            entry.insert("unclaimed".to_string(), json!(UNCLAIMED_TARGET_DECLARATION));
        }
        columns.insert(column.clone(), serde_json::Value::Object(entry));
    }
    let orphan: Vec<usize> = (0..db.shortcut_gates.len())
        .filter(|index| !reached.contains(index))
        .collect();
    json!({
        // Present and true on every certificate that carries this block.
        "indexed": true,
        "resolves": "published column name -> positions in this file's `verdicts` array",
        "target_set": "every column matching worldgen::scm::PUBLISHED_TARGET_PREFIXES, \
                       unioned with program_metadata",
        "columns": serde_json::Value::Object(columns),
        "published_columns": targets.len(),
        "columns_with_no_verdict": unclaimed_columns,
        "verdict_resolutions": resolutions,
        "verdicts_no_published_column_resolves_to": orphan,
    })
}

/// The shortcut-gate certificate, for the axis generator.
fn write_gate_certificate(db_dir: &Path, db: &GeneratedDb) -> Result<()> {
    if db.shortcut_gates.is_empty() {
        let _ = fs::remove_file(db_dir.join("shortcut_gates.json"));
        return Ok(());
    }
    const FILTER_NOT_APPLICABLE: &str = "not_applicable: the mean-shaped construction filter \
        runs on the count_ratio and order_stat axes only. This corpus did not run it, and the \
        verdicts in this file are what bound its published columns instead.";
    const FILTER_REQUIRED: &str =
        "required: this axis ran the mean-shaped construction filter, which writes its own \
         audit trail next to the data.";
    let mut scope = serde_json::Map::new();
    scope.insert(
        "shortcut_gates.json".to_string(),
        json!("required: written by worldgen"),
    );
    scope.insert(
        "mean_shaped_filter.json".to_string(),
        json!(if db.mean_shaped_filter.is_empty() {
            FILTER_NOT_APPLICABLE
        } else {
            FILTER_REQUIRED
        }),
    );
    write_json_atomic(
        &db_dir.join("shortcut_gates.json"),
        &json!({
            // 3 adds `published_column_index`.
            "schema_version": 3,
            "kind": "shortcut_gate_certificate",
            "written_by": "worldgen",
            "db_name": db.name,
            "seed": db.seed,
            "ceiling": crate::shortcut_gates::SHORTCUT_CEILING,
            "gated": true,
            "gate_sidecars": serde_json::Value::Object(scope),
            "published_column_index": published_column_index(db),
            "verdicts": db.shortcut_gates,
        }),
    )
}

/// The integrity anchor, for the axis generator.
fn write_integrity_anchor(db_dir: &Path, db: &GeneratedDb) -> Result<()> {
    let targets: HashSet<&str> = db
        .program_metadata
        .iter()
        .map(|meta| meta.target_col.as_str())
        .collect();
    let mut base_columns: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut base_digests: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    let mut target_digests: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    for table in &db.tables {
        for (name, data) in &table.columns {
            if targets.contains(name.as_str()) {
                // Recorded SEPARATELY, never folded into the base map.
                target_digests
                    .entry(table.name.clone())
                    .or_default()
                    .insert(name.clone(), base_value_digest(data));
                continue;
            }
            base_columns
                .entry(table.name.clone())
                .or_default()
                .push(name.clone());
            base_digests
                .entry(table.name.clone())
                .or_default()
                .insert(name.clone(), base_value_digest(data));
        }
    }
    if base_digests.is_empty() {
        let _ = fs::remove_file(db_dir.join("generator_integrity.json"));
        return Ok(());
    }
    let mut opaque_targets: Vec<&str> = targets.into_iter().collect();
    opaque_targets.sort_unstable();
    write_json_atomic(
        &db_dir.join("generator_integrity.json"),
        &json!({
            "schema_version": 2,
            "kind": "axis_db",
            "written_by": "worldgen",
            "opaque_target": opaque_targets.first(),
            "opaque_targets": opaque_targets,
            "immutable_base_columns": base_columns,
            "immutable_base_value_sha256": base_digests,
            "published_target_value_sha256": target_digests,
            "checks": serde_json::Value::Object(serde_json::Map::new()),
        }),
    )
}

/// Render the `database -> task -> task metadata` shape consumed by `rustler/src/pre.rs`.
pub(crate) fn task_metadata_json(db: &GeneratedDb) -> Result<serde_json::Value> {
    let mut tasks = BTreeMap::new();
    let mut task_list = Vec::new();
    let mut task_ids = HashSet::new();

    let declared_targets = db
        .program_metadata
        .iter()
        .map(|metadata| (metadata.target_table.as_str(), metadata.target_col.as_str()))
        .collect::<Vec<_>>();
    let mut metadata_order = (0..db.program_metadata.len()).collect::<Vec<_>>();
    metadata_order.sort_by_key(|&index| {
        let metadata = &db.program_metadata[index];
        format!(
            "{}:{}:{}",
            db.seed, metadata.target_col, metadata.target_table
        )
    });

    for metadata_index in metadata_order {
        let metadata = &db.program_metadata[metadata_index];
        let task_id = metadata.target_col.trim();
        if task_id.is_empty() {
            bail!(
                "program metadata for {} has an empty target_col; cannot create task metadata",
                metadata.target_table
            );
        }
        if !task_ids.insert(task_id.to_owned()) {
            bail!("duplicate generated task identifier {task_id:?}; target columns must be unique");
        }

        let table = db.table(&metadata.target_table).ok_or_else(|| {
            anyhow!(
                "program metadata task {task_id:?} references missing target table {:?}",
                metadata.target_table
            )
        })?;
        if table.column(task_id).is_none() {
            bail!(
                "program metadata task {task_id:?} references missing target column {:?} on table {:?}",
                task_id,
                metadata.target_table
            );
        }

        let rustler_type = rustler_task_type(metadata)?;
        // Evidence regime is public synthetic-program provenance, never a label-derived statistic.
        let evidence_regime = metadata
            .params
            .get("regression_evidence_regime")
            .and_then(serde_json::Value::as_str);
        if rustler_type == "regression"
            && metadata.params.contains_key("regression_geometry_contract")
        {
            let regime = evidence_regime.ok_or_else(|| {
                anyhow!("schema-v5 regression task {task_id:?} lacks regression_evidence_regime")
            })?;
            if !matches!(
                regime,
                "dense_continuous_control"
                    | "rank_like_20_40"
                    | "historical_rolling"
                    | "same_event_proxy"
            ) {
                bail!(
                    "schema-v5 regression task {task_id:?} has unknown evidence regime {regime:?}"
                );
            }
        } else if evidence_regime.is_some() {
            bail!("non-schema-v5 task {task_id:?} must not declare regression evidence regime");
        }
        let mut sibling_targets = declared_targets
            .iter()
            .filter(|(table, column)| {
                *table != metadata.target_table || *column != metadata.target_col
            })
            .map(|(table, column)| json!([table, column]))
            .collect::<Vec<_>>();
        for (table, column) in crate::sj_families::additional_task_drop_columns(
            &db,
            &metadata.target_table,
            &metadata.target_col,
        ) {
            let drop = json!([table, column]);
            if !sibling_targets.contains(&drop) {
                sibling_targets.push(drop);
            }
        }
        let mut task_object = json!({
            "task_type": rustler_type,
            "entity_table": metadata.target_table,
            "target_col": task_id,
            "drop_columns": sibling_targets.clone(),
        });
        if let Some(regime) = evidence_regime {
            task_object["evidence_regime"] = json!(regime);
        }
        let mut task_entry = json!({
            "table": metadata.target_table,
            "target_column": task_id,
            "task_type": rustler_type,
            "sem_type": if rustler_type == "regression" { "number" } else { "discrete" },
            "drop_columns": sibling_targets,
        });
        if let Some(regime) = evidence_regime {
            task_entry["evidence_regime"] = json!(regime);
        }
        if metadata.family == "p2_long_history" {
            // Preserve temporal task semantics through declaration/preprocessing.
            // Both representations are consumed: the task loader requires
            // these explicit forecast keys; the target table stays unchanged.
            for task in [&mut task_entry, &mut task_object] {
                task["task_kind"] = json!("forecast");
                task["task_view"] = json!("forecast");
                task["forecast_entity_table"] = json!("p2_entity");
                task["forecast_cutoff"] = json!("row_timestamp");
                task["label_history_rows"] = json!([1, 1]);
                task["label_horizon_seconds"] = json!(86400);
                task["time_legality"] = json!("strict_prior");
            }
        }
        if let Some(public) = metadata.params.get("public_task") {
            let public = public
                .as_object()
                .context("public_task must be an object")?;
            for (key, value) in public {
                task_entry[key] = value.clone();
                task_object[key] = value.clone();
            }
        }
        tasks.insert(task_id.to_owned(), task_object);
        task_list.push(task_entry);
    }

    let exact_aggregates: Vec<serde_json::Value> = db
        .exact_aggregates
        .iter()
        .map(|column| {
            json!({
                "table": column.table,
                "column": column.column,
                "fk": column.fk,
                "child_column": column.child_column,
                "stat": column.stat,
                "window": column.window,
                "mean": column.mean,
                "std": column.std,
            })
        })
        .collect();

    Ok(json!({
        db.name.clone(): tasks,
        "schema_version": if task_list.iter().any(|task| task.get("evidence_regime").is_some()) { 5 } else { 1 },
        "tasks": task_list,
        "exact_aggregates": exact_aggregates,
    }))
}

fn rustler_task_type(metadata: &LabelMetadata) -> Result<&'static str> {
    match metadata.task_type.trim() {
        "binary_classification" => Ok("binary_classification"),
        "multiclass_classification" => Ok("multiclass_classification"),
        "regression" | "reg" => Ok("regression"),
        "clf" | "classification" => match metadata.n_classes {
            2 => Ok("binary_classification"),
            3.. => Ok("multiclass_classification"),
            classes => bail!(
                "classification task {:?} has invalid n_classes={classes}; expected at least 2",
                metadata.target_col
            ),
        },
        task_type => bail!(
            "task {:?} has unsupported task_type {task_type:?}; expected clf, reg, binary_classification, multiclass_classification, or regression",
            metadata.target_col
        ),
    }
}

fn write_json_atomic(path: &Path, payload: &serde_json::Value) -> Result<()> {
    let parent = path.parent().context("json path has no parent")?;
    fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(
        ".{}.tmp.{}",
        path.file_name().and_then(|s| s.to_str()).unwrap_or("json"),
        std::process::id()
    ));
    {
        let file = File::create(&tmp)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, payload)?;
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Content digest of one column's VALUES, independent of file layout.
pub fn base_value_digest(data: &ColumnData) -> String {
    let mut digest = Sha256::new();
    match data {
        ColumnData::I64(items) | ColumnData::TimestampNs(items) => {
            digest.update([0]);
            for value in items {
                digest.update(value.to_le_bytes());
            }
        }
        ColumnData::F64(items) => {
            digest.update([1]);
            for value in items {
                match value {
                    Some(value) => {
                        digest.update([1]);
                        digest.update(value.to_bits().to_le_bytes());
                    }
                    None => digest.update([0]),
                }
            }
        }
        ColumnData::Bool(items) => {
            digest.update([2]);
            for value in items {
                digest.update([match value {
                    None => 0,
                    Some(false) => 1,
                    Some(true) => 2,
                }]);
            }
        }
        ColumnData::Utf8(items) => {
            digest.update([3]);
            for value in items {
                match value {
                    Some(value) => {
                        digest.update([1]);
                        digest.update((value.len() as u64).to_le_bytes());
                        digest.update(value.as_bytes());
                    }
                    None => digest.update([0]),
                }
            }
        }
    }
    format!("{:x}", digest.finalize())
}

/// Read back the columns of one written table.
pub fn read_table_columns(path: &Path) -> Result<Vec<(String, ColumnData)>> {
    use arrow_array::{Array, BooleanArray, Float64Array, Int64Array, StringArray};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let mut columns: Vec<(String, ColumnData)> = Vec::new();
    for batch in reader {
        let batch = batch?;
        let schema = batch.schema();
        for (index, field) in schema.fields().iter().enumerate() {
            let array = batch.column(index);
            let appended = match field.data_type() {
                DataType::Int64 => {
                    let typed = array
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .ok_or_else(|| anyhow!("{} is not an Int64Array", field.name()))?;
                    ColumnData::I64((0..typed.len()).map(|row| typed.value(row)).collect())
                }
                DataType::Timestamp(TimeUnit::Nanosecond, None) => {
                    let typed = array
                        .as_any()
                        .downcast_ref::<TimestampNanosecondArray>()
                        .ok_or_else(|| anyhow!("{} is not a timestamp array", field.name()))?;
                    ColumnData::TimestampNs((0..typed.len()).map(|row| typed.value(row)).collect())
                }
                DataType::Float64 => {
                    let typed = array
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .ok_or_else(|| anyhow!("{} is not a Float64Array", field.name()))?;
                    ColumnData::F64(
                        (0..typed.len())
                            .map(|row| (!typed.is_null(row)).then(|| typed.value(row)))
                            .collect(),
                    )
                }
                DataType::Boolean => {
                    let typed = array
                        .as_any()
                        .downcast_ref::<BooleanArray>()
                        .ok_or_else(|| anyhow!("{} is not a BooleanArray", field.name()))?;
                    ColumnData::Bool(
                        (0..typed.len())
                            .map(|row| (!typed.is_null(row)).then(|| typed.value(row)))
                            .collect(),
                    )
                }
                DataType::Utf8 => {
                    let typed = array
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| anyhow!("{} is not a StringArray", field.name()))?;
                    ColumnData::Utf8(
                        (0..typed.len())
                            .map(|row| (!typed.is_null(row)).then(|| typed.value(row).to_string()))
                            .collect(),
                    )
                }
                other => bail!(
                    "{}.{} has unsupported type {other}",
                    path.display(),
                    field.name()
                ),
            };
            match columns.iter_mut().find(|(name, _)| name == field.name()) {
                Some((_, existing)) => append_column(existing, appended)?,
                None => columns.push((field.name().clone(), appended)),
            }
        }
    }
    Ok(columns)
}

fn append_column(target: &mut ColumnData, extra: ColumnData) -> Result<()> {
    match (target, extra) {
        (ColumnData::I64(left), ColumnData::I64(right)) => left.extend(right),
        (ColumnData::TimestampNs(left), ColumnData::TimestampNs(right)) => left.extend(right),
        (ColumnData::F64(left), ColumnData::F64(right)) => left.extend(right),
        (ColumnData::Bool(left), ColumnData::Bool(right)) => left.extend(right),
        (ColumnData::Utf8(left), ColumnData::Utf8(right)) => left.extend(right),
        _ => bail!("parquet row groups disagree on column type"),
    }
    Ok(())
}

/// One column that failed verification against the recorded anchor.
#[derive(Debug, Clone)]
pub struct DigestMismatch {
    pub table: String,
    pub column: String,
    pub expected: String,
    pub observed: Option<String>,
}

impl std::fmt::Display for DigestMismatch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.observed {
            Some(observed) => write!(
                formatter,
                "{}.{}: recorded {} but the artifact hashes {}",
                self.table, self.column, self.expected, observed
            ),
            None => write!(
                formatter,
                "{}.{}: recorded {} but the column is absent from the artifact",
                self.table, self.column, self.expected
            ),
        }
    }
}

/// Recompute `immutable_base_value_sha256` from a written database directory and report every
/// disagreement.
pub fn verify_base_value_digests(
    db_dir: &Path,
    expected: &BTreeMap<String, BTreeMap<String, String>>,
) -> Result<Vec<DigestMismatch>> {
    let mut mismatches = Vec::new();
    for (table_name, columns) in expected {
        let path = db_dir.join("db").join(format!("{table_name}.parquet"));
        let observed = read_table_columns(&path).with_context(|| {
            format!(
                "failed reading {} for integrity verification",
                path.display()
            )
        })?;
        for (column, digest) in columns {
            let found = observed
                .iter()
                .find(|(name, _)| name == column)
                .map(|(_, data)| base_value_digest(data));
            if found.as_deref() != Some(digest.as_str()) {
                mismatches.push(DigestMismatch {
                    table: table_name.clone(),
                    column: column.clone(),
                    expected: digest.clone(),
                    observed: found,
                });
            }
        }
    }
    Ok(mismatches)
}
