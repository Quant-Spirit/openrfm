// Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
//! Reads a database's Parquet tables (`db/` and the task splits) into the converter's table map.
use crate::archive_v1::TableType;
use crate::pre::{Table, TableMap, TaskMetadataMap};
use crate::vocabulary::{
    ClassVocabulary, ClassificationTargetKey, cast_col_with_vocabulary, classification_target_specs,
};
use glob::glob;
use itertools::{self, Itertools};
use parquet::file::metadata::ParquetMetaDataReader;
use polars::prelude::*;
use serde_json::{Value, from_str};
use std::collections::HashMap;
use std::fs;
use std::hash::{BuildHasherDefault, DefaultHasher};
use std::iter;
use std::path::{Path, PathBuf};

/// The tables of `<dataset_path>` with their row offsets, and the total row and cell counts.
pub(crate) fn read_tables(
    db_name: &str,
    dataset_path: &str,
    task_metadata: &Option<TaskMetadataMap>,
    class_vocabularies: &HashMap<ClassificationTargetKey, ClassVocabulary>,
) -> (TableMap, i32, i32) {
    let mut table_map: TableMap = HashMap::with_hasher(BuildHasherDefault::<DefaultHasher>::new());
    let mut num_rows_sum: i32 = 0;
    let mut num_cells_sum = 0;
    let sorted = |pattern: String| glob(&pattern).unwrap().map(|p| p.unwrap()).sorted();
    for (is_db_table, pq_path) in itertools::chain(
        iter::repeat(true).zip(sorted(format!("{}/db/*.parquet", dataset_path))),
        iter::repeat(false).zip(sorted(format!("{}/tasks/*/*.parquet", dataset_path))),
    ) {
        if pq_path.to_str().unwrap().contains("-link-prediction") {
            eprintln!("Skipping link prediction task: {:?}", pq_path);
            continue;
        }
        let mut file = fs::File::open(&pq_path).unwrap();
        let df = ParquetReader::new(&mut file).finish().unwrap();
        let df = normalize_datetime_to_ns(df).unwrap();
        let (pcol_name, fcol_name_to_ptable_name, tcol_name) = key_metadata(&pq_path, db_name);
        let (table_name, table_type) = table_identity(&pq_path, is_db_table);
        let (df, discrete_target_num_classes) =
            apply_class_vocabularies(df, db_name, &table_name, task_metadata, class_vocabularies);
        let df = relbench_fixes(df, db_name, &table_name);
        let num_rows = df.height() as i32;
        let num_cells = num_rows * df.width() as i32;
        table_map.insert(
            (table_name.clone(), table_type),
            Table {
                table_name,
                df,
                col_stats: Vec::new(),
                pcol_name,
                fcol_name_to_ptable_name,
                tcol_name,
                node_idx_offset: num_rows_sum,
                discrete_target_num_classes,
            },
        );
        num_rows_sum += num_rows;
        num_cells_sum += num_cells;
    }
    (table_map, num_rows_sum, num_cells_sum)
}

fn normalize_datetime_to_ns(mut df: DataFrame) -> Result<DataFrame, PolarsError> {
    let datetime_cols: Vec<String> = df
        .get_columns()
        .iter()
        .filter_map(|col| match col.dtype() {
            DataType::Datetime(unit, _) if *unit != TimeUnit::Nanoseconds => Some(col.name().to_string()),
            _ => None,
        })
        .collect();
    for col_name in datetime_cols {
        df = df
            .lazy()
            .with_column(
                col(&col_name)
                    .cast(DataType::Datetime(TimeUnit::Nanoseconds, None))
                    .alias(&col_name),
            )
            .collect()?;
    }
    Ok(df)
}

fn optional_name(value: &str) -> Option<String> {
    match from_str(value).unwrap() {
        Value::Null => None,
        Value::String(s) => Some(s),
        _ => panic!(),
    }
}

/// The `pkey_col`, `fkey_col_to_pkey_table` and `time_col` key-value metadata of a Parquet file.
fn key_metadata(
    pq_path: &Path,
    db_name: &str,
) -> (Option<String>, HashMap<String, String>, Option<String>) {
    let mut reader = ParquetMetaDataReader::new();
    reader.try_parse(&fs::File::open(pq_path).unwrap()).unwrap();
    let metadata = reader.finish().unwrap();
    let metadata = metadata
        .file_metadata()
        .key_value_metadata()
        .unwrap()
        .iter()
        .map(|kv| (kv.key.clone(), kv.value.as_ref().unwrap().clone()))
        .collect::<HashMap<_, _>>();
    let fcol_name_to_ptable_name = match from_str(metadata.get("fkey_col_to_pkey_table").unwrap()).unwrap() {
        Value::Object(o) => o
            .into_iter()
            .map(|(k, v)| {
                let Value::String(mut v) = v else { panic!() };
                if db_name == "rel-avito" {
                    if k == "UserID" {
                        v = "UserInfo".to_string();
                    } else if k == "AdID" {
                        v = "AdsInfo".to_string();
                    }
                }
                (k, v)
            })
            .collect::<HashMap<_, _>>(),
        _ => panic!(),
    };
    (
        optional_name(metadata.get("pkey_col").unwrap()),
        fcol_name_to_ptable_name,
        optional_name(metadata.get("time_col").unwrap()),
    )
}

/// `db/<table>.parquet` is a database table; `tasks/<task>/<split>.parquet` a task split.
fn table_identity(pq_path: &PathBuf, is_db_table: bool) -> (String, TableType) {
    let stem = pq_path.file_stem().unwrap().to_str().unwrap().to_string();
    if is_db_table {
        return (stem, TableType::Db);
    }
    let task = pq_path.parent().unwrap().file_name().unwrap().to_str().unwrap().to_string();
    let split = match stem.as_str() {
        "train" => TableType::Train,
        "val" => TableType::Val,
        "test" => TableType::Test,
        _ => panic!(),
    };
    (task, split)
}

/// Declared classification targets of the table, encoded with their frozen vocabularies; the class counts of
/// those stored as discrete targets (at most 10 classes).
fn apply_class_vocabularies(
    mut df: DataFrame,
    db_name: &str,
    table_name: &str,
    task_metadata: &Option<TaskMetadataMap>,
    class_vocabularies: &HashMap<ClassificationTargetKey, ClassVocabulary>,
) -> (DataFrame, HashMap<String, i32>) {
    let mut discrete_target_num_classes = HashMap::new();
    for target in classification_target_specs(task_metadata, db_name, table_name) {
        let target_col = &target.target_col;
        if df.column(target_col).is_err() {
            panic!("declared classification target {target_col} is absent from {db_name}/{table_name}");
        }
        let vocabulary = class_vocabularies.get(&target).unwrap_or_else(|| {
            panic!(
                "missing frozen vocabulary for classification target {}/{}",
                target.entity_table, target.target_col
            )
        });
        let physical_type = if vocabulary.num_classes <= 10 { "Discrete" } else { "Number" };
        eprintln!(
            "Applying frozen {}-class vocabulary to categorical target '{}' for {}/{} as {}",
            vocabulary.num_classes, target_col, db_name, table_name, physical_type
        );
        let context = format!("{db_name}/{table_name}/{target_col}");
        df = cast_col_with_vocabulary(df, target_col, vocabulary, &context).unwrap();
        if vocabulary.num_classes <= 10 {
            discrete_target_num_classes.insert(target_col.clone(), vocabulary.num_classes);
        }
    }
    (df, discrete_target_num_classes)
}

/// The first value of the column becomes `true`, every other value `false`.
fn make_column_boolean(df: &mut DataFrame, col_name: &str) -> PolarsResult<()> {
    let s_str = df.column(col_name)?.cast(&DataType::String)?;
    let ca = s_str.str()?;
    let first = ca
        .get(0)
        .ok_or_else(|| PolarsError::NoData("column is empty".into()))?;
    let mut mask_series = ca.equal(first).into_series();
    mask_series.rename(col_name.into());
    df.replace(col_name, mask_series)?;
    Ok(())
}

/// RelBench fixes: binarised autocomplete columns, the first product category, dropped nulls and index columns.
fn relbench_fixes(mut df: DataFrame, db_name: &str, table_name: &str) -> DataFrame {
    if db_name == "rel-stack" && table_name == "postLinks" {
        make_column_boolean(&mut df, "LinkTypeId").unwrap();
    }
    if db_name == "rel-trial" && table_name == "studies" {
        make_column_boolean(&mut df, "has_dmc").unwrap();
    }
    if db_name == "rel-trial" && table_name == "eligibilities" {
        for col in ["adult", "child", "older_adult"] {
            make_column_boolean(&mut df, col).unwrap();
        }
    }
    if (db_name == "rel-amazon" || db_name == "rel-amazon-dense") && table_name == "product" {
        df.apply("category", |c| {
            c.list()
                .unwrap()
                .into_iter()
                .map(|l| l.map(|l| l.iter().next().unwrap().to_string()))
                .collect::<StringChunked>()
                .into_column()
        })
        .unwrap();
    }
    if db_name == "rel-stack"
        && table_name == "posts"
        && df.get_column_names().iter().any(|c| c.as_str() == "AcceptedAnswerId")
    {
        df = df.drop("AcceptedAnswerId").unwrap();
    }
    if db_name == "rel-event" {
        if table_name == "event_attendees" {
            df = df.drop_nulls::<String>(None).unwrap();
        }
        if table_name == "user_friends" {
            df = df.drop_nulls::<String>(None).unwrap();
            df = df.with_row_index("dummy".into(), None).unwrap();
        }
        if ["user-repeat", "user-ignore", "user-attendance"].contains(&table_name) {
            df = df.drop("index").unwrap();
        }
    }
    df
}
