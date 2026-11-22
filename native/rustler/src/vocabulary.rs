// Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
//! Class vocabularies of declared classification targets, fitted on training data only.
use crate::pre::TaskMetadataMap;
use polars::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

/// A classification target: its entity table and source column, shared by every task split.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct ClassificationTargetKey {
    pub(crate) entity_table: String,
    pub(crate) target_col: String,
}

#[derive(Debug, Clone)]
pub(crate) struct ClassVocabulary {
    value_to_idx: HashMap<String, u32>,
    pub(crate) num_classes: i32,
}

impl ClassVocabulary {
    fn from_labels(labels: impl IntoIterator<Item = String>, context: &str) -> Self {
        let mut labels: Vec<String> = labels.into_iter().collect();
        labels.sort();
        labels.dedup();
        if labels.len() < 2 {
            panic!(
                "categorical target {context} has {} classes in training data; expected at least 2",
                labels.len()
            );
        }
        let num_classes = labels.len() as i32;
        Self {
            value_to_idx: labels
                .into_iter()
                .enumerate()
                .map(|(idx, value)| (value, idx as u32))
                .collect(),
            num_classes,
        }
    }

    fn index_of(&self, label: &str) -> Option<u32> {
        self.value_to_idx.get(label).copied()
    }
}

/// Every declared classification target materialized in a table (its task table or its entity table).
pub(crate) fn classification_target_specs(
    task_metadata: &Option<TaskMetadataMap>,
    db_name: &str,
    table_name: &str,
) -> Vec<ClassificationTargetKey> {
    task_metadata
        .as_ref()
        .and_then(|meta| meta.get(db_name))
        .map(|db_tasks| {
            let normalized_table = table_name.to_lowercase();
            let mut targets: Vec<ClassificationTargetKey> = db_tasks
                .iter()
                .filter_map(|(task_name, task_info)| {
                    let is_classification = matches!(
                        task_info.task_type.as_str(),
                        "binary_classification" | "multiclass_classification"
                    );
                    let materializes_here = task_name.to_lowercase() == normalized_table
                        || task_info.entity_table.to_lowercase() == normalized_table;
                    (is_classification && materializes_here).then(|| ClassificationTargetKey {
                        entity_table: task_info.entity_table.clone(),
                        target_col: task_info.target_col.clone(),
                    })
                })
                .collect();
            targets.sort();
            targets.dedup();
            targets
        })
        .unwrap_or_default()
}

/// The per-row label strings of a classification target; a withheld target stays `None`.
fn class_labels(column: &Column, context: &str) -> Result<Vec<Option<String>>, PolarsError> {
    let strings = column.cast(&DataType::String).map_err(|err| {
        PolarsError::ComputeError(
            format!("discrete target {context} is not renderable as class labels: {err}").into(),
        )
    })?;
    Ok(strings
        .str()?
        .into_iter()
        .map(|value| value.map(str::to_owned))
        .collect())
}

/// Apply a vocabulary fitted on training data; withheld or unseen labels stay null.
pub(crate) fn cast_col_with_vocabulary(
    mut df: DataFrame,
    col_name: &str,
    vocabulary: &ClassVocabulary,
    context: &str,
) -> Result<DataFrame, PolarsError> {
    let labels = class_labels(df.column(col_name)?, context)?;
    // `Vec<Option<u32>>`, not `Vec<u32>`: a withheld or unseen target stays null so the
    // node encoder drops the cell rather than encoding a fabricated class.
    let class_indices: Vec<Option<u32>> = labels
        .iter()
        .map(|label| label.as_deref().and_then(|value| vocabulary.index_of(value)))
        .collect();
    df.with_column(Column::new(col_name.into(), class_indices))?;
    Ok(df)
}

#[cfg(test)]
mod class_vocabulary_tests {
    use super::{ClassVocabulary, cast_col_with_vocabulary, class_labels};
    use polars::prelude::*;

    #[test]
    fn a_withheld_boolean_target_is_null_rather_than_a_third_class() {
        let df = df![
            "target" => &[Some(true), Some(false), None, Some(true)]
        ]
        .unwrap();

        let labels = class_labels(df.column("target").unwrap(), "entity/target").unwrap();
        assert_eq!(labels.len(), 4);
        assert_eq!(labels.iter().filter(|label| label.is_none()).count(), 1);

        let vocabulary =
            ClassVocabulary::from_labels(labels.iter().flatten().cloned(), "entity/target");
        // Two classes, not three: the withheld row occupies none.
        assert_eq!(vocabulary.num_classes, 2);

        let cast = cast_col_with_vocabulary(df, "target", &vocabulary, "entity/target").unwrap();
        let column = cast.column("target").unwrap();
        assert_eq!(column.null_count(), 1);
        let values: Vec<Option<u32>> = column.u32().unwrap().into_iter().collect();
        // "false" < "true", so the frozen ids are false=0 and true=1, and the
        // withheld row carries no id at all.
        assert_eq!(values, vec![Some(1), Some(0), None, Some(1)]);
    }

    #[test]
    fn frozen_vocabulary_keeps_sparse_validation_ids_and_maps_unseen_to_null() {
        let train = ["A", "B", "C"];
        let vocabulary = ClassVocabulary::from_labels(
            train.into_iter().map(str::to_owned),
            "entity/target",
        );

        assert_eq!(vocabulary.num_classes, 3);
        assert_eq!(vocabulary.index_of("A"), Some(0));
        assert_eq!(vocabulary.index_of("C"), Some(2));
        assert_eq!(vocabulary.index_of("D"), None);

        let df = df!["target" => &[Some("A"), Some("D"), None]].unwrap();
        let cast = cast_col_with_vocabulary(df, "target", &vocabulary, "entity/target").unwrap();
        let values: Vec<Option<u32>> = cast.column("target").unwrap().u32().unwrap().into_iter().collect();
        assert_eq!(values, vec![Some(0), None, None]);
    }

    #[test]
    fn vocabulary_preserves_more_than_ten_classes_for_number_fallback() {
        let vocabulary = ClassVocabulary::from_labels(
            (0..11).map(|index| format!("class_{index:02}")),
            "entity/target",
        );
        assert_eq!(vocabulary.num_classes, 11);
        assert_eq!(vocabulary.index_of("class_10"), Some(10));
    }
}

/// One vocabulary per declared target fitted from its training split (never reading val or test),
/// or from the entity table for synthetic worlds without task splits.
pub(crate) fn build_class_vocabularies(
    dataset_path: &str,
    task_metadata: &Option<TaskMetadataMap>,
    db_name: &str,
) -> HashMap<ClassificationTargetKey, ClassVocabulary> {
    let Some(db_tasks) = task_metadata
        .as_ref()
        .and_then(|metadata| metadata.get(db_name))
    else {
        return HashMap::new();
    };
    let mut labels_by_target: HashMap<ClassificationTargetKey, HashSet<String>> = HashMap::new();
    for (task_name, task_info) in db_tasks {
        if !matches!(
            task_info.task_type.as_str(),
            "binary_classification" | "multiclass_classification"
        ) {
            continue;
        }
        let key = ClassificationTargetKey {
            entity_table: task_info.entity_table.clone(),
            target_col: task_info.target_col.clone(),
        };
        let labels = labels_by_target.entry(key.clone()).or_default();
        let task_dir = Path::new(dataset_path).join("tasks").join(task_name);
        let train_path = task_dir.join("train.parquet");
        let has_other_split = ["val", "test"]
            .into_iter()
            .any(|split| task_dir.join(format!("{split}.parquet")).exists());
        let (source_name, path): (&str, PathBuf) = if train_path.exists() {
            ("train", train_path)
        } else if !has_other_split {
            // Manifest-generated synthetic DBs expose targets directly on the entity table and
            // deliberately have no RelBench task splits.
            let path = Path::new(dataset_path)
                .join("db")
                .join(format!("{}.parquet", key.entity_table));
            if !path.exists() {
                panic!(
                    "classification task {task_name} has neither train split nor entity parquet {}; cannot freeze vocabulary",
                    path.display()
                );
            }
            ("entity", path)
        } else {
            panic!(
                "classification task {task_name} is missing train.parquet while other splits exist; cannot freeze vocabulary for {}/{}",
                key.entity_table, key.target_col
            );
        };
        let mut file = fs::File::open(&path).unwrap_or_else(|err| {
            panic!(
                "failed to read classification split {}: {err}",
                path.display()
            )
        });
        let df = ParquetReader::new(&mut file)
            .finish()
            .unwrap_or_else(|err| {
                panic!(
                    "failed to decode classification split {}: {err}",
                    path.display()
                )
            });
        let context = format!("{task_name}/{source_name}/{}", key.target_col);
        let column = df.column(&key.target_col).unwrap_or_else(|_| {
            panic!(
                "declared classification target {} is absent from {}",
                key.target_col,
                path.display()
            )
        });
        // `flatten()` drops the withheld rows: the frozen vocabulary is fitted over
        // observed training classes only, never touching val or test splits.
        labels.extend(
            class_labels(column, &context)
                .unwrap_or_else(|err| panic!("{err}"))
                .into_iter()
                .flatten(),
        );
    }
    labels_by_target
        .into_iter()
        .map(|(key, labels)| {
            let context = format!("{}/{}", key.entity_table, key.target_col);
            let vocabulary = ClassVocabulary::from_labels(labels, &context);
            (key, vocabulary)
        })
        .collect()
}
