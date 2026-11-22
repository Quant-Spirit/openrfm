// Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
//! Format-v1 archive records, shared by the `rustler pre` writer and the Python decoder.

use rkyv::{Archive, Deserialize, Serialize};
use serde::{Deserialize as SerdeDeserialize, Serialize as SerdeSerialize};
use std::collections::HashMap;

#[derive(SerdeSerialize, SerdeDeserialize, Default)]
pub struct TableInfo {
    pub node_idx_offset: i32,
    pub num_nodes: i32,
    #[serde(default)]
    pub pkey_col: Option<String>,
    #[serde(default)]
    pub fkey_col_to_pkey_table: HashMap<String, String>,
    #[serde(default)]
    pub column_types: HashMap<String, String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub column_aliases: HashMap<String, Vec<String>>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub column_display_overrides: HashMap<String, String>,
}

#[derive(Archive, Deserialize, Serialize, Hash, Eq, PartialEq, Default, Clone, Debug)]
#[rkyv(derive(PartialEq, Debug))]
pub enum TableType {
    #[default]
    Db,
    Train,
    Val,
    Test,
}

#[derive(Archive, Clone, Debug, Deserialize, Serialize, Default)]
#[rkyv(derive(Debug))]
pub struct Edge {
    pub node_idx: i32,
    pub table_name_idx: i32,
    pub table_type: TableType,
    pub timestamp: Option<i32>,
}

#[derive(Archive, Serialize, Deserialize, Default)]
pub struct Adj {
    pub adj: Vec<Vec<Edge>>,
}

/// CSR (Compressed Sparse Row) format for graph adjacency.
#[derive(Archive, Serialize, Deserialize, Default)]
pub struct CsrAdj {
    pub rowptr: Vec<i64>,
    pub col_idx: Vec<i32>,
    pub table_name_idxs: Vec<i32>,
    pub table_types: Vec<TableType>,
    /// Timestamp of each edge, -1 if none.
    pub timestamps: Vec<i32>,
}

#[derive(Archive, Deserialize, Serialize, Default)]
#[rkyv(derive(Clone, Debug))]
pub enum SemType {
    #[default]
    Number,
    Text,
    DateTime,
    Boolean,
    Discrete,
}

#[derive(Archive, Deserialize, Serialize, Default)]
pub struct Node {
    pub is_task_node: bool,
    pub node_idx: i32,
    pub f2p_nbr_idxs: Vec<i32>,
    pub f2p_edges: Vec<Edge>,
    pub timestamp: Option<i32>,
    pub table_name_idx: i32,
    pub col_name_idxs: Vec<i32>,
    pub sem_types: Vec<SemType>,
    pub number_values: Vec<f32>,
    pub text_values: Vec<i32>,
    pub datetime_values: Vec<f32>,
    pub boolean_values: Vec<f32>,
    pub class_value_idx: Vec<i32>,
}

/// For `SemType::Discrete` cells, `class_value_idx` packs the pair `(C, label)` as
/// `C * DISCRETE_LABEL_RADIX + label`.
pub const DISCRETE_LABEL_RADIX: i32 = 16;
pub const DISCRETE_MIN_CLASSES: i32 = 2;
pub const DISCRETE_MAX_CLASSES: i32 = 10;

pub fn pack_discrete_label(num_classes: i32, label: i32) -> Result<i32, String> {
    if !(DISCRETE_MIN_CLASSES..=DISCRETE_MAX_CLASSES).contains(&num_classes) {
        return Err(format!(
            "discrete class count {num_classes} must lie in [{DISCRETE_MIN_CLASSES}, {DISCRETE_MAX_CLASSES}]"
        ));
    }
    if !(0..num_classes).contains(&label) {
        return Err(format!(
            "discrete label {label} must lie in [0, {num_classes})"
        ));
    }
    Ok(num_classes * DISCRETE_LABEL_RADIX + label)
}

pub fn unpack_discrete_label(packed: i32) -> Result<(i32, i32), String> {
    if packed < 0 {
        return Err("discrete packed label must be nonnegative".to_string());
    }
    let num_classes = packed / DISCRETE_LABEL_RADIX;
    let label = packed % DISCRETE_LABEL_RADIX;
    if !(DISCRETE_MIN_CLASSES..=DISCRETE_MAX_CLASSES).contains(&num_classes)
        || !(0..num_classes).contains(&label)
    {
        return Err(format!("invalid packed discrete label {packed}"));
    }
    Ok((num_classes, label))
}

#[cfg(test)]
mod discrete_label_tests {
    use super::{pack_discrete_label, unpack_discrete_label};

    #[test]
    fn round_trips_binary_and_ten_way_labels() {
        assert_eq!(
            unpack_discrete_label(pack_discrete_label(2, 1).unwrap()),
            Ok((2, 1))
        );
        assert_eq!(
            unpack_discrete_label(pack_discrete_label(10, 9).unwrap()),
            Ok((10, 9))
        );
    }

    #[test]
    fn rejects_invalid_discrete_pairs() {
        assert!(pack_discrete_label(1, 0).is_err());
        assert!(pack_discrete_label(11, 0).is_err());
        assert!(pack_discrete_label(2, 2).is_err());
        assert!(unpack_discrete_label(16).is_err());
    }
}

#[derive(Archive, Deserialize, Serialize, Default)]
pub struct Offsets {
    pub offsets: Vec<i64>,
}
