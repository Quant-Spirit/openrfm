use crate::rng::{substream, zscore, GenRng, RngExt};
use crate::scm::{ColumnData, GenOptions, GeneratedDb, TableData, ENTITY_ID_COL};
use anyhow::{bail, Result};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

macro_rules! btreemap {
    ($($key:literal => $value:expr),* $(,)?) => {{
        let mut map = ::std::collections::BTreeMap::new();
        $(
            map.insert($key.to_string(), serde_json::json!($value));
        )*
        map
    }};
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
/// The discriminants are part of the generation contract: the relational operator family seeds
/// its search with `family as u64`, so they must not change.
pub enum LabelFamily {
    ChildSum = 0,
    ChildMean = 1,
    ChildCount = 2,
    ChildStd = 6,
    ChildNUnique = 7,
    GrandchildCount = 8,
    CooccurrenceCount = 9,
    RecentChildCount = 10,
    ParentLookup = 13,
    CooccurrenceSum = 16,
    TwohopGatedSum = 19,
    ExistsGrandchildGateSum = 20,
}

impl LabelFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ChildSum => "child_sum",
            Self::ChildMean => "child_mean",
            Self::ChildCount => "child_count",
            Self::ChildStd => "child_std",
            Self::ChildNUnique => "child_n_unique",
            Self::GrandchildCount => "grandchild_count",
            Self::CooccurrenceCount => "cooccurrence_count",
            Self::RecentChildCount => "recent_child_count",
            Self::ParentLookup => "parent_lookup",
            Self::CooccurrenceSum => "cooccurrence_sum",
            Self::TwohopGatedSum => "twohop_gated_sum",
            Self::ExistsGrandchildGateSum => "exists_grandchild_gate_sum",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct LabelMetadata {
    pub family: String,
    pub target_table: String,
    pub source_tables: Vec<String>,
    pub target_type: String,
    pub n_classes: usize,
    pub hops: usize,
    pub uses_time: bool,
    pub params: BTreeMap<String, serde_json::Value>,
    pub seed: u64,
    pub target_col: String,
    pub task_type: String,
    pub num_rows: usize,
}

/// Temporal-kernel label families: temporal kernels, parent lookups, feature readouts and the
/// additive families of `labels_v2`.
pub fn add_axis_labels(db: &mut GeneratedDb, seed: u64, options: &GenOptions) -> Result<()> {
    add_temporal_kernel_labels(db, seed, options.shortcut_gate_enforcement)?;
    add_parent_lookup_labels(db, seed, options.shortcut_gate_enforcement)?;
    add_feature_readout_labels(db, seed, options.shortcut_gate_enforcement)?;
    crate::labels_v2::add_v2_prior_labels(db, seed, options.exact_aggregates, options.nuisance_text)
}

/// FK-sibling neighbor edges: for each FK column on a table, rows sharing the
/// same FK parent value are siblings (neighbors). Returns flat (i, j) edges.
/// Each unordered pair contributes both directions for symmetric diffusion.
fn collect_fk_sibling_edges(table: &crate::scm::TableData) -> Vec<(usize, usize)> {
    use std::collections::HashMap;
    let mut edges = Vec::new();
    for (fk_col, _parent) in &table.fkey_col_to_pkey_table {
        let Some(ColumnData::I64(fk_values)) = table.column(fk_col) else {
            continue;
        };
        let mut by_parent: HashMap<i64, Vec<usize>> = HashMap::new();
        for (row, &fk) in fk_values.iter().enumerate() {
            by_parent.entry(fk).or_default().push(row);
        }
        for sibs in by_parent.values() {
            if sibs.len() < 2 {
                continue;
            }
            // Cap sibling-group size to avoid O(N²) for large groups.
            let cap = sibs.len().min(64);
            for i_idx in 0..cap {
                for j_idx in 0..cap {
                    if i_idx == j_idx {
                        continue;
                    }
                    edges.push((sibs[i_idx], sibs[j_idx]));
                }
            }
        }
    }
    edges
}

/// Two-hop neighbours: rows reachable by composing two FK-sibling steps.
fn collect_two_hop_edges(table: &crate::scm::TableData) -> Vec<(usize, usize)> {
    use std::collections::{HashMap, HashSet};
    let one_hop = collect_fk_sibling_edges(table);
    if one_hop.is_empty() {
        return Vec::new();
    }
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut direct: HashSet<(usize, usize)> = HashSet::new();
    for &(i, j) in &one_hop {
        adj.entry(i).or_default().push(j);
        direct.insert((i, j));
    }
    const PER_ROW_CAP: usize = 48;
    let mut edges = Vec::new();
    for (&i, mids) in adj.iter() {
        let mut seen: HashSet<usize> = HashSet::new();
        'outer: for &mid in mids.iter().take(24) {
            let Some(seconds) = adj.get(&mid) else {
                continue;
            };
            for &k in seconds.iter().take(24) {
                if k == i || direct.contains(&(i, k)) || !seen.insert(k) {
                    continue;
                }
                edges.push((i, k));
                if seen.len() >= PER_ROW_CAP {
                    break 'outer;
                }
            }
        }
    }
    edges
}

/// Z-score standardize: y_i ← (y_i − ȳ) / σ. Stable when σ > 0.
fn standardize_inplace(y: &mut [f64]) {
    let n = y.len() as f64;
    if n <= 1.0 {
        return;
    }
    let mean = y.iter().sum::<f64>() / n;
    let var = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    if std < 1e-9 {
        return;
    }
    for v in y.iter_mut() {
        *v = (*v - mean) / std;
    }
}

pub(crate) fn all_aggfam_candidates(
    db: &GeneratedDb,
    family: LabelFamily,
) -> Vec<(String, Vec<f64>, LabelMetadata)> {
    match family {
        LabelFamily::ChildSum => child_value_aggregate_all(db, family, Aggregate::Sum),
        LabelFamily::ChildMean => child_value_aggregate_all(db, family, Aggregate::Mean),
        LabelFamily::ChildCount => child_value_aggregate_all(db, family, Aggregate::Count),
        LabelFamily::ChildStd => child_value_aggregate_all(db, family, Aggregate::Std),
        LabelFamily::ChildNUnique => child_value_aggregate_all(db, family, Aggregate::NUnique),
        LabelFamily::ParentLookup => parent_lookup_all(db),
        LabelFamily::RecentChildCount => recent_child_count_all(db),
        LabelFamily::CooccurrenceCount => cooccurrence_count_all(db),
        LabelFamily::CooccurrenceSum => cooccurrence_sum_all(db),
        LabelFamily::GrandchildCount => grandchild_count_all(db),
        LabelFamily::TwohopGatedSum => twohop_gated_sum_all(db),
        LabelFamily::ExistsGrandchildGateSum => exists_grandchild_gate_sum_all(db),
    }
}

fn project_features(table: &TableData, rng: &mut GenRng) -> Vec<f64> {
    let cols = table.numeric_feature_columns();
    if cols.is_empty() {
        return vec![0.0; table.num_rows()];
    }
    let mut matrix = Vec::with_capacity(cols.len());
    for col in &cols {
        if let Some(values) = table.column(col).and_then(ColumnData::as_numeric_lossy) {
            matrix.push(fill_nan_with_mean(&values));
        }
    }
    if matrix.is_empty() {
        return vec![0.0; table.num_rows()];
    }
    let mut weights = (0..matrix.len())
        .map(|_| rng.standard_normal().abs())
        .collect::<Vec<_>>();
    let sum = weights.iter().sum::<f64>();
    if sum <= 1e-12 || !sum.is_finite() {
        weights.fill(1.0 / matrix.len() as f64);
    } else {
        for weight in &mut weights {
            *weight /= sum;
        }
    }
    let n = table.num_rows();
    let mut out = vec![0.0; n];
    for (col_values, weight) in matrix.iter().zip(weights.iter()) {
        for row in 0..n {
            out[row] += col_values[row] * *weight;
        }
    }
    zscore(&out)
}

fn fill_nan_with_mean(values: &[f64]) -> Vec<f64> {
    let finite = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    let mean = if finite.is_empty() {
        0.0
    } else {
        finite.iter().sum::<f64>() / finite.len() as f64
    };
    values
        .iter()
        .map(|v| if v.is_finite() { *v } else { mean })
        .collect()
}

#[derive(Debug, Clone, Copy)]
enum Aggregate {
    Sum,
    Mean,
    Count,
    Std,
    NUnique,
}

/// Return ALL eligible (parent_table, scores, metadata) candidates from the DB for
/// child_value_aggregate-style families.
fn child_value_aggregate_all(
    db: &GeneratedDb,
    family: LabelFamily,
    aggregate: Aggregate,
) -> Vec<(String, Vec<f64>, LabelMetadata)> {
    let mut out = Vec::new();
    for child in &db.tables {
        let value_cols: Vec<String> = child.numeric_feature_columns();
        if value_cols.is_empty() {
            continue;
        }
        for value_col in value_cols.iter().take(3) {
            for (fk_col, parent_name) in &child.fkey_col_to_pkey_table {
                let parent = match db.table(parent_name) {
                    Some(t) => t,
                    None => continue,
                };
                let fk_values = match child.column(fk_col).and_then(ColumnData::as_numeric_lossy) {
                    Some(v) => v,
                    None => continue,
                };
                let child_values = match child
                    .column(value_col)
                    .and_then(ColumnData::as_numeric_lossy)
                {
                    Some(v) => v,
                    None => continue,
                };
                let n_parent = parent.num_rows();
                let mut sums = vec![0.0; n_parent];
                let mut counts = vec![0usize; n_parent];
                let mut sum_sq = vec![0.0; n_parent];
                let mut uniques: Vec<std::collections::HashSet<u64>> = (0..n_parent)
                    .map(|_| std::collections::HashSet::new())
                    .collect();
                for row in 0..child.num_rows() {
                    let parent_idx = fk_values[row] as usize;
                    if parent_idx >= n_parent {
                        continue;
                    }
                    let value = child_values[row];
                    if !value.is_finite() {
                        continue;
                    }
                    sums[parent_idx] += value;
                    sum_sq[parent_idx] += value * value;
                    counts[parent_idx] += 1;
                    let bits = value.to_bits();
                    uniques[parent_idx].insert(bits);
                }
                let scores: Vec<f64> = match aggregate {
                    Aggregate::Sum => sums.clone(),
                    Aggregate::Mean => sums
                        .iter()
                        .zip(counts.iter())
                        .map(|(s, c)| if *c > 0 { *s / *c as f64 } else { 0.0 })
                        .collect(),
                    Aggregate::Count => counts.iter().map(|c| *c as f64).collect(),
                    Aggregate::Std => sums
                        .iter()
                        .zip(counts.iter())
                        .zip(sum_sq.iter())
                        .map(|((s, c), sq)| {
                            if *c > 1 {
                                let mean = *s / *c as f64;
                                let var = (*sq / *c as f64) - mean * mean;
                                var.max(0.0).sqrt()
                            } else {
                                0.0
                            }
                        })
                        .collect(),
                    Aggregate::NUnique => uniques.iter().map(|s| s.len() as f64).collect(),
                };
                let metadata = program_metadata(
                    family.as_str(),
                    parent_name,
                    vec![child.name.clone()],
                    1,
                    btreemap! {
                        "child_fk_col" => fk_col.clone(),
                        "child_value_col" => value_col.clone(),
                        "parent_pkey_col" => parent.pkey_col.clone(),
                    },
                );
                out.push((parent_name.clone(), scores, metadata));
            }
        }
    }
    out
}

/// All-candidates variant of grandchild_count.
fn grandchild_count_all(db: &GeneratedDb) -> Vec<(String, Vec<f64>, LabelMetadata)> {
    let mut out = Vec::new();
    for mid in &db.tables {
        for (mid_parent_fk, parent_name) in &mid.fkey_col_to_pkey_table {
            let parent = match db.table(parent_name) {
                Some(t) => t,
                None => continue,
            };
            for grandchild in &db.tables {
                for (gc_mid_fk, mid_name) in &grandchild.fkey_col_to_pkey_table {
                    if mid_name != &mid.name {
                        continue;
                    }
                    let mid_fk_values = match mid
                        .column(mid_parent_fk)
                        .and_then(ColumnData::as_numeric_lossy)
                    {
                        Some(v) => v,
                        None => continue,
                    };
                    let gc_fk_values = match grandchild
                        .column(gc_mid_fk)
                        .and_then(ColumnData::as_numeric_lossy)
                    {
                        Some(v) => v,
                        None => continue,
                    };
                    let mut mid_counts = vec![0.0; mid.num_rows()];
                    for gc_parent in &gc_fk_values {
                        let idx = *gc_parent as usize;
                        if idx < mid_counts.len() {
                            mid_counts[idx] += 1.0;
                        }
                    }
                    let mut parent_counts = vec![0.0; parent.num_rows()];
                    for (mid_row, parent_fk) in mid_fk_values.iter().enumerate() {
                        let parent_idx = *parent_fk as usize;
                        if parent_idx < parent_counts.len() {
                            parent_counts[parent_idx] += mid_counts[mid_row];
                        }
                    }
                    let metadata = program_metadata(
                        "grandchild_count",
                        parent_name,
                        vec![mid.name.clone(), grandchild.name.clone()],
                        2,
                        btreemap! {
                            "mid_parent_fk_col" => mid_parent_fk.clone(),
                            "mid_pkey_col" => mid.pkey_col.clone(),
                            "grandchild_mid_fk_col" => gc_mid_fk.clone(),
                            "parent_pkey_col" => parent.pkey_col.clone(),
                        },
                    );
                    out.push((parent_name.clone(), parent_counts, metadata));
                }
            }
        }
    }
    out
}

/// Extract i64 timestamps for a table's `time_col` if present.
/// Returns None if the table is non-Activity (no time_col).
fn time_values(table: &TableData) -> Option<Vec<i64>> {
    let col = table.time_col.as_ref()?;
    match table.column(col)? {
        ColumnData::TimestampNs(v) => Some(v.clone()),
        _ => None,
    }
}

/// Recent-child-count: per-parent count of children whose timestamp is in the upper half of the
/// child table's time range.
fn recent_child_count_all(db: &GeneratedDb) -> Vec<(String, Vec<f64>, LabelMetadata)> {
    let mut out = Vec::new();
    for child in &db.tables {
        let times = time_values(child);
        for (fk_col, parent_name) in &child.fkey_col_to_pkey_table {
            let parent = match db.table(parent_name) {
                Some(t) => t,
                None => continue,
            };
            let fk_values = match child.column(fk_col).and_then(ColumnData::as_numeric_lossy) {
                Some(v) => v,
                None => continue,
            };
            let n_child = child.num_rows();
            // Compute median timestamp; rows with t > median are "recent".
            let recent_mask: Vec<bool> = if let Some(ts) = &times {
                let mut sorted = ts.clone();
                sorted.sort();
                let median = sorted[sorted.len() / 2];
                ts.iter().map(|t| *t > median).collect()
            } else {
                // Fallback: row index in second half.
                (0..n_child).map(|i| i >= n_child / 2).collect()
            };
            let mut counts = vec![0.0; parent.num_rows()];
            for row in 0..n_child {
                if !recent_mask[row] {
                    continue;
                }
                let parent_idx = fk_values[row] as usize;
                if parent_idx < counts.len() {
                    counts[parent_idx] += 1.0;
                }
            }
            let mut metadata = program_metadata(
                "recent_child_count",
                parent_name,
                vec![child.name.clone()],
                1,
                btreemap! {
                    "child_fk_col" => fk_col.clone(),
                    "parent_pkey_col" => parent.pkey_col.clone(),
                    "time_col" => child.time_col.clone().unwrap_or_default(),
                    "uses_time" => times.is_some(),
                },
            );
            metadata.uses_time = times.is_some();
            out.push((parent_name.clone(), counts, metadata));
        }
    }
    out
}

/// All-candidates variant of cooccurrence_count.
fn cooccurrence_count_all(db: &GeneratedDb) -> Vec<(String, Vec<f64>, LabelMetadata)> {
    let mut out = Vec::new();
    for child in &db.tables {
        let cols = child.numeric_feature_columns();
        if cols.len() < 2 {
            continue;
        }
        for (i, a_col) in cols.iter().enumerate().take(3) {
            for b_col in cols.iter().skip(i + 1).take(2) {
                for (fk_col, parent_name) in &child.fkey_col_to_pkey_table {
                    let parent = match db.table(parent_name) {
                        Some(t) => t,
                        None => continue,
                    };
                    let fk_values =
                        match child.column(fk_col).and_then(ColumnData::as_numeric_lossy) {
                            Some(v) => v,
                            None => continue,
                        };
                    let a = match child.column(a_col).and_then(ColumnData::as_numeric_lossy) {
                        Some(v) => v,
                        None => continue,
                    };
                    let b = match child.column(b_col).and_then(ColumnData::as_numeric_lossy) {
                        Some(v) => v,
                        None => continue,
                    };
                    let thr_a = median(&a);
                    let thr_b = median(&b);
                    let mut counts = vec![0.0; parent.num_rows()];
                    for row in 0..child.num_rows() {
                        let parent_idx = fk_values[row] as usize;
                        if parent_idx < parent.num_rows() && a[row] > thr_a && b[row] > thr_b {
                            counts[parent_idx] += 1.0;
                        }
                    }
                    let metadata = program_metadata(
                        "cooccurrence_count",
                        parent_name,
                        vec![child.name.clone()],
                        1,
                        btreemap! {
                            "child_fk_col" => fk_col.clone(),
                            "pred_a_col" => a_col.clone(),
                            "pred_a_thr" => thr_a,
                            "pred_b_col" => b_col.clone(),
                            "pred_b_thr" => thr_b,
                            "parent_pkey_col" => parent.pkey_col.clone(),
                        },
                    );
                    out.push((parent_name.clone(), counts, metadata));
                }
            }
        }
    }
    out
}

fn median(values: &[f64]) -> f64 {
    let mut finite = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return 0.0;
    }
    finite.sort_by(|a, b| a.total_cmp(b));
    finite[finite.len() / 2]
}

fn program_metadata(
    family: &str,
    target_table: &str,
    source_tables: Vec<String>,
    hops: usize,
    params: BTreeMap<String, serde_json::Value>,
) -> LabelMetadata {
    LabelMetadata {
        family: family.to_string(),
        target_table: target_table.to_string(),
        source_tables,
        target_type: "classification".to_string(),
        n_classes: 2,
        hops,
        uses_time: false,
        params,
        seed: 0,
        target_col: String::new(),
        task_type: "clf".to_string(),
        num_rows: 0,
    }
}

/// parent_lookup: per-child-row label is the parent's value column (looked up
/// via FK). Target table is the CHILD (not parent).
fn parent_lookup_all(db: &GeneratedDb) -> Vec<(String, Vec<f64>, LabelMetadata)> {
    let mut out = Vec::new();
    for child in &db.tables {
        for (fk_col, parent_name) in &child.fkey_col_to_pkey_table {
            let parent = match db.table(parent_name) {
                Some(t) => t,
                None => continue,
            };
            let fk_values = match child.column(fk_col).and_then(ColumnData::as_numeric_lossy) {
                Some(v) => v,
                None => continue,
            };
            for value_col in parent.numeric_feature_columns().iter().take(2) {
                let parent_values = match parent
                    .column(value_col)
                    .and_then(ColumnData::as_numeric_lossy)
                {
                    Some(v) => v,
                    None => continue,
                };
                let mut scores = vec![0.0; child.num_rows()];
                for row in 0..child.num_rows() {
                    let pidx = fk_values[row] as usize;
                    if pidx < parent_values.len() && parent_values[pidx].is_finite() {
                        scores[row] = parent_values[pidx];
                    }
                }
                let metadata = program_metadata(
                    "parent_lookup",
                    &child.name, // target IS the child here
                    vec![parent_name.clone()],
                    1,
                    btreemap! {
                        "child_fk_col" => fk_col.clone(),
                        "parent_pkey_col" => parent.pkey_col.clone(),
                        "parent_value_col" => (*value_col).clone(),
                    },
                );
                out.push((child.name.clone(), scores, metadata));
            }
        }
    }
    out
}

/// cooccurrence_sum: per-parent SUM of child_value where joint predicate (a>thr_a AND b>thr_b).
fn cooccurrence_sum_all(db: &GeneratedDb) -> Vec<(String, Vec<f64>, LabelMetadata)> {
    let mut out = Vec::new();
    for child in &db.tables {
        let cols = child.numeric_feature_columns();
        if cols.len() < 3 {
            continue;
        }
        for (i, col_a) in cols.iter().enumerate().take(3) {
            for col_b in cols.iter().skip(i + 1).take(2) {
                for value_col in cols
                    .iter()
                    .filter(|c| c.as_str() != col_a && c.as_str() != col_b)
                    .take(2)
                {
                    for (fk_col, parent_name) in &child.fkey_col_to_pkey_table {
                        let parent = match db.table(parent_name) {
                            Some(t) => t,
                            None => continue,
                        };
                        let fk_values =
                            match child.column(fk_col).and_then(ColumnData::as_numeric_lossy) {
                                Some(v) => v,
                                None => continue,
                            };
                        let a = match child.column(col_a).and_then(ColumnData::as_numeric_lossy) {
                            Some(v) => v,
                            None => continue,
                        };
                        let b = match child.column(col_b).and_then(ColumnData::as_numeric_lossy) {
                            Some(v) => v,
                            None => continue,
                        };
                        let v = match child
                            .column(value_col)
                            .and_then(ColumnData::as_numeric_lossy)
                        {
                            Some(v) => v,
                            None => continue,
                        };
                        let thr_a = median(&a);
                        let thr_b = median(&b);
                        let mut scores = vec![0.0; parent.num_rows()];
                        for row in 0..child.num_rows() {
                            let pidx = fk_values[row] as usize;
                            if pidx < parent.num_rows()
                                && a[row] > thr_a
                                && b[row] > thr_b
                                && v[row].is_finite()
                            {
                                scores[pidx] += v[row];
                            }
                        }
                        let metadata = program_metadata(
                            "cooccurrence_sum",
                            parent_name,
                            vec![child.name.clone()],
                            1,
                            btreemap! {
                                "child_fk_col" => fk_col.clone(),
                                "pred_a_col" => col_a.clone(),
                                "pred_a_thr" => thr_a,
                                "pred_b_col" => col_b.clone(),
                                "pred_b_thr" => thr_b,
                                "child_value_col" => (*value_col).clone(),
                                "parent_pkey_col" => parent.pkey_col.clone(),
                            },
                        );
                        out.push((parent_name.clone(), scores, metadata));
                    }
                }
            }
        }
    }
    out
}

/// twohop_gated_sum: per-parent, sum mid_value over mid rows whose grandchildren
/// satisfy a predicate.
fn twohop_gated_sum_all(db: &GeneratedDb) -> Vec<(String, Vec<f64>, LabelMetadata)> {
    let mut out = Vec::new();
    for mid in &db.tables {
        let mid_cols = mid.numeric_feature_columns();
        if mid_cols.is_empty() {
            continue;
        }
        let mid_value_col = mid_cols[0].clone();
        let mid_value = match mid
            .column(&mid_value_col)
            .and_then(ColumnData::as_numeric_lossy)
        {
            Some(v) => v,
            None => continue,
        };
        for (mid_parent_fk, parent_name) in &mid.fkey_col_to_pkey_table {
            let parent = match db.table(parent_name) {
                Some(t) => t,
                None => continue,
            };
            let mid_fk_values = match mid
                .column(mid_parent_fk)
                .and_then(ColumnData::as_numeric_lossy)
            {
                Some(v) => v,
                None => continue,
            };
            for grandchild in &db.tables {
                let gc_cols = grandchild.numeric_feature_columns();
                if gc_cols.is_empty() {
                    continue;
                }
                let gate_col = gc_cols[0].clone();
                let gate_values = match grandchild
                    .column(&gate_col)
                    .and_then(ColumnData::as_numeric_lossy)
                {
                    Some(v) => v,
                    None => continue,
                };
                let gate_thr = median(&gate_values);
                for (gc_mid_fk, mid_name) in &grandchild.fkey_col_to_pkey_table {
                    if mid_name != &mid.name {
                        continue;
                    }
                    let gc_fk_values = match grandchild
                        .column(gc_mid_fk)
                        .and_then(ColumnData::as_numeric_lossy)
                    {
                        Some(v) => v,
                        None => continue,
                    };
                    // Mark mid rows with at least one gated grandchild.
                    let mut gated_mid = vec![false; mid.num_rows()];
                    for row in 0..grandchild.num_rows() {
                        if gate_values[row] > gate_thr {
                            let midx = gc_fk_values[row] as usize;
                            if midx < gated_mid.len() {
                                gated_mid[midx] = true;
                            }
                        }
                    }
                    let mut scores = vec![0.0; parent.num_rows()];
                    for row in 0..mid.num_rows() {
                        if !gated_mid[row] || !mid_value[row].is_finite() {
                            continue;
                        }
                        let pidx = mid_fk_values[row] as usize;
                        if pidx < parent.num_rows() {
                            scores[pidx] += mid_value[row];
                        }
                    }
                    let metadata = program_metadata(
                        "twohop_gated_sum",
                        parent_name,
                        vec![mid.name.clone(), grandchild.name.clone()],
                        2,
                        btreemap! {
                            "mid_parent_fk_col" => mid_parent_fk.clone(),
                            "mid_value_col" => mid_value_col.clone(),
                            "grandchild_mid_fk_col" => gc_mid_fk.clone(),
                            "gate_col" => gate_col.clone(),
                            "gate_thr" => gate_thr,
                            "parent_pkey_col" => parent.pkey_col.clone(),
                        },
                    );
                    out.push((parent_name.clone(), scores, metadata));
                }
            }
        }
    }
    out
}

const FEATREG_GATE_ATTEMPTS: usize = 8;

/// The two shortcuts an L0 feature-readout target has, and the liveness floor that keeps them
/// meaningful.
fn featreg_verdicts(
    table: &TableData,
    published: &[f64],
    prenoise: &[f64],
    rejected: usize,
    siblings: &[String],
) -> Vec<crate::shortcut_gates::ShortcutVerdict> {
    use crate::shortcut_gates::{
        attribute, ContinuousLiveness, ContinuousTargetGateInput, ExogenousChannel,
        SHORTCUT_CEILING,
    };
    let row_index: Vec<f64> = (0..table.num_rows()).map(|row| row as f64).collect();
    let channels = [ExogenousChannel {
        name: "row_index",
        values: &row_index,
    }];
    let mut verdicts =
        crate::shortcut_gates::evaluate_continuous_target(&ContinuousTargetGateInput {
            family: "featreg",
            arm: "feature_readout",
            target_table: table,
            target: "shared_latent",
            values: published,
            exogenous: &channels,
            liveness: ContinuousLiveness::Latent {
                name: "the readout before its additive noise draw",
                values: prenoise,
            },
            bag_ceiling: SHORTCUT_CEILING,
            bag_declaration: None,
            exogenous_ceiling: SHORTCUT_CEILING,
            exogenous_declaration: None,
            disclosure_exclusions: Some(siblings),
        });
    attribute(&mut verdicts, "feature_readout_shared_latent", rejected);
    verdicts
}

fn featreg_exhausted_verdict(
    table: &TableData,
    rejected: usize,
    first_rejection: Option<&str>,
) -> Vec<crate::shortcut_gates::ShortcutVerdict> {
    use crate::shortcut_gates::{
        attribute, GateKind, ShortcutVerdict, CONTINUOUS_BINS, LIVENESS_FLOOR,
    };
    let mut verdicts = vec![ShortcutVerdict {
        gate: GateKind::MechanismDeterminacy,
        family: "featreg".into(),
        arm: "feature_readout".into(),
        target_table: table.name.clone(),
        target: "shared_latent".into(),
        k: CONTINUOUS_BINS,
        measured_nats: 0.0,
        budget_nats: (CONTINUOUS_BINS as f64).ln(),
        ratio: 0.0,
        threshold_ratio: LIVENESS_FLOOR,
        passed: false,
        declared_exception: None,
        mechanism: None,
        candidates_rejected: None,
        read_as_coordinate_not_verdict: None,
        // This verdict is built by hand rather than by an `evaluate_*` entry,
        // so it stamps itself. See `ShortcutVerdict::bag_summary`.
        feature_contract: table.feature_contract,
        bag_summary: table.bag_summary,
        note: format!(
            "every one of the {FEATREG_GATE_ATTEMPTS} redraws was rejected, so this table \
             published no L0 target. First rejection: {}",
            first_rejection.unwrap_or("none recorded"),
        ),
    }];
    attribute(&mut verdicts, "feature_readout_exhausted", rejected);
    verdicts
}

/// Shared own-feature latent and noisy readout. `row_local` uses the Gaussian
/// population scale and identity member of the existing mechanism prior: no
/// empirical ranks, quantiles, graph filters, degree or clocks enter its draw.
pub(crate) fn own_feature_latent(
    table: &crate::scm::TableData,
    graph_free: &[String],
    rng: &mut GenRng,
    row_local: bool,
) -> Option<(usize, Vec<f64>, Vec<f64>)> {
    if graph_free.is_empty() {
        return None;
    }
    let row_count = table.num_rows();
    // One to four contributing features, capped by what the table has.
    let width = 1 + rng.choose_index(graph_free.len().min(4));
    let mut picks: Vec<String> = graph_free.to_vec();
    rng.shuffle(&mut picks);
    picks.truncate(width);

    let mut y = vec![0.0_f64; row_count];
    let mut used = 0usize;
    let mut variance = 0.0;
    for name in &picks {
        let Some(ColumnData::F64(values)) = table.column(name).cloned() else {
            continue;
        };
        if values.len() != row_count {
            continue;
        }
        let weight = rng.f64_range(-1.5, 1.5);
        let mut column: Vec<f64> = values.iter().map(|v| v.unwrap_or(0.0)).collect();
        if !row_local {
            standardize_inplace(&mut column);
        }
        variance += weight * weight;
        for (target, value) in y.iter_mut().zip(column.iter()) {
            *target += weight * value;
        }
        used += 1;
    }
    if used == 0 {
        return None;
    }
    if !row_local {
        standardize_inplace(&mut y);
    }

    let mech = TargetMechanism {
        choice: if row_local { 0 } else { rng.choose_index(12) },
        a: rng.f64_range(0.5, 2.0),
        b: rng.f64_range(-1.0, 1.0),
        bins: 3 + rng.choose_index(6),
        bin_values: (0..9).map(|_| rng.f64_range(-2.0, 2.0)).collect(),
        freqs: (0..3)
            .map(|_| {
                (
                    rng.f64_range(0.5, 3.0),
                    rng.f64_range(0.0, 6.283_185_307_179_586),
                    rng.f64_range(-1.0, 1.0),
                )
            })
            .collect(),
    };
    if row_local {
        for value in &mut y {
            *value /= variance.sqrt().max(1e-12);
        }
    }
    mech.apply(&mut y);

    // A gauss-rank readout on part of the family.
    if !row_local && rng.bernoulli(0.4) {
        gauss_rank_inplace(&mut y);
    }
    let prenoise = y.clone();
    let noise = rng.f64_range(0.02, 0.25);
    for value in y.iter_mut() {
        *value += noise * rng.standard_normal();
    }
    if !row_local {
        standardize_inplace(&mut y);
    }

    Some((used, y, prenoise))
}

pub(crate) fn cut_feature_classes(latent: &[f64], cuts: &[f64]) -> Vec<i64> {
    latent
        .iter()
        .map(|v| cuts.iter().filter(|c| *v > **c).count() as i64)
        .collect()
}

fn add_feature_readout_labels(db: &mut GeneratedDb, seed: u64, enforce_gates: bool) -> Result<()> {
    let table_count = db.tables.len();
    for table_idx in 0..table_count {
        let numeric: Vec<String> = db.tables[table_idx].numeric_feature_columns();
        if numeric.is_empty() {
            continue;
        }
        let rng = substream(
            seed + 202_021 * (table_idx as u64 + 1),
            "labels:feature_readout",
        );
        let row_count = db.tables[table_idx].num_rows();
        if row_count < 8 {
            continue;
        }
        // Not every numeric column is independent of the graph.
        let fk_groups: Option<Vec<i64>> = db.tables[table_idx]
            .fkey_col_to_pkey_table
            .keys()
            .next()
            .and_then(|c| match db.tables[table_idx].column(c) {
                Some(ColumnData::I64(v)) => Some(v.to_vec()),
                _ => None,
            });
        let graph_free: Vec<String> = numeric
            .iter()
            .filter(|name| {
                let Some(keys) = fk_groups.as_ref() else {
                    return true;
                };
                let Some(ColumnData::F64(values)) = db.tables[table_idx].column(name) else {
                    return false;
                };
                let column: Vec<f64> = values.iter().map(|v| v.unwrap_or(0.0)).collect();
                if column.len() != keys.len() || column.len() < 8 {
                    return false;
                }
                let mean = column.iter().sum::<f64>() / column.len() as f64;
                let total =
                    column.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / column.len() as f64;
                if total <= 0.0 {
                    return false;
                }
                let mut sums: std::collections::HashMap<i64, (f64, usize)> =
                    std::collections::HashMap::new();
                for (k, v) in keys.iter().zip(column.iter()) {
                    let entry = sums.entry(*k).or_insert((0.0, 0));
                    entry.0 += *v;
                    entry.1 += 1;
                }
                let mut within = 0.0_f64;
                for (k, v) in keys.iter().zip(column.iter()) {
                    let (sum, count) = sums[k];
                    let group_mean = sum / count as f64;
                    within += (v - group_mean).powi(2);
                }
                within / column.len() as f64 >= 0.5 * total
            })
            .cloned()
            .collect();
        let dispersed: Vec<String> = graph_free;
        let graph_free = dispersed;
        if graph_free.is_empty() {
            continue;
        }
        let mut selection: Option<(usize, Vec<f64>, Vec<f64>, GenRng)> = None;
        let mut rejected = 0usize;
        let mut first_rejection: Option<String> = None;
        let mut gate_verdicts = Vec::new();
        let mut unusable = false;
        for attempt in 0..FEATREG_GATE_ATTEMPTS {
            let mut rng = if attempt == 0 {
                rng.clone()
            } else {
                substream(
                    seed + 202_021 * (table_idx as u64 + 1) + 7_919 * attempt as u64,
                    "labels:feature_readout:retry",
                )
            };
            let Some((used, y, prenoise)) =
                own_feature_latent(&db.tables[table_idx], &graph_free, &mut rng, false)
            else {
                unusable = true;
                break;
            };

            let siblings = db.declared_targets_on_table(&db.tables[table_idx].name);
            let mut verdicts =
                featreg_verdicts(&db.tables[table_idx], &y, &prenoise, rejected, &siblings);
            if !enforce_gates || crate::shortcut_gates::all_passed(&verdicts) {
                gate_verdicts.append(&mut verdicts);
                selection = Some((used, y, prenoise, rng));
                break;
            }
            rejected += 1;
            if first_rejection.is_none() {
                first_rejection = verdicts
                    .iter()
                    .find(|verdict| !verdict.passed)
                    .map(|verdict| verdict.failure_message());
            }
        }
        if unusable {
            continue;
        }
        let Some((used, y, _prenoise, mut rng)) = selection else {
            db.record_shortcut_verdicts(featreg_exhausted_verdict(
                &db.tables[table_idx],
                rejected,
                first_rejection.as_deref(),
            ));
            continue;
        };
        // The verdicts are named for the shared latent, not for any of the three columns cut from
        // it, so the range is kept and each published name claims it below.
        let featreg_verdict_range = db.record_shortcut_verdicts(gate_verdicts);

        // The classification views are cut from the SAME latent the regression view uses.
        let latent = y.clone();

        let labels: Vec<Option<f64>> = y.iter().map(|v| Some(*v)).collect();
        let base = format!("featreg_w{used}_{}", db.tables[table_idx].name);
        let col_name = db.tables[table_idx].next_column_name(&base);
        db.tables[table_idx].push_column(col_name.clone(), ColumnData::F64(labels));
        db.claim_gate_verdicts(featreg_verdict_range.clone(), &col_name);

        // K-way view. K is drawn from the same ladder the aggregate families
        // use, so the L0 level spans the same cardinalities as L1 and L2.
        let mut sorted = latent.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let k_choices = [2usize, 2, 3, 3, 4, 4, 5, 6, 8, 10];
        let k = k_choices[rng.choose_index(k_choices.len())];
        let cuts: Vec<f64> = (1..k).map(|i| sorted[(sorted.len() * i) / k]).collect();
        let classes = cut_feature_classes(&latent, &cuts);
        if classes.iter().any(|c| *c != classes[0]) {
            let clf_base = format!("featclf_k{k}_w{used}_{}", db.tables[table_idx].name);
            let clf_name = db.tables[table_idx].next_column_name(&clf_base);
            db.tables[table_idx].push_column(clf_name.clone(), ColumnData::I64(classes.clone()));
            // Cut from the same latent, so the same verdicts cover it. The
            // gate set is stated on the latent, not on the readout.
            db.claim_gate_verdicts(featreg_verdict_range.clone(), &clf_name);
        }

        // Imbalanced binary view.
        let minority = {
            let lo = 0.002_f64.ln();
            let hi = 0.50_f64.ln();
            (lo + (hi - lo) * rng.f64_range(0.0, 1.0)).exp()
        };
        let positive_is_majority = rng.bernoulli(0.5);
        let share = if positive_is_majority {
            minority
        } else {
            1.0 - minority
        };
        let cut_index = ((sorted.len() as f64) * share) as usize;
        let cut = sorted[cut_index.min(sorted.len() - 1)];
        let imbalanced: Vec<i64> = latent
            .iter()
            .map(|v| i64::from((*v > cut) != positive_is_majority))
            .collect();
        let positives = imbalanced.iter().filter(|v| **v == 1).count();
        if positives > 0 && positives < imbalanced.len() {
            let imb_base = format!(
                "featimb_k2_q{:.2}_posmaj{}_w{used}_{}",
                1.0 - minority,
                i32::from(positive_is_majority),
                db.tables[table_idx].name
            );
            let imb_name = db.tables[table_idx].next_column_name(&imb_base);
            db.tables[table_idx].push_column(imb_name.clone(), ColumnData::I64(imbalanced.clone()));
            db.claim_gate_verdicts(featreg_verdict_range.clone(), &imb_name);
        }
    }
    Ok(())
}

/// Whether a candidate `plookup_` target is determined by the parent row.
fn parent_determines(y: &[f64], fk_values: &[i64]) -> bool {
    use std::collections::HashMap;
    if y.len() != fk_values.len() || y.len() < 40 {
        return true;
    }
    let mut groups: HashMap<i64, Vec<f64>> = HashMap::new();
    for (value, key) in y.iter().zip(fk_values) {
        groups.entry(*key).or_default().push(*value);
    }
    let multi: Vec<&Vec<f64>> = groups.values().filter(|group| group.len() >= 3).collect();
    if multi.len() < 5 {
        return true;
    }
    let mean = |group: &[f64]| group.iter().sum::<f64>() / group.len() as f64;
    let within: f64 = multi
        .iter()
        .map(|group| {
            let m = mean(group);
            group.iter().map(|value| (value - m).powi(2)).sum::<f64>() / group.len() as f64
        })
        .sum::<f64>()
        / multi.len() as f64;
    let group_means: Vec<f64> = multi.iter().map(|group| mean(group)).collect();
    let grand = mean(&group_means);
    let between: f64 = group_means
        .iter()
        .map(|value| (value - grand).powi(2))
        .sum::<f64>()
        / group_means.len() as f64;
    between <= 1e-9 || within / between < 0.5
}

fn add_parent_lookup_labels(db: &mut GeneratedDb, seed: u64, enforce_gates: bool) -> Result<()> {
    let table_count = db.tables.len();
    for table_idx in 0..table_count {
        let (fk_col, parent_name) = {
            let table = &db.tables[table_idx];
            match table.fkey_col_to_pkey_table.iter().next() {
                Some((c, p)) => (c.clone(), p.clone()),
                None => continue,
            }
        };
        let Some(parent_idx) = db.tables.iter().position(|t| t.name == parent_name) else {
            continue;
        };
        if parent_idx == table_idx {
            continue;
        }
        let mut rng = substream(
            seed + 101_009 * (table_idx as u64 + 1),
            "labels:parent_lookup",
        );
        // Which of the parent's numeric columns is being looked up. Drawn, so
        // the model cannot key on a fixed position.
        let parent_numeric: Vec<String> = db.tables[parent_idx].numeric_feature_columns();
        if parent_numeric.is_empty() {
            continue;
        }
        let first = rng.choose_index(parent_numeric.len());
        let Some(ColumnData::I64(fk_values)) = db.tables[table_idx].column(&fk_col).cloned() else {
            continue;
        };
        // Noise is deliberately small.
        let noise = rng.f64_range(0.02, 0.20);
        // The pick has to be a column the lookup is actually DETERMINED by.
        let mut chosen: Option<(String, Vec<f64>)> = None;
        // Ungated, the search is ONE candidate: the drawn one, taken or abandoned.
        let attempts = if enforce_gates {
            parent_numeric.len()
        } else {
            1
        };
        for offset in 0..attempts {
            let pick = parent_numeric[(first + offset) % parent_numeric.len()].clone();
            let Some(ColumnData::F64(parent_vals)) = db.tables[parent_idx].column(&pick).cloned()
            else {
                continue;
            };
            let n_parent = parent_vals.len() as i64;
            let looked_up: Vec<f64> = fk_values
                .iter()
                .map(|&fk| {
                    if fk >= 0 && fk < n_parent {
                        parent_vals[fk as usize].unwrap_or(0.0)
                    } else {
                        0.0
                    }
                })
                .collect();
            let scaled_noise = noise;
            let mut y: Vec<f64> = Vec::with_capacity(fk_values.len());
            for row in 0..fk_values.len() {
                let mut jitter = substream(
                    seed + 101_009 * (table_idx as u64 + 1) + row as u64,
                    "labels:parent_lookup:row",
                );
                y.push(looked_up[row] + scaled_noise * jitter.standard_normal());
            }
            if y.iter().all(|v| !v.is_finite()) {
                continue;
            }
            if enforce_gates && !parent_determines(&y, &fk_values) {
                continue;
            }
            chosen = Some((pick, y));
            break;
        }
        let Some((pick, y)) = chosen else {
            continue;
        };
        let published: Vec<f64> = y;
        if published.len() < 2 {
            continue;
        }
        let mut reg = published.clone();
        standardize_inplace(&mut reg);
        let scattered: Vec<Option<f64>> = reg.iter().copied().map(Some).collect();
        let base_name = format!("plookup_r_{}_{}", pick, db.tables[table_idx].name);
        let name = db.tables[table_idx].next_column_name(&base_name);
        db.tables[table_idx].push_column(name.clone(), ColumnData::F64(scattered.clone()));

        // A binary readout of the same lookup.
        let mut sorted = reg.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let cut = sorted[sorted.len() / 2];
        let positives = reg.iter().filter(|v| **v > cut).count();
        if positives > 0 && positives < reg.len() {
            let cbase = format!("plookup_k2_{}_{}", pick, db.tables[table_idx].name);
            let cname = db.tables[table_idx].next_column_name(&cbase);
            let clf: Vec<i64> = scattered
                .iter()
                .map(|value| i64::from(value.unwrap_or(f64::NAN) > cut))
                .collect();
            db.tables[table_idx].push_column(cname.clone(), ColumnData::I64(clf));

            // The binary readout is a median cut of the same column, so the
            // same verdicts are the ones about it.
        }
    }
    Ok(())
}

/// Mixture redraws allowed before a table gives up on a temporal-kernel target.
const TEMPK_MIX_ATTEMPTS: usize = 8;

/// Redraws of the DATABASE-level `(beta, omega, phi, w)` latent allowed before a database gives up
/// on the family entirely.
const TEMPK_GRID_ATTEMPTS: usize = 4;

/// The frequency grid `add_temporal_kernel_labels` draws `omega` from.
pub const TEMPK_OMEGA_GRID: [f64; 3] = [3.0, 7.0, 30.0];

/// The mixture with the relational (neighbour-label) weight removed.
fn label_free_mix(mix: [f64; 4]) -> [f64; 4] {
    [mix[0], mix[1], mix[2], 0.0]
}

pub const TEMPK_V1_PREFIX: &str = "tempk";

#[allow(clippy::too_many_arguments)]
fn tempk_verdicts(
    table: &TableData,
    times: &[f64],
    omega: f64,
    stem: &str,
    published: &[f64],
    label_free: &[f64],
    rejected: usize,
    siblings: &[String],
) -> Vec<crate::shortcut_gates::ShortcutVerdict> {
    use crate::shortcut_gates::{
        attribute, ContinuousLiveness, ContinuousTargetGateInput, ExogenousChannel,
        SHORTCUT_CEILING,
    };
    let n = times.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|left, right| times[*left].total_cmp(&times[*right]).then(left.cmp(right)));
    let mut t_norm = vec![0.0_f64; n];
    for (rank, &row) in order.iter().enumerate() {
        t_norm[row] = rank as f64 / (n as f64 - 1.0).max(1.0);
    }
    let mut channel_values: Vec<(String, Vec<f64>)> = vec![("clock".to_string(), times.to_vec())];
    for grid in TEMPK_OMEGA_GRID {
        channel_values.push((
            format!("season_phase_omega{grid:.0}"),
            t_norm.iter().map(|value| (value * grid).fract()).collect(),
        ));
    }
    let channels: Vec<ExogenousChannel<'_>> = channel_values
        .iter()
        .map(|(name, values)| ExogenousChannel {
            name,
            values: values.as_slice(),
        })
        .collect();
    let mut verdicts = crate::shortcut_gates::evaluate_continuous_target_with_exclusions(
        &ContinuousTargetGateInput {
            family: "tempk",
            arm: "bag",
            target_table: table,
            target: stem,
            values: published,
            exogenous: &channels,
            liveness: ContinuousLiveness::Ablation {
                name: "the same recursion with phi = 0 and the neighbour weight at 0",
                values: label_free,
            },
            bag_ceiling: SHORTCUT_CEILING,
            bag_declaration: None,
            exogenous_ceiling: SHORTCUT_CEILING,
            exogenous_declaration: None,
            disclosure_exclusions: Some(siblings),
        },
        &[],
    );
    attribute(
        &mut verdicts,
        &format!("temporal_kernel_shared_latent_omega{omega:.0}"),
        rejected,
    );
    verdicts
}

/// The column-name prefix a batch of `tempk*` verdicts routes its columns to.
fn tempk_routed_prefix(verdicts: &[crate::shortcut_gates::ShortcutVerdict]) -> Option<String> {
    verdicts
        .iter()
        .find_map(|verdict| verdict.read_as_coordinate_not_verdict.as_ref())
        .map(|coordinate| coordinate.column_prefix.clone())
}

pub fn add_temporal_kernel_labels(
    db: &mut GeneratedDb,
    seed: u64,
    enforce_gates: bool,
) -> Result<()> {
    const BETA: [f64; 3] = [-0.5, 0.0, 0.5];
    const OMEGA: [f64; 3] = [3.0, 7.0, 30.0];
    const PHI: [f64; 3] = [0.0, 0.3, 0.7];
    const W_FRAC: [f64; 2] = [0.25, 0.5];

    let mut rng_g = substream(seed + 41_113, "labels:tempk:grid");
    for grid_attempt in 0..TEMPK_GRID_ATTEMPTS {
        let beta = BETA[rng_g.choose_index(BETA.len())];
        let omega = OMEGA[rng_g.choose_index(OMEGA.len())];
        let phi = PHI[rng_g.choose_index(PHI.len())];
        let w_frac = W_FRAC[rng_g.choose_index(W_FRAC.len())];
        let mut candidate = db.clone();
        let (emitted, inapplicable) = temporal_kernel_grid_pass(
            &mut candidate,
            seed,
            beta,
            omega,
            phi,
            w_frac,
            enforce_gates,
        )?;
        let final_attempt = !enforce_gates || grid_attempt + 1 == TEMPK_GRID_ATTEMPTS;
        if emitted == 0 && !final_attempt {
            continue;
        }
        // The reason list has to carry the GATE rejections too.
        let rejected: Vec<String> = candidate
            .shortcut_gates
            .iter()
            .filter(|verdict| !verdict.passed)
            .map(|verdict| verdict.failure_message())
            .collect();
        *db = candidate;
        if emitted == 0 {
            let reason = if inapplicable.is_empty() && rejected.is_empty() {
                "database has no tables".to_string()
            } else {
                inapplicable
                    .into_iter()
                    .chain(rejected)
                    .collect::<Vec<_>>()
                    .join("; ")
            };
            bail!(
                "TemporalKernel selected but not applicable after {} grid draw(s): {}",
                grid_attempt + 1,
                reason
            );
        }
        return Ok(());
    }
    unreachable!("the final grid attempt always returns or bails")
}

/// One pass of the temporal-kernel family over every table, under one fixed
/// `(beta, omega, phi, w_frac)` draw.
#[allow(clippy::too_many_arguments)]
fn temporal_kernel_grid_pass(
    db: &mut GeneratedDb,
    seed: u64,
    beta: f64,
    omega: f64,
    phi: f64,
    w_frac: f64,
    enforce_gates: bool,
) -> Result<(usize, Vec<String>)> {
    let mut emitted = 0usize;
    let mut inapplicable = Vec::new();
    for table_idx in 0..db.tables.len() {
        let table = &db.tables[table_idx];
        let n = table.num_rows();
        if n < 8 {
            inapplicable.push(format!(
                "{} has only {n} rows (need at least 8)",
                table.name
            ));
            continue;
        }
        let Some(tc) = table.time_col.as_ref() else {
            inapplicable.push(format!("{} has no time_col", table.name));
            continue;
        };
        let t_vec: Vec<f64> = match table.column(tc) {
            Some(ColumnData::TimestampNs(values)) => values.iter().map(|v| *v as f64).collect(),
            Some(other) => match other.as_numeric_lossy() {
                Some(v) => v,
                None => bail!(
                    "TemporalKernel selected but {} time_col '{}' is not numeric or timestamp",
                    table.name,
                    tc
                ),
            },
            None => bail!(
                "TemporalKernel selected but {} declares missing time_col '{}'",
                table.name,
                tc
            ),
        };
        if t_vec.len() != n {
            bail!(
                "TemporalKernel selected but {} time_col '{}' has {} values for {n} rows",
                table.name,
                tc,
                t_vec.len()
            );
        }
        // The foreign key the panel is keyed on, by name.
        let entity_ids = match table.column(ENTITY_ID_COL) {
            Some(ColumnData::I64(values)) => Some(values.clone()),
            Some(_) => bail!(
                "TemporalKernel selected but {} column '{}' is not i64",
                table.name,
                ENTITY_ID_COL
            ),
            None => {
                // Fall back to the first foreign key as the entity identity.
                let fk_col = table
                    .fkey_col_to_pkey_table
                    .keys()
                    .find(|name| matches!(table.column(name.as_str()), Some(ColumnData::I64(_))))
                    .cloned();
                match fk_col {
                    Some(name) => match table.column(name.as_str()) {
                        Some(ColumnData::I64(values)) => Some(values.clone()),
                        _ => None,
                    },
                    None => None,
                }
            }
        };
        // The per-row weight that makes the neighbour aggregate non-separable.
        let mut rng_w = substream(
            seed + 83_047 * (table_idx as u64 + 1),
            "labels:tempk:coupling",
        );
        let leaked: Vec<String> = table
            .numeric_feature_columns()
            .into_iter()
            .filter(|name| name.starts_with("tempk"))
            .collect();
        if !leaked.is_empty() {
            bail!(
                "TemporalKernel coupling weights would be computed from label \
                 columns {leaked:?} on {}; the projection must not see the \
                 latent it is weighting",
                table.name
            );
        }
        let mut rng_comb = substream(
            seed + 94_007 * (table_idx as u64 + 1),
            "labels:tempk:combine",
        );
        let combine_choice = rng_comb.choose_index(11);
        let mut rng_mix = substream(seed + 98_009 * (table_idx as u64 + 1), "labels:tempk:mix");
        let draw_mix = |rng: &mut GenRng| -> [f64; 4] {
            // Gamma(0.5) marginals via a squared-normal, normalised: a cheap
            // Dirichlet(0.5,...) that needs no gamma sampler.
            let mut raw = [0.0_f64; 4];
            let mut total = 0.0_f64;
            for w in raw.iter_mut() {
                let z = rng.standard_normal();
                *w = z * z + 1e-6;
                total += *w;
            }
            for w in raw.iter_mut() {
                *w /= total;
            }
            [raw[0] * 4.0, raw[1] * 4.0, raw[2] * 4.0, raw[3] * 4.0]
        };
        let mix_weights: [f64; 4] = draw_mix(&mut rng_mix);
        let mut rng_op = substream(seed + 97_003 * (table_idx as u64 + 1), "labels:tempk:op");
        let op_choice = rng_op.choose_index(12);
        let mut projection = project_features(table, &mut rng_w);
        standardize_inplace(&mut projection);
        // The raw standardized projection IS the score; sharpness turns it into
        // a share.
        let coupling_weights: Vec<f64> = projection.clone();
        // Per-table latent, drawn from the mean end to the pointer end.
        const SHARPNESS: [f64; 3] = [2.0, 6.0, 16.0];
        let nb_sharpness = SHARPNESS[rng_w.choose_index(SHARPNESS.len())];
        let uniform_weights = vec![0.0_f64; n];

        // Borrowed rather than moved: the coupled variant below re-runs the same recursion with
        // feature-derived weights and needs the same entity identities.
        let bag_target = |mix: [f64; 4], phi: f64| -> Result<Vec<f64>> {
            let mut y = if let Some(entity_ids) = entity_ids.as_ref() {
                temporal_kernel_by_entity(
                    table,
                    &t_vec,
                    entity_ids,
                    beta,
                    omega,
                    phi,
                    w_frac,
                    seed,
                    table_idx,
                    &NeighbourAgg {
                        weights: &uniform_weights,
                        sharpness: 0.0,
                        alpha: 1.0,
                        split_entity: false,
                        combine: combine_choice,
                        op: op_choice,
                        mix,
                        times: &t_vec,
                    },
                )?
            } else {
                // Rank-normalize time → t̃ ∈ [0, 1].
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by(|a, b| t_vec[*a].total_cmp(&t_vec[*b]).then_with(|| a.cmp(b)));
                let mut t_norm = vec![0.0_f64; n];
                let mut rank_pos = vec![0_usize; n];
                for (rank, &row) in order.iter().enumerate() {
                    t_norm[row] = rank as f64 / (n as f64 - 1.0).max(1.0);
                    rank_pos[row] = rank;
                }
                // FK-sibling edges for neighbour-history term.
                let edges = collect_fk_sibling_edges(table);
                let mut neighbours: Vec<Vec<usize>> = vec![Vec::new(); n];
                for (i, j) in &edges {
                    neighbours[*i].push(*j);
                }
                // Window size in ranks.
                let w = ((n as f64) * w_frac).max(1.0) as usize;

                // Compute label sequentially in time-rank order.
                let mut y = vec![0.0_f64; n];
                // Random base shock per row (bounded) so the equation isn't degenerate.
                let mut rng_r =
                    substream(seed + 41_113 * (table_idx as u64 + 1), "labels:tempk:row");
                let shocks: Vec<f64> = (0..n).map(|_| rng_r.standard_normal() * 0.1).collect();
                let relational = if mix[3] > 0.0 { 0.5 } else { 0.0 };
                for (rank, &row) in order.iter().enumerate() {
                    let trend = beta * t_norm[row];
                    let season = (2.0 * std::f64::consts::PI * t_norm[row] * omega).sin();
                    let prev_y = if rank == 0 { 0.0 } else { y[order[rank - 1]] };
                    // Window-mean of neighbour history via rank-position lookup.
                    let mut nb_sum = 0.0_f64;
                    let mut nb_cnt = 0_usize;
                    let lo_rank = rank.saturating_sub(w);
                    for &nb in &neighbours[row] {
                        let nb_rank = rank_pos[nb];
                        if nb_rank >= lo_rank && nb_rank < rank {
                            nb_sum += y[nb];
                            nb_cnt += 1;
                        }
                    }
                    let nb_mean = if nb_cnt > 0 {
                        nb_sum / nb_cnt as f64
                    } else {
                        0.0
                    };
                    y[row] = trend + season + phi * prev_y + relational * nb_mean + shocks[row];
                }
                y
            };
            standardize_inplace(&mut y);
            Ok(y)
        };
        let mut mix_weights = mix_weights;
        let mut y = bag_target(mix_weights, phi)?;
        let mut y_nolabel = bag_target(label_free_mix(mix_weights), 0.0)?;
        // Regression transfers the value itself, unlike classification where the class index is an
        // arbitrary symbol, so the target's marginal shape is part of the prior.
        let mut rng_link = substream(seed + 91_009 * (table_idx as u64 + 1), "labels:tempk:link");
        // Match the skew QUANTILES, not the maximum.
        let zero_inflation_draw = if rng_link.bernoulli(0.5) {
            rng_link.f64_range(0.2, 0.9)
        } else {
            0.0
        };
        // Target mechanism class, drawn per table.
        let mut rng_mech = substream(seed + 93_011 * (table_idx as u64 + 1), "labels:tempk:mech");
        let mech_choice = rng_mech.choose_index(12);
        let mech_a = rng_mech.f64_range(0.5, 2.0);
        let mech_b = rng_mech.f64_range(-1.0, 1.0);
        let mech_bins = 3 + rng_mech.choose_index(6);
        let mech_bin_values: Vec<f64> = (0..mech_bins)
            .map(|_| rng_mech.f64_range(-2.0, 2.0))
            .collect();
        let mech_freqs: Vec<(f64, f64, f64)> = (0..3)
            .map(|_| {
                (
                    rng_mech.f64_range(0.5, 3.0),
                    rng_mech.f64_range(0.0, 6.283_185_307_179_586),
                    rng_mech.f64_range(-1.0, 1.0),
                )
            })
            .collect();
        // Quantile cut points for the piecewise-constant class are taken from
        // the latent's own order statistics, so the bins are populated whatever
        // the aggregate's scale happens to be.
        let mech = TargetMechanism {
            choice: mech_choice,
            a: mech_a,
            b: mech_b,
            bins: mech_bins,
            bin_values: mech_bin_values,
            freqs: mech_freqs,
        };
        let mech_apply = |values: &mut Vec<f64>| mech.apply(values);
        let mut rng_loc = substream(
            seed + 95_003 * (table_idx as u64 + 1),
            "labels:tempk:location",
        );
        let location_family = rng_loc.bernoulli(0.15);
        let loc_offset = rng_loc.f64_range(200.0, 3000.0);
        // The band is set to COVER the region, not to hit a number.
        let loc_spread = loc_offset * rng_loc.f64_range(0.002, 0.06);
        // The proper-fraction combination class produces a latent that IS a rate in [0,1].
        let link_choice = if combine_choice == 10 {
            7
        } else {
            rng_link.choose_index(10)
        };
        let heavy = link_choice >= 8 || zero_inflation_draw >= 0.2;
        let sigma = if heavy {
            rng_link.f64_range(1.2, 3.0)
        } else {
            rng_link.f64_range(0.15, 0.6)
        };
        let scale = rng_link.f64_range(1.0, 20.0);
        let floor_at_one = rng_link.bernoulli(0.3);
        // Zero inflation, drawn per table.
        let zero_inflation = zero_inflation_draw;
        // Extracted so the coupled variant can be pushed through the IDENTICAL link.
        let apply_link = |values: &mut Vec<f64>| {
            for value in values.iter_mut() {
                let z = *value;
                if location_family {
                    *value = (loc_offset + loc_spread * z).round();
                    continue;
                }
                *value = match link_choice {
                    // counts with a zero floor
                    0 | 1 | 2 | 3 | 4 => (scale * (sigma * z).exp()).round(),
                    // ranks/positions, floor at one
                    5 | 6 => (scale * (0.25 * z).exp()).round() + 1.0,
                    // bounded rates in [0,1]
                    7 => 1.0 / (1.0 + (0.0 - z).exp()),
                    // the heavy tail
                    _ => scale * (sigma * z).exp(),
                };
                if floor_at_one && *value < 1.0 {
                    *value = 1.0;
                }
            }
            // Applied AFTER the link and the floor, so the surviving values keep the shape the link
            // gave them and the zeros are a separate mass rather than a squashed tail.
            if zero_inflation > 0.0 && !location_family {
                for (row, value) in values.iter_mut().enumerate() {
                    let mut rng_z = substream(
                        seed + 97_001 * (table_idx as u64 + 1) + row as u64,
                        "labels:tempk:zeroinfl",
                    );
                    if rng_z.bernoulli(zero_inflation) {
                        *value = 0.0;
                    }
                }
            }
        };
        // Keep the pre-link latent for the classification views.
        mech_apply(&mut y);
        mech_apply(&mut y_nolabel);
        // The gates read the PRE-LINK latent, which is the object all three published views are cut
        // from and the one the family's own shared-latent claim is about.
        let neutral_stem =
            format!("{TEMPK_V1_PREFIX}_b{beta:+.1}_o{omega:.0}_p{phi:.1}_w{w_frac:.2}");
        let latent_stem = neutral_stem.clone();
        let tempk_siblings = db.declared_targets_on_table(&db.tables[table_idx].name);
        let mut verdicts = tempk_verdicts(
            &db.tables[table_idx],
            &t_vec,
            omega,
            &latent_stem,
            &y,
            &y_nolabel,
            0,
            &tempk_siblings,
        );
        let mut rejected = 0usize;
        while enforce_gates
            && !crate::shortcut_gates::all_passed(&verdicts)
            && rejected < TEMPK_MIX_ATTEMPTS - 1
        {
            rejected += 1;
            mix_weights = draw_mix(&mut rng_mix);
            let mut retry = bag_target(mix_weights, phi)?;
            let mut retry_nolabel = bag_target(label_free_mix(mix_weights), 0.0)?;
            mech_apply(&mut retry);
            mech_apply(&mut retry_nolabel);
            verdicts = tempk_verdicts(
                &db.tables[table_idx],
                &t_vec,
                omega,
                &latent_stem,
                &retry,
                &retry_nolabel,
                rejected,
                &tempk_siblings,
            );
            y = retry;
            y_nolabel = retry_nolabel;
        }
        let gates_cleared = !enforce_gates || crate::shortcut_gates::all_passed(&verdicts);
        // The prefix every column below is published under.
        let stem = tempk_routed_prefix(&verdicts).unwrap_or_else(|| TEMPK_V1_PREFIX.to_string());
        // Named for the routed STEM, never for a column: the columns append
        // the target table and a `next_column_name` suffix. Kept as a range
        // and claimed by each published readout below.
        let tempk_verdict_range = db.record_shortcut_verdicts(verdicts);
        if !gates_cleared {
            continue;
        }
        let _ = &y_nolabel;
        let mut pending: Vec<(String, ColumnData, Option<usize>)> = Vec::new();
        let latent = y.clone();
        apply_link(&mut y);
        let labels: Vec<Option<f64>> = y.iter().map(|v| Some(*v)).collect();
        let base = format!(
            "{stem}_b{beta:+.1}_o{omega:.0}_p{phi:.1}_w{w_frac:.2}_{}",
            db.tables[table_idx].name
        );
        let col_name = db.tables[table_idx].next_column_name(&base);
        pending.push((col_name, ColumnData::F64(labels), None));
        // Paired binary view of the same temporal process.
        let mut sorted = latent.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let mut rng_k = substream(seed + 77_003 * (table_idx as u64 + 1), "labels:tempk:k");
        let k_choices = [2usize, 2, 3, 3, 4, 4, 5, 6, 8, 10];
        let k = k_choices[rng_k.choose_index(k_choices.len())];
        let cuts: Vec<f64> = (1..k).map(|i| sorted[(sorted.len() * i) / k]).collect();
        let clf: Vec<i64> = latent
            .iter()
            .map(|v| cuts.iter().filter(|c| *v > **c).count() as i64)
            .collect();
        let clf_base = format!(
            "{stem}clf_k{k}_b{beta:+.1}_o{omega:.0}_p{phi:.1}_w{w_frac:.2}_{}",
            db.tables[table_idx].name
        );
        let clf_name = db.tables[table_idx].next_column_name(&clf_base);
        pending.push((clf_name, ColumnData::I64(clf), Some(k)));

        let mut rng_imb = substream(
            seed + 61_007 * (table_idx as u64 + 1),
            "labels:tempk:imbalance",
        );
        let minority = {
            let lo = 0.002_f64.ln();
            let hi = 0.50_f64.ln();
            (lo + (hi - lo) * rng_imb.f64_range(0.0, 1.0)).exp()
        };
        let majority = 1.0 - minority;
        let cut_index = (((sorted.len() as f64) * majority) as usize).min(sorted.len() - 1);
        let cut = sorted[cut_index];
        // Draw WHICH class is the majority, not only how large it is.
        let positive_is_majority = rng_imb.bernoulli(0.5);
        let imbalanced: Vec<i64> = latent
            .iter()
            .map(|v| i64::from((*v > cut) != positive_is_majority))
            .collect();
        // Ties in `y` can push every row to one side, which rustler refuses
        // downstream; emitting a single-class column would be a silent defect.
        let positives = imbalanced.iter().filter(|v| **v == 1).count();
        if positives > 0 && positives < imbalanced.len() {
            let imb_base = format!(
                "{stem}imb_k2_q{majority:.2}_posmaj{}_b{beta:+.1}_o{omega:.0}_p{phi:.1}_w{w_frac:.2}_{}",
                i32::from(positive_is_majority),
                db.tables[table_idx].name
            );
            let imb_name = db.tables[table_idx].next_column_name(&imb_base);
            pending.push((imb_name, ColumnData::I64(imbalanced), Some(2)));
        }

        if let Some(entity_ids) = entity_ids.as_ref() {
            let mut y_coupled = temporal_kernel_by_entity(
                &db.tables[table_idx],
                &t_vec,
                entity_ids,
                beta,
                omega,
                phi,
                w_frac,
                seed,
                table_idx,
                &NeighbourAgg {
                    weights: &coupling_weights,
                    sharpness: nb_sharpness,
                    alpha: 0.0,
                    split_entity: false,
                    combine: combine_choice,
                    op: op_choice,
                    mix: mix_weights,
                    times: &t_vec,
                },
            )?;
            // Same chain as the uniform side, in the same order: standardize, then the identical
            // link, then quantile cuts.
            standardize_inplace(&mut y_coupled);
            mech_apply(&mut y_coupled);
            // Same split as the uniform side: the classification views are cut from the PRE-link
            // latent and only the regression view goes through the link.
            let latent_coupled = y_coupled.clone();
            apply_link(&mut y_coupled);
            let cpl_labels: Vec<Option<f64>> = y_coupled.iter().map(|v| Some(*v)).collect();
            let cplreg_base = format!(
                "{stem}cplreg_b{beta:+.1}_o{omega:.0}_p{phi:.1}_w{w_frac:.2}_{}",
                db.tables[table_idx].name
            );
            let cplreg_name = db.tables[table_idx].next_column_name(&cplreg_base);
            pending.push((cplreg_name, ColumnData::F64(cpl_labels), None));

            let mut sorted_c = latent_coupled.clone();
            sorted_c.sort_by(|a, b| a.total_cmp(b));
            let cuts_c: Vec<f64> = (1..k).map(|i| sorted_c[(sorted_c.len() * i) / k]).collect();
            let clf_c: Vec<i64> = latent_coupled
                .iter()
                .map(|v| cuts_c.iter().filter(|c| *v > **c).count() as i64)
                .collect();
            let cpl_base = format!(
                "{stem}cpl_k{k}_b{beta:+.1}_o{omega:.0}_p{phi:.1}_w{w_frac:.2}_{}",
                db.tables[table_idx].name
            );
            let cpl_name = db.tables[table_idx].next_column_name(&cpl_base);
            pending.push((cpl_name, ColumnData::I64(clf_c), Some(k)));

            let cut_c = sorted_c[cut_index];
            let imb_c: Vec<i64> = latent_coupled
                .iter()
                .map(|v| i64::from((*v > cut_c) != positive_is_majority))
                .collect();
            let positives_c = imb_c.iter().filter(|v| **v == 1).count();
            if positives_c > 0 && positives_c < imb_c.len() {
                let cimb_base = format!(
                    "{stem}cplimb_k2_q{majority:.2}_posmaj{}_b{beta:+.1}_o{omega:.0}_p{phi:.1}_w{w_frac:.2}_{}",
                    i32::from(positive_is_majority),
                    db.tables[table_idx].name
                );
                let cimb_name = db.tables[table_idx].next_column_name(&cimb_base);
                pending.push((cimb_name, ColumnData::I64(imb_c), Some(2)));
            }
        }

        // The dual family: BOTH channels at a drawn ratio, and own-history and cross-entity rows
        // aggregated as separate terms.
        if let Some(entity_ids) = entity_ids.as_ref() {
            const ALPHA: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
            let mut rng_a = substream(
                seed + 91_393 * (table_idx as u64 + 1),
                "labels:tempk:dual_alpha",
            );
            let alpha = ALPHA[rng_a.choose_index(ALPHA.len())];
            let mut y_dual = temporal_kernel_by_entity(
                &db.tables[table_idx],
                &t_vec,
                entity_ids,
                beta,
                omega,
                phi,
                w_frac,
                seed,
                table_idx,
                &NeighbourAgg {
                    weights: &coupling_weights,
                    sharpness: nb_sharpness,
                    alpha,
                    split_entity: true,
                    combine: combine_choice,
                    op: op_choice,
                    mix: mix_weights,
                    times: &t_vec,
                },
            )?;
            standardize_inplace(&mut y_dual);
            mech_apply(&mut y_dual);
            let latent_dual = y_dual.clone();
            apply_link(&mut y_dual);
            let dual_labels: Vec<Option<f64>> = y_dual.iter().map(|v| Some(*v)).collect();
            let dreg_base = format!(
                "{stem}dualreg_a{alpha:.2}_b{beta:+.1}_o{omega:.0}_p{phi:.1}_w{w_frac:.2}_{}",
                db.tables[table_idx].name
            );
            let dreg_name = db.tables[table_idx].next_column_name(&dreg_base);
            pending.push((dreg_name, ColumnData::F64(dual_labels), None));

            let mut sorted_d = latent_dual.clone();
            sorted_d.sort_by(|a, b| a.total_cmp(b));
            let cuts_d: Vec<f64> = (1..k).map(|i| sorted_d[(sorted_d.len() * i) / k]).collect();
            let clf_d: Vec<i64> = latent_dual
                .iter()
                .map(|v| cuts_d.iter().filter(|c| *v > **c).count() as i64)
                .collect();
            let dual_base = format!(
                "{stem}dual_k{k}_a{alpha:.2}_b{beta:+.1}_o{omega:.0}_p{phi:.1}_w{w_frac:.2}_{}",
                db.tables[table_idx].name
            );
            let dual_name = db.tables[table_idx].next_column_name(&dual_base);
            pending.push((dual_name, ColumnData::I64(clf_d), Some(k)));

            let cut_d = sorted_d[cut_index];
            let imb_d: Vec<i64> = latent_dual
                .iter()
                .map(|v| i64::from((*v > cut_d) != positive_is_majority))
                .collect();
            let positives_d = imb_d.iter().filter(|v| **v == 1).count();
            if positives_d > 0 && positives_d < imb_d.len() {
                let dimb_base = format!(
                    "{stem}dualimb_k2_a{alpha:.2}_q{majority:.2}_posmaj{}_b{beta:+.1}_o{omega:.0}_p{phi:.1}_w{w_frac:.2}_{}",
                    i32::from(positive_is_majority),
                    db.tables[table_idx].name
                );
                let dimb_name = db.tables[table_idx].next_column_name(&dimb_base);
                pending.push((dimb_name, ColumnData::I64(imb_d), Some(2)));
            }
        }

        // Every PUBLISHED column carries its own disclosure verdict.
        let mut column_verdicts = Vec::with_capacity(pending.len());
        {
            let table = &db.tables[table_idx];
            // A published target is never in another target's context.
            let mut excluded: Vec<String> = table
                .column_names()
                .filter(|name| crate::scm::FeatureContract::is_published_target(name))
                .map(str::to_string)
                .collect();
            excluded.extend(db.declared_targets_on_table(&table.name));
            excluded.extend(pending.iter().map(|(name, _, _)| name.clone()));
            excluded.sort();
            excluded.dedup();
            for (name, data, declared_k) in &pending {
                // The readout type picks the estimator, and each estimator is the CANONICAL one for
                // its type -- never one re-derived here.
                if let (ColumnData::I64(values), Some(k)) = (data, declared_k) {
                    if values.len() >= crate::shortcut_gates::MIN_GATEABLE_ROWS {
                        let budget = crate::shortcut_gates::empirical_entropy(values, *k);
                        // The function `evaluate_discrete_target` itself calls.
                        let mut discrete = crate::shortcut_gates::discrete_disclosure_verdict(
                            "tempk", "bag", table, name, values, *k, budget, &excluded,
                        );
                        discrete.mechanism = Some("temporal_kernel_published_column".to_string());
                        discrete.candidates_rejected = Some(rejected);
                        column_verdicts.push(discrete);
                    }
                    continue;
                }
                let ColumnData::F64(_) = data else {
                    continue;
                };
                let Some(values) = data.as_numeric_lossy() else {
                    continue;
                };
                let rows = values.len();
                if rows < crate::shortcut_gates::MIN_GATEABLE_ROWS {
                    continue;
                }
                let m = crate::shortcut_gates::continuous_bins(rows);
                let bins = crate::shortcut_gates::balanced_bins(&values, m);
                let budget = crate::shortcut_gates::empirical_entropy(&bins, m);
                let (measured, note) = match crate::shortcut_gates::strongest_column_disclosure(
                    table, &bins, m, &excluded,
                ) {
                    Some(measurement) => (
                        measurement.information_nats,
                        format!(
                            "balanced {m}-bin view of the PUBLISHED column, not of the \
                                 latent it was cut from; strongest of {} in-context column(s) \
                                 is {}",
                            measurement.columns_scanned, measurement.column,
                        ),
                    ),
                    None => (
                        0.0,
                        "the target table carries no other numeric column left in context"
                            .to_string(),
                    ),
                };
                let ratio = if budget > 0.0 { measured / budget } else { 0.0 };
                column_verdicts.push(crate::shortcut_gates::ShortcutVerdict {
                    gate: crate::shortcut_gates::GateKind::SiblingColumnDisclosure,
                    family: "tempk".to_string(),
                    arm: "bag".to_string(),
                    target_table: table.name.clone(),
                    target: name.clone(),
                    k: m,
                    measured_nats: measured,
                    budget_nats: budget,
                    ratio,
                    threshold_ratio: crate::shortcut_gates::SHORTCUT_CEILING,
                    passed: ratio <= crate::shortcut_gates::SHORTCUT_CEILING + 1e-12,
                    declared_exception: None,
                    mechanism: Some("temporal_kernel_published_column".to_string()),
                    candidates_rejected: Some(rejected),
                    read_as_coordinate_not_verdict: None,
                    note,
                    feature_contract: table.feature_contract,
                    bag_summary: table.bag_summary,
                });
            }
        }
        // Which latent group a pending column belongs to. `stem` is the
        // published prefix, so what follows it is the group tag.
        let group_of = |name: &str| -> u8 {
            let rest = name.strip_prefix(stem.as_str()).unwrap_or(name);
            if rest.starts_with("cpl") {
                1
            } else if rest.starts_with("dual") {
                2
            } else {
                0
            }
        };
        // A group is refused when one of ITS continuous columns is refused.
        let mut refused = [false; 3];
        for verdict in &column_verdicts {
            if !verdict.passed {
                refused[group_of(&verdict.target) as usize] = true;
            }
        }
        // base and cpl are a matched pair and stand or fall together; dual is
        // independent.
        let pair_refused = refused[0] || refused[1];
        let withheld = |group: u8| -> bool {
            enforce_gates && if group == 2 { refused[2] } else { pair_refused }
        };
        let column_verdict_range = db.record_shortcut_verdicts(column_verdicts);
        let mut published_any = false;
        for (name, data, _) in pending {
            if withheld(group_of(&name)) {
                continue;
            }
            db.tables[table_idx].push_column(name.clone(), data);
            db.claim_gate_verdicts(tempk_verdict_range.clone(), &name);
            db.claim_gate_verdicts_named(column_verdict_range.clone(), &name, &name);
            published_any = true;
        }
        if !published_any {
            continue;
        }

        emitted += 1;
    }
    Ok((emitted, inapplicable))
}

/// How the neighbour-label term is aggregated.
#[derive(Clone, Copy)]
struct NeighbourAgg<'a> {
    /// Per-row score feeding the softmax. Ignored when `sharpness` is 0.
    weights: &'a [f64],
    /// Softmax temperature. 0 gives every neighbour an equal share.
    sharpness: f64,
    /// Weight on the SET channel; `1 - alpha` goes to the BINDING channel.
    alpha: f64,
    /// Aggregate own-entity and cross-entity neighbours as separate terms.
    split_entity: bool,
    /// How the label aggregate combines with the two structural aggregates.
    combine: usize,
    /// Which aggregation operator reads the neighbour list.
    op: usize,
    /// Mixing weights over [trend, season, persistence, relational].
    mix: [f64; 4],
    /// Timestamps, indexed like `weights`.
    times: &'a [f64],
}

impl NeighbourAgg<'_> {
    /// Set and binding readouts over one neighbour list.

    fn readout(&self, rows: &[usize], y: &[f64]) -> Option<f64> {
        if rows.is_empty() {
            return None;
        }
        let n = rows.len() as f64;
        let vals: Vec<f64> = rows.iter().map(|r| y[*r]).collect();
        let mean = vals.iter().sum::<f64>() / n;

        // softmax shares over the feature projection, shared by the binding
        // forms that need a weighting rather than a selection
        let mut peak = f64::NEG_INFINITY;
        for &r in rows {
            let score = self.sharpness * self.weights[r];
            if score > peak {
                peak = score;
            }
        }
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for &r in rows {
            let share = (self.sharpness * self.weights[r] - peak).exp();
            num += share * y[r];
            den += share;
        }
        let wmean = if den > 1e-12 { num / den } else { mean };

        #[allow(unused_variables)]
        let arg_hi = rows.iter().copied().fold(rows[0], |acc, r| {
            if self.weights[r] > self.weights[acc] {
                r
            } else {
                acc
            }
        });
        let vmax = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let vmin = vals.iter().copied().fold(f64::INFINITY, f64::min);
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;

        // top-k by feature versus top-k by label value: the same k, selected two
        // different ways, which is precisely the set/binding distinction
        let k = ((n * 0.3).round() as usize).clamp(1, rows.len());
        let mut by_val = vals.clone();
        by_val.sort_by(|a, b| b.total_cmp(a));
        let top_by_val = by_val[..k].iter().sum::<f64>() / k as f64;
        let mut by_w: Vec<usize> = rows.to_vec();
        by_w.sort_by(|a, b| self.weights[*b].total_cmp(&self.weights[*a]));
        let top_by_w = by_w[..k].iter().map(|r| y[*r]).sum::<f64>() / k as f64;

        // SOFT selection, not hard selection.
        let mut lo_peak = f64::NEG_INFINITY;
        for &r in rows {
            let sc = -self.sharpness * self.weights[r];
            if sc > lo_peak {
                lo_peak = sc;
            }
        }
        let (mut lnum, mut lden) = (0.0_f64, 0.0_f64);
        for &r in rows {
            let sh = (-self.sharpness * self.weights[r] - lo_peak).exp();
            lnum += sh * y[r];
            lden += sh;
        }
        let soft_hi = wmean;
        let soft_lo = if lden > 1e-12 { lnum / lden } else { mean };

        let (set, bind) = match self.op {
            0 => (mean, wmean),
            // sum -- scale carries the neighbourhood size, which a mean discards
            1 => (mean * n, wmean * n),
            // extremes: the set form takes the largest LABEL, the binding form
            // takes the label of the largest-FEATURE row. Same shape, opposite
            // sides of the hierarchy.
            2 => (vmax, soft_hi),
            3 => (vmin, soft_lo),
            // dispersion
            4 => (var.sqrt(), (soft_hi - soft_lo).abs()),
            // range
            5 => (vmax - vmin, soft_hi - soft_lo),
            // top-k, selected by label value versus by feature
            6 => (top_by_val, top_by_w),
            // Count above a FIXED threshold, not above the neighbourhood's own median.
            7 => {
                let c_set = vals.iter().filter(|v| **v > 0.0).count() as f64;
                let c_bind = rows
                    .iter()
                    .filter(|r| y[**r] > 0.0 && self.weights[**r] > 0.0)
                    .count() as f64;
                (c_set, c_bind)
            }
            // second moment about zero
            8 => (
                vals.iter().map(|v| v * v).sum::<f64>() / n,
                soft_hi * soft_hi,
            ),
            // the difference between the two selection rules, which is zero for
            // any predictor that cannot tell them apart
            9 => (mean, top_by_w - top_by_val + mean),
            // TEMPORAL DECAY.
            10 => {
                let t_ref = rows
                    .iter()
                    .map(|r| self.times[*r])
                    .fold(f64::NEG_INFINITY, f64::max);
                let (mut num, mut den) = (0.0_f64, 0.0_f64);
                for &r in rows {
                    let w = (-(t_ref - self.times[r]).abs() * 2.0).exp();
                    num += w * y[r];
                    den += w;
                }
                (mean, if den > 1e-12 { num / den } else { mean })
            }
            // MOST RECENT LABEL.
            _ => {
                let latest = rows.iter().copied().fold(rows[0], |acc, r| {
                    if self.times[r] > self.times[acc] {
                        r
                    } else {
                        acc
                    }
                });
                (mean, y[latest])
            }
        };
        if self.alpha >= 1.0 {
            return Some(set);
        }
        Some(self.alpha * set + (1.0 - self.alpha) * bind)
    }
}

fn temporal_kernel_by_entity(
    table: &TableData,
    t_vec: &[f64],
    entity_ids: &[i64],
    beta: f64,
    omega: f64,
    phi: f64,
    w_frac: f64,
    seed: u64,
    table_idx: usize,
    // Per-row weights for the neighbour-label aggregate.
    agg: &NeighbourAgg<'_>,
) -> Result<Vec<f64>> {
    let n = table.num_rows();
    if agg.weights.len() != n {
        bail!(
            "TemporalKernel neighbour weights have {} entries for {n} rows in {}",
            agg.weights.len(),
            table.name
        );
    }
    if entity_ids.len() != n {
        bail!(
            "TemporalKernel selected but {} column '{}' has {} values for {n} rows",
            table.name,
            ENTITY_ID_COL,
            entity_ids.len()
        );
    }

    let mut rows_by_entity: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (row, entity_id) in entity_ids.iter().copied().enumerate() {
        rows_by_entity.entry(entity_id).or_default().push(row);
    }
    let with_history = rows_by_entity
        .values()
        .filter(|rows| rows.len() >= 2)
        .count();
    if with_history == 0 {
        bail!(
            "TemporalKernel selected but no entity in {} has two observations; \
             a table of singletons carries no temporal binding",
            table.name
        );
    }

    let mut t_norm = vec![0.0_f64; n];
    let mut entity_rank = vec![0usize; n];
    let mut window = vec![1usize; n];
    for rows in rows_by_entity.values_mut() {
        rows.sort_by(|a, b| t_vec[*a].total_cmp(&t_vec[*b]).then_with(|| a.cmp(b)));
        let denom = (rows.len() as f64 - 1.0).max(1.0);
        let entity_window = ((rows.len() as f64) * w_frac).max(1.0) as usize;
        for (rank, &row) in rows.iter().enumerate() {
            t_norm[row] = rank as f64 / denom;
            entity_rank[row] = rank;
            window[row] = entity_window;
        }
    }

    let mut rng_hop = substream(seed + 99_013 * (table_idx as u64 + 1), "labels:tempk:hop");
    let two_hop = rng_hop.bernoulli(0.25);
    let edges = if two_hop {
        let e = collect_two_hop_edges(table);
        if e.is_empty() {
            collect_fk_sibling_edges(table)
        } else {
            e
        }
    } else {
        collect_fk_sibling_edges(table)
    };
    let mut neighbours: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, j) in &edges {
        neighbours[*i].push(*j);
    }

    let mut exposure = vec![0.0_f64; n];
    let mut nb_feat = vec![0.0_f64; n];
    // A PROPER FRACTION: how many of the time-valid neighbours clear a per-table feature threshold,
    // over how many there are.
    let mut nested_frac = vec![0.0_f64; n];
    for row in 0..n {
        let mut count = 0.0_f64;
        let mut fsum = 0.0_f64;
        for &nb in &neighbours[row] {
            if t_vec[nb] < t_vec[row] {
                count += 1.0;
                fsum += agg.weights[nb];
            }
        }
        exposure[row] = count;
        nb_feat[row] = if count > 0.0 { fsum / count } else { 0.0 };
        let mut hits = 0.0_f64;
        for &nb in &neighbours[row] {
            if t_vec[nb] < t_vec[row] && agg.weights[nb] > 0.0 {
                hits += 1.0;
            }
        }
        nested_frac[row] = if count > 0.0 { hits / count } else { 0.5 };
    }
    standardize_inplace(&mut exposure);
    standardize_inplace(&mut nb_feat);
    // not standardized: it stays a proper fraction in [0,1]

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| t_vec[*a].total_cmp(&t_vec[*b]).then_with(|| a.cmp(b)));
    let mut rng_r = substream(seed + 41_113 * (table_idx as u64 + 1), "labels:tempk:row");
    let shocks: Vec<f64> = (0..n).map(|_| rng_r.standard_normal() * 0.1).collect();
    let mut previous_by_entity: HashMap<i64, f64> = HashMap::new();
    let mut y = vec![0.0_f64; n];
    for &row in &order {
        let entity_id = entity_ids[row];
        let trend = beta * t_norm[row];
        let season = (2.0 * std::f64::consts::PI * t_norm[row] * omega).sin();
        let prev_y = previous_by_entity.get(&entity_id).copied().unwrap_or(0.0);
        let rank = entity_rank[row];
        let lo_rank = rank.saturating_sub(window[row]);
        // Weighted mean over sibling past labels.
        let mut own_rows: Vec<usize> = Vec::new();
        let mut xent_rows: Vec<usize> = Vec::new();
        for &nb in &neighbours[row] {
            if t_vec[nb] >= t_vec[row] {
                continue;
            }
            if entity_ids[nb] == entity_id {
                let nb_rank = entity_rank[nb];
                if nb_rank >= lo_rank && nb_rank < rank {
                    own_rows.push(nb);
                }
            } else if agg.split_entity {
                xent_rows.push(nb);
            }
        }
        if !agg.split_entity {
            // Single pooled term, which is what the bag and coupled families
            // use; cross-entity rows join the same list.
            for &nb in &neighbours[row] {
                if t_vec[nb] < t_vec[row] && entity_ids[nb] != entity_id {
                    let nb_rank = entity_rank[nb];
                    if nb_rank >= lo_rank && nb_rank < rank {
                        own_rows.push(nb);
                    }
                }
            }
        }
        let own_term = agg.readout(&own_rows, &y);
        let xent_term = if agg.split_entity {
            agg.readout(&xent_rows, &y)
        } else {
            None
        };
        let present = own_term.is_some() as u32 + xent_term.is_some() as u32;
        let nb_total = if present > 0 {
            (own_term.unwrap_or(0.0) + xent_term.unwrap_or(0.0)) * agg.mix[3] / present as f64
        } else {
            0.0
        };
        // The combination class decides HOW the label aggregate meets the two structural ones.
        let base =
            agg.mix[0] * trend + agg.mix[1] * season + agg.mix[2] * phi * prev_y + shocks[row];
        let e = exposure[row];
        let f = nb_feat[row];
        y[row] = match agg.combine {
            // additive
            0..=5 => base + nb_total,
            // ratio against exposure -- the shape of a rate: an event aggregate
            // over an exposure count. The denominator is offset away from zero
            // so the map stays finite.
            6 => base + nb_total / (1.5 + e.abs()),
            // product with exposure: the label term matters more where the
            // neighbourhood is large, an interaction no sum can express
            7 => base + nb_total * (0.5 + 0.5 * e),
            // gated by the feature aggregate: the sign of the label term flips
            // on a feature threshold
            8 => base + if f > 0.0 { nb_total } else { -nb_total },
            // ratio against the feature aggregate
            9 => base + nb_total / (1.5 + f.abs()),
            _ => {
                let r = nested_frac[row] + 0.15 * nb_total;
                r.clamp(0.0, 1.0)
            }
        };
        previous_by_entity.insert(entity_id, y[row]);
    }

    // A STATIC ENTITY ATTRIBUTE: one value per entity, constant across all its rows and across
    // time.
    let mut rng_static = substream(
        seed + 96_017 * (table_idx as u64 + 1),
        "labels:tempk:static",
    );
    if rng_static.bernoulli(0.12) {
        let mut sum_by_entity: HashMap<i64, (f64, f64)> = HashMap::new();
        for row in 0..n {
            let e = sum_by_entity.entry(entity_ids[row]).or_insert((0.0, 0.0));
            e.0 += y[row];
            e.1 += 1.0;
        }
        for row in 0..n {
            if let Some((total, count)) = sum_by_entity.get(&entity_ids[row]) {
                if *count > 0.0 {
                    y[row] = total / count;
                }
            }
        }
    }
    Ok(y)
}

/// exists_grandchild_gate_sum: per-parent, sum mid_value over mid rows that
/// have AT LEAST ONE grandchild (existence gate).
fn exists_grandchild_gate_sum_all(db: &GeneratedDb) -> Vec<(String, Vec<f64>, LabelMetadata)> {
    let mut out = Vec::new();
    for mid in &db.tables {
        let mid_cols = mid.numeric_feature_columns();
        if mid_cols.is_empty() {
            continue;
        }
        let mid_value_col = mid_cols[0].clone();
        let mid_value = match mid
            .column(&mid_value_col)
            .and_then(ColumnData::as_numeric_lossy)
        {
            Some(v) => v,
            None => continue,
        };
        for (mid_parent_fk, parent_name) in &mid.fkey_col_to_pkey_table {
            let parent = match db.table(parent_name) {
                Some(t) => t,
                None => continue,
            };
            let mid_fk_values = match mid
                .column(mid_parent_fk)
                .and_then(ColumnData::as_numeric_lossy)
            {
                Some(v) => v,
                None => continue,
            };
            for grandchild in &db.tables {
                for (gc_mid_fk, mid_name) in &grandchild.fkey_col_to_pkey_table {
                    if mid_name != &mid.name {
                        continue;
                    }
                    let gc_fk_values = match grandchild
                        .column(gc_mid_fk)
                        .and_then(ColumnData::as_numeric_lossy)
                    {
                        Some(v) => v,
                        None => continue,
                    };
                    let mut has_gc = vec![false; mid.num_rows()];
                    for v in gc_fk_values {
                        let midx = v as usize;
                        if midx < has_gc.len() {
                            has_gc[midx] = true;
                        }
                    }
                    let mut scores = vec![0.0; parent.num_rows()];
                    for row in 0..mid.num_rows() {
                        if !has_gc[row] || !mid_value[row].is_finite() {
                            continue;
                        }
                        let pidx = mid_fk_values[row] as usize;
                        if pidx < parent.num_rows() {
                            scores[pidx] += mid_value[row];
                        }
                    }
                    let metadata = program_metadata(
                        "exists_grandchild_gate_sum",
                        parent_name,
                        vec![mid.name.clone(), grandchild.name.clone()],
                        2,
                        btreemap! {
                            "mid_parent_fk_col" => mid_parent_fk.clone(),
                            "mid_value_col" => mid_value_col.clone(),
                            "grandchild_mid_fk_col" => gc_mid_fk.clone(),
                            "parent_pkey_col" => parent.pkey_col.clone(),
                        },
                    );
                    out.push((parent_name.clone(), scores, metadata));
                }
            }
        }
    }
    out
}

fn gauss_rank_inplace(values: &mut [f64]) {
    let n = values.len();
    if n < 2 {
        return;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
    let mut out = vec![0.0_f64; n];
    for (rank, &idx) in order.iter().enumerate() {
        let q = (rank as f64 + 0.5) / n as f64;
        out[idx] = inverse_standard_normal_cdf(q);
    }
    values.copy_from_slice(&out);
}

/// Acklam's rational approximation to the standard normal quantile, absolute
/// error below 1.15e-9 over the open unit interval. Used only by
/// `gauss_rank_inplace`.
fn inverse_standard_normal_cdf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e+01,
        2.209_460_984_245_205e+02,
        -2.759_285_104_469_687e+02,
        1.383_577_518_672_690e+02,
        -3.066_479_806_614_716e+01,
        2.506_628_277_459_239e+00,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e+01,
        1.615_858_368_580_409e+02,
        -1.556_989_798_598_866e+02,
        6.680_131_188_771_972e+01,
        -1.328_068_155_288_572e+01,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-03,
        -3.223_964_580_411_365e-01,
        -2.400_758_277_161_838e+00,
        -2.549_732_539_343_734e+00,
        4.374_664_141_464_968e+00,
        2.938_163_982_698_783e+00,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-03,
        3.224_671_290_700_398e-01,
        2.445_134_137_142_996e+00,
        3.754_408_661_907_416e+00,
    ];
    const P_LOW: f64 = 0.024_25;
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Target mechanism classes of `add_temporal_kernel_labels`.
pub struct TargetMechanism {
    pub choice: usize,
    pub a: f64,
    pub b: f64,
    pub bins: usize,
    pub bin_values: Vec<f64>,
    pub freqs: Vec<(f64, f64, f64)>,
}

impl TargetMechanism {
    /// True when this draw leaves the latent's ordering untouched.
    pub(crate) fn is_identity(&self) -> bool {
        self.choice < 6
    }

    pub(crate) fn apply(&self, values: &mut Vec<f64>) {
        if self.is_identity() {
            return;
        }
        let cuts: Vec<f64> = if self.choice == 8 {
            let mut sorted = values.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));
            (1..self.bins)
                .map(|i| sorted[(sorted.len() * i) / self.bins])
                .collect()
        } else {
            Vec::new()
        };
        for value in values.iter_mut() {
            let z = *value;
            *value = match self.choice {
                // quadratic: non-monotone with an interior extremum
                6 => self.a * z * z + self.b * z,
                // absolute value: non-monotone with a kink
                7 => self.a * (z - self.b).abs(),
                // piecewise-constant over quantile bins: the tree and
                // discretization class
                8 => {
                    let k = cuts.iter().filter(|c| z > **c).count();
                    self.bin_values[k.min(self.bin_values.len() - 1)]
                }
                // product of the latent with a shifted copy of itself
                9 => z * (z - self.b) * self.a,
                // random Fourier sum, standing in for the Gaussian-process class
                10 => self
                    .freqs
                    .iter()
                    .map(|(w, phi, amp)| amp * (w * z + phi).sin())
                    .sum::<f64>(),
                // nearest-centroid distance: the clustering class
                _ => {
                    let mut best = f64::INFINITY;
                    for (i, c) in self.bin_values.iter().enumerate() {
                        let d = (z - c * (1.0 + i as f64 * 0.1)).abs();
                        if d < best {
                            best = d;
                        }
                    }
                    self.a * best + self.b
                }
            };
        }
        // Gauss-rank, not a z-score.
        gauss_rank_inplace(values);
    }
}
