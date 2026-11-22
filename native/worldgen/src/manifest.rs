use crate::labels::{all_aggfam_candidates, LabelFamily, LabelMetadata};
use crate::rng::{substream, RngExt};
use crate::scm::{ColumnData, GeneratedDb, TableData};
use crate::shortcut_gates::{
    all_passed, attribute, evaluate_discrete_target_with_row_order, evaluate_support_bindings,
    operator_arm, DiscreteTargetGateInput, MECH_FAMILIES, OPERATOR_ARM_POINTER,
    POINTER_ARM_CEILING, POINTER_ARM_DECLARATION, SHORTCUT_CEILING,
};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};

/// Families `worldgen-materialize` builds.
pub const MATERIALIZED_FAMILIES: [&str; 3] = [
    "tabicl_v2_like_discrete",
    "support_required_multiclass",
    "relational_operator_discrete",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixtureManifest {
    pub schema_version: usize,
    pub name: String,
    pub intent_only: bool,
    pub manifest_sha256: String,
    pub allowed_discrete_k: Vec<usize>,
    pub records: Vec<MixtureRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixtureRecord {
    pub split: String,
    pub family: String,
    pub db_name: String,
    pub generation_seed: u64,
    pub opaque_target_name: String,
    pub task_type: String,
    pub target_k: Option<usize>,
    pub class_id_permutation_seed: Option<u64>,
    #[serde(default)]
    pub opaque_target_names: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CorpusScaleCoverage {
    pub rows_per_table: BTreeMap<String, usize>,
    #[serde(default)]
    pub rows_per_tier: BTreeMap<String, usize>,
    #[serde(default)]
    pub cells_per_db: usize,
    #[serde(default)]
    pub nuisance_cell_fraction: f64,
    pub mean_fanout: f64,
    pub p95_fanout: usize,
    pub depth: usize,
    pub label_relevant_cells_within: BTreeMap<String, usize>,
    pub expected_label_relevant_cells_within_target: Option<usize>,
}

pub fn corpus_scale_coverage(
    db: &GeneratedDb,
    target_context_len: Option<usize>,
) -> CorpusScaleCoverage {
    let rows_per_table = db
        .tables
        .iter()
        .map(|table| (table.name.clone(), table.num_rows()))
        .collect();
    let mut rows_per_tier = BTreeMap::new();
    let mut cells_per_db = 0usize;
    let mut nuisance_cells = 0usize;
    let v4_tables: std::collections::BTreeSet<&str> = db
        .program_metadata
        .iter()
        .filter(|program| program.family.starts_with("p2_v4_"))
        .flat_map(|program| program.source_tables.iter().map(String::as_str))
        .collect();
    let v4_tier = db
        .program_metadata
        .iter()
        .find(|program| program.family.starts_with("p2_v4_"))
        .map(|program| program.family.as_str());
    for table in &db.tables {
        let tier = match table.name.as_str() {
            "p2_entity" | "p2_group" => "p2_root",
            "p2_leaf" => "p2_long_range",
            "p2_history" => "p2_long_history",
            "p2_branch" | "p2_twig" => "p2_peer_subtree",
            "p2_nuisance" | "p5_nuisance" => "p2_nuisance",
            "p5_entity" => "p2_root",
            "p5_query" | "p5_history" | "p5_event" => "p2_v5_task",
            name if name.starts_with("p5_") => "p2_v5_evidence",
            name if v4_tables.contains(name) => v4_tier.unwrap_or("p2_v4"),
            _ => "base",
        };
        *rows_per_tier.entry(tier.to_string()).or_default() += table.num_rows();
        let width = table
            .columns
            .iter()
            .filter(|(name, _)| !crate::scm::FeatureContract::is_published_target(name))
            .count();
        let table_cells = table.num_rows().saturating_mul(width);
        cells_per_db = cells_per_db.saturating_add(table_cells);
        nuisance_cells = nuisance_cells.saturating_add(
            if table.name == "p2_nuisance" || table.name == "p5_nuisance" {
                table_cells
            } else {
                table.num_rows().saturating_mul(
                    table
                        .columns
                        .iter()
                        .filter(|(name, _)| {
                            name.starts_with("p2_nuisance_") || name.starts_with("p2_text_")
                        })
                        .count(),
                )
            },
        );
    }
    let nuisance_cell_fraction = nuisance_cells as f64 / cells_per_db.max(1) as f64;
    let table_index: HashMap<&str, usize> = db
        .tables
        .iter()
        .enumerate()
        .map(|(idx, table)| (table.name.as_str(), idx))
        .collect();
    let mut fanouts = Vec::new();
    for child in &db.tables {
        for (fk, parent_name) in &child.fkey_col_to_pkey_table {
            let Some(&parent_idx) = table_index.get(parent_name.as_str()) else {
                continue;
            };
            let mut counts = vec![0usize; db.tables[parent_idx].num_rows()];
            if let Some(ColumnData::I64(values)) = child.column(fk) {
                for &value in values {
                    if value >= 0 {
                        if let Some(count) = counts.get_mut(value as usize) {
                            *count += 1;
                        }
                    }
                }
            }
            fanouts.extend(counts);
        }
    }
    fanouts.sort_unstable();
    let mean_fanout = if fanouts.is_empty() {
        0.0
    } else {
        fanouts.iter().sum::<usize>() as f64 / fanouts.len() as f64
    };
    let p95_fanout = if fanouts.is_empty() {
        0
    } else {
        let rank = ((95 * fanouts.len() + 99) / 100).saturating_sub(1);
        fanouts[rank.min(fanouts.len() - 1)]
    };
    let mut depths = vec![0usize; db.tables.len()];
    for (child_idx, child) in db.tables.iter().enumerate() {
        for parent_name in child.fkey_col_to_pkey_table.values() {
            if let Some(&parent_idx) = table_index.get(parent_name.as_str()) {
                depths[child_idx] = depths[child_idx].max(depths[parent_idx] + 1);
            }
        }
    }
    let depth = depths.into_iter().max().unwrap_or(0);
    let mut mechanisms: BTreeMap<String, f64> = BTreeMap::new();
    for metadata in db
        .program_metadata
        .iter()
        .filter(|metadata| metadata.family == "population_law_long_range")
    {
        let Some(id) = metadata.params.get("mechanism_id").and_then(Value::as_str) else {
            continue;
        };
        let Some(cells) = metadata
            .params
            .get("expected_label_relevant_cells")
            .and_then(Value::as_f64)
        else {
            continue;
        };
        mechanisms.insert(id.to_string(), cells);
    }
    let expected = mechanisms.values().sum::<f64>().round() as usize;
    let label_relevant_cells_within = [1_000usize, 4_000, 8_000, 16_000]
        .into_iter()
        .map(|budget| (budget.to_string(), expected.min(budget)))
        .collect();
    CorpusScaleCoverage {
        rows_per_table,
        rows_per_tier,
        cells_per_db,
        nuisance_cell_fraction,
        mean_fanout,
        p95_fanout,
        depth,
        label_relevant_cells_within,
        expected_label_relevant_cells_within_target: target_context_len
            .map(|target| expected.min(target)),
    }
}

pub fn print_corpus_scale_coverage(db_name: &str, report: &CorpusScaleCoverage) {
    let rows = report
        .rows_per_table
        .iter()
        .map(|(table, rows)| format!("{table}:{rows}"))
        .collect::<Vec<_>>()
        .join(",");
    eprintln!("[coverage] db={db_name} rows/table={rows} cells/db={} nuisance_cell_fraction={:.4} mean_fanout={:.3} p95_fanout={} depth={} label_relevant_cells={{1k:{},4k:{},8k:{},16k:{}}}",
        report.cells_per_db, report.nuisance_cell_fraction, report.mean_fanout, report.p95_fanout, report.depth,
        report.label_relevant_cells_within.get("1000").copied().unwrap_or(0),
        report.label_relevant_cells_within.get("4000").copied().unwrap_or(0),
        report.label_relevant_cells_within.get("8000").copied().unwrap_or(0),
        report.label_relevant_cells_within.get("16000").copied().unwrap_or(0));
}

impl MixtureManifest {
    /// Checks what materializing each record needs: unique DB names, seeds and target names, a
    /// supported family, and a target K the family can publish.
    pub fn validate(&self) -> Result<()> {
        let allowed: std::collections::BTreeSet<_> =
            self.allowed_discrete_k.iter().copied().collect();
        let mut dbs = std::collections::HashSet::new();
        let mut seeds = std::collections::HashSet::new();
        let mut targets = std::collections::HashSet::new();
        for record in &self.records {
            if !dbs.insert(record.db_name.as_str()) || !seeds.insert(record.generation_seed) {
                bail!("manifest DB names and seeds must be unique");
            }
            if !MATERIALIZED_FAMILIES.contains(&record.family.as_str()) {
                bail!("{}: unsupported family {}", record.db_name, record.family);
            }
            match record.task_type.as_str() {
                "binary" if record.target_k == Some(2) => {}
                "discrete" if record.target_k.is_some_and(|k| allowed.contains(&k)) => {}
                _ => bail!("invalid task/K contract for {}", record.db_name),
            }
            if record.class_id_permutation_seed.is_none() {
                bail!("{} lacks a class permutation seed", record.db_name);
            }
            if record.opaque_target_name.trim().is_empty()
                || !targets.insert(record.opaque_target_name.as_str())
            {
                bail!("manifest target names must be non-empty and unique");
            }
            validate_declared_target_names(record)?;
        }
        Ok(())
    }
}

/// Which construction contract to generate under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionContract {
    /// Gates are preconditions on publication.
    Gated,
    LegacyUngated,
}

/// Six alpha levels, one target column each.
pub const RELATIONAL_FL_LEVELS: usize = 6;

pub fn add_manifest_target(
    db: &mut GeneratedDb,
    record: &MixtureRecord,
    contract: SelectionContract,
) -> Result<()> {
    db.apply_mechanism_contract();
    match record.family.as_str() {
        "tabicl_v2_like_discrete" => add_discrete_target(db, record, contract),
        "support_required_multiclass" => add_support_bound_targets(db, record, contract),
        "relational_operator_discrete" => add_relational_operator_target(db, record, contract),
        family => bail!("unsupported manifest family {family:?}"),
    }
}

fn add_relational_operator_target(
    db: &mut GeneratedDb,
    record: &MixtureRecord,
    contract: SelectionContract,
) -> Result<()> {
    let k_requested = record
        .target_k
        .context("relational operator record is missing target_k")?;
    if !(2..=10).contains(&k_requested) {
        bail!("relational operator K must be in [2,10]");
    }
    let k = k_requested;
    // A single independently sampled operator per DB teaches generic graph execution.
    const FAMILIES: [LabelFamily; 12] = [
        LabelFamily::ChildSum,
        LabelFamily::ChildMean,
        LabelFamily::ChildCount,
        LabelFamily::ChildStd,
        LabelFamily::ChildNUnique,
        LabelFamily::ParentLookup,
        LabelFamily::RecentChildCount,
        LabelFamily::CooccurrenceCount,
        LabelFamily::CooccurrenceSum,
        LabelFamily::GrandchildCount,
        LabelFamily::TwohopGatedSum,
        LabelFamily::ExistsGrandchildGateSum,
    ];
    let mut permutation_v1: Vec<usize> = (0..k).collect();
    let mut permutation_rng = substream(
        record
            .class_id_permutation_seed
            .context("missing permutation seed")?,
        "manifest:operator-permutation",
    );
    permutation_rng.shuffle(&mut permutation_v1);

    // Deterministic feasibility search over (operator family, candidate).
    let start = (record.generation_seed as usize) % FAMILIES.len();
    let mut rejected = 0usize;
    let mut first_rejection: Option<String> = None;
    for offset in 0..FAMILIES.len() {
        let family = FAMILIES[(start + offset) % FAMILIES.len()];
        let candidates = all_aggfam_candidates(db, family);
        if candidates.is_empty() {
            continue;
        }
        let mut rng = substream(
            record.generation_seed.wrapping_add(family as u64 * 31),
            "labels:manifest-operator",
        );
        let head = rng.choose_index(candidates.len());
        for step in 0..candidates.len() {
            let (target_table, scores, metadata) = &candidates[(head + step) % candidates.len()];
            // `k_published` is what the score could resolve; `k` is what the manifest asked for.
            let (latent, k_published) = (balanced_quantiles(scores, k), k);
            let mut permutation: Vec<usize> = (0..k_published).collect();
            permutation.clone_from_slice(&permutation_v1[..k_published]);

            let labels: Vec<i64> = latent
                .into_iter()
                .map(|label| permutation[label] as i64)
                .collect();
            let table_idx = db
                .tables
                .iter()
                .position(|table| &table.name == target_table)
                .context("operator target table disappeared")?;
            // Gate BEFORE publishing, on the arm the operator's shape puts it in.
            let arm = operator_arm(&db.tables[table_idx], &metadata.source_tables);
            let (bag_ceiling, bag_declaration) = if arm == OPERATOR_ARM_POINTER {
                (POINTER_ARM_CEILING, Some(POINTER_ARM_DECLARATION))
            } else {
                (SHORTCUT_CEILING, None)
            };
            let disclosure_exclusions = sibling_targets_on_table(
                db,
                &db.tables[table_idx].name,
                &[],
                &record.opaque_target_name,
            );
            let gate_input = DiscreteTargetGateInput {
                family: &record.family,
                arm,
                target_table: &db.tables[table_idx],
                target: &record.opaque_target_name,
                k: k_published,
                scores,
                labels: &labels,
                bag_ceiling,
                bag_declaration,
                focal_observable: None,
                disclosure_exclusions: Some(&disclosure_exclusions),
            };
            // Both branches, not just the tie-respecting one.
            let mut verdicts = evaluate_discrete_target_with_row_order(&gate_input);
            attribute(&mut verdicts, &metadata.family, rejected);
            if contract == SelectionContract::Gated && !all_passed(&verdicts) {
                rejected += 1;
                if first_rejection.is_none() {
                    first_rejection = verdicts
                        .iter()
                        .find(|verdict| !verdict.passed)
                        .map(|verdict| verdict.failure_message());
                }
                continue;
            }
            let target_table = target_table.clone();
            db.tables[table_idx]
                .push_column(record.opaque_target_name.clone(), ColumnData::I64(labels));
            db.record_shortcut_verdicts(verdicts);
            db.program_metadata.push(opaque_metadata(
                record,
                target_table,
                if k_published == 2 {
                    "binary_classification"
                } else {
                    "multiclass_classification"
                },
                k_published,
                db.tables[table_idx].num_rows(),
            ));
            return Ok(());
        }
    }
    bail!(
        "no relational operator candidate for {} cleared the shortcut gates at K={k}: \
         {rejected} candidates were rejected. First rejection: {}",
        record.db_name,
        first_rejection.as_deref().unwrap_or("none recorded"),
    )
}

/// Every `mech_*` record publishes a K-way view and its regression partner.
pub const MECH_TARGETS: usize = 2;

/// Every OTHER declared target that lives on `table`, plus any name the record being gated is about
/// to publish there.
fn sibling_targets_on_table(
    db: &GeneratedDb,
    table: &str,
    about_to_publish: &[String],
    me: &str,
) -> Vec<String> {
    let mut out: Vec<String> = db
        .program_metadata
        .iter()
        .filter(|metadata| metadata.target_table == table)
        .map(|metadata| metadata.target_col.clone())
        .chain(about_to_publish.iter().cloned())
        .filter(|name| name != me)
        .collect();
    out.sort();
    out.dedup();
    out
}

fn add_support_bound_targets(
    db: &mut GeneratedDb,
    record: &MixtureRecord,
    contract: SelectionContract,
) -> Result<()> {
    let k = record
        .target_k
        .context("support-bound record is missing target_k")?;
    let permutation_seed = record
        .class_id_permutation_seed
        .context("support-bound record is missing class permutation seed")?;
    // Same X and same latent partition, up to four incompatible label alphabets.
    let binding_count = support_binding_count(k);
    let permutations = distinct_permutations(k, binding_count, permutation_seed, contract);
    // The names come from the manifest contract, not from a second derivation here.
    let target_names = expected_target_names(record)?;
    if target_names.len() != binding_count {
        bail!(
            "{} declares {} support bindings but K={k} materialises {binding_count}",
            record.db_name,
            target_names.len()
        );
    }

    let candidates = target_table_candidates(db, false);
    if candidates.is_empty() {
        bail!("no eligible target table for {}", record.db_name);
    }
    let mut rejected = 0usize;
    let mut first_rejection: Option<String> = None;
    for table_idx in candidates {
        let table = &db.tables[table_idx];
        let table_name = table.name.clone();
        let scores = tabular_score(table, record.generation_seed);
        let latent = balanced_quantiles(&scores, k);

        let mut names = Vec::with_capacity(binding_count);
        let mut published: Vec<Vec<i64>> = Vec::with_capacity(binding_count);
        let mut verdicts = Vec::new();
        for (binding, permutation) in permutations.iter().enumerate() {
            let labels: Vec<i64> = latent
                .iter()
                .map(|label| permutation[*label] as i64)
                .collect();
            let target_name = target_names[binding].clone();
            // The whole binding group is published on this one table, so every
            // OTHER binding is a sibling view of the same latent partition and
            // is dropped from this target's context by the task contract.
            let disclosure_exclusions = sibling_targets_on_table(
                db,
                &db.tables[table_idx].name,
                &target_names,
                &target_name,
            );
            let mut per_column =
                evaluate_discrete_target_with_row_order(&DiscreteTargetGateInput {
                    family: &record.family,
                    arm: "binding",
                    target_table: &db.tables[table_idx],
                    target: &target_name,
                    k,
                    scores: &scores,
                    labels: &labels,
                    bag_ceiling: SHORTCUT_CEILING,
                    bag_declaration: None,
                    focal_observable: None,
                    disclosure_exclusions: Some(&disclosure_exclusions),
                });
            attribute(&mut per_column, "tabular_score", rejected);
            verdicts.append(&mut per_column);
            names.push(target_name);
            published.push(labels);
        }
        let mut group = evaluate_support_bindings(
            &record.family,
            &db.tables[table_idx],
            names.first().map(String::as_str).unwrap_or_default(),
            k,
            &published,
        );
        attribute(std::slice::from_mut(&mut group), "tabular_score", rejected);
        verdicts.push(group);

        if contract == SelectionContract::Gated && !all_passed(&verdicts) {
            rejected += 1;
            if first_rejection.is_none() {
                first_rejection = verdicts
                    .iter()
                    .find(|verdict| !verdict.passed)
                    .map(|verdict| verdict.failure_message());
            }
            continue;
        }

        db.record_shortcut_verdicts(verdicts);
        for (target_name, labels) in names.into_iter().zip(published) {
            db.tables[table_idx].push_column(target_name.clone(), ColumnData::I64(labels));
            let mut target_record = record.clone();
            target_record.opaque_target_name = target_name;
            db.program_metadata.push(opaque_metadata(
                &target_record,
                table_name.clone(),
                if k == 2 {
                    "binary_classification"
                } else {
                    "multiclass_classification"
                },
                k,
                db.tables[table_idx].num_rows(),
            ));
        }
        return Ok(());
    }
    bail!(
        "no target table for {} cleared the shortcut gates at K={k}: {rejected} rejected. \
         First rejection: {}",
        record.db_name,
        first_rejection.as_deref().unwrap_or("none recorded"),
    )
}

/// The names of every column `add_manifest_target` will materialise for a record, and therefore the
/// names a manifest must declare.
pub fn expected_target_names(record: &MixtureRecord) -> Result<Vec<String>> {
    let derived = |domain: &'static str, count: usize| -> Vec<String> {
        std::iter::once(record.opaque_target_name.clone())
            .chain((1..count).map(|index| derived_target_name(record, domain, index)))
            .collect()
    };
    Ok(match record.family.as_str() {
        "support_required_multiclass" => {
            let k = record
                .target_k
                .context("support-bound record is missing target_k")?;
            derived("binding", support_binding_count(k))
        }
        "relational_fl_discrete" => derived("relation", RELATIONAL_FL_LEVELS),
        // Slot 0 is the K-way view and carries the record's declared
        // `task_type`/`target_k`; slot 1 is its regression partner. The order
        // is the materialisation order `worldgen-materialize` compares against.
        family if MECH_FAMILIES.contains(&family) => derived("mech", MECH_TARGETS),
        _ => match record.opaque_target_names.as_ref() {
            Some(names) => names.clone(),
            None => vec![record.opaque_target_name.clone()],
        },
    })
}

/// Fail closed, in both directions, on what a record says it trains.
pub fn validate_declared_target_names(record: &MixtureRecord) -> Result<Vec<String>> {
    let expected = expected_target_names(record)?;
    if matches!(
        record.family.as_str(),
        "support_required_multiclass" | "relational_fl_discrete"
    ) || MECH_FAMILIES.contains(&record.family.as_str())
    {
        let declared = record.opaque_target_names.as_ref().with_context(|| {
            format!(
                "{} ({}) materialises {} targets and must declare all of them",
                record.db_name,
                record.family,
                expected.len()
            )
        })?;
        if declared != &expected {
            bail!(
                "{} declares target names that are not the ones it materialises: declared={declared:?}, materialises={expected:?}",
                record.db_name
            );
        }
    }
    Ok(expected)
}

/// K=2 has only 2! distinct bijections, so it produces two bindings.
pub fn support_binding_count(k: usize) -> usize {
    if k == 2 {
        2
    } else {
        4
    }
}

/// Hash prefix of the derived target names; its value fixes the published column names.
const TARGET_NAME_SALT: &str = "plurel-v2-target";

fn derived_target_name(record: &MixtureRecord, domain: &str, index: usize) -> String {
    let digest = Sha256::digest(format!(
        "{TARGET_NAME_SALT}::{domain}::{}::{index}::{}",
        record.generation_seed, record.opaque_target_name
    ));
    format!("target_{:x}", digest)[..27].to_string()
}

/// The `count` label alphabets published on one latent partition.
fn distinct_permutations(
    k: usize,
    count: usize,
    seed: u64,
    contract: SelectionContract,
) -> Vec<Vec<usize>> {
    assert!(k >= 2 && count <= if k == 2 { 2 } else { 4 });
    let mut rng = substream(seed, "manifest:distinct-support-bindings");
    if contract == SelectionContract::LegacyUngated {
        let mut result: Vec<Vec<usize>> = Vec::with_capacity(count);
        while result.len() < count {
            let mut candidate: Vec<usize> = (0..k).collect();
            rng.shuffle(&mut candidate);
            if !result.contains(&candidate) {
                result.push(candidate);
            }
        }
        return result;
    }
    let mut base: Vec<usize> = (0..k).collect();
    rng.shuffle(&mut base);
    let mut result: Vec<Vec<usize>> = (0..count.min(k))
        .map(|shift| base.iter().map(|value| (value + shift) % k).collect())
        .collect();
    // Only reachable at K = 3, count = 4. Any distinct permutation attains the
    // optimum here, so the draw stays random rather than pretending to choose.
    while result.len() < count {
        let mut candidate: Vec<usize> = (0..k).collect();
        rng.shuffle(&mut candidate);
        if !result.contains(&candidate) {
            result.push(candidate);
        }
    }
    result
}

/// Every table a discrete target could live on, best first.
fn target_table_candidates(db: &GeneratedDb, relational: bool) -> Vec<usize> {
    let mut eligible: Vec<usize> = db
        .tables
        .iter()
        .enumerate()
        .filter(|(_, table)| table.num_rows() >= 20 && !table.numeric_feature_columns().is_empty())
        .filter(|(_, table)| !relational || !table.fkey_col_to_pkey_table.is_empty())
        .map(|(idx, _)| idx)
        .collect();
    eligible.sort_by(|left, right| {
        db.tables[*right]
            .num_rows()
            .cmp(&db.tables[*left].num_rows())
            .then(right.cmp(left))
    });
    eligible
}

fn numeric_matrix(table: &TableData) -> Vec<Vec<f64>> {
    table
        .numeric_feature_columns()
        .into_iter()
        .filter_map(|name| table.column(&name).and_then(ColumnData::as_numeric_lossy))
        .map(|values| standardize(&values))
        .collect()
}

fn tabular_score(table: &TableData, seed: u64) -> Vec<f64> {
    let matrix = numeric_matrix(table);
    let mut rng = substream(seed, "manifest:tabular-rule");
    let weights: Vec<f64> = (0..matrix.len()).map(|_| rng.standard_normal()).collect();
    (0..table.num_rows())
        .map(|row| {
            let linear: f64 = matrix
                .iter()
                .zip(&weights)
                .map(|(column, weight)| column[row] * weight)
                .sum();
            let interaction = if matrix.len() >= 2 {
                matrix[0][row] * matrix[1][row]
            } else {
                0.0
            };
            linear.sin() + 0.7 * interaction + 0.2 * linear
        })
        .collect()
}

fn published_latent(record: &MixtureRecord, scores: &[f64], k: usize) -> (Vec<usize>, f64) {
    let _ = record;
    (balanced_quantiles(scores, k), 1.0 / k as f64)
}

fn add_discrete_target(
    db: &mut GeneratedDb,
    record: &MixtureRecord,
    contract: SelectionContract,
) -> Result<()> {
    let k = record
        .target_k
        .context("discrete record is missing target_k")?;
    if !(2..=10).contains(&k) {
        bail!("discrete K must be in [2,10]");
    }
    let permutation_seed = record
        .class_id_permutation_seed
        .context("discrete record is missing class permutation seed")?;
    let mut permutation: Vec<usize> = (0..k).collect();
    let mut permutation_rng = substream(permutation_seed, "manifest:class-permutation");
    permutation_rng.shuffle(&mut permutation);

    let gated = record.family == "tabicl_v2_like_discrete";
    let candidates = target_table_candidates(db, false);
    if candidates.is_empty() {
        bail!("no eligible target table for {}", record.db_name);
    }
    let mut rejected = 0usize;
    let mut first_rejection: Option<String> = None;
    for table_idx in candidates {
        let table = &db.tables[table_idx];
        let scores = tabular_score(table, record.generation_seed);
        let (latent, _minority_rate) = published_latent(record, &scores, k);
        let labels: Vec<i64> = latent
            .into_iter()
            .map(|label| permutation[label] as i64)
            .collect();
        let mut verdicts = Vec::new();
        if gated {
            let disclosure_exclusions = sibling_targets_on_table(
                db,
                &db.tables[table_idx].name,
                &[],
                &record.opaque_target_name,
            );
            // `_with_row_order`, same reason as the other two discrete
            // manifest families: the published column comes out of
            // `balanced_quantiles`, whose tie break is the row index.
            verdicts = evaluate_discrete_target_with_row_order(&DiscreteTargetGateInput {
                family: &record.family,
                arm: "tabular",
                target_table: &db.tables[table_idx],
                target: &record.opaque_target_name,
                k,
                scores: &scores,
                labels: &labels,
                bag_ceiling: SHORTCUT_CEILING,
                bag_declaration: None,
                focal_observable: None,
                disclosure_exclusions: Some(&disclosure_exclusions),
            });
            attribute(&mut verdicts, "tabular_score", rejected);
            if contract == SelectionContract::Gated && !all_passed(&verdicts) {
                rejected += 1;
                if first_rejection.is_none() {
                    first_rejection = verdicts
                        .iter()
                        .find(|verdict| !verdict.passed)
                        .map(|verdict| verdict.failure_message());
                }
                continue;
            }
        }
        let table_name = db.tables[table_idx].name.clone();
        db.record_shortcut_verdicts(verdicts);
        db.tables[table_idx]
            .push_column(record.opaque_target_name.clone(), ColumnData::I64(labels));
        db.program_metadata.push(opaque_metadata(
            record,
            table_name,
            if k == 2 {
                "binary_classification"
            } else {
                "multiclass_classification"
            },
            k,
            db.tables[table_idx].num_rows(),
        ));
        return Ok(());
    }
    bail!(
        "no target table for {} cleared the shortcut gates at K={k}: {rejected} rejected. \
         First rejection: {}",
        record.db_name,
        first_rejection.as_deref().unwrap_or("none recorded"),
    )
}

fn opaque_metadata(
    record: &MixtureRecord,
    target_table: String,
    task_type: &str,
    n_classes: usize,
    num_rows: usize,
) -> LabelMetadata {
    LabelMetadata {
        family: "opaque".to_string(),
        target_table,
        source_tables: Vec::new(),
        target_type: if task_type == "regression" {
            "continuous"
        } else {
            "class"
        }
        .to_string(),
        n_classes,
        hops: 0,
        uses_time: false,
        params: BTreeMap::new(),
        seed: record.generation_seed,
        target_col: record.opaque_target_name.clone(),
        task_type: task_type.to_string(),
        num_rows,
    }
}

use crate::scm::balanced_quantiles;

fn standardize(values: &[f64]) -> Vec<f64> {
    let finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let mean = if finite.is_empty() {
        0.0
    } else {
        finite.iter().sum::<f64>() / finite.len() as f64
    };
    let variance = if finite.is_empty() {
        1.0
    } else {
        finite
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / finite.len() as f64
    };
    let std = variance.sqrt().max(1e-6);
    values
        .iter()
        .map(|value| {
            if value.is_finite() {
                (value - mean) / std
            } else {
                0.0
            }
        })
        .collect()
}

/// One integrity check, with the evidence that decided it.
pub struct IntegrityCheck {
    pub name: &'static str,
    pub passed: bool,
    pub evidence: Value,
}

pub fn integrity_checks(
    db: &GeneratedDb,
    record: &MixtureRecord,
    base_columns: &BTreeMap<String, Vec<String>>,
    expected_targets: usize,
    manifest_sha256: &str,
) -> Vec<IntegrityCheck> {
    let materialized: Vec<String> = db
        .program_metadata
        .iter()
        .map(|metadata| metadata.target_col.clone())
        .collect();

    // 1. The family contract fixes how many targets a record materializes.
    let count_ok = materialized.len() == expected_targets;

    let leaked: Vec<String> = materialized
        .iter()
        .filter(|target| {
            base_columns
                .values()
                .any(|columns| columns.iter().any(|column| &column == target))
        })
        .cloned()
        .collect();

    // 3. The artifact is the one the manifest record asked for: same database
    //    name, the record's generation seed on every target's provenance, and
    //    every explicitly declared target name actually materialized.
    let declared_missing: Vec<String> = match record.opaque_target_names.as_ref() {
        Some(names) => names
            .iter()
            .filter(|name| !materialized.contains(name))
            .cloned()
            .collect(),
        None => Vec::new(),
    };
    let seed_ok = db
        .program_metadata
        .iter()
        .all(|metadata| metadata.seed == record.generation_seed);
    let bound = db.name == record.db_name
        && seed_ok
        && declared_missing.is_empty()
        && !materialized.is_empty();

    vec![
        IntegrityCheck {
            name: "target_count_matches_family_contract",
            passed: count_ok,
            evidence: json!({
                "expected_targets": expected_targets,
                "materialized_targets": materialized.len(),
            }),
        },
        IntegrityCheck {
            name: "target_excluded_from_base_features",
            passed: leaked.is_empty(),
            evidence: json!({
                "base_columns_checked": base_columns.values().map(Vec::len).sum::<usize>(),
                "targets_checked": materialized.len(),
                "targets_found_among_base_columns": leaked,
            }),
        },
        IntegrityCheck {
            name: "manifest_bound",
            passed: bound,
            evidence: json!({
                "manifest_sha256": manifest_sha256,
                "record_db_name": record.db_name,
                "artifact_db_name": db.name,
                "record_generation_seed": record.generation_seed,
                "every_target_carries_the_record_seed": seed_ok,
                "declared_target_names_missing": declared_missing,
            }),
        },
    ]
}
