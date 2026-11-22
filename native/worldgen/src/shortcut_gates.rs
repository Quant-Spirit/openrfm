//! Construction-time shortcut gates for the manifest classification families.

use crate::scm::{ColumnData, TableData};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

/// The SHORTCUT direction only: no mechanism-free channel may carry more than this fraction of the
/// published column's own entropy.
pub const SHORTCUT_CEILING: f64 = 0.50;

/// The LIVENESS direction: the mechanism must supply at least this fraction of the published
/// column's own entropy.
pub const LIVENESS_FLOOR: f64 = 0.50;

/// [`GateKind::PriorCoverage`]'s threshold, as a multiple of the estimators' own in-sample bias
/// bound.
pub const PRIOR_COVERAGE_FLOOR: f64 = 1.0;

pub const RELBENCH_MINORITY_RATE_MIN: f64 = 0.0154;
/// See [`RELBENCH_MINORITY_RATE_MIN`].
pub const RELBENCH_MINORITY_RATE_MAX: f64 = 0.4472;

/// The three manifest families backed by a `worldgen --axis` mechanism.
pub const MECH_FAMILIES: [&str; 3] = ["mech_group_rank", "mech_count_ratio", "mech_order_stat"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GateKind {
    LatentDeterminacy,
    #[serde(alias = "balanced_prior")]
    PriorCoverage,
    /// Shortcut. Rows sharing the focal row's foreign key are visible in the
    /// same sampled subgraph with their own labels, so copying them is a
    /// mechanism-free predictor available to any sequence model.
    NeighbourLabelBag,
    /// Liveness, specific to `support_required_multiclass`.
    SupportBindingResidual,
    /// Shortcut, continuous families.
    ExogenousChannel,
    /// Liveness, continuous families whose readout ends in a noise draw.
    MechanismDeterminacy,
    /// Liveness, `tempk*`.
    LabelChannelResidual,
    /// Shortcut, specific to `mech_group_rank`.
    FocalRowReadability,
    ProjectedResolution,
    SiblingColumnDisclosure,
}

impl GateKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LatentDeterminacy => "latent_determinacy",
            Self::PriorCoverage => "prior_coverage",
            Self::NeighbourLabelBag => "neighbour_label_bag",
            Self::SupportBindingResidual => "support_binding_residual",
            Self::ExogenousChannel => "exogenous_channel",
            Self::MechanismDeterminacy => "mechanism_determinacy",
            Self::LabelChannelResidual => "label_channel_residual",
            Self::FocalRowReadability => "focal_row_readability",
            Self::SiblingColumnDisclosure => "sibling_column_disclosure",
            Self::ProjectedResolution => "projected_resolution",
        }
    }

    /// `true` when passing means staying UNDER the budget.
    pub fn is_shortcut(self) -> bool {
        matches!(
            self,
            Self::NeighbourLabelBag
                | Self::ExogenousChannel
                | Self::FocalRowReadability
                | Self::SiblingColumnDisclosure
        )
    }
}

/// A gate read as a POSITION ON AN AXIS instead of as a threshold.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GateCoordinate {
    /// The axis the measurement is a position on.
    pub axis: String,
    /// [`LABEL_SIDE`] or [`FEATURE_SIDE`].
    pub side: String,
    /// The rule, written out, so the side can be recomputed from the numbers
    /// already on the verdict without reading this crate.
    pub decision_rule: String,
    /// `ratio` at or above this is [`LABEL_SIDE`]; below it, [`FEATURE_SIDE`].
    pub threshold_ratio: f64,
    /// The consumer this side is for, in the certificate's own words.
    pub routes_to: String,
    pub column_prefix: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unresolved_reason: Option<String>,
}

impl GateCoordinate {}

/// One gate applied to one published column (or, for
/// [`GateKind::SupportBindingResidual`], to one binding group).
#[derive(Debug, Clone, Serialize)]
pub struct ShortcutVerdict {
    pub gate: GateKind,
    /// Manifest family this column belongs to.
    pub family: String,
    /// Sub-arm inside the family, when the family holds arms to different standards.
    pub arm: String,
    pub target_table: String,
    /// The published column, or the binding-group stem for the support gate.
    pub target: String,
    pub k: usize,
    pub measured_nats: f64,
    pub budget_nats: f64,
    /// `measured_nats / budget_nats`, the scale-free number the ceiling reads.
    pub ratio: f64,
    #[serde(rename = "ceiling")]
    pub threshold_ratio: f64,
    pub passed: bool,
    /// Present when the arm is admitted above the general ceiling on a stated
    /// argument rather than on its measurement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub declared_exception: Option<String>,
    /// The executable rule behind the target, where the family selects one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mechanism: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidates_rejected: Option<usize>,
    #[serde(default)]
    pub feature_contract: crate::scm::FeatureContract,
    #[serde(default)]
    pub bag_summary: BagSummary,
    /// Present exactly when this row is NOT a verdict.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub read_as_coordinate_not_verdict: Option<GateCoordinate>,
    pub note: String,
}

impl ShortcutVerdict {
    pub fn failure_message(&self) -> String {
        // A shortcut gate has a CEILING at `ceiling * budget`; a liveness gate has a FLOOR at `(1 -
        // ceiling) * budget`.
        let (direction, threshold) = if self.gate.is_shortcut() {
            ("above its", self.threshold_ratio * self.budget_nats)
        } else {
            ("below its", self.threshold_ratio * self.budget_nats)
        };
        format!(
            "{}: gate {} on {}.{} (arm {}, K={}) measured {:.4} nats, {} {:.4}-nat threshold \
             ({:.1}% of the {:.4}-nat budget), against a {:.2} threshold. {}",
            self.family,
            self.gate.as_str(),
            self.target_table,
            self.target,
            self.arm,
            self.k,
            self.measured_nats,
            direction,
            threshold,
            100.0 * self.ratio,
            self.budget_nats,
            self.threshold_ratio,
            self.note,
        )
    }
}

/// The arm of `relational_operator_discrete` whose target table is the MANY side of the foreign key
/// the operator reads.
pub const OPERATOR_ARM_POINTER: &str = "pointer";
/// Every other operator in `relational_operator_discrete`.
pub const OPERATOR_ARM_AGGREGATE: &str = "aggregate";

/// The pointer arm's ceiling, and the reason it is not [`SHORTCUT_CEILING`].
pub const POINTER_ARM_CEILING: f64 = 1.0;

pub const POINTER_ARM_DECLARATION: &str = "parent_lookup publishes on the many side of the \
    foreign key it reads, so the label is sigma(parent)-measurable and sibling labels are the \
    operator's own output rather than a route around it; admitted with the measurement recorded \
    so downstream analysis can exclude the arm, and narrowed to 'tests pointer following, not \
    the parent value read'";

/// Shannon entropy of an empirical distribution over `k` classes, in nats.
pub fn empirical_entropy(labels: &[i64], k: usize) -> f64 {
    if labels.is_empty() || k == 0 {
        return 0.0;
    }
    let mut counts = vec![0usize; k];
    let mut total = 0usize;
    for &label in labels {
        if label >= 0 && (label as usize) < k {
            counts[label as usize] += 1;
            total += 1;
        }
    }
    if total == 0 {
        return 0.0;
    }
    counts
        .into_iter()
        .filter(|count| *count > 0)
        .map(|count| {
            let p = count as f64 / total as f64;
            -p * p.ln()
        })
        .sum()
}

/// `ln K - H(Y)`: what a constant predictor gets for free.
pub fn class_prior_information(labels: &[i64], k: usize) -> f64 {
    (k as f64).ln() - empirical_entropy(labels, k)
}

/// The share of rows in the LEAST frequent class that actually occurs.
pub fn minority_rate(labels: &[i64], k: usize) -> f64 {
    if labels.is_empty() || k == 0 {
        return 0.0;
    }
    let mut counts = vec![0usize; k];
    let mut total = 0usize;
    for &label in labels {
        if label >= 0 && (label as usize) < k {
            counts[label as usize] += 1;
            total += 1;
        }
    }
    if total == 0 {
        return 0.0;
    }
    let smallest = counts
        .into_iter()
        .filter(|count| *count > 0)
        .min()
        .unwrap_or(0);
    smallest as f64 / total as f64
}

/// The entropy below which no gate on this column is a measurement.
pub fn prior_resolvability_floor_nats(k: usize, n: usize) -> f64 {
    if n == 0 {
        return f64::INFINITY;
    }
    ((k as f64) - 1.0).powi(2) / (2.0 * n as f64)
}

/// `ln K - H(Y | s)`, where `s` is the score the family quantised.
pub fn latent_determinacy_information(scores: &[f64], labels: &[i64], k: usize) -> f64 {
    debug_assert_eq!(scores.len(), labels.len());
    let mut groups: BTreeMap<u64, Vec<i64>> = BTreeMap::new();
    for (score, label) in scores.iter().zip(labels) {
        // `to_bits` on the canonicalised value so that -0.0 and 0.0 group
        // together and every NaN payload lands in one bucket.
        let key = if score.is_nan() {
            f64::NAN.to_bits()
        } else {
            (score + 0.0).to_bits()
        };
        groups.entry(key).or_default().push(*label);
    }
    let n = labels.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let conditional: f64 = groups
        .values()
        .map(|bucket| (bucket.len() as f64 / n) * empirical_entropy(bucket, k))
        .sum();
    empirical_entropy(labels, k) - conditional
}

#[derive(Debug, Clone, Serialize)]
pub struct BagMeasurement {
    pub fk_column: String,
    pub information_nats: f64,
    /// Rows that had at least one labelled sibling. A bag gate over a table
    /// where nobody has siblings is trivially zero, and that is worth being
    /// able to see rather than having to infer.
    pub rows_with_siblings: usize,
    pub rows: usize,
    /// Which function of the bag this number conditions on.
    #[serde(default)]
    pub summary: BagSummary,
    /// The OCCUPIED alphabet of that summary; see [`BagSummary`].
    #[serde(default)]
    pub alphabet: usize,
}

impl BagMeasurement {
    /// The plug-in bias bound for this measurement, in nats.
    pub(crate) fn bias_bound_nats(&self, k: usize) -> f64 {
        bag_information_bias_bound(self.alphabet, k, self.rows)
    }
}

/// Which function of the leave-one-out sibling bag the gate conditions on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BagSummary {
    #[default]
    ModeUnanimity,
    ModeAntimodeUnanimity,
    /// The leave-one-out count vector in full: the sufficient statistic for any exchangeable
    /// neighbour-label predictor.
    FullMultiset,
}

impl BagSummary {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ModeUnanimity => "mode_unanimity",
            Self::ModeAntimodeUnanimity => "mode_antimode_unanimity",
            Self::FullMultiset => "full_multiset",
        }
    }
}

/// [`bag_shortcut_information`] under a named summary, with the alphabet it actually occupied.
pub fn bag_shortcut_measurement(
    labels: &[i64],
    groups: &[i64],
    k: usize,
    summary: BagSummary,
) -> BagInformation {
    debug_assert_eq!(labels.len(), groups.len());
    let n = labels.len();
    if n == 0 || k == 0 {
        return BagInformation {
            information_nats: 0.0,
            alphabet: 0,
        };
    }
    let mut counts: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    let mut sizes: BTreeMap<i64, usize> = BTreeMap::new();
    for (label, group) in labels.iter().zip(groups) {
        let entry = counts.entry(*group).or_insert_with(|| vec![0usize; k]);
        if *label >= 0 && (*label as usize) < k {
            entry[*label as usize] += 1;
        }
        *sizes.entry(*group).or_default() += 1;
    }
    // Bucket key: `None` for "no siblings", else the summary of the leave-one-out bag.
    let mut buckets: BTreeMap<Option<Vec<usize>>, Vec<i64>> = BTreeMap::new();
    for (label, group) in labels.iter().zip(groups) {
        let bucket = &counts[group];
        let bag_size = sizes[group] - 1;
        let key = if bag_size == 0 || *label < 0 || (*label as usize) >= k {
            None
        } else {
            let mut loo = bucket.clone();
            loo[*label as usize] -= 1;
            // `max_by_key` returns the LAST maximum.
            let mode = (0..k).max_by_key(|class| loo[*class]).unwrap_or(0);
            let unanimous = loo[mode] == bag_size;
            // Every key in one call comes from the same arm, so the shared
            // `Vec<usize>` key type cannot mix lengths within a measurement.
            match summary {
                BagSummary::FullMultiset => Some(loo),
                BagSummary::ModeAntimodeUnanimity => {
                    // `min_by_key` returns the FIRST minimum, so the anti-mode ties to the lowest
                    // class id.
                    let antimode = (0..k).min_by_key(|class| loo[*class]).unwrap_or(0);
                    Some(vec![mode, antimode, usize::from(unanimous)])
                }
                BagSummary::ModeUnanimity => Some(vec![mode, usize::from(unanimous)]),
            }
        };
        buckets.entry(key).or_default().push(*label);
    }
    let alphabet = buckets.len();
    let conditional: f64 = buckets
        .values()
        .map(|bucket| (bucket.len() as f64 / n as f64) * empirical_entropy(bucket, k))
        .sum();
    // `H(Y) - H(Y | S_bag)`, NOT `ln K - H(Y | S_bag)`. The two differ by
    // `class_prior_information`, which is a different gate's quantity.
    BagInformation {
        information_nats: empirical_entropy(labels, k) - conditional,
        alphabet,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BagInformation {
    pub information_nats: f64,
    /// Distinct values the summary took, INCLUDING the "no siblings" value.
    /// This is the occupied `|A|` in the bias bound, not the a-priori one.
    pub alphabet: usize,
}

/// The largest value [`bag_shortcut_information`] may return on a table where no row has a labelled
/// sibling.
pub const BAG_ZERO_TOLERANCE: f64 = 1e-9;

/// Upper bound on the in-sample upward bias of a pooled plug-in bag estimate, in nats: `(|A| - 1)(K
/// - 1) / (2n)` for an OCCUPIED alphabet of `|A|`.
pub fn bag_information_bias_bound(alphabet: usize, k: usize, n: usize) -> f64 {
    if n == 0 || alphabet == 0 {
        return 0.0;
    }
    (alphabet as f64 - 1.0) * (k as f64 - 1.0) / (2.0 * n as f64)
}

/// The larger of the two BOUNDED coarsenings, with the summary it came from.
pub fn strongest_bounded_bag_estimate(
    labels: &[i64],
    groups: &[i64],
    k: usize,
    declared: BagSummary,
) -> (BagInformation, BagSummary) {
    if declared == BagSummary::FullMultiset {
        return (
            bag_shortcut_measurement(labels, groups, k, declared),
            declared,
        );
    }
    let plain = bag_shortcut_measurement(labels, groups, k, BagSummary::ModeUnanimity);
    let antimode = bag_shortcut_measurement(labels, groups, k, BagSummary::ModeAntimodeUnanimity);
    if antimode.information_nats > plain.information_nats {
        (antimode, BagSummary::ModeAntimodeUnanimity)
    } else {
        (plain, BagSummary::ModeUnanimity)
    }
}

/// Measure the bag shortcut on every foreign key of the target table and keep the strongest.
pub fn strongest_bag_shortcut(
    table: &TableData,
    labels: &[i64],
    k: usize,
) -> Option<BagMeasurement> {
    strongest_bag_shortcut_excluding(table, labels, k, &[]).0
}

/// [`strongest_bag_shortcut`], with named foreign keys held out of the maximum and returned
/// separately.
pub fn strongest_bag_shortcut_excluding(
    table: &TableData,
    labels: &[i64],
    k: usize,
    excluded: &[&str],
) -> (Option<BagMeasurement>, Vec<BagMeasurement>) {
    let mut best: Option<BagMeasurement> = None;
    let mut held_out: Vec<BagMeasurement> = Vec::new();
    for fk_column in table.fkey_col_to_pkey_table.keys() {
        let Some(ColumnData::I64(keys)) = table.column(fk_column) else {
            continue;
        };
        if keys.len() != labels.len() {
            continue;
        }
        let mut sizes: HashMap<i64, usize> = HashMap::new();
        for key in keys {
            *sizes.entry(*key).or_default() += 1;
        }
        let rows_with_siblings = keys.iter().filter(|key| sizes[key] > 1).count();
        // The STRONGER of the two bounded coarsenings, not the one the table
        // declares; see `strongest_bounded_bag_estimate`.
        let (estimate, summary) =
            strongest_bounded_bag_estimate(labels, keys, k, table.bag_summary);
        let measurement = BagMeasurement {
            fk_column: fk_column.clone(),
            information_nats: estimate.information_nats,
            rows_with_siblings,
            rows: labels.len(),
            summary,
            alphabet: estimate.alphabet,
        };
        assert!(
            measurement.rows_with_siblings > 0
                || measurement.information_nats.abs() <= BAG_ZERO_TOLERANCE,
            "bag gate on foreign key {} measured {:.6} nats of sibling \
             information while reporting that 0 of {} rows have a labelled \
             sibling; the note and the verdict cannot both be right",
            measurement.fk_column,
            measurement.information_nats,
            measurement.rows,
        );
        if excluded.contains(&fk_column.as_str()) {
            held_out.push(measurement);
            continue;
        }
        // Strictly greater, else the lexicographically smaller foreign-key name wins.
        let take = match best.as_ref() {
            None => true,
            Some(current) => {
                let delta = measurement.information_nats - current.information_nats;
                if delta.abs() <= TIE_TOLERANCE_NATS {
                    measurement.fk_column < current.fk_column
                } else {
                    delta > 0.0
                }
            }
        };
        if take {
            best = Some(measurement);
        }
    }
    (best, held_out)
}

/// Two channel measurements this close are treated as EQUAL and separated by name instead.
pub const TIE_TOLERANCE_NATS: f64 = 1e-10;

/// `I(Y ; label of the nearest row in `observable`)`, leave-one-out.
pub fn nearest_neighbour_label_information(observable: &[f64], labels: &[i64], k: usize) -> f64 {
    debug_assert_eq!(observable.len(), labels.len());
    let n = labels.len();
    if n < 2 || k == 0 {
        return 0.0;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| {
        observable[*a]
            .total_cmp(&observable[*b])
            .then_with(|| a.cmp(b))
    });
    // In a sorted column the nearest other row is one of the two adjacent
    // ones, so the whole predictor is a single pass.
    // `BTreeMap`: see `TIE_TOLERANCE_NATS`.
    let mut buckets: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
    for position in 0..n {
        let row = order[position];
        let neighbour = match (position.checked_sub(1), position + 1 < n) {
            (None, true) => order[position + 1],
            (Some(previous), false) => order[previous],
            (Some(previous), true) => {
                let left = (observable[row] - observable[order[previous]]).abs();
                let right = (observable[order[position + 1]] - observable[row]).abs();
                // Ties go left, so the map is a deterministic function of the
                // column and not of the iteration order.
                if right < left {
                    order[position + 1]
                } else {
                    order[previous]
                }
            }
            (None, false) => continue,
        };
        buckets
            .entry(labels[neighbour])
            .or_default()
            .push(labels[row]);
    }
    let conditional: f64 = buckets
        .values()
        .map(|bucket| (bucket.len() as f64 / n as f64) * empirical_entropy(bucket, k))
        .sum();
    empirical_entropy(labels, k) - conditional
}

#[derive(Debug, Clone, Serialize)]
pub struct DisclosureMeasurement {
    pub column: String,
    pub information_nats: f64,
    /// How many columns the scan compared.
    pub columns_scanned: usize,
}

/// `I(Y ; q_K(c))`: what a reader of one other column gets at the label's own resolution.
pub fn quantised_column_information(values: &[f64], labels: &[i64], k: usize) -> f64 {
    if values.len() != labels.len() || values.is_empty() || k == 0 {
        return 0.0;
    }
    let buckets = crate::scm::balanced_quantiles(values, k);
    // `BTreeMap`: see `TIE_TOLERANCE_NATS`.
    let mut pooled: BTreeMap<usize, Vec<i64>> = BTreeMap::new();
    for (bucket, label) in buckets.into_iter().zip(labels) {
        pooled.entry(bucket).or_default().push(*label);
    }
    let n = labels.len() as f64;
    let conditional: f64 = pooled
        .values()
        .map(|bucket| (bucket.len() as f64 / n) * empirical_entropy(bucket, k))
        .sum();
    empirical_entropy(labels, k) - conditional
}

/// Scan every column of the target table that will still be in context and keep the strongest.
pub fn strongest_column_disclosure(
    table: &TableData,
    labels: &[i64],
    k: usize,
    excluded: &[String],
) -> Option<DisclosureMeasurement> {
    let mut best: Option<DisclosureMeasurement> = None;
    let mut scanned = 0usize;
    for name in table.numeric_feature_columns() {
        if excluded.iter().any(|column| column == &name) {
            continue;
        }
        let Some(values) = table.column(&name).and_then(ColumnData::as_numeric_lossy) else {
            continue;
        };
        if values.len() != labels.len() {
            continue;
        }
        scanned += 1;
        let information = quantised_column_information(&values, labels, k);
        if best
            .as_ref()
            .is_none_or(|current| information > current.information_nats)
        {
            best = Some(DisclosureMeasurement {
                column: name,
                information_nats: information,
                columns_scanned: 0,
            });
        }
    }
    best.map(|mut measurement| {
        measurement.columns_scanned = scanned;
        measurement
    })
}

pub fn support_binding_residual(bindings: &[Vec<i64>], k: usize) -> f64 {
    let b = bindings.len();
    if b == 0 || k == 0 {
        return 0.0;
    }
    let n = bindings[0].len();
    if n == 0 || bindings.iter().any(|column| column.len() != n) {
        return 0.0;
    }
    // `BTreeMap`: see `TIE_TOLERANCE_NATS`.
    let mut classes: BTreeMap<Vec<i64>, usize> = BTreeMap::new();
    for row in 0..n {
        let tuple: Vec<i64> = bindings.iter().map(|column| column[row]).collect();
        *classes.entry(tuple).or_default() += 1;
    }
    classes
        .into_iter()
        .map(|(tuple, count)| {
            // The column identity is opaque, so a learner that has pinned the
            // latent still faces the uniform mixture over which binding it is
            // looking at.
            let entropy = empirical_entropy(&tuple, k);
            (count as f64 / n as f64) * entropy
        })
        .sum()
}

/// Which arm of `relational_operator_discrete` a published column belongs to.
pub fn operator_arm(target_table: &TableData, source_tables: &[String]) -> &'static str {
    let reads_own_parent = source_tables.iter().any(|source| {
        target_table
            .fkey_col_to_pkey_table
            .values()
            .any(|parent| parent == source)
    });
    if reads_own_parent {
        OPERATOR_ARM_POINTER
    } else {
        OPERATOR_ARM_AGGREGATE
    }
}

fn verdict(
    gate: GateKind,
    family: &str,
    arm: &str,
    target_table: &str,
    target: &str,
    k: usize,
    measured: f64,
    budget: f64,
    threshold_ratio: f64,
    declared_exception: Option<String>,
    note: &str,
) -> ShortcutVerdict {
    let ratio = if budget > 0.0 { measured / budget } else { 0.0 };
    // One threshold, read in the gate's own direction.
    let passed = if gate.is_shortcut() {
        ratio <= threshold_ratio + 1e-12
    } else {
        ratio >= threshold_ratio - 1e-12
    };
    ShortcutVerdict {
        gate,
        family: family.to_string(),
        arm: arm.to_string(),
        target_table: target_table.to_string(),
        target: target.to_string(),
        k,
        measured_nats: measured,
        budget_nats: budget,
        ratio,
        threshold_ratio,
        passed,
        declared_exception,
        mechanism: None,
        candidates_rejected: None,
        read_as_coordinate_not_verdict: None,
        note: note.to_string(),
        // Overwritten by `stamp_contract` at the entry point that holds the table.
        feature_contract: crate::scm::FeatureContract::default(),
        bag_summary: BagSummary::default(),
    }
}

/// Copy the table's contract onto a batch of verdicts.
fn stamp_contract(verdicts: &mut [ShortcutVerdict], table: &TableData) {
    for verdict in verdicts {
        verdict.feature_contract = table.feature_contract;
        verdict.bag_summary = table.bag_summary;
    }
}

/// Stamp the mechanism and the feasibility-search yield onto a batch of
/// verdicts. Kept separate from [`evaluate_discrete_target`] so the estimator
/// never depends on provenance it might otherwise be tempted to read.
pub fn attribute(verdicts: &mut [ShortcutVerdict], mechanism: &str, candidates_rejected: usize) {
    for verdict in verdicts {
        verdict.mechanism = Some(mechanism.to_string());
        verdict.candidates_rejected = Some(candidates_rejected);
    }
}

/// `true` when every verdict in the batch passed.
pub fn all_passed(verdicts: &[ShortcutVerdict]) -> bool {
    verdicts.iter().all(|verdict| verdict.passed)
}

/// Everything a discrete manifest target needs to be gated on.
pub struct DiscreteTargetGateInput<'a> {
    pub family: &'a str,
    pub arm: &'a str,
    pub target_table: &'a TableData,
    pub target: &'a str,
    pub k: usize,
    /// The pre-quantisation score. Required: without it the liveness gate
    /// cannot tell a live mechanism from a shuffled row index.
    pub scores: &'a [f64],
    pub labels: &'a [i64],
    /// Set for an arm admitted above the general ceiling on a stated argument.
    pub bag_ceiling: f64,
    pub bag_declaration: Option<&'a str>,
    /// The focal cell the family deliberately writes part of its latent into.
    pub focal_observable: Option<&'a [f64]>,
    /// The columns that will NOT be in this target's context.
    pub disclosure_exclusions: Option<&'a [String]>,
}

/// Run the per-column gates: three always, plus the two the mechanism
/// families turn on by supplying the inputs they need.
pub fn evaluate_discrete_target(input: &DiscreteTargetGateInput<'_>) -> Vec<ShortcutVerdict> {
    // Every per-column gate is normalised by the published column's OWN entropy, never by `ln K`.
    let budget = empirical_entropy(input.labels, input.k);
    let mut out = Vec::with_capacity(5);
    out.push(verdict(
        GateKind::LatentDeterminacy,
        input.family,
        input.arm,
        &input.target_table.name,
        input.target,
        input.k,
        latent_determinacy_information(input.scores, input.labels, input.k),
        budget,
        LIVENESS_FLOOR,
        None,
        "the published class must be a function of the score the mechanism computed; a \
         constant score still quantises to a balanced column, so balance alone cannot \
         detect a dead mechanism",
    ));
    let minority = minority_rate(input.labels, input.k);
    let resolvability_floor = prior_resolvability_floor_nats(input.k, input.labels.len());
    out.push(verdict(
        GateKind::PriorCoverage,
        input.family,
        input.arm,
        &input.target_table.name,
        input.target,
        input.k,
        budget,
        resolvability_floor,
        PRIOR_COVERAGE_FLOOR,
        None,
        &format!(
            "minority-class rate {minority:.6} over K={} at n={}; H(Y)={:.4} nats against \
             ln K={:.4}, so the marginal hands a constant predictor {:.4} nats. That number \
             is REPORTED, not bounded: class imbalance is a target property RelBench has \
             (binary minority rates {RELBENCH_MINORITY_RATE_MIN} to \
             {RELBENCH_MINORITY_RATE_MAX}, median 0.2949), and the majority-class shortcut \
             is caught by latent_determinacy, which is normalised by H(Y) and so demands \
             the same FRACTION of the entropy at every marginal. What is bounded here is \
             resolvability: H(Y) must clear the estimators' own in-sample bias bound \
             (K-1)^2/2n, or no other gate on this column is a measurement",
            input.k,
            input.labels.len(),
            budget,
            (input.k as f64).ln(),
            class_prior_information(input.labels, input.k),
        ),
    ));
    let bag = strongest_bag_shortcut(input.target_table, input.labels, input.k);
    let (measured, note) = match &bag {
        Some(measurement) => (
            measurement.information_nats,
            format!(
                "strongest foreign key {} ({} of {} rows have a labelled sibling); copying \
                 sibling labels is a mechanism-free predictor available to any sequence model. \
                 Bag summary {} over an occupied alphabet of {}; in-sample upward bias is at \
                 most {:.4} nats at this K and row count",
                measurement.fk_column,
                measurement.rows_with_siblings,
                measurement.rows,
                measurement.summary.as_str(),
                measurement.alphabet,
                measurement.bias_bound_nats(input.k),
            ),
        ),
        None => (
            0.0,
            "the target table has no foreign key, so no sibling bag exists to copy from"
                .to_string(),
        ),
    };
    out.push(verdict(
        GateKind::NeighbourLabelBag,
        input.family,
        input.arm,
        &input.target_table.name,
        input.target,
        input.k,
        measured,
        budget,
        input.bag_ceiling,
        input.bag_declaration.map(str::to_string),
        &note,
    ));
    if let Some(observable) = input.focal_observable {
        out.push(verdict(
            GateKind::FocalRowReadability,
            input.family,
            input.arm,
            &input.target_table.name,
            input.target,
            input.k,
            nearest_neighbour_label_information(observable, input.labels, input.k),
            budget,
            SHORTCUT_CEILING,
            None,
            "the family writes part of its latent into a focal cell on purpose, so the \
             focal cell alone must stay worthless: the label of the nearest row in that \
             column may not recover more than half the label's entropy. Its liveness \
             partner is latent_determinacy, which requires the same label to be \
             recoverable from the same column read ACROSS the foreign-key group",
        ));
    }
    if let Some(excluded) = input.disclosure_exclusions {
        out.push(discrete_disclosure_verdict(
            input.family,
            input.arm,
            input.target_table,
            input.target,
            input.labels,
            input.k,
            budget,
            excluded,
        ));
    }
    stamp_contract(&mut out, input.target_table);
    out
}

/// [`GateKind::SiblingColumnDisclosure`] on a K-way discrete target.
#[allow(clippy::too_many_arguments)]
pub fn discrete_disclosure_verdict(
    family: &str,
    arm: &str,
    target_table: &TableData,
    target: &str,
    labels: &[i64],
    k: usize,
    budget: f64,
    excluded: &[String],
) -> ShortcutVerdict {
    let disclosure = strongest_column_disclosure(target_table, labels, k, excluded);
    let (measured, note) = match &disclosure {
        Some(measurement) => (
            measurement.information_nats,
            format!(
                "strongest of {} in-context column(s) is {}; the family publishes a \
                 regression view and a K-way view of the same numbers, so a partner \
                 view left in context would be the exact answer. In-sample upward \
                 bias is at most {:.4} nats at this K and row count",
                measurement.columns_scanned,
                measurement.column,
                ((k as f64) - 1.0).powi(2) / (2.0 * labels.len().max(1) as f64),
            ),
        ),
        None => (
            0.0,
            "the target table carries no other numeric column, so nothing is left in \
             context to read the answer off"
                .to_string(),
        ),
    };
    let mut out = verdict(
        GateKind::SiblingColumnDisclosure,
        family,
        arm,
        &target_table.name,
        target,
        k,
        measured,
        budget,
        SHORTCUT_CEILING,
        None,
        &note,
    );
    // An entry point in its own right, so it stamps rather than relying on a
    // caller exit stamp -- the same reason `quantised_disclosure_verdict` does.
    stamp_contract(std::slice::from_mut(&mut out), target_table);
    out
}

/// [`evaluate_discrete_target`], plus the row-order shortcut.
pub fn evaluate_discrete_target_with_row_order(
    input: &DiscreteTargetGateInput<'_>,
) -> Vec<ShortcutVerdict> {
    let mut out = evaluate_discrete_target(input);
    let n = input.labels.len();
    let buckets = exogenous_buckets(n);
    let row_index: Vec<f64> = (0..n).map(|row| row as f64).collect();
    let measured = bucketed_channel_information(&row_index, input.labels, input.k, buckets);
    let budget = empirical_entropy(input.labels, input.k);
    out.push(verdict(
        GateKind::ExogenousChannel,
        input.family,
        input.arm,
        &input.target_table.name,
        input.target,
        input.k,
        measured,
        budget,
        SHORTCUT_CEILING,
        None,
        &format!(
            "row order at {buckets} balanced-rank buckets. The quantiser breaks ties on the row              index, so a score too coarse to fill K bins publishes row order under an operator's              name; in-sample upward bias is at most {:.4} nats",
            bucketed_information_bias_bound(input.k, buckets, n),
        ),
    ));
    stamp_contract(&mut out, input.target_table);
    out
}

/// Bins a continuous published column enters the budget algebra through.
pub const CONTINUOUS_BINS: usize = 8;

/// Bins actually used on a table of `n` rows, and the smallest table any of this module's
/// continuous gates will speak about.
pub fn continuous_bins(n: usize) -> usize {
    (n / MIN_BUCKET_ROWS).clamp(2, CONTINUOUS_BINS)
}

/// Smallest table these gates will make a claim about: two buckets' worth.
pub const MIN_GATEABLE_ROWS: usize = 2 * MIN_BUCKET_ROWS;

/// What a verdict carries instead of a measurement when the table is too small.
pub const UNGATEABLE_DECLARATION: &str = "the table has fewer rows than the smallest at which a \
    bucketed conditional has any resolution, so this gate makes no claim about it; a one-bucket \
    conditional is the marginal, and the marginal of a balanced view is the whole budget";

pub const EXOGENOUS_BUCKETS: usize = 64;

/// Fewest rows a bucket may hold.
pub const MIN_BUCKET_ROWS: usize = 32;

/// Buckets actually used on a table of `n` rows.
pub fn exogenous_buckets(n: usize) -> usize {
    EXOGENOUS_BUCKETS.min(n / MIN_BUCKET_ROWS).max(1)
}

/// The exactly-balanced `m`-bin view of a continuous column.
pub fn balanced_bins(values: &[f64], m: usize) -> Vec<i64> {
    let n = values.len();
    if n == 0 || m == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|left, right| {
        values[*left]
            .total_cmp(&values[*right])
            .then(left.cmp(right))
    });
    let mut out = vec![0i64; n];
    for (rank, row) in order.into_iter().enumerate() {
        out[row] = ((rank * m / n).min(m - 1)) as i64;
    }
    out
}

/// Balanced-rank buckets that keep TIED CHANNEL VALUES TOGETHER.
fn tied_rank_buckets(values: &[f64], buckets: usize) -> Vec<i64> {
    let n = values.len();
    if n == 0 || buckets == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|left, right| values[*left].total_cmp(&values[*right]));
    let mut out = vec![0i64; n];
    let mut start = 0usize;
    while start < n {
        let mut end = start + 1;
        while end < n && values[order[start]].total_cmp(&values[order[end]]).is_eq() {
            end += 1;
        }
        let bucket = ((start * buckets / n).min(buckets - 1)) as i64;
        for row in &order[start..end] {
            out[*row] = bucket;
        }
        start = end;
    }
    out
}

/// `I(q_m(Y) ; b_B(c))` in nats: what a predictor that reads only the channel `c` can recover of
/// the balanced `m`-bin view of the target.
pub fn bucketed_channel_information(
    channel: &[f64],
    labels: &[i64],
    m: usize,
    buckets: usize,
) -> f64 {
    let n = labels.len();
    if n == 0 || m == 0 || buckets == 0 || channel.len() != n {
        return 0.0;
    }
    let keys = tied_rank_buckets(channel, buckets.min(n));
    // `BTreeMap`: see `TIE_TOLERANCE_NATS`; the sum's last bits must not depend on a hash seed.
    let mut groups: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
    for (key, label) in keys.iter().zip(labels) {
        groups.entry(*key).or_default().push(*label);
    }
    let conditional: f64 = groups
        .values()
        .map(|bucket| (bucket.len() as f64 / n as f64) * empirical_entropy(bucket, m))
        .sum();
    empirical_entropy(labels, m) - conditional
}

/// Upper bound on the in-sample upward bias of
/// [`bucketed_channel_information`], in nats: `(m - 1)(B - 1) / 2n`.
pub fn bucketed_information_bias_bound(m: usize, buckets: usize, n: usize) -> f64 {
    if n == 0 || m == 0 || buckets == 0 {
        return 0.0;
    }
    (m as f64 - 1.0) * (buckets.min(n) as f64 - 1.0) / (2.0 * n as f64)
}

/// One channel a continuous family declares it must not be a relabelling of.
pub struct ExogenousChannel<'a> {
    /// What the channel is, in the certificate. Never inferred from the values.
    pub name: &'a str,
    pub values: &'a [f64],
}

/// The strongest of a family's declared exogenous channels.
pub fn strongest_exogenous_channel<'a>(
    channels: &'a [ExogenousChannel<'a>],
    labels: &[i64],
    m: usize,
) -> Option<(&'a str, f64)> {
    let buckets = exogenous_buckets(labels.len());
    channels
        .iter()
        .map(|channel| {
            (
                channel.name,
                bucketed_channel_information(channel.values, labels, m, buckets),
            )
        })
        .fold(None, |best: Option<(&str, f64)>, current| match best {
            // Strictly greater wins; a TIE is broken by name, ascending.
            Some(previous) => {
                let delta = current.1 - previous.1;
                if delta.abs() <= TIE_TOLERANCE_NATS {
                    if current.0 < previous.0 {
                        Some(current)
                    } else {
                        Some(previous)
                    }
                } else if delta > 0.0 {
                    Some(current)
                } else {
                    Some(previous)
                }
            }
            None => Some(current),
        })
}

/// The liveness input a continuous family supplies, and which of the two liveness gates it asks
/// for.
pub enum ContinuousLiveness<'a> {
    Latent {
        name: &'a str,
        values: &'a [f64],
    },
    /// [`GateKind::LabelChannelResidual`]: `Y_0`, the same mechanism with the
    /// label channel deleted.
    Ablation {
        name: &'a str,
        values: &'a [f64],
    },
}

/// Everything a continuous target needs to be gated on.
pub struct ContinuousTargetGateInput<'a> {
    pub family: &'a str,
    pub arm: &'a str,
    pub target_table: &'a TableData,
    /// The published column, or the shared-latent stem when a family cuts
    /// several views from one latent and gates the latent once.
    pub target: &'a str,
    pub values: &'a [f64],
    /// Every mechanism-free channel this family concedes a model can read.
    /// The gate reports the strongest.
    pub exogenous: &'a [ExogenousChannel<'a>],
    pub liveness: ContinuousLiveness<'a>,
    /// Normally [`SHORTCUT_CEILING`]; an arm admitted above it on a stated
    /// argument carries its own and says so.
    pub bag_ceiling: f64,
    pub bag_declaration: Option<&'a str>,
    pub exogenous_ceiling: f64,
    pub exogenous_declaration: Option<&'a str>,
    /// The columns that will NOT be in this target's context.
    pub disclosure_exclusions: Option<&'a [String]>,
}

/// The three verdicts a table below [`MIN_GATEABLE_ROWS`] gets.
fn ungateable_verdicts(input: &ContinuousTargetGateInput<'_>, n: usize) -> Vec<ShortcutVerdict> {
    let mut kinds = vec![
        GateKind::NeighbourLabelBag,
        GateKind::ExogenousChannel,
        match input.liveness {
            ContinuousLiveness::Latent { .. } => GateKind::MechanismDeterminacy,
            ContinuousLiveness::Ablation { .. } => GateKind::LabelChannelResidual,
        },
    ];
    // A family that turned the disclosure gate ON gets a declared non-claim
    // here rather than a missing row: the whole point of the gate is that a
    // missing verdict reads like a passing one.
    if input.disclosure_exclusions.is_some() {
        kinds.push(GateKind::SiblingColumnDisclosure);
    }
    kinds
        .into_iter()
        .map(|gate| ShortcutVerdict {
            gate,
            family: input.family.to_string(),
            arm: input.arm.to_string(),
            target_table: input.target_table.name.clone(),
            target: input.target.to_string(),
            k: 0,
            measured_nats: 0.0,
            budget_nats: 0.0,
            ratio: 0.0,
            threshold_ratio: if gate.is_shortcut() {
                SHORTCUT_CEILING
            } else {
                LIVENESS_FLOOR
            },
            passed: true,
            declared_exception: Some(UNGATEABLE_DECLARATION.to_string()),
            mechanism: None,
            read_as_coordinate_not_verdict: None,
            candidates_rejected: None,
            feature_contract: input.target_table.feature_contract,
            bag_summary: input.target_table.bag_summary,
            note: format!("{n} rows, below the {MIN_GATEABLE_ROWS}-row minimum; no claim made"),
        })
        .collect()
}

/// [`GateKind::SiblingColumnDisclosure`] on an ALREADY QUANTISED view.
#[allow(clippy::too_many_arguments)]
pub fn quantised_disclosure_verdict(
    family: &str,
    arm: &str,
    target_table: &TableData,
    target: &str,
    labels: &[i64],
    m: usize,
    budget: f64,
    excluded: &[String],
) -> ShortcutVerdict {
    let disclosure = strongest_column_disclosure(target_table, labels, m, excluded);
    let (measured, note) = match &disclosure {
        Some(measurement) => (
            measurement.information_nats,
            format!(
                "balanced {m}-bin view; strongest of {} in-context column(s) is {}. A                      continuous target discloses through a sibling exactly when the sibling                      pins its RANK, so the statistic is I(q_m(Y); q_m(c)) and not an R^2: a                      monotone recoding of either side leaves it unchanged, and no row can                      carry more than 1/n of the budget. In-sample upward bias is at most                      {:.4} nats at this m and row count",
                measurement.columns_scanned,
                measurement.column,
                ((m as f64) - 1.0).powi(2) / (2.0 * labels.len().max(1) as f64),
            ),
        ),
        None => (
            0.0,
            "the target table carries no other numeric column, so nothing is left in \
             context to read the answer off"
                .to_string(),
        ),
    };
    let mut out = verdict(
        GateKind::SiblingColumnDisclosure,
        family,
        arm,
        &target_table.name,
        target,
        m,
        measured,
        budget,
        SHORTCUT_CEILING,
        None,
        &note,
    );
    // This function is an entry point in its own right -- `evaluate_mech_*`
    // calls it without going through any `evaluate_*_target` -- so it stamps
    // rather than relying on a caller's exit stamp.
    stamp_contract(std::slice::from_mut(&mut out), target_table);
    out
}

/// [`GateKind::ExogenousChannel`] for one already-quantised target.
#[allow(clippy::too_many_arguments)]
pub fn exogenous_channel_verdict(
    family: &str,
    arm: &str,
    target_table: &TableData,
    target: &str,
    labels: &[i64],
    m: usize,
    budget: f64,
    channels: &[ExogenousChannel<'_>],
    ceiling: f64,
    declaration: Option<String>,
) -> ShortcutVerdict {
    let n = labels.len();
    let buckets = exogenous_buckets(n);
    let (measured, note) = match strongest_exogenous_channel(channels, labels, m) {
        Some((name, measured)) => (
            measured,
            format!(
                "strongest of {} declared channel(s), {name}, at {buckets} balanced-rank \
                 buckets; a target this channel already determines needs neither a label nor \
                 an edge. In-sample upward bias is at most {:.4} nats",
                channels.len(),
                bucketed_information_bias_bound(m, buckets, n),
            ),
        ),
        None => (
            0.0,
            "the family declared no mechanism-free channel, so this gate measured nothing"
                .to_string(),
        ),
    };
    let mut out = verdict(
        GateKind::ExogenousChannel,
        family,
        arm,
        &target_table.name,
        target,
        m,
        measured,
        budget,
        ceiling,
        declaration,
        &note,
    );
    stamp_contract(std::slice::from_mut(&mut out), target_table);
    out
}

/// How many distinct values a continuous column still has after `rustler pre` writes it into a
/// bundle.
pub fn f32_projection_distinct(values: &[f64]) -> usize {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return 0;
    }
    let n = finite.len() as f64;
    let mean = finite.iter().sum::<f64>() / n;
    let std = if finite.len() > 1 {
        (finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    } else {
        0.0
    };
    let std = if std == 0.0 { 1.0 } else { std };
    let mut projected: Vec<f32> = finite.iter().map(|v| ((v - mean) / std) as f32).collect();
    projected.sort_by(f32::total_cmp);
    projected.dedup();
    projected.len()
}

/// [`GateKind::ProjectedResolution`] on one published continuous column.
pub fn projected_resolution_verdict(
    family: &str,
    arm: &str,
    target_table: &TableData,
    target: &str,
    values: &[f64],
    m: usize,
) -> ShortcutVerdict {
    let distinct = f32_projection_distinct(values);
    let raw = {
        let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        sorted.sort_by(f64::total_cmp);
        sorted.dedup();
        sorted.len()
    };
    let measured = (distinct.max(1) as f64).ln();
    let budget = (m.max(2) as f64).ln();
    verdict(
        GateKind::ProjectedResolution,
        family,
        arm,
        &target_table.name,
        target,
        m,
        measured,
        budget,
        1.0,
        None,
        &format!(
            "{raw} distinct f64 values in the generated column, {distinct} after the \
             (v - mean) / std -> f32 projection rustler pre performs. The floor is the view \
             width m = {m}: below it the bundle cannot separate the bins this target is \
             gated on, and the balanced view is filled from row order instead"
        ),
    )
}

/// Run the three continuous gates: the bag shortcut, the exogenous-channel
/// shortcut, and whichever liveness floor the family declared.
pub fn evaluate_continuous_target(input: &ContinuousTargetGateInput<'_>) -> Vec<ShortcutVerdict> {
    evaluate_continuous_target_with_exclusions(input, &[])
}

/// [`evaluate_continuous_target`], with extra bag columns held out of the shortcut measurement.
pub fn evaluate_continuous_target_with_exclusions(
    input: &ContinuousTargetGateInput<'_>,
    bag_exclusions: &[&str],
) -> Vec<ShortcutVerdict> {
    let n = input.values.len();
    if n < MIN_GATEABLE_ROWS {
        return ungateable_verdicts(input, n);
    }
    let m = continuous_bins(n);
    let labels = balanced_bins(input.values, m);
    // The same rule as the discrete path.
    let budget = empirical_entropy(&labels, m);
    let mut out = Vec::with_capacity(4);

    // Before any of the three: the column has to survive the preprocessor.
    out.push(projected_resolution_verdict(
        input.family,
        input.arm,
        input.target_table,
        input.target,
        input.values,
        m,
    ));

    let (bag, held_out) =
        strongest_bag_shortcut_excluding(input.target_table, &labels, m, bag_exclusions);
    let (bag_measured, bag_note) = match &bag {
        Some(measurement) => (
            measurement.information_nats,
            format!(
                "balanced {m}-bin view; strongest foreign key {} ({} of {} rows have a labelled \
                 sibling). Bag summary {} over an occupied alphabet of {}; in-sample upward \
                 bias is at most {:.4} nats at this m and row count",
                measurement.fk_column,
                measurement.rows_with_siblings,
                measurement.rows,
                measurement.summary.as_str(),
                measurement.alphabet,
                measurement.bias_bound_nats(m),
            ),
        ),
        None if held_out.is_empty() => (
            0.0,
            "the target table has no foreign key, so no sibling bag exists to copy from"
                .to_string(),
        ),
        None => (
            0.0,
            "every foreign key on this table is one the family aggregates over, so this gate \
             makes no independent claim here; the sibling term is measured by the \
             label-channel coordinate instead"
                .to_string(),
        ),
    };
    // The held-out measurements are carried on the verdict, never dropped: an
    // exclusion nobody can see is an exemption.
    let held_out_note = if held_out.is_empty() {
        String::new()
    } else {
        let rendered: Vec<String> = held_out
            .iter()
            .map(|measurement| {
                format!(
                    "{}={:.4} nats ({:.1}% of budget)",
                    measurement.fk_column,
                    measurement.information_nats,
                    100.0 * measurement.information_nats / budget,
                )
            })
            .collect();
        format!(
            ". Held out of the maximum because the family aggregates over them, measured \
             anyway: {}",
            rendered.join(", "),
        )
    };
    let bag_declaration = if held_out.is_empty() {
        input.bag_declaration.map(str::to_string)
    } else {
        Some(
            input
                .bag_declaration
                .unwrap_or(
                    "the family's own equation contains an aggregate over this foreign key's \
                     siblings, so a bag measurement on it is the mechanism rather than a route \
                     around it",
                )
                .to_string(),
        )
    };
    out.push(verdict(
        GateKind::NeighbourLabelBag,
        input.family,
        input.arm,
        &input.target_table.name,
        input.target,
        m,
        bag_measured,
        budget,
        input.bag_ceiling,
        bag_declaration,
        &format!("{bag_note}{held_out_note}"),
    ));

    let buckets = exogenous_buckets(n);
    out.push(exogenous_channel_verdict(
        input.family,
        input.arm,
        input.target_table,
        input.target,
        &labels,
        m,
        budget,
        input.exogenous,
        input.exogenous_ceiling,
        input.exogenous_declaration.map(str::to_string),
    ));

    if let Some(excluded) = input.disclosure_exclusions {
        out.push(quantised_disclosure_verdict(
            input.family,
            input.arm,
            input.target_table,
            input.target,
            &labels,
            m,
            budget,
            excluded,
        ));
    }

    let (kind, measured, note) = match &input.liveness {
        ContinuousLiveness::Latent { name, values } => (
            GateKind::MechanismDeterminacy,
            bucketed_channel_information(values, &labels, m, buckets),
            format!(
                "the published column must still be a function of {name}, the latent the \
                 mechanism computed before the family's noise draw; a mechanism that degenerated \
                 leaves a column of noise, and every shortcut ceiling passes that"
            ),
        ),
        ContinuousLiveness::Ablation { name, values } => (
            GateKind::LabelChannelResidual,
            budget - bucketed_channel_information(values, &labels, m, buckets),
            format!(
                "{name} is the same mechanism with the label channel deleted; at least half the \
                 budget has to be destroyed by deleting it, or the family's support-required \
                 claim is false for this table"
            ),
        ),
    };
    let liveness = verdict(
        kind,
        input.family,
        input.arm,
        &input.target_table.name,
        input.target,
        m,
        measured,
        budget,
        LIVENESS_FLOOR,
        None,
        &note,
    );
    out.push(liveness);
    stamp_contract(&mut out, input.target_table);
    out
}

/// Run the support-binding liveness gate over one binding group.
pub fn evaluate_support_bindings(
    family: &str,
    target_table: &TableData,
    stem: &str,
    k: usize,
    bindings: &[Vec<i64>],
) -> ShortcutVerdict {
    let ln_b = (bindings.len() as f64).ln();
    let mut out = verdict(
        GateKind::SupportBindingResidual,
        family,
        "binding",
        &target_table.name,
        stem,
        k,
        support_binding_residual(bindings, k),
        ln_b,
        LIVENESS_FLOOR,
        None,
        "the class identity must stay unidentifiable from features alone: at least half of \
         the ln B nats that distinguish the published bindings has to be left for the \
         support labels to supply",
    );
    // The table, not just its name, so that this gate's verdict carries the same provenance every
    // other one does.
    stamp_contract(std::slice::from_mut(&mut out), target_table);
    out
}
