use crate::hsbm::{assign_cluster_at_levels, fast_get_bipartite_hsbm, BipartiteHsbm};
use crate::labels::{add_axis_labels, LabelMetadata};
use crate::rng::{child_substream, substream, zscore, GenRng, RngExt};
use crate::schema::{build_random_schema, ColumnSpec, SchemaOptions, SemType, TableKind};
use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

pub const ENTITY_ID_COL: &str = "entity_id";

#[derive(Debug, Clone, Serialize)]
pub enum ColumnData {
    I64(Vec<i64>),
    F64(Vec<Option<f64>>),
    Bool(Vec<Option<bool>>),
    Utf8(Vec<Option<String>>),
    TimestampNs(Vec<i64>),
}

impl ColumnData {
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::I64(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::Utf8(v) => v.len(),
            Self::TimestampNs(v) => v.len(),
        }
    }

    pub(crate) fn as_numeric_lossy(&self) -> Option<Vec<f64>> {
        match self {
            Self::I64(values) => Some(values.iter().map(|v| *v as f64).collect()),
            Self::F64(values) => Some(values.iter().map(|v| v.unwrap_or(f64::NAN)).collect()),
            Self::Bool(values) => Some(
                values
                    .iter()
                    .map(|v| match v {
                        Some(true) => 1.0,
                        Some(false) => 0.0,
                        None => f64::NAN,
                    })
                    .collect(),
            ),
            Self::Utf8(_) => None,
            Self::TimestampNs(_) => None,
        }
    }
}

/// Every column-name prefix under which some label family PUBLISHES a target.
pub const PUBLISHED_TARGET_PREFIXES: &[&str] = &[
    "v6_y_",
    "p2v5_y_",
    "p2v4_y_",
    "p2_y_",
    "hsbm_",
    "scm_",
    "oplabel_",
    "hop_",
    "aggident_",
    "bagc",
    "bagns",
    "bipartiteC_",
    "comembpeer_",
    "clf\u{3b1}",
    "cntclf_",
    "cntreg_",
    "evx_",
    // evs_ is the latent-state family's OBSERVABLE child payload (same status
    // and same caveat as evx_).
    "evs_",
    "featclf_",
    "featimb_",
    "featreg_",
    "satfeat_r",
    "satfeat_k2",
    "satnull_r",
    "satnull_k2",
    "junmix_r",
    "junmix_k2",
    "juncount_r",
    "juncount_k2",
    "hubhist_r",
    "hubhist_k2",
    "hubnull_r",
    "hubnull_k2",
    "xhub_r",
    "xhub_k2",
    "xhub_add_r",
    "xhub_add_k2",
    "streamrec_r",
    "streamrec_k2",
    "streamnull_r",
    "streamnull_k2",
    "remote_evidence_v1_",
    "remote_evidence_long_v1_",
    "hubrate_r",
    "hubrate_k2",
    "hubrate_null_r",
    "hubrate_null_k2",
    "hubhistj_r",
    "hubhistj_k2",
    "hubhistj_null_r",
    "hubhistj_null_k2",
    "sj_", // S-J observables never become another generator family input
    "relnull_r",
    "relnull_k2",
    "relnull_imb",
    "relnull_x_", // scoped observable, excluded from other generator families
    "relnull_evx_",
    "aggcol_", // readouts and scoped DFS observables; never generator inputs
    "grankclf_",
    "grankreg_",
    "hazgapreg_",
    "haznext_",
    "mident_",
    "mrflab_",
    "noisyimb_",
    "orth6_",
    "osclf_",
    "osreg_",
    "paired30_",
    "pathhomo_",
    "phi_p",
    "phw10_",
    "plookup2_",
    "xfun",
    "xstate",
    "xroute",
    "plookup_",
    "plookuprec_",
    "plookupt_",
    "proto_k",
    "qsel",
    "ratioclf_",
    "ratioreg_",
    "reg\u{3b1}",
    "route",
    "tempk",
    "vec_",
    // The manifest families publish under opaque names drawn per record.
    "target_",
    "regression_target_",
    "xagg_",
    // condrel/xrep targets only. Predicate/evidence columns deliberately use other
    // names so they remain encoder-visible ordinary inputs.
    "condrel_r",
    "condrel_k2",
    "condrel_permuted_r",
    "condrel_permuted_k2",
    "xrep_y_",
    "mpnn_",
    "poplaw_",
    "evc_",
];

impl FeatureContract {
    /// `true` when `name` is a column some family publishes as a target.
    pub(crate) fn is_published_target(name: &str) -> bool {
        PUBLISHED_TARGET_PREFIXES
            .iter()
            .any(|prefix| name.starts_with(prefix))
    }
}

/// Which definition of "this column is a model input" a table answers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureContract {
    /// The four prefixes the generator has always excluded.
    #[default]
    V1FourPrefixes,
    /// [`PUBLISHED_TARGET_PREFIXES`] in full.
    ExcludeEveryPublishedTarget,
}

/// How a `mech_group_rank` contest lays its members out on the rung ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RankFieldContract {
    /// Every rung of the ladder used exactly once.
    #[default]
    V1FullLadder,
    /// The field is `m` independent draws on the ladder's scale.
    SampledField,
}

#[derive(Debug, Clone, Serialize)]
pub struct TableData {
    pub name: String,
    pub pkey_col: String,
    pub fkey_col_to_pkey_table: BTreeMap<String, String>,
    pub time_col: Option<String>,
    pub columns: Vec<(String, ColumnData)>,
    pub hsbm_hierarchy: Vec<Vec<i64>>,
    pub fk_hsbm: BTreeMap<String, BipartiteHsbm>,
    /// See [`FeatureContract`].
    #[serde(default)]
    pub feature_contract: FeatureContract,
    /// Which summary of a foreign-key sibling bag
    /// [`crate::shortcut_gates::GateKind::NeighbourLabelBag`] conditions on for targets published
    /// on this table.
    #[serde(default)]
    pub bag_summary: crate::shortcut_gates::BagSummary,
    /// See [`RankFieldContract`].
    #[serde(default)]
    pub rank_field: RankFieldContract,
}

impl TableData {
    pub(crate) fn num_rows(&self) -> usize {
        self.columns
            .first()
            .map(|(_, col)| col.len())
            .unwrap_or_default()
    }

    pub fn column_names(&self) -> impl Iterator<Item = &str> {
        self.columns.iter().map(|(name, _)| name.as_str())
    }

    pub fn column(&self, name: &str) -> Option<&ColumnData> {
        self.columns
            .iter()
            .find_map(|(col_name, data)| (col_name == name).then_some(data))
    }

    pub(crate) fn push_column(&mut self, name: String, data: ColumnData) {
        assert_eq!(
            data.len(),
            self.num_rows(),
            "new column length must match table length"
        );
        self.columns.push((name, data));
    }

    pub(crate) fn next_column_name(&self, base: &str) -> String {
        let existing: std::collections::HashSet<&str> = self.column_names().collect();
        if !existing.contains(base) {
            return base.to_string();
        }
        for suffix in 1usize.. {
            let candidate = format!("{base}_{suffix}");
            if !existing.contains(candidate.as_str()) {
                return candidate;
            }
        }
        unreachable!()
    }

    /// The columns this table offers as model inputs.
    pub(crate) fn numeric_feature_columns(&self) -> Vec<String> {
        self.columns
            .iter()
            .filter_map(|(name, data)| {
                if name == &self.pkey_col
                    || self.fkey_col_to_pkey_table.contains_key(name)
                    || name == ENTITY_ID_COL
                    || name == "date"
                {
                    return None;
                }
                let excluded = match self.feature_contract {
                    FeatureContract::V1FourPrefixes => {
                        name.starts_with("hsbm_")
                            || name.starts_with("scm_")
                            || name.starts_with("oplabel_")
                            || name.starts_with("hop_")
                    }
                    FeatureContract::ExcludeEveryPublishedTarget => {
                        FeatureContract::is_published_target(name)
                    }
                };
                // Additive scoped families never enter another generator
                // family's inputs, under either feature contract.
                if excluded
                    || name.starts_with("relnull_")
                    || name.starts_with("plookupt_")
                    || name.starts_with("aggcol_")
                    || name.starts_with("sj_")
                    || name.starts_with("mpnn_")
                    || name.starts_with("poplaw_")
                    || name.starts_with("p2_y_")
                    || name.starts_with("p2v4_y_")
                    || crate::sj::is_target(name)
                {
                    return None;
                }
                data.as_numeric_lossy().map(|_| name.clone())
            })
            .collect()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GeneratedDb {
    pub name: String,
    pub seed: u64,
    pub tables: Vec<TableData>,
    pub program_metadata: Vec<LabelMetadata>,
    /// What the mean-shaped construction filter rejected, if it ran.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mean_shaped_filter: Vec<crate::mean_shaped::MeanShapedFilterReport>,
    /// Verdicts from the construction-time shortcut gates that [`crate::shortcut_gates`] applies to
    /// the discrete manifest families.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shortcut_gates: Vec<crate::shortcut_gates::ShortcutVerdict>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub gate_verdict_claims: BTreeMap<String, Vec<usize>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exact_aggregates: Vec<crate::labels_v2::ExactAggregateColumn>,
    /// Paired-neighbourhood receipt of the `condrel_*` targets. It is never materialised as a table.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condrel_receipt: Option<crate::sj_families::CondrelReceipt>,
    /// Rendering manifest of the `xrep_*` targets. It is never materialised as a table.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub xrep_manifest: Option<crate::sj_families::XrepManifest>,
}

impl GeneratedDb {
    /// Every target already declared on `table`.
    pub(crate) fn declared_targets_on_table(&self, table: &str) -> Vec<String> {
        let mut out: Vec<String> = self
            .program_metadata
            .iter()
            .filter(|metadata| metadata.target_table == table)
            .map(|metadata| metadata.target_col.clone())
            .collect();
        out.sort();
        out.dedup();
        out
    }

    pub fn table(&self, name: &str) -> Option<&TableData> {
        self.tables.iter().find(|table| table.name == name)
    }

    /// Record the shortcut-gate verdicts for one published target.
    pub(crate) fn record_shortcut_verdicts(
        &mut self,
        verdicts: impl IntoIterator<Item = crate::shortcut_gates::ShortcutVerdict>,
    ) -> std::ops::Range<usize> {
        let start = self.shortcut_gates.len();
        self.shortcut_gates.extend(verdicts);
        start..self.shortcut_gates.len()
    }

    /// Say that the whole verdict batch in `range` is ABOUT the column published as
    /// `published_column`.
    pub(crate) fn claim_gate_verdicts(
        &mut self,
        range: std::ops::Range<usize>,
        published_column: &str,
    ) {
        let claimed: Vec<usize> = range
            .filter(|index| *index < self.shortcut_gates.len())
            .collect();
        self.insert_gate_claim(published_column, claimed);
    }

    /// The same, narrowed to the verdicts in `range` that name `verdict_target`.
    pub(crate) fn claim_gate_verdicts_named(
        &mut self,
        range: std::ops::Range<usize>,
        verdict_target: &str,
        published_column: &str,
    ) {
        let claimed: Vec<usize> = range
            .filter(|index| {
                self.shortcut_gates
                    .get(*index)
                    .is_some_and(|verdict| verdict.target == verdict_target)
            })
            .collect();
        self.insert_gate_claim(published_column, claimed);
    }

    fn insert_gate_claim(&mut self, published_column: &str, claimed: Vec<usize>) {
        if claimed.is_empty() {
            return;
        }
        let entry = self
            .gate_verdict_claims
            .entry(published_column.to_string())
            .or_default();
        entry.extend(claimed);
        entry.sort_unstable();
        entry.dedup();
    }

    pub(crate) fn apply_mechanism_contract(&mut self) {
        let feature = FeatureContract::V1FourPrefixes;
        let bag = crate::shortcut_gates::BagSummary::ModeUnanimity;
        let rank_field = RankFieldContract::V1FullLadder;
        for table in &mut self.tables {
            table.feature_contract = feature;
            table.bag_summary = bag;
            table.rank_field = rank_field;
        }
    }

    /// Every recorded verdict that did not pass, in record order.
    pub fn failed_shortcut_gates(&self) -> Vec<&crate::shortcut_gates::ShortcutVerdict> {
        self.shortcut_gates
            .iter()
            .filter(|verdict| !verdict.passed)
            .collect()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GenOptions {
    pub schema: SchemaOptions,
    pub hsbm_levels_range: (usize, usize),
    pub hsbm_clusters_per_level_range: (usize, usize),
    /// Refuse to publish a continuous label column that hands a shortcut more than half its budget,
    /// or that survives having its mechanism deleted.
    #[serde(default = "default_true")]
    pub shortcut_gate_enforcement: bool,
    /// Emit `xagg_*` exact-aggregate sidecar columns (`labels_v2::add_exact_aggregate_sidecars`)
    /// on the `TemporalKernel` axis.
    #[serde(default)]
    pub exact_aggregates: bool,
    #[serde(default)]
    pub nuisance_text: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub v6: Option<crate::v6::V6World>,
}

impl GenOptions {
    pub fn validate_corpus_scale(&self) -> Result<()> {
        self.schema.validate_scale()
    }

    pub fn corpus_scale_enabled(&self) -> bool {
        self.schema.fanout_tail.is_some()
            || self.schema.max_depth.is_some()
            || self.schema.min_depth.is_some()
    }
}

impl Default for GenOptions {
    fn default() -> Self {
        Self {
            schema: SchemaOptions::default(),
            hsbm_levels_range: (2, 3),
            hsbm_clusters_per_level_range: (2, 4),
            shortcut_gate_enforcement: true,
            exact_aggregates: false,
            nuisance_text: false,
            v6: None,
        }
    }
}

/// Assign each row an exactly equal-frequency bucket of `values`.
pub fn balanced_quantiles(values: &[f64], k: usize) -> Vec<usize> {
    let n = values.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| values[*a].total_cmp(&values[*b]).then_with(|| a.cmp(b)));
    let mut labels = vec![0usize; n];
    for (rank, row) in order.into_iter().enumerate() {
        labels[row] = (rank * k / n).min(k - 1);
    }
    labels
}

pub fn generate_db(seed: u64, options: &GenOptions) -> Result<GeneratedDb> {
    let mut db = generate_db_skeleton(seed, options)?;
    let sj_streams = crate::sj::take_streams(&mut db);
    db.apply_mechanism_contract();
    add_axis_labels(&mut db, seed, options)?;
    db.tables.extend(sj_streams);
    crate::labels_v2::add_sj_extension_labels(&mut db, seed);
    if let Some(stats) = crate::sj::statistics(&db) {
        anyhow::ensure!(
            (options.schema.num_tables_range.0..=options.schema.num_tables_range.1)
                .contains(&stats.total_tables),
            "S-J materialised total is outside table budget"
        );
        anyhow::ensure!(
            stats.junction_tables <= options.schema.max_junction_tables,
            "S-J materialised junction count exceeds cap"
        );
    }
    validate_corpus_scale_db(&db, options)?;
    Ok(db)
}

pub fn validate_corpus_scale_db(db: &GeneratedDb, options: &GenOptions) -> Result<()> {
    if let Some(min) = options.schema.min_depth {
        let actual = crate::manifest::corpus_scale_coverage(db, None).depth;
        anyhow::ensure!(
            actual >= min,
            "generated depth {actual} below min-depth {min}"
        );
    }
    if let Some(max_depth) = options.schema.max_depth {
        let actual = crate::manifest::corpus_scale_coverage(db, None).depth;
        anyhow::ensure!(
            actual <= max_depth,
            "generated FK depth {actual} exceeds --depth {max_depth}"
        );
    }
    if let Some(tail) = options.schema.fanout_tail {
        for child in &db.tables {
            for fk in child.fkey_col_to_pkey_table.keys() {
                let Some(ColumnData::I64(values)) = child.column(fk) else {
                    continue;
                };
                let mut counts: HashMap<i64, usize> = HashMap::new();
                for &value in values {
                    if value < 0 {
                        continue;
                    }
                    let count = counts.entry(value).or_default();
                    *count += 1;
                    anyhow::ensure!(
                        *count <= tail,
                        "generated fan-out {} exceeds --fanout-tail {tail} on {}.{fk}",
                        *count,
                        child.name
                    );
                }
            }
        }
    }
    Ok(())
}

/// Build schema + tables but DO NOT attach axis labels, so callers can apply
/// multiple z's to the same (G, X).
pub fn generate_db_skeleton(seed: u64, options: &GenOptions) -> Result<GeneratedDb> {
    generate_db_skeleton_with_schema_seed(seed, seed, options)
}

/// Build a skeleton with independent data and schema RNG streams.
pub(crate) fn generate_db_skeleton_with_schema_seed(
    data_seed: u64,
    schema_seed: u64,
    options: &GenOptions,
) -> Result<GeneratedDb> {
    options.validate_corpus_scale()?;
    if options.schema.sj_enabled() {
        options.schema.validate_sj()?;
    }
    let mut schema_rng = substream(schema_seed, "schema");
    let table_options = options;
    let schema = build_random_schema(&mut schema_rng, &options.schema)?;

    let mut timestamp_rng = substream(data_seed, "timestamps");
    let (min_ts_ns, max_ts_ns) = sample_timestamp_window(&mut timestamp_rng);

    let mut tables: Vec<Option<TableData>> = vec![None; schema.tables.len()];
    for &table_id in &schema.topo_order {
        let table_spec = &schema.tables[table_id];
        let mut table_rng = child_substream(data_seed, "table", table_id);
        let table = generate_table(
            data_seed,
            &mut table_rng,
            table_spec,
            &tables,
            min_ts_ns,
            max_ts_ns,
            table_options,
        )
        .with_context(|| format!("failed generating {}", table_spec.name))?;
        tables[table_id] = Some(table);
    }

    if options.schema.sj_enabled() {
        anyhow::ensure!(
            tables.iter().flatten().any(|t| t
                .time_col
                .as_ref()
                .is_some_and(|c| matches!(t.column(c), Some(ColumnData::TimestampNs(_))))),
            "S-J corpus contract: every database must carry a real time column"
        );
    }
    Ok(GeneratedDb {
        name: format!("rel-synthetic-{data_seed}"),
        seed: data_seed,
        tables: tables
            .into_iter()
            .map(|table| table.expect("all tables generated in topo order"))
            .collect(),
        program_metadata: Vec::new(),
        mean_shaped_filter: Vec::new(),
        shortcut_gates: Vec::new(),
        gate_verdict_claims: BTreeMap::new(),
        exact_aggregates: Vec::new(),
        condrel_receipt: None,
        xrep_manifest: None,
    })
}

fn generate_table(
    seed: u64,
    rng: &mut GenRng,
    spec: &crate::schema::TableSpec,
    prior_tables: &[Option<TableData>],
    min_ts_ns: i64,
    max_ts_ns: i64,
    options: &GenOptions,
) -> Result<TableData> {
    if options.schema.sj_enabled() {
        return crate::sj::generate_table(seed, spec, prior_tables, min_ts_ns, max_ts_ns, options);
    }
    let n = spec.num_rows;
    let mut columns: Vec<(String, ColumnData)> = Vec::new();
    columns.push((
        spec.pkey_col.clone(),
        ColumnData::I64((0..n).map(|idx| idx as i64).collect()),
    ));

    let mut fkey_col_to_pkey_table = BTreeMap::new();
    let mut fk_hsbm = BTreeMap::new();
    let mut primary_hierarchy: Option<Vec<Vec<i64>>> = None;
    let mut parent_feature_cache: HashMap<String, Vec<f64>> = HashMap::new();

    for (fk_idx, fk) in spec.fkeys.iter().enumerate() {
        let parent = prior_tables
            .get(fk.parent_id)
            .and_then(Option::as_ref)
            .with_context(|| format!("parent {} unavailable", fk.parent_table))?;
        let levels = rng.usize_inclusive(options.hsbm_levels_range.0, options.hsbm_levels_range.1);
        let hierarchy_a = (0..levels)
            .map(|_| {
                rng.usize_inclusive(
                    options.hsbm_clusters_per_level_range.0,
                    options.hsbm_clusters_per_level_range.1,
                )
            })
            .collect::<Vec<_>>();
        let hierarchy_b = (0..levels)
            .map(|_| {
                rng.usize_inclusive(
                    options.hsbm_clusters_per_level_range.0,
                    options.hsbm_clusters_per_level_range.1,
                )
            })
            .collect::<Vec<_>>();
        let mut topo_rng =
            child_substream(seed, &format!("topology:{}:{fk_idx}", spec.name), fk_idx);
        let hsbm = fast_get_bipartite_hsbm(
            &mut topo_rng,
            parent.num_rows(),
            n,
            &hierarchy_a,
            &hierarchy_b,
        );
        let mut fk_values = hsbm
            .parent_for_child
            .iter()
            .map(|idx| *idx as i64)
            .collect::<Vec<_>>();
        if let Some(tail) = options.schema.fanout_tail {
            fk_values =
                capped_long_tail_fkeys(seed, &spec.name, fk_idx, parent.num_rows(), n, tail)?;
        }
        // Opt-in heavy-tail degree law (see SchemaOptions::heavy_tail_fanout).
        if options.schema.heavy_tail_fanout && options.schema.fanout_tail.is_none() {
            let mut hub_rng = child_substream(
                seed,
                &format!("topology:heavy_tail:{}:{fk_idx}", spec.name),
                fk_idx,
            );
            let n_parent = parent.num_rows();
            if n_parent >= 4 {
                let tail_index = hub_rng.f64_range(1.1_f64.ln(), 2.5_f64.ln()).exp();
                let weights: Vec<f64> = (0..n_parent)
                    .map(|_| {
                        let u: f64 = hub_rng.f64_range(f64::MIN_POSITIVE, 1.0);
                        // Pareto(alpha) via inverse CDF; capped so one parent
                        // cannot absorb the whole table.
                        u.powf(-1.0 / tail_index).min(1e6)
                    })
                    .collect();
                let reassign_share = hub_rng.f64_range(0.35, 0.65);
                for value in fk_values.iter_mut() {
                    if hub_rng.bernoulli(reassign_share) {
                        *value = hub_rng.choose_weighted(&weights) as i64;
                    }
                }
            }
        }
        columns.push((fk.col_name.clone(), ColumnData::I64(fk_values)));
        fkey_col_to_pkey_table.insert(fk.col_name.clone(), fk.parent_table.clone());
        if primary_hierarchy.is_none() {
            primary_hierarchy = Some(
                hsbm.hierarchy_b
                    .iter()
                    .map(|row| row.iter().map(|v| *v as i64).collect())
                    .collect(),
            );
        }
        fk_hsbm.insert(fk.col_name.clone(), hsbm);
    }

    let v6_times = match (&options.v6, spec.kind) {
        (Some(world), TableKind::Activity) if !spec.fkeys.is_empty() => {
            let mut fk_columns: Vec<&mut Vec<i64>> = columns
                .iter_mut()
                .filter(|(name, _)| fkey_col_to_pkey_table.contains_key(name))
                .filter_map(|(_, data)| match data {
                    ColumnData::I64(values) => Some(values),
                    _ => None,
                })
                .collect();
            Some(crate::v6::bursty_order(
                seed,
                &spec.name,
                world,
                &mut fk_columns,
                min_ts_ns,
                max_ts_ns,
            ))
        }
        _ => None,
    };

    let hsbm_hierarchy = primary_hierarchy.unwrap_or_else(|| {
        let levels = rng.usize_inclusive(options.hsbm_levels_range.0, options.hsbm_levels_range.1);
        let hierarchy = (0..levels)
            .map(|_| {
                rng.usize_inclusive(
                    options.hsbm_clusters_per_level_range.0,
                    options.hsbm_clusters_per_level_range.1,
                )
            })
            .collect::<Vec<_>>();
        assign_cluster_at_levels(n, &hierarchy)
            .into_iter()
            .map(|row| row.into_iter().map(|v| v as i64).collect())
            .collect()
    });

    for (feature_idx, feature) in spec
        .features
        .iter()
        .enumerate()
        .filter(|(_, feature)| !matches!(feature.sem_type, SemType::Class))
    {
        let mut col_rng = child_substream(seed, &format!("columns:{}", spec.name), feature_idx);
        let parent_signal =
            parent_signal_for_table(&mut parent_feature_cache, &columns, prior_tables, spec, n);
        let data = match feature.sem_type {
            SemType::Numerical => generate_numeric_column(
                &mut col_rng,
                feature,
                spec.kind,
                n,
                parent_signal.as_deref(),
            ),
            SemType::Boolean => {
                generate_boolean_column(&mut col_rng, spec.kind, n, parent_signal.as_deref())
            }
            SemType::Class => unreachable!("class columns are generated after numeric inputs"),
        };
        columns.push((feature.name.clone(), data));
    }

    for (feature_idx, feature) in spec
        .features
        .iter()
        .enumerate()
        .filter(|(_, feature)| matches!(feature.sem_type, SemType::Class))
    {
        let mut col_rng = child_substream(seed, &format!("columns:{}", spec.name), feature_idx);
        let parent_signal =
            parent_signal_for_table(&mut parent_feature_cache, &columns, prior_tables, spec, n);
        let data = generate_class_column(
            &mut col_rng,
            &columns,
            spec,
            feature.num_classes.unwrap_or(2),
            n,
            parent_signal.as_deref(),
        );
        columns.push((feature.name.clone(), data));
    }

    if matches!(spec.kind, TableKind::Activity) {
        let timestamps = if let Some(times) = v6_times {
            times
        } else {
            linspace_i64(min_ts_ns, max_ts_ns, n)
        };
        columns.push(("date".to_string(), ColumnData::TimestampNs(timestamps)));
    }

    Ok(TableData {
        name: spec.name.clone(),
        pkey_col: spec.pkey_col.clone(),
        fkey_col_to_pkey_table,
        time_col: (matches!(spec.kind, TableKind::Activity)).then(|| "date".to_string()),
        columns,
        hsbm_hierarchy,
        fk_hsbm,
        feature_contract: FeatureContract::default(),
        bag_summary: crate::shortcut_gates::BagSummary::default(),
        rank_field: RankFieldContract::default(),
    })
}

fn capped_long_tail_fkeys(
    seed: u64,
    table_name: &str,
    fk_idx: usize,
    n_parent: usize,
    n_child: usize,
    tail: usize,
) -> Result<Vec<i64>> {
    anyhow::ensure!(n_parent > 0, "cannot assign FK into an empty parent table");
    let capacity = n_parent
        .checked_mul(tail)
        .context("fanout tail capacity overflow")?;
    anyhow::ensure!(
        n_child <= capacity,
        "fanout-tail {tail} cannot hold {n_child} child rows across {n_parent} parents"
    );
    let mut rng = child_substream(
        seed,
        &format!("topology:capped_long_tail:{table_name}:{fk_idx}"),
        fk_idx,
    );
    let alpha = rng.f64_range(1.15, 2.0);
    let weights: Vec<f64> = (0..n_parent)
        .map(|_| {
            let u = rng.f64_range(f64::MIN_POSITIVE, 1.0);
            u.powf(-1.0 / alpha).min(1e9)
        })
        .collect();
    let mut counts = vec![0usize; n_parent];
    let mut active: Vec<usize> = (0..n_parent).collect();
    let mut remaining = n_child;
    while remaining > 0 {
        let weight_sum: f64 = active.iter().map(|&p| weights[p]).sum();
        let proposals: Vec<(usize, usize)> = active
            .iter()
            .map(|&parent| {
                let room = tail - counts[parent];
                let quota =
                    ((remaining as f64 * weights[parent] / weight_sum).floor() as usize).min(room);
                (parent, quota)
            })
            .collect();
        let proposed_total: usize = proposals.iter().map(|(_, quota)| quota).sum();
        if proposed_total == 0 {
            active.sort_by(|a, b| weights[*b].total_cmp(&weights[*a]));
            let start = rng.choose_index(active.len());
            for offset in 0..active.len().min(remaining) {
                let parent = active[(start + offset) % active.len()];
                if counts[parent] < tail {
                    counts[parent] += 1;
                    remaining -= 1;
                    if remaining == 0 {
                        break;
                    }
                }
            }
        } else {
            for (parent, quota) in proposals {
                counts[parent] += quota;
            }
            remaining -= proposed_total;
        }
        active.retain(|&parent| counts[parent] < tail);
        anyhow::ensure!(
            !active.is_empty() || remaining == 0,
            "fanout allocation exhausted capacity"
        );
    }
    let mut values = Vec::with_capacity(n_child);
    for (parent, count) in counts.into_iter().enumerate() {
        values.extend(std::iter::repeat(parent as i64).take(count));
    }
    rng.shuffle(&mut values);
    Ok(values)
}

fn parent_signal_for_table(
    cache: &mut HashMap<String, Vec<f64>>,
    columns: &[(String, ColumnData)],
    prior_tables: &[Option<TableData>],
    spec: &crate::schema::TableSpec,
    n: usize,
) -> Option<Vec<f64>> {
    if spec.fkeys.is_empty() {
        return None;
    }
    let mut out = vec![0.0; n];
    let mut counts = vec![0usize; n];
    for fk in &spec.fkeys {
        let Some((_, ColumnData::I64(fk_values))) =
            columns.iter().find(|(name, _)| name == &fk.col_name)
        else {
            continue;
        };
        let Some(parent) = prior_tables.get(fk.parent_id).and_then(Option::as_ref) else {
            continue;
        };
        let signal = cache.entry(fk.parent_table.clone()).or_insert_with(|| {
            first_numeric_signal(parent).unwrap_or_else(|| vec![0.0; parent.num_rows()])
        });
        for row_idx in 0..n {
            let parent_idx = fk_values[row_idx] as usize;
            if let Some(value) = signal.get(parent_idx) {
                out[row_idx] += *value;
                counts[row_idx] += 1;
            }
        }
    }
    let mut any = false;
    for (value, count) in out.iter_mut().zip(counts.iter()) {
        if *count > 0 {
            *value /= *count as f64;
            any = true;
        }
    }
    any.then_some(zscore(&out))
}

fn first_numeric_signal(table: &TableData) -> Option<Vec<f64>> {
    for col_name in table.numeric_feature_columns() {
        if let Some(data) = table
            .column(&col_name)
            .and_then(ColumnData::as_numeric_lossy)
        {
            return Some(zscore(&data));
        }
    }
    None
}

fn generate_numeric_column(
    rng: &mut GenRng,
    _feature: &ColumnSpec,
    kind: TableKind,
    n: usize,
    parent_signal: Option<&[f64]>,
) -> ColumnData {
    let mut values = match rng.choose_slice(&["ts", "uniform", "gaussian", "beta", "mixed"]) {
        "uniform" => {
            let scale = rng.choose_slice(&[0.01, 0.1, 1.0, 10.0, 100.0]);
            let a = rng.f64_range(0.0, 1.0) * scale;
            let b = rng.f64_range(0.0, 1.0) * scale;
            let (low, high) = if a <= b { (a, b) } else { (b, a) };
            (0..n)
                .map(|_| rng.f64_range(low, high.max(low + 1e-9)))
                .collect::<Vec<_>>()
        }
        "gaussian" => {
            let scale = rng.choose_slice(&[0.01, 0.1, 1.0, 10.0, 100.0]);
            let a = rng.f64_range(0.0, 1.0) * scale;
            let b = rng.f64_range(0.0, 1.0) * scale;
            let (low, high) = if a <= b { (a, b) } else { (b, a) };
            let mean = (low + high) / 2.0;
            let std = ((high - low) / 4.0).max(1e-6);
            (0..n)
                .map(|_| rng.normal(mean, std).clamp(low, high))
                .collect::<Vec<_>>()
        }
        "beta" => {
            let scale = rng.choose_slice(&[0.01, 0.1, 1.0, 10.0, 100.0]);
            let a = rng.f64_range(0.0, 1.0) * scale;
            let b = rng.f64_range(0.0, 1.0) * scale;
            let (low, high) = if a <= b { (a, b) } else { (b, a) };
            let alpha = rng.f64_range(0.5, 5.0);
            let beta = rng.f64_range(0.5, 5.0);
            (0..n)
                .map(|_| rng.beta(alpha, beta) * (high - low) + low)
                .collect::<Vec<_>>()
        }
        "mixed" => {
            let mut out = bulk_ts_values(rng, kind, n);
            let mix = generate_numeric_column(rng, _feature, kind, n, None)
                .as_numeric_lossy()
                .unwrap_or_else(|| vec![0.0; n]);
            for (lhs, rhs) in out.iter_mut().zip(mix.iter()) {
                *lhs = 0.5 * *lhs + 0.5 * *rhs;
            }
            out
        }
        _ => bulk_ts_values(rng, kind, n),
    };

    if let Some(parent) = parent_signal {
        let weight = rng.f64_range(0.1, 0.6);
        let z = zscore(&values);
        values = z
            .iter()
            .zip(parent.iter())
            .map(|(self_v, parent_v)| {
                (1.0 - weight) * *self_v + weight * *parent_v + rng.normal(0.0, 0.03)
            })
            .collect();
    }

    if rng.bernoulli(0.18) && !values.is_empty() {
        let constant = rng.normal(0.0, 1.0);
        values.fill(constant);
    }

    apply_transform_and_nan(rng, values)
}

fn generate_boolean_column(
    rng: &mut GenRng,
    kind: TableKind,
    n: usize,
    parent_signal: Option<&[f64]>,
) -> ColumnData {
    let mut base = bulk_ts_values(rng, kind, n);
    base = zscore(&base);
    if let Some(parent) = parent_signal {
        for (value, parent_value) in base.iter_mut().zip(parent.iter()) {
            *value = 0.65 * *value + 0.35 * *parent_value;
        }
    }
    let bias = rng.f64_range(-1.25, 1.25);
    let mut values = base
        .iter()
        .map(|score| Some(*score + bias > 0.0))
        .collect::<Vec<_>>();
    let true_count = values.iter().filter(|v| matches!(v, Some(true))).count();
    if true_count == 0 && !values.is_empty() {
        values[0] = Some(true);
    } else if true_count == values.len() && !values.is_empty() {
        values[0] = Some(false);
    }
    ColumnData::Bool(values)
}

fn generate_class_column(
    rng: &mut GenRng,
    columns: &[(String, ColumnData)],
    spec: &crate::schema::TableSpec,
    num_classes: usize,
    n: usize,
    parent_signal: Option<&[f64]>,
) -> ColumnData {
    let k = num_classes.clamp(2, 32);
    let matrix = class_feature_matrix(columns, spec, n, rng);
    let num_inputs = matrix.len().max(1);
    let mut weights = vec![vec![0.0; num_inputs]; k];
    let mut biases = vec![0.0; k];
    for class_idx in 0..k {
        biases[class_idx] = rng.normal(0.0, 0.35);
        for input_idx in 0..num_inputs {
            weights[class_idx][input_idx] = rng.normal(0.0, 1.0 / (num_inputs as f64).sqrt());
        }
    }
    if let Some(parent) = parent_signal {
        for class_idx in 0..k {
            let sign = if class_idx % 2 == 0 { 1.0 } else { -1.0 };
            weights[class_idx].push(sign * rng.f64_range(0.15, 0.45));
        }
        let mut extended = matrix;
        extended.push(parent.to_vec());
        return sample_class_values(rng, &extended, &weights, &biases, k, n);
    }
    sample_class_values(rng, &matrix, &weights, &biases, k, n)
}

fn class_feature_matrix(
    columns: &[(String, ColumnData)],
    spec: &crate::schema::TableSpec,
    n: usize,
    rng: &mut GenRng,
) -> Vec<Vec<f64>> {
    let mut matrix = Vec::new();
    for (name, data) in columns {
        if name == &spec.pkey_col
            || spec.fkeys.iter().any(|fk| fk.col_name == *name)
            || name == ENTITY_ID_COL
            || name == "date"
        {
            continue;
        }
        if let Some(values) = data.as_numeric_lossy() {
            matrix.push(zscore(&values));
        }
    }
    if matrix.is_empty() {
        matrix.push(zscore(&bulk_ts_values(rng, spec.kind, n)));
    }
    matrix
}

fn sample_class_values(
    rng: &mut GenRng,
    matrix: &[Vec<f64>],
    weights: &[Vec<f64>],
    biases: &[f64],
    k: usize,
    n: usize,
) -> ColumnData {
    let mut out = Vec::with_capacity(n);
    for row in 0..n {
        let mut logits = vec![0.0; k];
        for class_idx in 0..k {
            let mut logit = biases[class_idx];
            for (input_idx, values) in matrix.iter().enumerate() {
                logit += weights[class_idx][input_idx] * values[row];
            }
            logits[class_idx] = logit;
        }
        let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let probs = logits
            .iter()
            .map(|logit| (*logit - max_logit).exp())
            .collect::<Vec<_>>();
        let class_idx = rng.choose_weighted(&probs);
        out.push(Some(format!("class_{class_idx}")));
    }
    ColumnData::Utf8(out)
}

fn apply_transform_and_nan(rng: &mut GenRng, values: Vec<f64>) -> ColumnData {
    let values = maybe_apply_heavy_tail_affine(rng, values);
    let mut values =
        match rng.choose_slice(&["identity", "rank_uniform", "log", "sqrt", "standardize"]) {
            "rank_uniform" => rank_uniform(&values),
            "log" => values
                .iter()
                .map(|v| v.signum() * v.abs().ln_1p())
                .collect::<Vec<_>>(),
            "sqrt" => values
                .iter()
                .map(|v| v.signum() * v.abs().sqrt())
                .collect::<Vec<_>>(),
            "standardize" => zscore(&values),
            _ => values,
        };
    let nan_perc = rng.f64_range(0.01, 0.1);
    let num_nan = ((nan_perc * values.len() as f64).floor() as usize).min(values.len());
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    rng.shuffle(&mut indices);
    let mut out = values.drain(..).map(Some).collect::<Vec<_>>();
    for idx in indices.into_iter().take(num_nan) {
        out[idx] = None;
    }
    ColumnData::F64(out)
}

fn maybe_apply_heavy_tail_affine(rng: &mut GenRng, values: Vec<f64>) -> Vec<f64> {
    if !rng.bernoulli(0.70) {
        return values;
    }
    let magnitude = 10.0_f64.powf(rng.f64_range(-2.0, 7.0));
    let scale = magnitude * rng.choose_slice(&[-1.0, 1.0]);
    let offset_std = magnitude * rng.f64_range(0.0, 3.0);
    let offset = rng.normal(0.0, offset_std);
    values
        .into_iter()
        .map(|value| value * scale + offset)
        .collect()
}

fn rank_uniform(values: &[f64]) -> Vec<f64> {
    let mut order = values.iter().copied().enumerate().collect::<Vec<_>>();
    order.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let mut ranks = vec![0.0; values.len()];
    for (rank, (idx, _)) in order.iter().enumerate() {
        ranks[*idx] = (rank + 1) as f64 / (values.len() + 1) as f64;
    }
    ranks
}

pub fn bulk_ts_values(rng: &mut GenRng, kind: TableKind, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    let scale = rng.choose_slice(&[0.01, 0.1, 1.0, 10.0, 100.0]);
    let a = rng.f64_range(0.0, 1.0) * scale;
    let b = rng.f64_range(0.0, 1.0) * scale;
    let (min_value, max_value) = if a <= b { (a, b) } else { (b, a) };
    let trend_alpha = rng.f64_range(0.0, 2.0);
    let trend_scale = match kind {
        TableKind::Activity => rng.choose_slice(&[-1.0, 1.0]),
        TableKind::Entity => 0.0,
    };
    let cycle_frequency = ((n as f64)
        * rng.choose_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
    .ceil()
    .max(1.0);
    let cycle_scale = match kind {
        TableKind::Activity => rng.choose_slice(&[-1.0, 1.0]),
        TableKind::Entity => 0.0,
    };
    let noise_scale = match kind {
        TableKind::Activity => 0.05,
        TableKind::Entity => 1.0,
    };
    let ar_rho = rng.f64_range(0.0, 0.9);
    let mut state = 0.0;
    let mut out = Vec::with_capacity(n);
    for row_idx in 0..n {
        let x = row_idx as f64 / n as f64;
        let trend = (trend_scale * x.powf(trend_alpha) + min_value).clamp(min_value, max_value);
        let cycle = (cycle_scale
            * ((row_idx as f64 / cycle_frequency) * std::f64::consts::PI).sin())
        .clamp(min_value, max_value);
        state = ar_rho * state + rng.standard_normal() * noise_scale;
        let noise = state.clamp(min_value, max_value);
        out.push((trend + cycle + noise) / 3.0);
    }
    out
}

fn linspace_i64(start: i64, end: i64, n: usize) -> Vec<i64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    (0..n)
        .map(|idx| {
            let t = idx as f64 / (n - 1) as f64;
            (start as f64 + (end - start) as f64 * t).round() as i64
        })
        .collect()
}

fn sample_timestamp_window(rng: &mut GenRng) -> (i64, i64) {
    const DAY_NS: i64 = 86_400_000_000_000;
    const START_NS: i64 = 631_152_000_000_000_000; // 1990-01-01 UTC
    const DAYS: i64 = 12_784;
    let a = rng.i64_inclusive(0, DAYS - 1);
    let mut b = rng.i64_inclusive(0, DAYS - 1);
    if a == b {
        b = (b + 1) % DAYS;
    }
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    (START_NS + lo * DAY_NS, START_NS + hi * DAY_NS)
}

/// A table with a `row_idx` key and no other columns.
pub(crate) fn table(name: &str, n: usize) -> TableData {
    TableData {
        name: name.into(),
        pkey_col: "row_idx".into(),
        fkey_col_to_pkey_table: BTreeMap::new(),
        time_col: None,
        columns: vec![("row_idx".into(), ColumnData::I64((0..n as i64).collect()))],
        hsbm_hierarchy: Vec::new(),
        fk_hsbm: BTreeMap::new(),
        feature_contract: FeatureContract::ExcludeEveryPublishedTarget,
        bag_summary: Default::default(),
        rank_field: Default::default(),
    }
}
/// Append a dense numeric column.
pub(crate) fn numeric(table: &mut TableData, name: &str, values: Vec<f64>) {
    table.push_column(
        name.into(),
        ColumnData::F64(values.into_iter().map(Some).collect()),
    );
}
/// Append a foreign-key column into `parent`.
pub(crate) fn foreign(table: &mut TableData, name: &str, parent: &str, values: Vec<i64>) {
    table
        .fkey_col_to_pkey_table
        .insert(name.into(), parent.into());
    table.push_column(name.into(), ColumnData::I64(values));
}
