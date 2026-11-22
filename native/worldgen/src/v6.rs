use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::json;
use std::collections::BTreeMap;

use crate::labels::LabelMetadata;
use crate::rng::{substream, GenRng, RngExt};
use crate::scm::{ColumnData, FeatureContract, GenOptions, GeneratedDb, TableData};

pub const V6_PROFILE: &str = "v6_rpg_20260923";
pub const V6_FORECAST_CONTRACT: &str = "v6_forecast_v1";
pub const V6_TARGET_PREFIX: &str = "v6_y_";
pub const V6_TASK_TABLE_PREFIX: &str = "v6_task_";
pub const V6_TASK_ENTITY_FK: &str = "entity_id";
pub const V6_TASK_CUTOFF_COLUMN: &str = "date";
pub const KEY_TASK_KIND: &str = "task_kind";
pub const KEY_ENTITY_TABLE: &str = "forecast_entity_table";
pub const KEY_ENTITY_FK: &str = "forecast_entity_fk";
pub const KEY_CUTOFF_COLUMN: &str = "cutoff_column";
pub const KEY_HORIZON: &str = "label_horizon_seconds";
pub const KEY_TIME_LEGALITY: &str = "time_legality";
pub const KEY_HISTORY_ROWS: &str = "label_history_rows";
pub const KEY_COLD_START: &str = "cold_start_fraction";
pub const KEY_EVENT_TABLE: &str = "forecast_event_table";
pub const KEY_ARCHETYPE: &str = "v6_archetype";
pub const KEY_LABEL_KIND: &str = "v6_label_kind";
pub const KEY_SNR: &str = "v6_snr";
pub const KEY_RELEVANT: &str = "v6_relevant_channel";
pub const KEY_CONTRACT: &str = "v6_contract";
pub const KEY_TASK_VIEW: &str = "task_view";
pub const KEY_FORECAST_CUTOFF: &str = "forecast_cutoff";
pub const FORECAST_CUTOFF_ROW_TIMESTAMP: &str = "row_timestamp";
pub const KEY_HISTORY_MEAN: &str = "v6_label_history_mean";
pub const KEY_Q24_CHANNELS: &str = "q24_channels";

const DAY_NS: i64 = 86_400_000_000_000;
pub const V6_MAX_EVENT_ROWS: usize = 200_000;
/// Hard cell cap per world.
pub const V6_MAX_CELLS: usize = 450_000;
/// Cell cap of the long-context VARIANT worlds (few entities, long histories).
pub const V6_VARIANT_MAX_CELLS: usize = 2_000_000;

/// Long-context teaching variants: each has its own seed range and DB range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum V6Variant {
    /// A1 with 500-3000 events per entity on ~50-120 entities; the label reads the
    /// own event count over a long window (30-60% of the span).
    LongHistory,
    /// One row per entity; label = a rank-d nonlinear map of fresh own-row features plus
    /// warped distractor columns; no history, no peers, binary, calibrated oracle.
    FRank,
}

impl V6Variant {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LongHistory => "long_history",
            Self::FRank => "f_rank",
        }
    }
    pub(crate) fn is_q24(self) -> bool {
        matches!(self, Self::FRank)
    }
    /// Base DB number of the variant (stage adds 10,000 per step).
    pub(crate) fn db_offset(self) -> u64 {
        match self {
            Self::LongHistory => 3_600_000,
            Self::FRank => 4_100_000,
        }
    }
    /// Seed salt: disjoint from the base 60M-67M slot seeds.
    pub(crate) fn seed_base(self) -> u64 {
        match self {
            Self::LongHistory => 80_000_000,
            Self::FRank => 110_000_000,
        }
    }
    pub fn archetype(self) -> V6Archetype {
        match self {
            Self::LongHistory => V6Archetype::A1,
            Self::FRank => V6Archetype::A2,
        }
    }
}

fn stage_index(stage: V6Stage) -> u64 {
    match stage {
        V6Stage::S1 => 0,
        V6Stage::S2 => 1,
    }
}

/// Generation seed of a VARIANT slot (same attempt layout as `slot_seed`).
pub fn variant_slot_seed(variant: V6Variant, stage: V6Stage, j: usize, attempt: usize) -> u64 {
    variant.seed_base() + stage_index(stage) * 1_000_000 + (j as u64) * 64 + attempt as u64
}

/// DB number of a VARIANT slot.
pub fn variant_db_number(variant: V6Variant, stage: V6Stage, j: usize) -> u64 {
    variant.db_offset() + stage_index(stage) * 10_000 + j as u64
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum V6Stage {
    S1,
    S2,
}

impl V6Stage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::S1 => "s1",
            Self::S2 => "s2",
        }
    }
    fn events_per_entity(self) -> (f64, f64) {
        match self {
            Self::S1 => (3.0, 20.0),
            Self::S2 => (8.0, 50.0),
        }
    }
    fn history_rows(self) -> (usize, usize) {
        match self {
            Self::S1 => (1, 8),
            Self::S2 => (1, 12),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum V6Archetype {
    /// Dense own-history forecast (future activity of the own event process).
    A1,
    /// Entity-row / weak relational (text-, categorical-, missing-rich rows).
    A2,
    /// Peer / hub collaborative through a heavy-tailed hub.
    A4,
}

impl V6Archetype {
    pub const ALL: [Self; 3] = [Self::A1, Self::A2, Self::A4];
    pub fn as_str(self) -> &'static str {
        match self {
            Self::A1 => "a1",
            Self::A2 => "a2",
            Self::A4 => "a4",
        }
    }
}

/// Generation seed of attempt `attempt` for stage slot `j` (refused seeds are
/// retried with the next attempt; the first successful attempt defines the
/// slot's world in every scale).
pub fn slot_seed(stage: V6Stage, j: usize, attempt: usize) -> u64 {
    let base: u64 = match stage {
        V6Stage::S1 => 60_000_000,
        V6Stage::S2 => 61_000_000,
    };
    if attempt < V6_PRIMARY_ATTEMPTS {
        base + (j as u64) * 16 + attempt as u64
    } else {
        base + 4_000_000 + (j as u64) * 48 + (attempt - V6_PRIMARY_ATTEMPTS) as u64
    }
}

/// Primary generation seeds per slot; `V6_MAX_ATTEMPTS` adds fallback seeds.
pub const V6_PRIMARY_ATTEMPTS: usize = 16;
pub const V6_MAX_ATTEMPTS: usize = 64;

/// Final DB number of stage slot `j` (identical in every nested scale).
pub fn slot_db_number(stage: V6Stage, j: usize) -> u64 {
    let offset: u64 = match stage {
        V6Stage::S1 => 3_000_000,
        V6Stage::S2 => 3_100_000,
    };
    offset + j as u64
}

fn degenerate_task(db: &GeneratedDb, m: &LabelMetadata) -> bool {
    let Some(table) = db.table(&m.target_table) else {
        return true;
    };
    let keys: Vec<String> = match table.column(&m.target_col) {
        Some(ColumnData::I64(v)) => v.iter().map(|x| x.to_string()).collect(),
        Some(ColumnData::F64(v)) => v.iter().flatten().map(|x| format!("{:.6e}", x)).collect(),
        Some(ColumnData::Bool(v)) => v.iter().flatten().map(|x| x.to_string()).collect(),
        Some(ColumnData::Utf8(v)) => v.iter().flatten().cloned().collect(),
        _ => return true,
    };
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for k in &keys {
        *counts.entry(k.as_str()).or_default() += 1;
    }
    let classification = m.n_classes > 0
        || m.target_type == "classification"
        || m.task_type == "clf"
        || m.task_type.contains("classification");
    keys.len() < 100 || counts.len() < 2 || (classification && counts.values().any(|c| *c < 10))
}

pub fn generate_world(seed: u64, options: &GenOptions, slot: usize) -> Result<GeneratedDb> {
    let world = options
        .v6
        .clone()
        .context("generate_world requires GenOptions.v6")?;
    let mut db = crate::scm::generate_db(seed, options)?;
    add_link_tables(&mut db, seed, &world);
    let written = add_v6_tasks(&mut db, seed, &world)?;
    ensure!(written > 0, "v6 world declared no forecast task");
    if world.static_aggcol {
        // 0 when the schema publishes no aggcol readout; that is not a refusal.
        crate::labels_v2::declare_aggcol_tasks(&mut db)?;
        // A tiny focal table can leave a static readout single-class (rustler
        // pre refuses those); the world then simply has no static view.
        let degenerate = db
            .program_metadata
            .iter()
            .filter(|m| m.family == "aggcol")
            .any(|m| degenerate_task(&db, m));
        if degenerate {
            db.program_metadata.retain(|m| m.family != "aggcol");
        }
    }
    // Same validity rule for every other declared task (e.g. an S-J readout on
    // a small table): a degenerate task is dropped, never the world, unless no
    // forecast task is left.
    let keep: Vec<bool> = db
        .program_metadata
        .iter()
        .map(|m| !degenerate_task(&db, m))
        .collect();
    let mut keep_iter = keep.into_iter();
    db.program_metadata
        .retain(|_| keep_iter.next().unwrap_or(true));
    ensure!(
        db.program_metadata
            .iter()
            .any(|m| m.family == "v6_forecast"),
        "v6 world lost every forecast task to degeneracy"
    );
    let cells: usize = db
        .tables
        .iter()
        .map(|t| t.num_rows() * t.columns.len())
        .sum();
    let cap = if world.variant.is_some() {
        V6_VARIANT_MAX_CELLS
    } else {
        V6_MAX_CELLS
    };
    ensure!(cells <= cap, "v6 world has {cells} cells > {cap}");
    let events: usize = db
        .tables
        .iter()
        .filter(|t| t.time_col.is_some() && !t.name.starts_with(V6_TASK_TABLE_PREFIX))
        .map(|t| t.num_rows())
        .max()
        .unwrap_or(0);
    ensure!(
        events <= V6_MAX_EVENT_ROWS,
        "v6 world event table has {events} rows"
    );
    db.name = match world.variant {
        Some(variant) => format!(
            "rel-synthetic-{}",
            variant_db_number(variant, world.stage, slot)
        ),
        None => format!("rel-synthetic-{}", slot_db_number(world.stage, slot)),
    };
    Ok(db)
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct V6TimeAxes {
    pub gap_cv: f64,
    /// Clock granularity in seconds (1 s .. 7 days): coarse clocks tie.
    pub tick_seconds: i64,
    /// Share of events emitted as same-tick bursts.
    pub burst_share: f64,
    /// Arrival skew: entity starts ~ U^(1/growth), growth >= 1 = later arrivals.
    pub growth: f64,
    /// Lifespan exponent: end = start + (T - start) * U^lifespan.
    pub lifespan: f64,
    /// Probability an event table is occasion-driven: every row of one primary
    /// parent shares one timestamp (a race's results, an order's lines).
    pub occasion_prob: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct V6AttributeAxes {
    pub categorical_columns: usize,
    pub max_cardinality: usize,
    pub zipf_exponent: f64,
    pub integer_coded_share: f64,
    /// Mean column-level missingness on entity rows.
    pub missing_mean: f64,
    /// Free-text columns per entity table, drawn from the shared topic pool.
    #[serde(default)]
    pub text_columns: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct V6World {
    pub stage: V6Stage,
    pub archetype: V6Archetype,
    pub time: V6TimeAxes,
    pub attributes: V6AttributeAxes,
    pub sj_band: bool,
    pub heavy_tail_fanout: bool,
    pub fanout_tail: Option<usize>,
    pub entity_rows: (usize, usize),
    pub event_rows: (usize, usize),
    pub max_depth: usize,
    pub nuisance_text: bool,
    pub forecast_tasks: usize,
    pub static_aggcol: bool,
    #[serde(default)]
    pub link_tables: usize,
    #[serde(default)]
    pub max_cols: usize,
    /// Long-context teaching variant (None = a base world; omitted from JSON).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant: Option<V6Variant>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q24_rank: Option<usize>,
}

fn log_uniform(rng: &mut GenRng, lo: f64, hi: f64) -> f64 {
    rng.f64_range(lo.ln(), hi.ln()).exp()
}

pub fn sample_world(seed: u64, stage: V6Stage, archetype: V6Archetype) -> V6World {
    let mut rng = substream(seed, "v6:axes:20260923");
    let max_cols = log_uniform(&mut rng, 3.0, 60.0).round() as usize;
    let cells = match stage {
        V6Stage::S1 => log_uniform(&mut rng, 60_000.0, 220_000.0),
        V6Stage::S2 => log_uniform(&mut rng, 80_000.0, 260_000.0),
    };
    let budget = cells / ((2 + max_cols) as f64 / 2.0 + 4.0);
    let (epe_lo, epe_hi) = stage.events_per_entity();
    let epe = log_uniform(&mut rng, epe_lo, epe_hi);
    let per_table = (budget / 3.0).min(V6_MAX_EVENT_ROWS as f64);
    let entity = (per_table / epe).clamp(100.0, 3162.0);
    let entity_lo = (entity / log_uniform(&mut rng, 1.0, 300.0)).max(4.0);
    let sj_band = rng.bernoulli(0.30);
    let heavy_tail_fanout = rng.bernoulli(0.50);
    let fanout_tail =
        (!heavy_tail_fanout).then(|| log_uniform(&mut rng, 100.0, 10_000.0).round() as usize);
    let tick_choices = [1i64, 60, 3_600, 86_400, 86_400, 604_800, 2_592_000];
    let time = V6TimeAxes {
        gap_cv: log_uniform(&mut rng, 1.5, 14.0),
        tick_seconds: tick_choices[rng.choose_index(tick_choices.len())],
        burst_share: rng.f64_range(0.02, 0.6),
        growth: rng.f64_range(1.0, 8.0),
        lifespan: rng.f64_range(0.3, 1.5),
        occasion_prob: rng.f64_range(0.0, 0.7),
    };
    let attributes = V6AttributeAxes {
        categorical_columns: (log_uniform(&mut rng, 1.0, 21.0).floor() as usize).saturating_sub(1),
        max_cardinality: log_uniform(&mut rng, 10.0, 100_000.0).round() as usize,
        zipf_exponent: rng.f64_range(0.7, 1.6),
        integer_coded_share: rng.f64_range(0.0, 0.6),
        missing_mean: if rng.bernoulli(0.25) {
            if rng.bernoulli(0.5) {
                0.0
            } else {
                rng.f64_range(0.0, 0.02)
            }
        } else {
            rng.f64_range(0.0, 0.5)
        },
        text_columns: rng.usize_inclusive(0, 5),
    };
    V6World {
        stage,
        archetype,
        time,
        attributes,
        sj_band,
        heavy_tail_fanout,
        fanout_tail,
        entity_rows: (entity_lo.round() as usize, entity.round() as usize),
        event_rows: (
            (per_table / 2.0).round() as usize,
            per_table.round() as usize,
        ),
        max_depth: rng.usize_inclusive(2, 4),
        nuisance_text: false,
        forecast_tasks: rng.usize_inclusive(1, 3),
        static_aggcol: rng.bernoulli(0.5),
        link_tables: [0usize, 0, 1, 1, 2, 3][rng.choose_index(6)],
        max_cols,
        variant: None,
        q24_rank: None,
    }
}

pub fn assign_variant_slot(world: &mut V6World, slot: usize) {
    world.q24_rank = match world.variant {
        Some(V6Variant::FRank) => Some(crate::info_prior::f_rank_for_slot(slot)),
        _ => None,
    };
}

/// Draw a VARIANT world: the base axes of `sample_world`, then the variant's size law.
pub fn sample_variant_world(seed: u64, stage: V6Stage, variant: V6Variant) -> V6World {
    let mut world = sample_world(seed, stage, variant.archetype());
    let mut rng = substream(seed, "v6:variant:20260925");
    world.variant = Some(variant);
    match variant {
        V6Variant::LongHistory => {
            // Few, deep entities: ~50-120 entities x 500-3000 events on narrow tables.
            // The activity-row budget applies to EVERY activity table (<= 3 of them under the
            // 2-4 table band), so the per-table cap keeps the world under the variant cell cap.
            let entities = rng.usize_inclusive(40, 100);
            let epe = log_uniform(&mut rng, 500.0, 2000.0);
            let events = ((entities as f64) * epe).min(80_000.0);
            world.entity_rows = (entities, entities);
            world.event_rows = ((events * 0.7).round() as usize, events.round() as usize);
            world.max_cols = rng.usize_inclusive(3, 5);
            world.heavy_tail_fanout = false;
            world.fanout_tail = None;
            world.max_depth = 2;
            world.sj_band = false;
            world.link_tables = 0;
            world.static_aggcol = false;
        }
        V6Variant::FRank => {
            // Modest F-only worlds: 1-3k entities, a few events each (the events carry nothing).
            let entities = rng.usize_inclusive(1000, 3000);
            let events = (entities as f64 * log_uniform(&mut rng, 1.5, 4.0))
                .min(V6_MAX_EVENT_ROWS as f64 * 0.5);
            world.entity_rows = (entities / 2, entities);
            world.event_rows = ((events * 0.7).round() as usize, events.round() as usize);
            world.max_cols = rng.usize_inclusive(3, 6);
            world.attributes.text_columns = rng.usize_inclusive(0, 1);
            world.attributes.missing_mean = 0.0;
            world.forecast_tasks = rng.usize_inclusive(1, 2);
            world.static_aggcol = false;
            world.link_tables = 0;
        }
    }
    world
}

/// Write a world's axes into the base generator options.
pub fn apply_world(world: &V6World, options: &mut GenOptions) {
    let schema = &mut options.schema;
    schema.rows_entity_range = world.entity_rows;
    schema.rows_activity_range = world.event_rows;
    schema.num_tables_range = match world.variant {
        Some(V6Variant::LongHistory) => (2, 4),
        Some(V6Variant::FRank) => (2, 5),
        None => (2, 12),
    };
    schema.num_cols_range = (
        2,
        if world.max_cols > 2 {
            world.max_cols
        } else {
            14
        },
    );
    schema.min_parents = 1;
    schema.max_depth = Some(world.max_depth);
    schema.min_depth = None;
    schema.heavy_tail_fanout = world.heavy_tail_fanout;
    schema.fanout_tail = world.fanout_tail;
    if world.sj_band {
        let band = crate::schema::SchemaOptions::sj_band();
        schema.satellite_tables = band.satellite_tables;
        schema.junction_prob = band.junction_prob;
        schema.hub_fan_in = band.hub_fan_in;
        // S-J needs >= 4 informative columns on its tables.
        schema.num_cols_range = (5, world.max_cols.max(12));
        schema.num_tables_range = (6, 16);
    }
    options.nuisance_text = world.nuisance_text;
    options.v6 = Some(world.clone());
}

/// Time-sort an event table by a per-parent bursty point process.
pub fn bursty_order(
    seed: u64,
    table: &str,
    world: &V6World,
    fk_columns: &mut [&mut Vec<i64>],
    min_ts: i64,
    max_ts: i64,
) -> Vec<i64> {
    let n = fk_columns.first().map_or(0, |c| c.len());
    if n == 0 {
        return Vec::new();
    }
    let mut rng = substream(seed, &format!("v6:time:{table}"));
    let axes = &world.time;
    let span = (max_ts - min_ts).max(DAY_NS) as f64;
    let sigma = (1.0 + axes.gap_cv * axes.gap_cv).ln().sqrt();
    let primary = fk_columns[0].clone();
    let mut by_parent: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (row, parent) in primary.iter().enumerate() {
        by_parent.entry(*parent).or_default().push(row);
    }
    let tick = (axes.tick_seconds * 1_000_000_000).max(1);
    let mut times = vec![0i64; n];
    let occasion = rng.bernoulli(axes.occasion_prob);
    let tick = if rng.bernoulli(0.2) {
        (tick * 30).max(DAY_NS)
    } else {
        tick
    };
    for rows in by_parent.values() {
        let start =
            min_ts as f64 + span * (1.0 - rng.f64_range(0.0, 1.0).powf(1.0 / axes.growth)) * 0.9;
        if occasion {
            let t = (start as i64).clamp(min_ts, max_ts);
            let t = min_ts + ((t - min_ts) / tick) * tick;
            for row in rows {
                times[*row] = t;
            }
            continue;
        }
        let life = rng.f64_range(0.0, 1.0).powf(axes.lifespan).max(0.02);
        let end = start + (max_ts as f64 - start) * life;
        let mut gaps = Vec::with_capacity(rows.len());
        for _ in 0..rows.len() {
            if !gaps.is_empty() && rng.bernoulli(axes.burst_share) {
                gaps.push(0.0);
            } else {
                gaps.push((sigma * rng.standard_normal() - 0.5 * sigma * sigma).exp());
            }
        }
        let total: f64 = gaps.iter().sum::<f64>().max(1e-12);
        let mut acc = 0.0;
        for (row, gap) in rows.iter().zip(gaps.iter()) {
            acc += gap;
            let t = start + (end - start) * (acc / total);
            let t = t.clamp(min_ts as f64, max_ts as f64) as i64;
            times[*row] = min_ts + ((t - min_ts) / tick) * tick;
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&row| (times[row], row));
    for column in fk_columns.iter_mut() {
        let permuted: Vec<i64> = order.iter().map(|&row| column[row]).collect();
        **column = permuted;
    }
    order.iter().map(|&row| times[row]).collect()
}

/// Topics x texts per topic of the shared free-text pool.
pub const V6_TEXT_TOPICS: u64 = 256;
pub const V6_TEXT_PER_TOPIC: u64 = 256;
const SYLLABLES: [&str; 32] = [
    "ka", "lo", "mi", "ren", "tu", "sa", "vek", "no", "pi", "dar", "el", "fo", "gri", "ha", "jun",
    "zo", "bel", "cu", "dri", "em", "ta", "ol", "pra", "qui", "ros", "sil", "um", "var", "wen",
    "xo", "yal", "ne",
];

fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn pool_word(id: u64) -> String {
    let h = mix64(id ^ 0x7E57_0000_0000);
    let syllables = 2 + (h % 2) as usize;
    (0..syllables)
        .map(|i| SYLLABLES[((h >> (8 + 6 * i)) % 32) as usize])
        .collect()
}

/// Text `index` of `topic` in the shared pool (6-24 synthetic words, ~40-150
/// chars; 60% of the words come from the topic's own 16-word vocabulary).
pub fn pool_text(topic: u64, index: u64) -> String {
    let (topic, index) = (topic % V6_TEXT_TOPICS, index % V6_TEXT_PER_TOPIC);
    let mut state = mix64((topic << 20) | index);
    let words = 6 + (state % 19) as usize;
    let mut out = Vec::with_capacity(words);
    for _ in 0..words {
        state = mix64(state);
        let id = if state % 10 < 6 {
            1_000_000 + topic * 16 + (state >> 8) % 16
        } else {
            (state >> 8) % 512
        };
        out.push(pool_word(id));
    }
    out.join(" ")
}

fn zipf_draws(rng: &mut GenRng, n: usize, cardinality: usize, exponent: f64) -> Vec<usize> {
    let cardinality = cardinality.max(2);
    let weights: Vec<f64> = (1..=cardinality)
        .map(|k| (k as f64).powf(-exponent))
        .collect();
    let mut cumulative = Vec::with_capacity(cardinality);
    let mut acc = 0.0;
    for w in &weights {
        acc += w;
        cumulative.push(acc);
    }
    (0..n)
        .map(|_| {
            let u = rng.f64_range(0.0, acc);
            cumulative.partition_point(|c| *c < u).min(cardinality - 1)
        })
        .collect()
}

fn is_entity_table(table: &TableData) -> bool {
    table.time_col.is_none() && !table.name.starts_with(V6_TASK_TABLE_PREFIX)
}

fn protected(table: &TableData, name: &str) -> bool {
    name == table.pkey_col
        || table.fkey_col_to_pkey_table.contains_key(name)
        || table.time_col.as_deref() == Some(name)
        || FeatureContract::is_published_target(name)
        || name.starts_with("aggcol_")
        || name.starts_with("sj_")
        || name == crate::scm::ENTITY_ID_COL
}

fn push_categorical(
    table: &mut TableData,
    rng: &mut GenRng,
    column: &str,
    cardinality: usize,
    axes: &V6AttributeAxes,
) {
    let n = table.num_rows();
    let draws = zipf_draws(rng, n, cardinality, axes.zipf_exponent);
    if rng.bernoulli(axes.integer_coded_share) {
        table.push_column(
            column.into(),
            ColumnData::I64(draws.iter().map(|v| *v as i64).collect()),
        );
    } else {
        table.push_column(
            column.into(),
            ColumnData::Utf8(draws.iter().map(|v| Some(format!("c{v}"))).collect()),
        );
    }
}

/// Column-level MNAR: a column's missing rate r_c ~ Beta around the world's
/// mean (some columns > 50%), and the missing rows are those with the highest
/// value-dependent score b * z(value) + noise.
fn apply_mnar(table: &mut TableData, rng: &mut GenRng, missing_mean: f64) {
    let n = table.num_rows();
    if missing_mean <= 0.0 || n == 0 {
        return;
    }
    let candidates: Vec<usize> = (0..table.columns.len())
        .filter(|&c| !protected(table, &table.columns[c].0))
        .collect();
    for c in candidates {
        let concentration = 2.0;
        let rate = rng
            .beta(
                missing_mean * concentration + 1e-3,
                (1.0 - missing_mean) * concentration + 1e-3,
            )
            .clamp(0.0, 0.95);
        let masked = (rate * n as f64).round() as usize;
        if masked == 0 {
            continue;
        }
        let dependence = rng.f64_range(0.0, 1.5);
        let values: Vec<f64> = match &table.columns[c].1 {
            ColumnData::F64(v) => v.iter().map(|x| x.unwrap_or(0.0)).collect(),
            ColumnData::I64(v) => v.iter().map(|x| *x as f64).collect(),
            ColumnData::Bool(v) => v
                .iter()
                .map(|x| f64::from(x.unwrap_or(false) as u8))
                .collect(),
            ColumnData::Utf8(v) => v
                .iter()
                .map(|x| x.as_ref().map_or(0.0, |s| s.len() as f64))
                .collect(),
            ColumnData::TimestampNs(_) => continue,
        };
        let mean = values.iter().sum::<f64>() / n as f64;
        let sd = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64)
            .sqrt()
            .max(1e-12);
        let mut score: Vec<(f64, usize)> = values
            .iter()
            .enumerate()
            .map(|(r, v)| (dependence * (v - mean) / sd + rng.standard_normal(), r))
            .collect();
        score.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut hide = vec![false; n];
        for (_, r) in score.iter().take(masked) {
            hide[*r] = true;
        }
        let data = std::mem::replace(&mut table.columns[c].1, ColumnData::F64(Vec::new()));
        table.columns[c].1 = match data {
            ColumnData::F64(v) => ColumnData::F64(
                v.into_iter()
                    .enumerate()
                    .map(|(r, x)| if hide[r] { None } else { x })
                    .collect(),
            ),
            // Integer columns become nullable floats (as a real integer
            // column with nulls reads after a parquet round trip).
            ColumnData::I64(v) => ColumnData::F64(
                v.into_iter()
                    .enumerate()
                    .map(|(r, x)| (!hide[r]).then_some(x as f64))
                    .collect(),
            ),
            ColumnData::Bool(v) => ColumnData::Bool(
                v.into_iter()
                    .enumerate()
                    .map(|(r, x)| if hide[r] { None } else { x })
                    .collect(),
            ),
            ColumnData::Utf8(v) => ColumnData::Utf8(
                v.into_iter()
                    .enumerate()
                    .map(|(r, x)| if hide[r] { None } else { x })
                    .collect(),
            ),
            other => other,
        };
    }
}

/// Topic-pool text column on an entity table. The topic is a noisy quantile
/// bin of one numeric entity column (a redundant view, as a real description
/// echoes structured fields) or, without one, pure nuisance.
fn push_text(table: &mut TableData, rng: &mut GenRng, column: &str) {
    let n = table.num_rows();
    let anchor: Option<Vec<f64>> = table
        .columns
        .iter()
        .filter(|(name, _)| !protected(table, name) && !name.starts_with("v6_"))
        .find_map(|(_, data)| match data {
            ColumnData::F64(v) => Some(v.iter().map(|x| x.unwrap_or(0.0)).collect()),
            _ => None,
        });
    let topics = log_uniform(rng, 4.0, V6_TEXT_TOPICS as f64)
        .round()
        .max(2.0) as usize;
    let base = rng.choose_index(V6_TEXT_TOPICS as usize) as u64;
    let bins = anchor.map(|values| crate::scm::balanced_quantiles(&values, topics));
    let texts: Vec<Option<String>> = (0..n)
        .map(|r| {
            let local = match &bins {
                Some(b) if !rng.bernoulli(0.3) => b[r] as u64,
                _ => rng.choose_index(topics) as u64,
            };
            Some(pool_text(
                base + local,
                rng.choose_index(V6_TEXT_PER_TOPIC as usize) as u64,
            ))
        })
        .collect();
    table.push_column(column.into(), ColumnData::Utf8(texts));
}

pub fn add_entity_attributes(
    db: &mut GeneratedDb,
    seed: u64,
    world: &V6World,
) -> BTreeMap<String, Vec<String>> {
    let axes = &world.attributes;
    let mut added = BTreeMap::new();
    for index in 0..db.tables.len() {
        let table = &mut db.tables[index];
        // Task tables are labels; link tables stay pure junctions (<= 2 attributes).
        if table.name.starts_with(V6_TASK_TABLE_PREFIX) || table.name.starts_with("v6_link_") {
            continue;
        }
        let n = table.num_rows();
        if n == 0 {
            continue;
        }
        let name = table.name.clone();
        let mut rng = substream(seed, &format!("v6:attributes:{name}"));
        let entity = is_entity_table(table);
        let count = if entity {
            axes.categorical_columns
        } else {
            rng.usize_inclusive(0, axes.categorical_columns.div_ceil(2))
        };
        let mut names = Vec::new();
        for k in 0..count {
            let card =
                log_uniform(&mut rng, 3.0, axes.max_cardinality.max(4) as f64).round() as usize;
            let column = if entity {
                format!("v6_cat_{k}")
            } else {
                format!("v6_ecat_{k}")
            };
            push_categorical(table, &mut rng, &column, card.clamp(2, n.max(2)), axes);
            names.push(column);
        }
        let texts = if entity {
            axes.text_columns
        } else if n <= 50_000 {
            rng.usize_inclusive(0, axes.text_columns.div_ceil(2))
        } else {
            0
        };
        for k in 0..texts {
            push_text(table, &mut rng, &format!("v6_text_{k}"));
        }
        apply_mnar(table, &mut rng, axes.missing_mean);
        if entity {
            added.insert(name, names);
        }
    }
    added
}

pub fn add_link_tables(db: &mut GeneratedDb, seed: u64, world: &V6World) -> usize {
    let entities: Vec<(String, usize)> = db
        .tables
        .iter()
        .filter(|t| is_entity_table(t) && t.num_rows() >= 2)
        .map(|t| (t.name.clone(), t.num_rows()))
        .collect();
    if entities.is_empty() {
        return 0;
    }
    let (t_min, t_max) = db
        .tables
        .iter()
        .filter(|t| t.time_col.is_some() && !t.name.starts_with(V6_TASK_TABLE_PREFIX))
        .filter_map(|t| i64_column(t, t.time_col.as_deref()?))
        .fold((i64::MAX, i64::MIN), |(lo, hi), v| {
            (
                v.iter().copied().min().unwrap_or(lo).min(lo),
                v.iter().copied().max().unwrap_or(hi).max(hi),
            )
        });
    let mut rng = substream(seed, "v6:links:20260923");
    let mut written = 0;
    for i in 0..world.link_tables {
        let (a, na) = entities[rng.choose_index(entities.len())].clone();
        let (b, nb) = entities[rng.choose_index(entities.len())].clone();
        let rows = log_uniform(&mut rng, 0.5 * na.max(nb) as f64, 5.0 * (na + nb) as f64)
            .min(V6_MAX_EVENT_ROWS as f64 / 4.0)
            .max(20.0) as usize;
        let name = format!("v6_link_{i}");
        let mut table = crate::scm::table(&name, rows);
        let ends = [(format!("{a}_a"), &a, na), (format!("{b}_b"), &b, nb)];
        let mut fks: Vec<Vec<i64>> = Vec::new();
        for (_, _, n) in &ends {
            let mut ids: Vec<i64> = (0..*n as i64).collect();
            rng.shuffle(&mut ids);
            let exponent = rng.f64_range(0.2, 1.2);
            fks.push(
                zipf_draws(&mut rng, rows, *n, exponent)
                    .into_iter()
                    .map(|d| ids[d])
                    .collect(),
            );
        }
        let timed = t_min < t_max && rng.bernoulli(0.5);
        if timed {
            let mut columns: Vec<&mut Vec<i64>> = fks.iter_mut().collect();
            let times = bursty_order(seed, &name, world, &mut columns, t_min, t_max);
            for ((column, parent, _), values) in ends.iter().zip(fks) {
                crate::scm::foreign(&mut table, column, parent, values);
            }
            table.time_col = Some("date".into());
            table.push_column("date".into(), ColumnData::TimestampNs(times));
        } else {
            for ((column, parent, _), values) in ends.iter().zip(fks) {
                crate::scm::foreign(&mut table, column, parent, values);
            }
        }
        for k in 0..rng.usize_inclusive(0, 2) {
            let values: Vec<f64> = (0..rows).map(|_| rng.standard_normal()).collect();
            crate::scm::numeric(&mut table, &format!("weight_{k}"), values);
        }
        db.tables.push(table);
        written += 1;
    }
    written
}

fn i64_column<'a>(table: &'a TableData, name: &str) -> Option<&'a Vec<i64>> {
    match table.column(name) {
        Some(ColumnData::I64(v)) | Some(ColumnData::TimestampNs(v)) => Some(v),
        _ => None,
    }
}

pub(crate) fn standardize(values: &[f64]) -> Vec<f64> {
    let n = values.len().max(1) as f64;
    let mean = values.iter().sum::<f64>() / n;
    let sd = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
    if !(sd > 1e-12) {
        return vec![0.0; values.len()];
    }
    values.iter().map(|v| (v - mean) / sd).collect()
}

/// AUROC of `score` for binary `labels` (ties get half credit).
pub fn auroc(score: &[f64], labels: &[bool]) -> Option<f64> {
    let positives = labels.iter().filter(|y| **y).count();
    let negatives = labels.len() - positives;
    if positives == 0 || negatives == 0 {
        return None;
    }
    let mut order: Vec<usize> = (0..score.len()).collect();
    order.sort_by(|a, b| score[*a].total_cmp(&score[*b]));
    let mut rank_sum = 0.0;
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && score[order[j + 1]] == score[order[i]] {
            j += 1;
        }
        let average = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            if labels[order[k]] {
                rank_sum += average;
            }
        }
        i = j + 1;
    }
    let p = positives as f64;
    Some((rank_sum - p * (p + 1.0) / 2.0) / (p * negatives as f64))
}

fn pearson_r2(a: &[f64], b: &[f64]) -> f64 {
    let za = standardize(a);
    let zb = standardize(b);
    let r = za.iter().zip(&zb).map(|(x, y)| x * y).sum::<f64>() / a.len().max(1) as f64;
    r * r
}

fn threshold_labels(z: &[f64], base_rate: f64) -> Vec<bool> {
    let mut sorted: Vec<f64> = z.to_vec();
    sorted.sort_by(f64::total_cmp);
    let k = ((1.0 - base_rate) * sorted.len() as f64).floor() as usize;
    let cut = sorted[k.min(sorted.len() - 1)];
    z.iter().map(|v| *v >= cut).collect()
}

/// Calibrate the noise scale so the realised oracle AUROC of `signal` for the
/// thresholded labels is close to `target` (bisection on a fixed noise draw).
pub(crate) fn calibrate_binary(
    signal: &[f64],
    noise: &[f64],
    base_rate: f64,
    target: f64,
) -> (Vec<bool>, f64, f64) {
    let (mut lo, mut hi) = (1e-3f64, 50.0f64);
    let mut best = (threshold_labels(signal, base_rate), 0.0);
    for _ in 0..30 {
        let sigma = (lo * hi).sqrt();
        let z: Vec<f64> = signal
            .iter()
            .zip(noise)
            .map(|(s, e)| s + sigma * e)
            .collect();
        let y = threshold_labels(&z, base_rate);
        let a = auroc(signal, &y).unwrap_or(0.5);
        best = (y, sigma);
        if a > target {
            lo = sigma;
        } else {
            hi = sigma;
        }
    }
    let realised = auroc(signal, &best.0).unwrap_or(0.5);
    (best.0, best.1, realised)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LabelKind {
    Churn,
    AnyEvent,
    Count,
    ZeroInflatedSum,
    Multiclass,
}

impl LabelKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Churn => "churn",
            Self::AnyEvent => "any_event",
            Self::Count => "count",
            Self::ZeroInflatedSum => "zero_inflated_sum",
            Self::Multiclass => "multiclass",
        }
    }
    fn binary(self) -> bool {
        matches!(self, Self::Churn | Self::AnyEvent)
    }
}

struct Candidate {
    entity: usize,
    event: usize,
    fk: String,
}

fn candidates(db: &GeneratedDb) -> Vec<Candidate> {
    let mut out = Vec::new();
    for (ei, entity) in db.tables.iter().enumerate() {
        if !is_entity_table(entity) || entity.num_rows() < 50 {
            continue;
        }
        for (ci, child) in db.tables.iter().enumerate() {
            if child.time_col.is_none() || child.name.starts_with(V6_TASK_TABLE_PREFIX) {
                continue;
            }
            for (col, parent) in &child.fkey_col_to_pkey_table {
                if parent == &entity.name {
                    out.push(Candidate {
                        entity: ei,
                        event: ci,
                        fk: col.clone(),
                    });
                }
            }
        }
    }
    out
}

fn published_hidden(db: &GeneratedDb, except: &str) -> Vec<serde_json::Value> {
    let mut hidden = Vec::new();
    for table in &db.tables {
        for (name, _) in &table.columns {
            let registered =
                FeatureContract::is_published_target(name) || name.starts_with("aggcol_");
            if registered && name != except {
                hidden.push(json!([table.name, name]));
            }
        }
    }
    hidden
}

pub fn add_v6_tasks(db: &mut GeneratedDb, seed: u64, world: &V6World) -> Result<usize> {
    let cats = add_entity_attributes(db, seed, world);
    let pool = candidates(db);
    ensure!(
        !pool.is_empty(),
        "v6 world has no (entity, timed event table) pair"
    );
    let mut rng = substream(seed, "v6:tasks:20260923");
    // Variant-only draws come from their own stream: a base world's draws are untouched.
    let mut vrng = substream(seed, "v6:variant-tasks:20260925");
    let variant = world.variant;
    let mut written = 0;
    let entity_choice = pool[rng.choose_index(pool.len())].entity;
    let pool: Vec<&Candidate> = pool.iter().filter(|c| c.entity == entity_choice).collect();
    let q24 = variant.filter(|v| v.is_q24());
    let q24_block = if q24.is_some() {
        let rank = world
            .q24_rank
            .context("an f-rank world needs q24_rank (assign_variant_slot)")?;
        let mut frng = substream(seed, "q24:features:20260926");
        let p_inf = rank + crate::info_prior::INFORMATIVE_EXTRA;
        let p_dis = frng.usize_inclusive(8, 16);
        let n = db.tables[entity_choice].num_rows();
        let (latent, published, warps) =
            crate::info_prior::feature_block(&mut frng, n, p_inf, p_dis);
        let mut names = Vec::new();
        let mut index = 100usize;
        for column in published {
            while db.tables[entity_choice]
                .column(&format!("feature_{index}"))
                .is_some()
            {
                index += 1;
            }
            let name = format!("feature_{index}");
            crate::scm::numeric(&mut db.tables[entity_choice], &name, column);
            names.push(name);
            index += 1;
        }
        Some((rank, p_inf, p_dis, latent, names, warps))
    } else {
        None
    };
    for k in 0..world.forecast_tasks {
        let pick = pool[rng.choose_index(pool.len())];
        let (ei, ci, fk) = (pick.entity, pick.event, pick.fk.clone());
        let entity_name = db.tables[ei].name.clone();
        let event_name = db.tables[ci].name.clone();
        let n_entities = db.tables[ei].num_rows();
        let event_fk = i64_column(&db.tables[ci], &fk)
            .context("event FK column")?
            .clone();
        let time_col = db.tables[ci]
            .time_col
            .clone()
            .context("event time column")?;
        let event_time = i64_column(&db.tables[ci], &time_col)
            .context("event time")?
            .clone();
        let t_min = *event_time.iter().min().unwrap_or(&0);
        let t_max = *event_time.iter().max().unwrap_or(&1);
        let span = (t_max - t_min).max(DAY_NS);
        // Per-entity sorted event times, and a positive mark (amount).
        let mut per_entity: Vec<Vec<i64>> = vec![Vec::new(); n_entities];
        for (row, parent) in event_fk.iter().enumerate() {
            if *parent >= 0 && (*parent as usize) < n_entities {
                per_entity[*parent as usize].push(event_time[row]);
            }
        }
        for times in per_entity.iter_mut() {
            times.sort_unstable();
        }
        let tick = (world.time.tick_seconds * 1_000_000_000).max(1_000_000_000);
        let horizon = ((span as f64 * rng.f64_range(0.01, 0.08)) as i64 / tick).max(1) * tick;
        // Cutoff grid (shared timestamps).
        let grid_n = 16usize;
        let lo = t_min + span / 3;
        let hi = (t_max - horizon).max(lo + tick);
        let grid: Vec<i64> = (0..grid_n)
            .map(|g| lo + ((hi - lo) as f64 * g as f64 / (grid_n - 1) as f64) as i64)
            .collect();
        let archetype = world.archetype;
        let cold_target = rng.f64_range(0.0, 0.2);
        let (h_lo, h_hi) = match q24 {
            Some(_) => (1, 1),
            None => world.stage.history_rows(),
        };
        let mut rows: Vec<(usize, i64, bool)> = Vec::new();
        let max_entities = 3_000usize;
        let mut entity_ids: Vec<usize> = (0..n_entities).collect();
        rng.shuffle(&mut entity_ids);
        entity_ids.truncate(max_entities);
        entity_ids.sort_unstable();
        for &e in &entity_ids {
            let first = per_entity[e].first().copied();
            match q24 {
                Some(V6Variant::FRank) => {
                    rows.push((e, grid[vrng.choose_index(grid.len())], false));
                    continue;
                }
                _ => {}
            }
            let (cold_pool, warm_pool): (Vec<i64>, Vec<i64>) = grid
                .iter()
                .copied()
                .partition(|t| first.map_or(true, |f| *t < f));
            let want = rng.usize_inclusive(h_lo, h_hi);
            for _ in 0..want {
                let cold =
                    !cold_pool.is_empty() && (warm_pool.is_empty() || rng.bernoulli(cold_target));
                let pool = if cold { &cold_pool } else { &warm_pool };
                if pool.is_empty() {
                    continue;
                }
                rows.push((e, pool[rng.choose_index(pool.len())], cold));
            }
        }
        rows.sort_by_key(|(e, t, _)| (*t, *e));
        rows.dedup_by_key(|(e, t, _)| (*e, *t));
        ensure!(rows.len() >= 50, "v6 task has too few rows");
        let count_in = |e: usize, a: i64, b: i64| -> f64 {
            let times = &per_entity[e];
            (times.partition_point(|t| *t <= b) - times.partition_point(|t| *t <= a)) as f64
        };
        let future: Vec<f64> = rows
            .iter()
            .map(|(e, t, _)| count_in(*e, *t, *t + horizon))
            .collect();
        // LongHistory: the history statistic covers 30-60% of the span (hundreds of own
        // events), so only a context holding most of the own history reads it.
        let window = match variant {
            Some(V6Variant::LongHistory) => ((span as f64) * vrng.f64_range(0.3, 0.6)) as i64,
            _ => horizon * 4,
        };
        // LongHistory: the history statistic is the normalised SUM of an i.i.d. N(0, 1) event
        // mark over the window (no per-entity mean), so recent events do not predict it: the
        // information about the label grows with the share of the window's own events in context.
        let mut mark_note = None;
        let history: Vec<f64> = if variant == Some(V6Variant::LongHistory) {
            let marks: Vec<f64> = (0..event_fk.len())
                .map(|_| vrng.standard_normal())
                .collect();
            let mut per_entity_marks: Vec<Vec<(i64, f64)>> = vec![Vec::new(); n_entities];
            for (row, parent) in event_fk.iter().enumerate() {
                if *parent >= 0 && (*parent as usize) < n_entities {
                    per_entity_marks[*parent as usize].push((event_time[row], marks[row]));
                }
            }
            for events in per_entity_marks.iter_mut() {
                events.sort_by_key(|(t, _)| *t);
            }
            let mark_name = format!("v6_mark_{k}");
            crate::scm::numeric(&mut db.tables[ci], &mark_name, marks);
            mark_note = Some(mark_name);
            rows.iter()
                .map(|(e, t, _)| {
                    let events = &per_entity_marks[*e];
                    let a = events.partition_point(|(u, _)| *u <= *t - window);
                    let b = events.partition_point(|(u, _)| *u <= *t);
                    let n = (b - a) as f64;
                    events[a..b].iter().map(|(_, m)| m).sum::<f64>() / n.max(1.0).sqrt()
                })
                .collect()
        } else {
            rows.iter()
                .map(|(e, t, _)| count_in(*e, *t - window, *t))
                .collect()
        };
        // Entity-row signal: numeric entity columns + a random effect per
        // category of one categorical column.
        let entity_signal = {
            let table = &db.tables[ei];
            let mut signal = vec![0.0; n_entities];
            let mut used = 0;
            for (name, data) in &table.columns {
                if protected(table, name) {
                    continue;
                }
                if let ColumnData::F64(v) = data {
                    let w = rng.normal(0.0, 1.0);
                    let filled: Vec<f64> = v.iter().map(|x| x.unwrap_or(0.0)).collect();
                    let z = standardize(&filled);
                    for (s, x) in signal.iter_mut().zip(z) {
                        *s += w * x;
                    }
                    used += 1;
                    if used >= 6 {
                        break;
                    }
                }
            }
            if let Some(cat) = cats.get(&entity_name).and_then(|c| c.first()) {
                let codes: Vec<i64> = match table.column(cat) {
                    Some(ColumnData::I64(v)) => v.clone(),
                    Some(ColumnData::F64(v)) => {
                        v.iter().map(|x| x.map_or(-1, |x| x as i64)).collect()
                    }
                    Some(ColumnData::Utf8(v)) => v
                        .iter()
                        .map(|x| {
                            x.as_ref()
                                .and_then(|s| s.trim_start_matches('c').parse::<i64>().ok())
                                .unwrap_or(-1)
                        })
                        .collect(),
                    _ => Vec::new(),
                };
                let mut effect: BTreeMap<i64, f64> = BTreeMap::new();
                for (s, code) in signal.iter_mut().zip(codes) {
                    let value = *effect.entry(code).or_insert_with(|| rng.normal(0.0, 1.5));
                    *s += value;
                }
            }
            standardize(&signal)
        };
        // A4: heavy-tailed hub (a new parent of the entity table).
        let mut hub_note = None;
        let make_hub = q24.is_none() && (archetype == V6Archetype::A4 || rng.bernoulli(0.25));
        let hub_signal = if make_hub {
            let hubs = (n_entities / rng.usize_inclusive(5, 50)).max(2);
            let alpha = rng.f64_range(1.1, 2.5);
            let weights: Vec<f64> = (0..hubs)
                .map(|_| {
                    rng.f64_range(f64::MIN_POSITIVE, 1.0)
                        .powf(-1.0 / alpha)
                        .min(1e6)
                })
                .collect();
            let mut cumulative = Vec::with_capacity(hubs);
            let mut acc = 0.0;
            for w in &weights {
                acc += w;
                cumulative.push(acc);
            }
            let assignment: Vec<i64> = (0..n_entities)
                .map(|_| {
                    cumulative
                        .partition_point(|c| *c < rng.f64_range(0.0, acc))
                        .min(hubs - 1) as i64
                })
                .collect();
            let latent: Vec<f64> = (0..hubs).map(|_| rng.standard_normal()).collect();
            let hub_name = format!("v6_hub_{k}");
            let mut hub = crate::scm::table(&hub_name, hubs);
            // A noisy public proxy only: the hub latent itself is never a column.
            crate::scm::numeric(
                &mut hub,
                "hub_attribute",
                latent.iter().map(|h| h + rng.normal(0.0, 2.0)).collect(),
            );
            db.tables.push(hub);
            let fk_name = format!("v6_hub_{k}_id");
            let entity = &mut db.tables[ei];
            entity.push_column(fk_name.clone(), ColumnData::I64(assignment.clone()));
            entity
                .fkey_col_to_pkey_table
                .insert(fk_name.clone(), hub_name.clone());
            hub_note = Some(
                json!({"hub_table": hub_name, "hub_fk": fk_name, "hubs": hubs, "pareto_alpha": alpha}),
            );
            rows.iter()
                .map(|(e, _, _)| latent[assignment[*e] as usize])
                .collect::<Vec<f64>>()
        } else {
            Vec::new()
        };
        // Archetype signal mix.
        let lf: Vec<f64> = standardize(&future.iter().map(|v| v.ln_1p()).collect::<Vec<_>>());
        let lh: Vec<f64> = if variant == Some(V6Variant::LongHistory) {
            standardize(&history) // a signed mark sum: no log transform
        } else {
            standardize(&history.iter().map(|v| v.ln_1p()).collect::<Vec<_>>())
        };
        let ent: Vec<f64> = rows.iter().map(|(e, _, _)| entity_signal[*e]).collect();
        let hub: Vec<f64> = if hub_signal.is_empty() || archetype != V6Archetype::A4 {
            vec![0.0; rows.len()]
        } else {
            standardize(&hub_signal)
        };
        let (wf, wh, we, wu, relevant) = match (variant, archetype) {
            (Some(V6Variant::LongHistory), _) => (0.15, 1.0, 0.1, 0.0, "own_mark_sum_long"),
            (Some(v), _) => (0.0, 0.0, 0.0, 0.0, v.as_str()),
            (None, _) => match archetype {
                V6Archetype::A1 => (1.0, 0.3, 0.2, 0.0, "own_event_process"),
                V6Archetype::A2 => (0.25, 0.0, 1.0, 0.0, "entity_row"),
                V6Archetype::A4 => (0.3, 0.0, 0.1, 1.0, "hub_peers"),
            },
        };
        let mut signal: Vec<f64> = (0..rows.len())
            .map(|i| wf * lf[i] + wh * lh[i] + we * ent[i] + wu * hub[i])
            .collect();
        let mut q24_record = None;
        let mut q24_private = None;
        if let Some((rank, p_inf, p_dis, latent, names, warps)) = q24_block.as_ref() {
            let map = crate::info_prior::RankMap::sample(&mut vrng, *rank);
            let f = standardize(
                &rows
                    .iter()
                    .map(|(e, _, _)| map.score(&latent[*e]))
                    .collect::<Vec<_>>(),
            );
            let shift = vec![0.0; rows.len()];
            let extra = json!({"rank": rank, "p_informative": p_inf, "p_distractor": p_dis,
                                   "map": "mlp_tanh_2layer_plus_linear", "horizon_seconds": horizon / 1_000_000_000});
            let components: Vec<(&str, f64, Vec<f64>)> = vec![("F", 1.0, f)];
            signal = (0..rows.len())
                .map(|i| components.iter().map(|(_, w, c)| w * c[i]).sum::<f64>() + shift[i])
                .collect();
            let refs: Vec<(&str, f64, &[f64])> = components
                .iter()
                .map(|(c, w, v)| (*c, *w, v.as_slice()))
                .collect();
            let mut record = crate::info_prior::channel_record(&refs, extra);
            let shifted: Vec<f64> = shift.clone();
            record["cold_shift_share"] = json!(if shifted.iter().any(|x| *x != 0.0) {
                let total: Vec<f64> = signal.clone();
                let var = |v: &[f64]| {
                    let m = v.iter().sum::<f64>() / v.len().max(1) as f64;
                    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len().max(1) as f64
                };
                var(&shifted) / var(&total).max(1e-12)
            } else {
                0.0
            });
            q24_record = Some(record);
            q24_private = Some(json!({"informative_columns": names[..*p_inf].to_vec(),
                                      "distractor_columns": names[*p_inf..].to_vec(),
                                      "warps": warps.iter().map(|w| w.as_str()).collect::<Vec<_>>()}));
        }
        let signal = standardize(&signal);
        let noise: Vec<f64> = (0..rows.len()).map(|_| rng.standard_normal()).collect();
        let kinds = [
            LabelKind::Churn,
            LabelKind::AnyEvent,
            LabelKind::Count,
            LabelKind::ZeroInflatedSum,
            LabelKind::Multiclass,
        ];
        let kind_weights = [0.3, 0.2, 0.15, 0.2, 0.15];
        let kind = kinds[rng.choose_weighted(&kind_weights)];
        let kind = if q24.is_some() {
            LabelKind::AnyEvent
        } else {
            kind
        };
        let target_name = format!("{V6_TARGET_PREFIX}{k}");
        let (column, n_classes, snr) = if kind.binary() {
            let base_rate = log_uniform(&mut rng, 0.05, 0.9);
            let target = rng.f64_range(0.55, 0.95);
            let (base_rate, target) = match q24 {
                Some(_) => (log_uniform(&mut vrng, 0.2, 0.8), vrng.f64_range(0.75, 0.95)),
                None => (base_rate, target),
            };
            // Churn = LOW future activity; any-event = high.
            let oriented: Vec<f64> = if kind == LabelKind::Churn {
                signal.iter().map(|s| -s).collect()
            } else {
                signal.clone()
            };
            let (labels, sigma, realised) = calibrate_binary(&oriented, &noise, base_rate, target);
            (
                ColumnData::I64(labels.iter().map(|y| i64::from(*y)).collect()),
                2usize,
                json!({"metric": "auroc", "target": target, "realised_oracle": realised,
                    "noise_scale": if sigma.is_finite() { json!(sigma) } else { json!(null) },
                    "base_rate": base_rate, "null": false}),
            )
        } else {
            let target = rng.f64_range(0.1, 0.7);
            let sigma = ((1.0 - target) / target).sqrt();
            let latent: Vec<f64> = signal
                .iter()
                .zip(&noise)
                .map(|(s, e)| s + sigma * e)
                .collect();
            let realised_link = pearson_r2(&signal, &latent);
            match kind {
                LabelKind::Multiclass => {
                    let k_classes = rng.usize_inclusive(3, 10);
                    let bins = crate::scm::balanced_quantiles(&latent, k_classes);
                    (
                        ColumnData::I64(bins.into_iter().map(|b| b as i64).collect()),
                        k_classes,
                        json!({"metric": "r2_link", "target": target, "realised_oracle": realised_link, "null": false}),
                    )
                }
                _ => {
                    let zero_share = if kind == LabelKind::ZeroInflatedSum {
                        rng.f64_range(0.0, 0.5)
                    } else {
                        0.0
                    };
                    let zl = standardize(&latent);
                    let mut sorted = zl.clone();
                    sorted.sort_by(f64::total_cmp);
                    let cut =
                        sorted[((zero_share * sorted.len() as f64) as usize).min(sorted.len() - 1)];
                    let scale = rng.f64_range(0.4, 1.4);
                    let affine = 10f64.powf(rng.f64_range(-3.0, 7.0));
                    let values: Vec<f64> = zl
                        .iter()
                        .map(|z| {
                            if kind == LabelKind::Count {
                                (scale * (z + 1.0).max(0.0)).exp_m1().round()
                            } else if *z < cut {
                                0.0
                            } else {
                                (scale * (z - cut)).exp_m1() * affine
                            }
                        })
                        .collect();
                    let realised = pearson_r2(
                        &signal,
                        &values.iter().map(|v| v.ln_1p()).collect::<Vec<_>>(),
                    );
                    (
                        ColumnData::F64(values.into_iter().map(Some).collect()),
                        0usize,
                        json!({"metric": "r2_link", "target": target, "realised_oracle": realised_link,
                            "realised_log1p_r2": realised, "zero_share": zero_share, "affine": affine,
                            "null": false}),
                    )
                }
            }
        };
        let n_rows = rows.len();
        let task_name = format!("{V6_TASK_TABLE_PREFIX}{k}");
        let mut task = crate::scm::table(&task_name, n_rows);
        crate::scm::foreign(
            &mut task,
            V6_TASK_ENTITY_FK,
            &entity_name,
            rows.iter().map(|(e, _, _)| *e as i64).collect(),
        );
        task.time_col = Some(V6_TASK_CUTOFF_COLUMN.into());
        task.push_column(
            V6_TASK_CUTOFF_COLUMN.into(),
            ColumnData::TimestampNs(rows.iter().map(|(_, t, _)| *t).collect()),
        );
        let history_rows = {
            let mut per: BTreeMap<usize, usize> = BTreeMap::new();
            for (e, _, _) in &rows {
                *per.entry(*e).or_default() += 1;
            }
            let counts: Vec<usize> = per.values().copied().collect();
            let lo = counts.iter().copied().min().unwrap_or(1).max(1);
            let hi = counts.iter().copied().max().unwrap_or(1).clamp(lo, 16);
            (
                lo,
                hi,
                counts.iter().sum::<usize>() as f64 / counts.len().max(1) as f64,
            )
        };
        let cold_share = rows.iter().filter(|(_, _, c)| *c).count() as f64 / n_rows as f64;
        task.push_column(target_name.clone(), column);
        db.tables.push(task);
        let task_index = db.tables.len() - 1;
        let hidden = published_hidden(db, &target_name);
        // NOTE: never an "entity_table" key -- parquet.rs copies public_task
        // keys onto the task object, whose entity_table is the TARGET table.
        let mut public_task = serde_json::Map::new();
        let mut put = |key: &str, value: serde_json::Value| {
            public_task.insert(key.to_string(), value);
        };
        put(KEY_TASK_KIND, json!("forecast"));
        put(KEY_ENTITY_TABLE, json!(entity_name.clone()));
        put(KEY_ENTITY_FK, json!(V6_TASK_ENTITY_FK));
        put(KEY_CUTOFF_COLUMN, json!(V6_TASK_CUTOFF_COLUMN));
        put(KEY_HORIZON, json!(horizon / 1_000_000_000));
        put(KEY_TIME_LEGALITY, json!("strict_prior"));
        put(KEY_TASK_VIEW, json!("forecast"));
        put(KEY_FORECAST_CUTOFF, json!(FORECAST_CUTOFF_ROW_TIMESTAMP));
        put(KEY_HISTORY_ROWS, json!([history_rows.0, history_rows.1]));
        put(KEY_HISTORY_MEAN, json!(history_rows.2));
        put(KEY_COLD_START, json!(cold_share));
        put(KEY_EVENT_TABLE, json!(event_name.clone()));
        put(KEY_ARCHETYPE, json!(archetype.as_str()));
        put(KEY_LABEL_KIND, json!(kind.as_str()));
        put(KEY_SNR, snr);
        let mut relevant_note = json!({"kind": relevant, "event_table": event_name.clone(), "event_fk": fk.clone(), "hub": hub_note.clone()});
        if let Some(variant) = variant {
            relevant_note["variant"] = json!(variant.as_str());
            relevant_note["history_window_seconds"] = json!(window / 1_000_000_000);
            if let Some(mark) = &mark_note {
                relevant_note["mark_column"] = json!(mark);
            }
        }
        put(KEY_RELEVANT, relevant_note);
        put(KEY_CONTRACT, json!(V6_FORECAST_CONTRACT));
        if let Some(record) = &q24_record {
            put(KEY_Q24_CHANNELS, record.clone());
        }
        put("drop_columns", json!(hidden));
        if n_classes > 0 {
            put("num_classes", json!(n_classes));
        }
        let public_task = serde_json::Value::Object(public_task);
        let mut params = BTreeMap::new();
        params.insert("public_task".to_string(), public_task);
        params.insert("v6_profile".to_string(), json!(V6_PROFILE));
        params.insert("v6_world".to_string(), serde_json::to_value(world)?);
        if let Some(private) = q24_private {
            // PRIVATE generator receipt (program_info.json): the informative columns and the
            // per-row noiseless oracle score in task-row order. Never public task metadata.
            let mut private = private;
            private["oracle_score"] = json!(signal);
            params.insert("q24_private".to_string(), private);
        }
        let (target_type, task_type) = if n_classes == 0 {
            ("continuous", "regression")
        } else {
            ("classification", "clf")
        };
        db.program_metadata.push(LabelMetadata {
            family: "v6_forecast".into(),
            target_table: task_name.clone(),
            source_tables: vec![entity_name.clone(), event_name.clone()],
            target_type: target_type.into(),
            n_classes,
            hops: 1,
            uses_time: true,
            params,
            seed: db.seed,
            target_col: target_name,
            task_type: task_type.into(),
            num_rows: db.tables[task_index].num_rows(),
        });
        written += 1;
    }
    Ok(written)
}
