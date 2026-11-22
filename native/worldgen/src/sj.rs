//! Default-off S-J schema/materialization.
use crate::rng::{substream, GenRng, RngExt};
use crate::schema::{
    ColumnSpec, ForeignKeySpec, SchemaOptions, SchemaSpec, SemType, TableKind, TableSpec,
};
use crate::scm::{ColumnData, GenOptions, TableData};
use anyhow::{ensure, Result};
use std::collections::BTreeMap;

impl SchemaOptions {
    pub(crate) fn validate_sj(&self) -> Result<()> {
        ensure!(
            self.satellite_tables.0 >= 1 && self.satellite_tables.0 <= self.satellite_tables.1,
            "S-J requires at least one satellite per entity"
        );
        ensure!(
            self.hub_fan_in && self.junction_prob > 0.0 && self.junction_prob <= 1.0,
            "S-J requires hub_fan_in and junction_prob in (0,1]"
        );
        ensure!(
            self.num_tables_range.0 >= 6 && self.num_tables_range.1 <= 24,
            "S-J total table bounds must be within 6..24"
        );
        ensure!(
            self.num_cols_range.0 >= 4,
            "S-J needs at least four informative columns to move 30..60 percent"
        );
        ensure!(
            self.num_tables_range.0 <= self.num_tables_range.1,
            "reversed total table bounds"
        );
        ensure!(
            self.satellite_tables.1 <= self.max_satellites_per_entity,
            "satellite request exceeds max_satellites_per_entity"
        );
        ensure!(
            self.max_junction_tables >= 3,
            "S-J requires two membership junctions and one click-history junction"
        );
        ensure!(
            self.hub_fanin_zipf.is_finite() && self.hub_fanin_zipf >= 0.0,
            "hub_fanin_zipf must be finite and nonnegative"
        );
        let minimum = self
            .satellite_tables
            .0
            .checked_add(2)
            .and_then(|width| 3usize.checked_mul(width))
            .and_then(|n| n.checked_add(7))
            .ok_or_else(|| anyhow::anyhow!("S-J table budget overflow"))?;
        ensure!(
            minimum <= self.num_tables_range.1,
            "S-J request cannot fit total table budget: needs at least {minimum}"
        );
        Ok(())
    }
}
fn fk(name: &str, parent: &TableSpec) -> ForeignKeySpec {
    ForeignKeySpec {
        col_name: name.into(),
        parent_table: parent.name.clone(),
        parent_id: parent.id,
    }
}
pub fn schema(rng: &mut GenRng, options: &SchemaOptions) -> Result<SchemaSpec> {
    options.validate_sj()?;
    // Reserve five label sidecars and two mandatory membership junctions.
    let budget = options.num_tables_range.1;
    let n = rng.usize_inclusive(3, (budget - 7) / (2 + options.satellite_tables.0));
    let mut tables: Vec<TableSpec> = (0..n)
        .map(|id| {
            let width = rng.usize_inclusive(options.num_cols_range.0, options.num_cols_range.1);
            let base_rows =
                rng.usize_inclusive(options.rows_entity_range.0, options.rows_entity_range.1);
            TableSpec {
                id,
                name: if id < 2 {
                    format!("sj_hub_{id}")
                } else {
                    format!("sj_entity_{id}")
                },
                pkey_col: "row_idx".into(),
                fkeys: vec![],
                kind: TableKind::Entity,
                num_rows: base_rows,
                features: (0..width)
                    .map(|j| ColumnSpec {
                        name: format!("feature_{j}"),
                        sem_type: SemType::Numerical,
                        num_classes: None,
                    })
                    .collect(),
            }
        })
        .collect();
    for p in 0..n {
        let id = tables.len();
        tables.push(TableSpec {
            id,
            name: format!("sj_stream_{p}"),
            pkey_col: "row_idx".into(),
            fkeys: vec![fk("entity", &tables[p])],
            kind: TableKind::Activity,
            num_rows: tables[p].num_rows,
            features: vec![],
        });
    }
    let mut remaining = budget - 2 * n - 7;
    for p in 0..n {
        let hi = options
            .satellite_tables
            .1
            .min(remaining - (n - p - 1) * options.satellite_tables.0);
        let k = rng.usize_inclusive(options.satellite_tables.0, hi);
        remaining -= k;
        let width = tables[p].features.len();
        let count = rng.usize_inclusive(
            (0.3 * width as f64).ceil() as usize,
            (0.6 * width as f64).floor() as usize,
        );
        let moved: Vec<_> = tables[p].features.drain(..count).collect();
        for j in 0..k {
            let id = tables.len();
            tables.push(TableSpec {
                id,
                name: format!("sj_sat_{p}_{j}"),
                pkey_col: "row_idx".into(),
                fkeys: vec![fk("entity", &tables[p])],
                kind: TableKind::Entity,
                num_rows: tables[p].num_rows,
                features: moved
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % k == j)
                    .map(|(_, c)| c.clone())
                    .collect(),
            });
        }
    }
    // Mandatory focal junctions first; optional pairs cannot consume their slots.
    let mut pairs = vec![(0, 2), (1, 2)];
    let mut candidates: Vec<_> = (0..n)
        .flat_map(|a| (a + 1..n).map(move |b| (a, b)))
        .filter(|pair| !pairs.contains(pair))
        .collect();
    rng.shuffle(&mut candidates);
    for pair in candidates {
        if pairs.len() >= options.max_junction_tables - 1 || remaining == 0 {
            break;
        }
        if rng.bernoulli(options.junction_prob) {
            pairs.push(pair);
            remaining -= 1;
        }
    }
    for (a, b) in pairs {
        let id = tables.len();
        tables.push(TableSpec {
            id,
            name: format!("sj_junction_{a}_{b}"),
            pkey_col: "row_idx".into(),
            fkeys: vec![fk("fk_a", &tables[a]), fk("fk_b", &tables[b])],
            kind: TableKind::Activity,
            num_rows: tables[b].num_rows,
            features: if b == 2 || rng.bernoulli(0.5) {
                vec![ColumnSpec {
                    name: "tag".into(),
                    sem_type: SemType::Class,
                    num_classes: Some(4),
                }]
            } else {
                vec![]
            },
        });
    }
    ensure!(
        tables.len() + 5 >= options.num_tables_range.0,
        "S-J draw cannot meet total table lower bound with requested caps"
    );
    ensure!(
        tables.len() + 5 <= budget,
        "S-J total table budget exceeded"
    );
    Ok(SchemaSpec {
        topo_order: (0..tables.len()).collect(),
        tables,
    })
}

fn hub_weights(n: usize, exponent: f64) -> Vec<f64> {
    (1..=n).map(|r| (r as f64).powf(-exponent)).collect()
}
pub fn generate_table(
    seed: u64,
    spec: &TableSpec,
    previous: &[Option<TableData>],
    start: i64,
    end: i64,
    options: &GenOptions,
) -> Result<TableData> {
    let parent = |j: usize| previous[spec.fkeys[j].parent_id].as_ref().unwrap();
    if spec.name.starts_with("sj_stream_") {
        return generate_stream(seed, spec, parent(0), start, end);
    }
    let satellite = spec.name.starts_with("sj_sat_");
    let junction = spec.name.starts_with("sj_junction_");
    let mut n = spec.num_rows;
    let mut columns = vec![];
    let mut links = BTreeMap::new();
    if satellite {
        n = parent(0).num_rows();
        columns.push((
            "entity".into(),
            parent(0).column(&parent(0).pkey_col).unwrap().clone(),
        ));
        links.insert("entity".into(), parent(0).name.clone());
    } else if junction {
        let mut rng = substream(seed, &format!("sj:edges:{}", spec.name));
        let a = parent(0);
        let b = parent(1);
        let ak = match a.column(&a.pkey_col).unwrap() {
            ColumnData::I64(x) => x,
            _ => unreachable!(),
        };
        let bk = match b.column(&b.pkey_col).unwrap() {
            ColumnData::I64(x) => x,
            _ => unreachable!(),
        };
        let a_cold = zero_history_rows(seed, &a.name, ak.len());
        let mut ranks: Vec<_> = ak
            .iter()
            .enumerate()
            .filter_map(|(r, &k)| (!a_cold[r]).then_some(k))
            .collect();
        substream(seed, &format!("sj:hub-ranks:{}", a.name)).shuffle(&mut ranks);
        let weights = if ranks.is_empty() {
            vec![]
        } else {
            hub_weights(ranks.len(), options.schema.hub_fanin_zipf)
        };
        let mut fa = vec![];
        let mut fb = vec![];
        let mut tags = vec![];
        let cold = zero_history_rows(seed, &b.name, bk.len());
        let mut order: Vec<_> = (0..bk.len()).collect();
        if let Some(ColumnData::TimestampNs(ts)) = b.time_col.as_ref().and_then(|c| b.column(c)) {
            order.sort_by_key(|&r| (ts[r], r));
        }
        let mut available = std::collections::BTreeSet::new();
        let mut leaders = BTreeMap::<i64, usize>::new();
        let mut assignment = vec![None; bk.len()];
        for r in order {
            if cold[r] {
                available.insert(r);
                continue;
            }
            if ranks.is_empty() {
                continue;
            }
            let mut hub = ranks[rng.choose_weighted(&weights)];
            if !leaders.contains_key(&hub) {
                if let Some(&leader) = available.iter().next() {
                    available.remove(&leader);
                    leaders.insert(hub, leader);
                    assignment[leader] = Some(hub);
                } else if let Some((&existing, _)) = leaders.iter().next() {
                    hub = existing;
                } else {
                    continue;
                }
            }
            assignment[r] = Some(hub);
        }
        let mut singleton_keys = ak
            .iter()
            .enumerate()
            .filter_map(|(r, &key)| a_cold[r].then_some(key));
        for (r, hub) in assignment.iter_mut().enumerate() {
            if cold[r] && hub.is_none() {
                *hub = singleton_keys.next();
            }
        }
        for (r, hub) in assignment.into_iter().enumerate() {
            if let Some(hub) = hub {
                for _ in 0..rng.usize_inclusive(1, 6) {
                    fa.push(hub);
                    fb.push(bk[r]);
                    tags.push(rng.choose_index(4) as i64);
                }
            }
        }
        n = fa.len();
        columns.push(("fk_a".into(), ColumnData::I64(fa)));
        columns.push(("fk_b".into(), ColumnData::I64(fb)));
        links.insert("fk_a".into(), a.name.clone());
        links.insert("fk_b".into(), b.name.clone());
        if !spec.features.is_empty() {
            let mut shuffled = tags.clone();
            substream(seed, &format!("sj:nulltags:{}", spec.name)).shuffle(&mut shuffled);
            columns.push(("tag".into(), ColumnData::I64(tags)));
            columns.push(("sj_null_tag".into(), ColumnData::I64(shuffled)));
        }
    }
    columns.insert(
        0,
        (
            spec.pkey_col.clone(),
            ColumnData::I64((0..n as i64).collect()),
        ),
    );
    if !junction {
        // Moved columns keep the entity's named Gaussian stream, but occur ONLY
        // in their satellite. Own and satellite coordinates are independent.
        let source = if satellite {
            &parent(0).name
        } else {
            &spec.name
        };
        for f in &spec.features {
            let mut rng = substream(seed, &format!("sj:feature:{source}:{}", f.name));
            columns.push((
                f.name.clone(),
                ColumnData::F64((0..n).map(|_| Some(rng.standard_normal())).collect()),
            ));
        }
        for j in 0..4 {
            let mut rng = substream(seed, &format!("sj:own:{}:{j}", spec.name));
            columns.push((
                format!("sj_x_{j}"),
                ColumnData::F64((0..n).map(|_| Some(rng.standard_normal())).collect()),
            ));
        }
    }
    // Junctions/satellites are static attributes/links. Only member entity rows
    // carry event clocks; this permits strict past filtering at the endpoint
    // without requiring a query row to precede itself at a DFS intermediate hop.
    let timed = spec.name.starts_with("sj_entity_");
    if timed {
        let mut order: Vec<_> = (0..n).collect();
        substream(seed, &format!("sj:clock:{}", spec.name)).shuffle(&mut order);
        columns.push((
            "date".into(),
            ColumnData::TimestampNs(
                order
                    .iter()
                    .map(|&r| {
                        start
                            + (((end as i128 - start as i128) * r as i128) / (n.max(2) - 1) as i128)
                                as i64
                    })
                    .collect(),
            ),
        ));
    }
    Ok(TableData {
        name: spec.name.clone(),
        pkey_col: spec.pkey_col.clone(),
        fkey_col_to_pkey_table: links,
        time_col: timed.then(|| "date".into()),
        columns,
        hsbm_hierarchy: vec![],
        fk_hsbm: BTreeMap::new(),
        feature_contract: Default::default(),
        bag_summary: Default::default(),
        rank_field: Default::default(),
    })
}

pub fn is_target(name: &str) -> bool {
    [
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
        "hubrate_r",
        "hubrate_k2",
        "hubrate_null_r",
        "hubrate_null_k2",
        "hubhistj_r",
        "hubhistj_k2",
        "hubhistj_null_r",
        "hubhistj_null_k2",
    ]
    .iter()
    .any(|stem| name == *stem || name.starts_with(&format!("{stem}_")))
        || name.starts_with("xrep_y_")
        || [
            "condrel_r",
            "condrel_k2",
            "condrel_permuted_r",
            "condrel_permuted_k2",
        ]
        .iter()
        .any(|stem| name == *stem || name.starts_with(&format!("{stem}_")))
}

pub fn streamnull_draw(seed: u64, own: &[f64]) -> Vec<f64> {
    let mut w = substream(seed, "sj:weights:streamnull");
    let sign = w.f64_range(-1.5, 1.5).signum();
    let mut rng = substream(seed, "sj:readout:streamnull");
    let sigma = rng.f64_range(0.02, 0.25);
    own.iter()
        .map(|x| sign * x + sigma * rng.standard_normal())
        .collect()
}
fn generate_stream(
    seed: u64,
    spec: &TableSpec,
    parent: &TableData,
    start: i64,
    end: i64,
) -> Result<TableData> {
    let own = parent.column("sj_x_0").unwrap().as_numeric_lossy().unwrap();
    let _already_drawn_null_y = streamnull_draw(seed, &own);
    let ColumnData::I64(keys) = parent.column(&parent.pkey_col).unwrap() else {
        unreachable!()
    };
    let n = keys.len();
    let mut rng = substream(seed, &format!("sj:stream-events:{}", spec.name));
    let cold = zero_history_rows(seed, &parent.name, n);
    let rows: Vec<_> = (0..n).filter(|&r| !cold[r]).collect();
    let weights: Vec<_> = (1..=64).map(|k| (k as f64).powf(-1.1)).collect();
    let times = parent.time_col.as_ref().and_then(|c| parent.column(c));
    let mut events = vec![];
    for &r in &rows {
        let cutoff = if let Some(ColumnData::TimestampNs(t)) = times {
            t[r]
        } else {
            start + ((end as i128 - start as i128) / 2) as i64
        };
        let count = rng.choose_weighted(&weights) + 1;
        for _ in 0..count {
            // Ages relative to the entity cutoff prevent row-time shortcuts.
            // Future events are present, but never enter the label aggregates.
            let age = rng.f64_range(-30.0, 365.0);
            let ts = (cutoff as i128 - (age * 86_400_000_000_000.0) as i128)
                .clamp(i64::MIN as i128 + 1, i64::MAX as i128) as i64;
            events.push((ts, keys[r]));
        }
    }
    events.sort_unstable();
    let count = events.len();
    Ok(TableData {
        name: spec.name.clone(),
        pkey_col: spec.pkey_col.clone(),
        fkey_col_to_pkey_table: [("entity".into(), parent.name.clone())].into(),
        time_col: Some("date".into()),
        columns: vec![
            (
                spec.pkey_col.clone(),
                ColumnData::I64((0..count as i64).collect()),
            ),
            (
                "entity".into(),
                ColumnData::I64(events.iter().map(|e| e.1).collect()),
            ),
            (
                "date".into(),
                ColumnData::TimestampNs(events.iter().map(|e| e.0).collect()),
            ),
        ],
        hsbm_hierarchy: vec![],
        fk_hsbm: BTreeMap::new(),
        feature_contract: Default::default(),
        bag_summary: Default::default(),
        rank_field: Default::default(),
    })
}

/// Keep these event tables out of every other recipe's discovery paths.
pub fn take_streams(db: &mut crate::scm::GeneratedDb) -> Vec<TableData> {
    let (streams, old) = std::mem::take(&mut db.tables)
        .into_iter()
        .partition(|t: &TableData| t.name.starts_with("sj_stream_"));
    db.tables = old;
    streams
}

pub fn zero_history_rows(seed: u64, table: &str, n: usize) -> Vec<bool> {
    let mut rows: Vec<_> = (0..n).collect();
    substream(seed, &format!("sj:zero-history:{table}")).shuffle(&mut rows);
    if n > 0 {
        let first = if table.starts_with("sj_entity_") {
            let mut clock: Vec<_> = (0..n).collect();
            substream(seed, &format!("sj:clock:{table}")).shuffle(&mut clock);
            clock.iter().position(|&rank| rank == 0).unwrap()
        } else {
            0
        };
        let pos = rows.iter().position(|&r| r == first).unwrap();
        rows.swap(0, pos);
    }
    let mut cold = vec![false; n];
    for &r in &rows[..((n as f64 * 0.4).ceil() as usize)] {
        cold[r] = true;
    }
    cold
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SjStatistics {
    pub total_tables: usize,
    pub junction_tables: usize,
    pub max_satellites_per_entity: usize,
    pub hub_top5_share: f64,
}

pub fn statistics(db: &crate::scm::GeneratedDb) -> Option<SjStatistics> {
    if db.table("sj_hub_0").is_none() {
        return None;
    }
    let mut satellites = BTreeMap::<&str, usize>::new();
    let mut degrees = BTreeMap::<&str, BTreeMap<i64, usize>>::new();
    for table in &db.tables {
        if table.name.starts_with("sj_sat_") {
            *satellites
                .entry(&table.fkey_col_to_pkey_table["entity"])
                .or_default() += 1;
        }
        if table.name.starts_with("sj_junction_") {
            let parent = table.fkey_col_to_pkey_table["fk_a"].as_str();
            let counts = degrees.entry(parent).or_default();
            if let Some(ColumnData::I64(keys)) = table.column("fk_a") {
                for &key in keys {
                    *counts.entry(key).or_default() += 1;
                }
            }
        }
    }
    let mut top = 0usize;
    let mut total = 0usize;
    for (parent, counts) in degrees {
        let n = db.table(parent)?.num_rows();
        let mut counts: Vec<_> = counts.into_values().collect();
        counts.sort_unstable_by(|a, b| b.cmp(a));
        top += counts
            .iter()
            .take((n as f64 * 0.05).ceil() as usize)
            .sum::<usize>();
        total += counts.iter().sum::<usize>();
    }
    Some(SjStatistics {
        total_tables: db.tables.len(),
        junction_tables: db
            .tables
            .iter()
            .filter(|t| t.fkey_col_to_pkey_table.len() >= 2)
            .count(),
        max_satellites_per_entity: satellites.into_values().max().unwrap_or(0),
        hub_top5_share: if total == 0 {
            0.0
        } else {
            top as f64 / total as f64
        },
    })
}
