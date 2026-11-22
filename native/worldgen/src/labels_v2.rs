use crate::rng::{child_substream, substream, RngExt};
use crate::scm::{ColumnData, FeatureContract, GeneratedDb, TableData, ENTITY_ID_COL};
use anyhow::Result;
use serde::Serialize;

pub fn add_v2_prior_labels(
    db: &mut GeneratedDb,
    seed: u64,
    enable_exact_aggregates: bool,
    enable_nuisance_text: bool,
) -> Result<()> {
    let streams = crate::sj::take_streams(db);
    add_grandparent_lookup_labels(db, seed);
    add_recent_event_lookup_labels(db, seed);
    add_conjugate_bag_labels(db, seed);
    add_nonregular_bag_labels(db, seed);
    add_hazard_labels(db);
    add_query_selection_labels(db, seed);
    add_mechanism_identification_labels(db, seed);
    add_atomic_route_labels(db, seed);
    add_comembership_peer_label_labels(db, seed);
    add_cross_entity_functional_labels(db, seed);
    add_latent_state_link_labels(db, seed);
    if enable_exact_aggregates {
        add_exact_aggregate_sidecars(db, seed);
    }
    add_relation_null_labels(db, seed);
    add_transformed_pointer_labels(db, seed);
    add_aggcol_labels(db, seed);
    add_sj_labels(db, seed);
    add_mpnn_readout_labels(db, seed);
    add_population_law_labels(db, seed);
    // Keep the opt-in S-J stream tables out of every other recipe's relation discovery.
    db.tables.extend(streams);
    add_sj_extension_labels(db, seed);
    if enable_nuisance_text {
        add_nuisance_text_columns(db, seed);
    }
    Ok(())
}

fn apply_observation_link(
    latent: &[f64],
    rng: &mut crate::rng::GenRng,
) -> (Vec<f64>, &'static str) {
    let branch = rng.choose_weighted(&[0.5, 0.25, 0.25]);
    match branch {
        0 => {
            let mut y = latent.to_vec();
            standardize(&mut y);
            (y, "lid")
        }
        1 => {
            // Zero-inflated hurdle: everything below a drawn quantile clamps to an exact 0 (the
            // mode), the rest keeps its ordering above it.
            let p0 = rng.f64_range(0.3, 0.7);
            let cut = quantile_of(latent, p0);
            let y: Vec<f64> = latent.iter().map(|v| (v - cut).max(0.0)).collect();
            if y.iter().any(|v| *v != y[0]) {
                (y, "lzi")
            } else {
                let mut y = latent.to_vec();
                standardize(&mut y);
                (y, "lid")
            }
        }
        _ => {
            // Heavy affine: standardized shape at a drawn scale and offset.
            let scale = rng.f64_range(0.1_f64.ln(), 1_000.0_f64.ln()).exp();
            let offset = rng.f64_range(-3.0, 3.0) * scale;
            let mut y = latent.to_vec();
            standardize(&mut y);
            for v in y.iter_mut() {
                *v = *v * scale + offset;
            }
            (y, "lha")
        }
    }
}

fn contaminate_payload(payload: &mut [f64], salt: u64, rng: &mut crate::rng::GenRng) {
    if !rng.bernoulli(0.5) {
        return;
    }
    let epsilon = rng.f64_range(0.02, 0.2);
    let lo = quantile_of(payload, 0.25);
    let hi = quantile_of(payload, 0.75);
    let spread = (hi - lo).max(1e-9);
    let center = quantile_of(payload, 0.5);
    for (row, value) in payload.iter_mut().enumerate() {
        let mut r = child_substream(salt, "labels_v2:contaminate", row);
        if r.bernoulli(epsilon) {
            *value = center + 10.0 * spread * r.standard_normal();
        }
    }
}

/// MNAR missingness for a GENERATED event payload: larger values are more
/// likely to go missing, so the missingness indicator itself carries signal.
/// Returns per-row missingness; the caller publishes None cells.
fn mnar_mask(payload: &[f64], salt: u64, rng: &mut crate::rng::GenRng) -> Option<Vec<bool>> {
    if !rng.bernoulli(0.5) {
        return None;
    }
    let base_rate = rng.f64_range(0.05, 0.3);
    let cut = quantile_of(payload, 0.75);
    Some(
        payload
            .iter()
            .enumerate()
            .map(|(row, v)| {
                let mut r = child_substream(salt, "labels_v2:mnar", row);
                let rate = if *v > cut { 3.0 * base_rate } else { base_rate };
                r.bernoulli(rate.min(0.9))
            })
            .collect(),
    )
}

/// Entity ids and event times for a panel table, or None when the table has
/// neither an entity column nor a usable time column.
fn entity_and_time(table: &TableData) -> Option<(Vec<i64>, Vec<f64>)> {
    let n = table.num_rows();
    let entity = match table.column(ENTITY_ID_COL) {
        Some(ColumnData::I64(values)) if values.len() == n => values.clone(),
        _ => {
            // Fall back to the first foreign key: grouping by it is grouping by parent entity.
            let fk = table.fkey_col_to_pkey_table.keys().next()?;
            match table.column(fk) {
                Some(ColumnData::I64(values)) if values.len() == n => values.clone(),
                _ => return None,
            }
        }
    };
    let tc = table.time_col.as_ref()?;
    let t_vec: Vec<f64> = match table.column(tc) {
        Some(ColumnData::TimestampNs(values)) if values.len() == n => {
            values.iter().map(|v| *v as f64).collect()
        }
        Some(other) => {
            let v = other.as_numeric_lossy()?;
            if v.len() != n || v.iter().any(|x| !x.is_finite()) {
                return None;
            }
            v
        }
        None => return None,
    };
    Some((entity, t_vec))
}

/// A numeric FEATURE column (never a published target of any family, this module's included) with
/// real dispersion.
fn pick_payload(
    table: &TableData,
    candidates: &[String],
    rng: &mut crate::rng::GenRng,
) -> Option<(String, Vec<f64>)> {
    if candidates.is_empty() {
        return None;
    }
    let first = rng.choose_index(candidates.len());
    for offset in 0..candidates.len() {
        let name = &candidates[(first + offset) % candidates.len()];
        if FeatureContract::is_published_target(name) {
            continue;
        }
        let Some(values) = table.column(name).and_then(ColumnData::as_numeric_lossy) else {
            continue;
        };
        if values.len() != table.num_rows() {
            continue;
        }
        let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        if finite.len() < table.num_rows() / 2 || finite.len() < 4 {
            continue;
        }
        let mean = finite.iter().sum::<f64>() / finite.len() as f64;
        let var = finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / finite.len() as f64;
        if !(var.is_finite() && var > 1e-12) {
            continue;
        }
        // NaN cells are median-filled so downstream arithmetic stays total.
        let median = quantile_of(&finite, 0.5);
        let filled: Vec<f64> = values
            .iter()
            .map(|v| if v.is_finite() { *v } else { median })
            .collect();
        return Some((name.clone(), filled));
    }
    None
}

fn quantile_of(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let idx = ((sorted.len() as f64 - 1.0) * q.clamp(0.0, 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn is_constant(values: &[f64]) -> bool {
    values.windows(2).all(|w| w[0] == w[1])
}

fn standardize(y: &mut [f64]) {
    let n = y.len() as f64;
    if n < 1.0 {
        return;
    }
    let mean = y.iter().sum::<f64>() / n;
    let var = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let sd = var.sqrt();
    if sd.is_finite() && sd > 1e-12 {
        for v in y.iter_mut() {
            *v = (*v - mean) / sd;
        }
    }
}

/// Median-cut binary view of a latent, published beside the regression view.
fn median_k2(latent: &[f64]) -> Option<Vec<i64>> {
    let cut = quantile_of(latent, 0.5);
    let labels: Vec<i64> = latent.iter().map(|v| i64::from(*v > cut)).collect();
    let positives = labels.iter().filter(|v| **v == 1).count();
    (positives > 0 && positives < labels.len()).then_some(labels)
}

fn push_f64(table: &mut TableData, base: &str, values: Vec<f64>) -> String {
    let name = table.next_column_name(base);
    let data: Vec<Option<f64>> = values.into_iter().map(Some).collect();
    table.push_column(name.clone(), ColumnData::F64(data));
    name
}

fn push_f64_opt(table: &mut TableData, base: &str, values: Vec<Option<f64>>) -> String {
    let name = table.next_column_name(base);
    table.push_column(name.clone(), ColumnData::F64(values));
    name
}

fn push_i64(table: &mut TableData, base: &str, values: Vec<i64>) -> String {
    let name = table.next_column_name(base);
    table.push_column(name.clone(), ColumnData::I64(values));
    name
}

/// Strictly-prior same-entity row indices, per row.
fn prior_same_entity_rows(entity: &[i64], t_vec: &[f64]) -> Vec<Vec<usize>> {
    use std::collections::BTreeMap;
    let n = entity.len();
    let mut rows_by_entity: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (row, id) in entity.iter().enumerate() {
        rows_by_entity.entry(*id).or_default().push(row);
    }
    let mut prior: Vec<Vec<usize>> = vec![Vec::new(); n];
    for rows in rows_by_entity.values_mut() {
        rows.sort_by(|a, b| t_vec[*a].total_cmp(&t_vec[*b]).then_with(|| a.cmp(b)));
        for (pos, &row) in rows.iter().enumerate() {
            // Ties in time stay out of the bag: "prior" means strictly earlier.
            prior[row] = rows[..pos]
                .iter()
                .copied()
                .filter(|&r| t_vec[r] < t_vec[row])
                .collect();
        }
    }
    prior
}

/// Small per-row jitter at a stated fraction of the answers' IQR.
fn iqr_noise(latent: &[f64], fraction: f64, seed: u64, tag: &str) -> Vec<f64> {
    let lo = quantile_of(latent, 0.25);
    let hi = quantile_of(latent, 0.75);
    let spread = (hi - lo).max(1e-9);
    latent
        .iter()
        .enumerate()
        .map(|(row, v)| {
            let mut rng = child_substream(seed, tag, row);
            v + fraction * spread * rng.standard_normal()
        })
        .collect()
}

fn add_grandparent_lookup_labels(db: &mut GeneratedDb, seed: u64) {
    add_fk_lookup_labels(db, seed, 2, false);
}

pub fn add_transformed_pointer_labels(db: &mut GeneratedDb, seed: u64) {
    add_fk_lookup_labels(db, seed, 1, true);
    add_fk_lookup_labels(db, seed, 2, true);
    add_recent_event_lookup_impl(db, seed, true);
}

fn publish_transformed_pointer(
    table: &mut TableData,
    pointed: &[f64],
    seed: u64,
    geometry: &str,
    payload: &str,
) {
    let mut rng = substream(seed, &format!("plookupt:own:{geometry}:{}", table.name));
    let candidates = table.numeric_feature_columns();
    let Some((own_name, own)) = pick_payload(table, &candidates, &mut rng) else {
        return;
    };
    let y: Vec<f64> = pointed
        .iter()
        .zip(&own)
        .map(|(p, x)| p.sin() + 0.5 * x.tanh())
        .collect();
    let k2 = y.iter().map(|v| i64::from(*v > 0.0)).collect();
    let suffix = format!("{geometry}_{payload}_{own_name}_{}", table.name);
    push_f64(table, &format!("plookupt_r_{suffix}"), y);
    push_i64(table, &format!("plookupt_k2_{suffix}"), k2);
}

fn add_fk_lookup_labels(db: &mut GeneratedDb, seed: u64, hops: usize, transform: bool) {
    let table_count = db.tables.len();
    for table_idx in 0..table_count {
        let Some((fk_col, parent_name)) = db.tables[table_idx]
            .fkey_col_to_pkey_table
            .iter()
            .next()
            .map(|(c, p)| (c.clone(), p.clone()))
        else {
            continue;
        };
        let Some(parent_idx) = db.tables.iter().position(|t| t.name == parent_name) else {
            continue;
        };
        if parent_idx == table_idx {
            continue;
        }
        let (gp_idx, parent_fk) = if hops == 1 {
            // Identity second hop reuses the exact same lookup/fallback below.
            (
                parent_idx,
                (0..db.tables[parent_idx].num_rows() as i64).collect(),
            )
        } else {
            let Some((gp_fk_col, gp_name)) = db.tables[parent_idx]
                .fkey_col_to_pkey_table
                .iter()
                .next()
                .map(|(c, p)| (c.clone(), p.clone()))
            else {
                continue;
            };
            let Some(gp_idx) = db.tables.iter().position(|t| t.name == gp_name) else {
                continue;
            };
            if gp_idx == table_idx || gp_idx == parent_idx {
                continue;
            }
            let Some(ColumnData::I64(parent_fk)) =
                db.tables[parent_idx].column(&gp_fk_col).cloned()
            else {
                continue;
            };
            (gp_idx, parent_fk)
        };
        let tag = if transform {
            "labels_v2:plookupt:fk"
        } else {
            "labels_v2:plookup2"
        };
        let mut rng = substream(seed + 131_071 * (table_idx as u64 + 1), tag);
        let gp_candidates = db.tables[gp_idx].numeric_feature_columns();
        let Some((gp_col, gp_vals)) = pick_payload(&db.tables[gp_idx], &gp_candidates, &mut rng)
        else {
            continue;
        };
        let Some(ColumnData::I64(child_fk)) = db.tables[table_idx].column(&fk_col).cloned() else {
            continue;
        };
        let n_parent = parent_fk.len() as i64;
        let n_gp = gp_vals.len() as i64;
        let median = quantile_of(&gp_vals, 0.5);
        let latent: Vec<f64> = child_fk
            .iter()
            .map(|&fk| {
                if fk >= 0 && fk < n_parent {
                    let gp = parent_fk[fk as usize];
                    if gp >= 0 && gp < n_gp {
                        return gp_vals[gp as usize];
                    }
                }
                median
            })
            .collect();
        if transform {
            publish_transformed_pointer(
                &mut db.tables[table_idx],
                &latent,
                seed,
                if hops == 1 { "h1" } else { "h2" },
                &gp_col,
            );
            continue;
        }
        let lo = quantile_of(&latent, 0.25);
        let hi = quantile_of(&latent, 0.75);
        if !((hi - lo).is_finite() && hi - lo > 0.0) {
            continue;
        }
        let noise = rng.f64_range(0.02, 0.15);
        let noised = iqr_noise(
            &latent,
            noise,
            seed + 131_071 * (table_idx as u64 + 1),
            "labels_v2:plookup2:row",
        );
        if is_constant(&noised) {
            continue;
        }
        let (y, link) = apply_observation_link(&noised, &mut rng);
        let table_name = db.tables[table_idx].name.clone();
        push_f64(
            &mut db.tables[table_idx],
            &format!("plookup2_r_{link}_{gp_col}_{table_name}"),
            y,
        );
        if let Some(k2) = median_k2(&latent) {
            push_i64(
                &mut db.tables[table_idx],
                &format!("plookup2_k2_{gp_col}_{table_name}"),
                k2,
            );
        }
    }
}

fn add_recent_event_lookup_labels(db: &mut GeneratedDb, seed: u64) {
    add_recent_event_lookup_impl(db, seed, false);
}

fn add_recent_event_lookup_impl(db: &mut GeneratedDb, seed: u64, transform: bool) {
    let table_count = db.tables.len();
    for table_idx in 0..table_count {
        if db.tables[table_idx].num_rows() < 8 {
            continue;
        }
        let Some((entity, t_vec)) = entity_and_time(&db.tables[table_idx]) else {
            continue;
        };
        let mut rng = substream(
            seed + 149_057 * (table_idx as u64 + 1),
            if transform {
                "labels_v2:plookupt:recent"
            } else {
                "labels_v2:plookuprec"
            },
        );
        let candidates = db.tables[table_idx].numeric_feature_columns();
        let Some((col, payload)) = pick_payload(&db.tables[table_idx], &candidates, &mut rng)
        else {
            continue;
        };
        let prior = prior_same_entity_rows(&entity, &t_vec);
        if prior.iter().filter(|p| !p.is_empty()).count() < db.tables[table_idx].num_rows() / 4 {
            // A table where almost every row is its entity's first event has
            // no recency pointer to speak of.
            continue;
        }
        let median = quantile_of(&payload, 0.5);
        let latent: Vec<f64> = prior
            .iter()
            .map(|rows| {
                rows.last() // rows are time-sorted; last is the most recent prior event
                    .map(|&r| payload[r])
                    .unwrap_or(median)
            })
            .collect();
        if transform {
            publish_transformed_pointer(&mut db.tables[table_idx], &latent, seed, "recent", &col);
            continue;
        }
        let noise = rng.f64_range(0.02, 0.15);
        let noised = iqr_noise(
            &latent,
            noise,
            seed + 149_057 * (table_idx as u64 + 1),
            "labels_v2:plookuprec:row",
        );
        if is_constant(&noised) {
            continue;
        }
        let (y, link) = apply_observation_link(&noised, &mut rng);
        let table_name = db.tables[table_idx].name.clone();
        push_f64(
            &mut db.tables[table_idx],
            &format!("plookuprec_r_{link}_{col}_{table_name}"),
            y,
        );
        if let Some(k2) = median_k2(&latent) {
            push_i64(
                &mut db.tables[table_idx],
                &format!("plookuprec_k2_{col}_{table_name}"),
                k2,
            );
        }
    }
}

fn add_conjugate_bag_labels(db: &mut GeneratedDb, seed: u64) {
    let table_count = db.tables.len();
    let mut scans = 0usize;
    for table_idx in 0..table_count {
        if scans >= 2 {
            break; // at most two conjugate scans per DB
        }
        let Some((fk_col, parent_name)) = db.tables[table_idx]
            .fkey_col_to_pkey_table
            .iter()
            .next()
            .map(|(c, p)| (c.clone(), p.clone()))
        else {
            continue;
        };
        let Some(parent_idx) = db.tables.iter().position(|t| t.name == parent_name) else {
            continue;
        };
        if parent_idx == table_idx {
            continue;
        }
        let Some(ColumnData::I64(child_fk)) = db.tables[table_idx].column(&fk_col).cloned() else {
            continue;
        };
        let n_parent = db.tables[parent_idx].num_rows();
        let n_child = child_fk.len();
        if n_parent < 8 || n_child < 4 * n_parent / 2 {
            continue;
        }
        let salt = seed + 163_841 * (table_idx as u64 + 1);
        let mut rng = substream(salt, "labels_v2:bagc");
        let core = rng.choose_index(3); // 0 beta-bern, 1 gamma-pois, 2 normal-normal
                                        // Per-parent latents, each from its own child substream so parent
                                        // count changes cannot shift another parent's draw.
        let (theta, payload): (Vec<f64>, Vec<f64>) = match core {
            0 => {
                let a = rng.usize_inclusive(1, 5) as f64;
                let b = rng.usize_inclusive(1, 5) as f64;
                let theta: Vec<f64> = (0..n_parent)
                    .map(|p| child_substream(salt, "labels_v2:bagc:theta", p).beta(a, b))
                    .collect();
                let payload = child_fk
                    .iter()
                    .enumerate()
                    .map(|(row, &fk)| {
                        let mut r = child_substream(salt, "labels_v2:bagc:event", row);
                        let p = if fk >= 0 && (fk as usize) < n_parent {
                            theta[fk as usize]
                        } else {
                            r.beta(a, b)
                        };
                        f64::from(u8::from(r.bernoulli(p)))
                    })
                    .collect();
                (theta, payload)
            }
            1 => {
                let shape = rng.usize_inclusive(2, 4);
                let rate = (rng.f64_range(0.5_f64.ln(), 5.0_f64.ln())).exp();
                let draw_lambda = |r: &mut crate::rng::GenRng| {
                    // Gamma(shape, rate) as a sum of exponentials (integer shape).
                    let mut total = 0.0;
                    for _ in 0..shape {
                        let u: f64 = r.f64_range(f64::MIN_POSITIVE, 1.0);
                        total += -u.ln() / rate;
                    }
                    total
                };
                let theta: Vec<f64> = (0..n_parent)
                    .map(|p| draw_lambda(&mut child_substream(salt, "labels_v2:bagc:theta", p)))
                    .collect();
                let payload = child_fk
                    .iter()
                    .enumerate()
                    .map(|(row, &fk)| {
                        let mut r = child_substream(salt, "labels_v2:bagc:event", row);
                        let lambda = if fk >= 0 && (fk as usize) < n_parent {
                            theta[fk as usize]
                        } else {
                            draw_lambda(&mut r)
                        };
                        // Knuth's Poisson sampler; lambda is bounded by
                        // construction (shape<=4, rate>=0.5 => mean<=8).
                        let cap = lambda.min(64.0);
                        let limit = (-cap).exp();
                        let mut k = 0u32;
                        let mut prod: f64 = 1.0;
                        loop {
                            prod *= r.f64_range(f64::MIN_POSITIVE, 1.0);
                            if prod <= limit || k >= 512 {
                                break;
                            }
                            k += 1;
                        }
                        f64::from(k)
                    })
                    .collect();
                (theta, payload)
            }
            _ => {
                let mu0 = rng.standard_normal();
                let tau = (rng.f64_range(0.2_f64.ln(), 2.0_f64.ln())).exp();
                let sigma = (rng.f64_range(0.2_f64.ln(), 2.0_f64.ln())).exp();
                let theta: Vec<f64> = (0..n_parent)
                    .map(|p| {
                        let mut r = child_substream(salt, "labels_v2:bagc:theta", p);
                        mu0 + tau * r.standard_normal()
                    })
                    .collect();
                let payload = child_fk
                    .iter()
                    .enumerate()
                    .map(|(row, &fk)| {
                        let mut r = child_substream(salt, "labels_v2:bagc:event", row);
                        let mean = if fk >= 0 && (fk as usize) < n_parent {
                            theta[fk as usize]
                        } else {
                            mu0 + tau * r.standard_normal()
                        };
                        mean + sigma * r.standard_normal()
                    })
                    .collect();
                (theta, payload)
            }
        };
        let core_tag = ["beta", "pois", "norm"][core];
        let child_name = db.tables[table_idx].name.clone();
        let parent_table_name = db.tables[parent_idx].name.clone();
        let mut payload = payload;
        contaminate_payload(&mut payload, salt, &mut rng);
        let missing = mnar_mask(&payload, salt, &mut rng);
        let cells: Vec<Option<f64>> = payload
            .iter()
            .enumerate()
            .map(|(row, v)| match &missing {
                Some(mask) if mask[row] => None,
                _ => Some(*v),
            })
            .collect();
        push_f64_opt(
            &mut db.tables[table_idx],
            &format!("evx_{core_tag}_{child_name}"),
            cells,
        );
        push_f64(
            &mut db.tables[parent_idx],
            &format!("bagc{core_tag}reg_{fk_col}_{parent_table_name}"),
            theta.clone(),
        );
        if let Some(k2) = median_k2(&theta) {
            push_i64(
                &mut db.tables[parent_idx],
                &format!("bagc{core_tag}_k2_{fk_col}_{parent_table_name}"),
                k2,
            );
        }
        scans += 1;
    }
}

fn add_nonregular_bag_labels(db: &mut GeneratedDb, seed: u64) {
    let table_count = db.tables.len();
    for table_idx in 0..table_count {
        if db.tables[table_idx].num_rows() < 8 {
            continue;
        }
        let Some((entity, t_vec)) = entity_and_time(&db.tables[table_idx]) else {
            continue;
        };
        let mut rng = substream(seed + 174_763 * (table_idx as u64 + 1), "labels_v2:bagns");
        let candidates = db.tables[table_idx].numeric_feature_columns();
        let Some((col, payload)) = pick_payload(&db.tables[table_idx], &candidates, &mut rng)
        else {
            continue;
        };
        let prior = prior_same_entity_rows(&entity, &t_vec);
        let with_bag = prior.iter().filter(|p| p.len() >= 2).count();
        if with_bag < db.tables[table_idx].num_rows() / 8 {
            continue;
        }
        let stat = rng.choose_index(3);
        let threshold = quantile_of(&payload, 0.75);
        let latent_raw: Vec<Option<f64>> = prior
            .iter()
            .map(|rows| {
                if rows.is_empty() {
                    return None;
                }
                let bag: Vec<f64> = rows.iter().map(|&r| payload[r]).collect();
                Some(match stat {
                    0 => bag.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                    1 => quantile_of(&bag, 0.9),
                    _ => bag.iter().filter(|v| **v > threshold).count() as f64 / bag.len() as f64,
                })
            })
            .collect();
        let defined: Vec<f64> = latent_raw.iter().filter_map(|v| *v).collect();
        if defined.len() < 4 {
            continue;
        }
        let fill = quantile_of(&defined, 0.5);
        let latent: Vec<f64> = latent_raw.iter().map(|v| v.unwrap_or(fill)).collect();
        let stat_tag = ["max", "q90", "exc"][stat];
        if is_constant(&latent) {
            continue;
        }
        let table_name = db.tables[table_idx].name.clone();
        // Exceedance is a proper fraction in [0,1], so it skips the link and publishes raw.
        let (y, link) = if stat == 2 {
            (latent.clone(), "lid")
        } else {
            apply_observation_link(&latent, &mut rng)
        };
        push_f64(
            &mut db.tables[table_idx],
            &format!("bagns{stat_tag}reg_{link}_{col}_{table_name}"),
            y,
        );
        if let Some(k2) = median_k2(&latent) {
            push_i64(
                &mut db.tables[table_idx],
                &format!("bagns{stat_tag}_k2_{col}_{table_name}"),
                k2,
            );
        }
    }
}

fn add_hazard_labels(db: &mut GeneratedDb) {
    use std::collections::BTreeMap;
    let table_count = db.tables.len();
    for table_idx in 0..table_count {
        let n = db.tables[table_idx].num_rows();
        if n < 8 {
            continue;
        }
        let Some((entity, t_vec)) = entity_and_time(&db.tables[table_idx]) else {
            continue;
        };
        let mut rows_by_entity: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        for (row, id) in entity.iter().enumerate() {
            rows_by_entity.entry(*id).or_default().push(row);
        }
        // Forward gap to the entity's next event; None = censored (last event).
        let mut gap: Vec<Option<f64>> = vec![None; n];
        for rows in rows_by_entity.values_mut() {
            rows.sort_by(|a, b| t_vec[*a].total_cmp(&t_vec[*b]).then_with(|| a.cmp(b)));
            for pair in rows.windows(2) {
                let g = t_vec[pair[1]] - t_vec[pair[0]];
                if g > 0.0 {
                    gap[pair[0]] = Some(g);
                }
            }
        }
        let observed: Vec<f64> = gap.iter().filter_map(|v| *v).collect();
        if observed.len() < n / 4 || observed.len() < 4 {
            continue;
        }
        // Window = median observed inter-event gap, so the classification view
        // is balanced by construction over the uncensored rows.
        let horizon = quantile_of(&observed, 0.5);
        if !(horizon.is_finite() && horizon > 0.0) {
            continue;
        }
        let clf: Vec<i64> = gap
            .iter()
            .map(|g| i64::from(matches!(g, Some(v) if *v <= horizon)))
            .collect();
        let positives = clf.iter().filter(|v| **v == 1).count();
        if positives == 0 || positives == n {
            continue;
        }
        // Censored regression: min(gap, 2H)/2H in [0,1]; censored rows sit at
        // the censoring boundary 1.0 — a bounded survival-style readout.
        let reg: Vec<f64> = gap
            .iter()
            .map(|g| match g {
                Some(v) => (v / (2.0 * horizon)).min(1.0),
                None => 1.0,
            })
            .collect();
        let table_name = db.tables[table_idx].name.clone();
        push_i64(
            &mut db.tables[table_idx],
            &format!("haznext_k2_w50_{table_name}"),
            clf,
        );
        push_f64(
            &mut db.tables[table_idx],
            &format!("hazgapreg_w50_{table_name}"),
            reg,
        );
    }
}

fn add_query_selection_labels(db: &mut GeneratedDb, seed: u64) {
    let table_count = db.tables.len();
    for table_idx in 0..table_count {
        if db.tables[table_idx].num_rows() < 8 {
            continue;
        }
        let Some((entity, t_vec)) = entity_and_time(&db.tables[table_idx]) else {
            continue;
        };
        let mut rng = substream(seed + 196_613 * (table_idx as u64 + 1), "labels_v2:qsel");
        let candidates = db.tables[table_idx].numeric_feature_columns();
        // Three roles: the neighbour score s, the query gate q, the payload p being read.
        let Some((s_col, s_vals)) = pick_payload(&db.tables[table_idx], &candidates, &mut rng)
        else {
            continue;
        };
        let remaining: Vec<String> = candidates
            .iter()
            .filter(|c| **c != s_col)
            .cloned()
            .collect();
        let (q_col, q_vals) = match pick_payload(&db.tables[table_idx], &remaining, &mut rng) {
            Some(found) => found,
            None => (s_col.clone(), s_vals.clone()),
        };
        let remaining2: Vec<String> = remaining.iter().filter(|c| **c != q_col).cloned().collect();
        let (_p_col, p_vals) = match pick_payload(&db.tables[table_idx], &remaining2, &mut rng) {
            Some(found) => found,
            None => (s_col.clone(), s_vals.clone()),
        };
        let prior = prior_same_entity_rows(&entity, &t_vec);
        let with_bag = prior.iter().filter(|p| p.len() >= 2).count();
        if with_bag < db.tables[table_idx].num_rows() / 8 {
            continue;
        }
        // Selection sharpness: log-uniform from a nearly uniform softmax to a near-argmax selection.
        let sharpness = (rng.f64_range(0.5_f64.ln(), 32.0_f64.ln())).exp();
        let s_mean = s_vals.iter().sum::<f64>() / s_vals.len() as f64;
        let s_sd = (s_vals.iter().map(|v| (v - s_mean).powi(2)).sum::<f64>() / s_vals.len() as f64)
            .sqrt()
            .max(1e-9);
        let q_median = quantile_of(&q_vals, 0.5);
        let latent_raw: Vec<Option<f64>> = prior
            .iter()
            .enumerate()
            .map(|(row, rows)| {
                if rows.is_empty() {
                    return None;
                }
                // The focal row's own feature flips the sign of the selection score: which
                // neighbour carries the answer depends on the query.
                let gate = if q_vals[row] > q_median { 1.0 } else { -1.0 };
                let mut weights: Vec<f64> = rows
                    .iter()
                    .map(|&j| (sharpness * gate * (s_vals[j] - s_mean) / s_sd).exp())
                    .collect();
                let total: f64 = weights.iter().sum();
                if !(total.is_finite() && total > 0.0) {
                    let best = rows
                        .iter()
                        .copied()
                        .max_by(|&a, &b| (gate * s_vals[a]).total_cmp(&(gate * s_vals[b])))
                        .expect("rows is non-empty");
                    return Some(p_vals[best]);
                }
                for w in weights.iter_mut() {
                    *w /= total;
                }
                Some(
                    rows.iter()
                        .zip(weights.iter())
                        .map(|(&j, w)| w * p_vals[j])
                        .sum(),
                )
            })
            .collect();
        let defined: Vec<f64> = latent_raw.iter().filter_map(|v| *v).collect();
        if defined.len() < 4 {
            continue;
        }
        let fill = quantile_of(&defined, 0.5);
        let latent: Vec<f64> = latent_raw.iter().map(|v| v.unwrap_or(fill)).collect();
        if is_constant(&latent) {
            continue;
        }
        let (y, link) = apply_observation_link(&latent, &mut rng);
        let table_name = db.tables[table_idx].name.clone();
        push_f64(
            &mut db.tables[table_idx],
            &format!("qselreg_s{sharpness:.1}_{link}_{table_name}"),
            y,
        );
        let k = [2usize, 3, 4][rng.choose_index(3)];
        let mut sorted = latent.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let cuts: Vec<f64> = (1..k).map(|i| sorted[(sorted.len() * i) / k]).collect();
        let classes: Vec<i64> = latent
            .iter()
            .map(|v| cuts.iter().filter(|c| *v > **c).count() as i64)
            .collect();
        if classes.iter().any(|c| *c != classes[0]) {
            push_i64(
                &mut db.tables[table_idx],
                &format!("qsel_k{k}_s{sharpness:.1}_{table_name}"),
                classes,
            );
        }
    }
}

/// The mechanism library the identification family draws from.
fn mident_latent(
    mechanism: usize,
    prior: &[Vec<usize>],
    payload: &[f64],
    threshold: f64,
) -> Vec<Option<f64>> {
    prior
        .iter()
        .map(|rows| {
            if rows.is_empty() {
                return None;
            }
            let bag: Vec<f64> = rows.iter().map(|&r| payload[r]).collect();
            Some(match mechanism {
                0 => bag.iter().sum::<f64>() / bag.len() as f64,
                1 => bag.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                2 => payload[*rows.last().expect("non-empty")],
                _ => bag.iter().filter(|v| **v > threshold).count() as f64 / bag.len() as f64,
            })
        })
        .collect()
}

fn add_mechanism_identification_labels(db: &mut GeneratedDb, seed: u64) {
    let table_count = db.tables.len();
    for table_idx in 0..table_count {
        if db.tables[table_idx].num_rows() < 8 {
            continue;
        }
        let Some((entity, t_vec)) = entity_and_time(&db.tables[table_idx]) else {
            continue;
        };
        let mut rng = substream(seed + 219_937 * (table_idx as u64 + 1), "labels_v2:mident");
        let candidates = db.tables[table_idx].numeric_feature_columns();
        let Some((_col, payload)) = pick_payload(&db.tables[table_idx], &candidates, &mut rng)
        else {
            continue;
        };
        let prior = prior_same_entity_rows(&entity, &t_vec);
        if prior.iter().filter(|p| p.len() >= 2).count() < db.tables[table_idx].num_rows() / 8 {
            continue;
        }
        let threshold = quantile_of(&payload, 0.75);
        let k = [2usize, 3, 4][rng.choose_index(3)];
        for variant in 0..2usize {
            let mechanism = rng.choose_index(4);
            let ambiguous = rng.bernoulli(0.15);
            let latent_raw = if ambiguous {
                let other = (mechanism + 1 + rng.choose_index(3)) % 4;
                let a = mident_latent(mechanism, &prior, &payload, threshold);
                let b = mident_latent(other, &prior, &payload, threshold);
                let salt = seed + 219_937 * (table_idx as u64 + 1) + variant as u64;
                a.iter()
                    .zip(b.iter())
                    .enumerate()
                    .map(|(row, (x, y))| match (x, y) {
                        (Some(x), Some(y)) => {
                            let mut r = child_substream(salt, "labels_v2:mident:mix", row);
                            Some(if r.bernoulli(0.5) { *x } else { *y })
                        }
                        _ => None,
                    })
                    .collect::<Vec<Option<f64>>>()
            } else {
                mident_latent(mechanism, &prior, &payload, threshold)
            };
            let defined: Vec<f64> = latent_raw.iter().filter_map(|v| *v).collect();
            if defined.len() < 4 {
                continue;
            }
            let fill = quantile_of(&defined, 0.5);
            let latent: Vec<f64> = latent_raw.iter().map(|v| v.unwrap_or(fill)).collect();
            let mut sorted = latent.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let cuts: Vec<f64> = (1..k).map(|i| sorted[(sorted.len() * i) / k]).collect();
            let classes: Vec<i64> = latent
                .iter()
                .map(|v| cuts.iter().filter(|c| *v > **c).count() as i64)
                .collect();
            if classes.iter().all(|c| *c == classes[0]) {
                continue;
            }
            // The name carries K and the variant index and nothing about the mechanism.
            let table_name = db.tables[table_idx].name.clone();
            push_i64(
                &mut db.tables[table_idx],
                &format!("mident_k{k}_v{variant}_{table_name}"),
                classes,
            );
        }
    }
}

fn add_atomic_route_labels(db: &mut GeneratedDb, seed: u64) {
    use std::collections::{BTreeMap, BTreeSet};
    let table_count = db.tables.len();
    let mut scans = 0usize;
    for bridge_idx in 0..table_count {
        if scans >= 2 {
            break;
        }
        // A bridge is any table with two foreign keys; its two parents are the route's endpoint
        // tables (when they coincide, the route reaches other rows of one table).
        let fks: Vec<(String, String)> = db.tables[bridge_idx]
            .fkey_col_to_pkey_table
            .iter()
            .map(|(c, p)| (c.clone(), p.clone()))
            .collect();
        if fks.len() < 2 {
            continue;
        }
        let (fk_a, parent_a_name) = fks[0].clone();
        let (fk_b, parent_b_name) = fks[1].clone();
        let Some(a_idx) = db.tables.iter().position(|t| t.name == parent_a_name) else {
            continue;
        };
        let Some(b_idx) = db.tables.iter().position(|t| t.name == parent_b_name) else {
            continue;
        };
        if a_idx == bridge_idx || b_idx == bridge_idx {
            continue;
        }
        let (Some(ColumnData::I64(a_vals)), Some(ColumnData::I64(b_vals))) = (
            db.tables[bridge_idx].column(&fk_a).cloned(),
            db.tables[bridge_idx].column(&fk_b).cloned(),
        ) else {
            continue;
        };
        let n_a = db.tables[a_idx].num_rows();
        let n_b = db.tables[b_idx].num_rows();
        if n_a < 8 || n_b < 4 {
            continue;
        }
        let salt = seed + 233_021 * (bridge_idx as u64 + 1);
        let mut rng = substream(salt, "labels_v2:route");
        let b_candidates = db.tables[b_idx].numeric_feature_columns();
        let Some((b_col, b_payload)) = pick_payload(&db.tables[b_idx], &b_candidates, &mut rng)
        else {
            continue;
        };
        // Distinct co-entities per focal A row: the atomic route with the
        // bridge as relay and endpoints de-duplicated.
        let mut co: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
        for (a, b) in a_vals.iter().zip(b_vals.iter()) {
            if *a >= 0 && (*a as usize) < n_a && *b >= 0 && (*b as usize) < n_b {
                co.entry(*a).or_default().insert(*b);
            }
        }
        let reached = co.values().filter(|s| !s.is_empty()).count();
        if reached < n_a / 8 || reached < 4 {
            continue;
        }
        let a_name = db.tables[a_idx].name.clone();
        let empty = BTreeSet::new();

        let counts: Vec<f64> = (0..n_a)
            .map(|a| co.get(&(a as i64)).unwrap_or(&empty).len() as f64)
            .collect();
        if counts.iter().any(|c| *c != counts[0]) {
            let (y, link) = apply_observation_link(&counts, &mut rng);
            push_f64(
                &mut db.tables[a_idx],
                &format!("routecntreg_{link}_{fk_a}_{a_name}"),
                y,
            );
            if let Some(k2) = median_k2(&counts) {
                push_i64(
                    &mut db.tables[a_idx],
                    &format!("routecnt_k2_{fk_a}_{a_name}"),
                    k2,
                );
            }
        }

        let mean_raw: Vec<Option<f64>> = (0..n_a)
            .map(|a| {
                let set = co.get(&(a as i64)).unwrap_or(&empty);
                if set.is_empty() {
                    return None;
                }
                Some(set.iter().map(|b| b_payload[*b as usize]).sum::<f64>() / set.len() as f64)
            })
            .collect();
        let defined: Vec<f64> = mean_raw.iter().filter_map(|v| *v).collect();
        if defined.len() >= 4 {
            let fill = quantile_of(&defined, 0.5);
            let latent: Vec<f64> = mean_raw.iter().map(|v| v.unwrap_or(fill)).collect();
            if is_constant(&latent) {
                continue;
            }
            let (y, link) = apply_observation_link(&latent, &mut rng);
            push_f64(
                &mut db.tables[a_idx],
                &format!("routemeanreg_{link}_{b_col}_{a_name}"),
                y,
            );
            if let Some(k2) = median_k2(&latent) {
                push_i64(
                    &mut db.tables[a_idx],
                    &format!("routemean_k2_{b_col}_{a_name}"),
                    k2,
                );
            }
        }

        // Query-gated route: the focal row's own feature decides
        // which half of the co-entity payload distribution contributes.
        let a_candidates = db.tables[a_idx].numeric_feature_columns();
        if let Some((_qa_col, qa)) = pick_payload(&db.tables[a_idx], &a_candidates, &mut rng) {
            let qa_median = quantile_of(&qa, 0.5);
            let b_median = quantile_of(&b_payload, 0.5);
            let gated_raw: Vec<Option<f64>> = (0..n_a)
                .map(|a| {
                    let set = co.get(&(a as i64)).unwrap_or(&empty);
                    if set.is_empty() {
                        return None;
                    }
                    let want_high = qa[a] > qa_median;
                    let selected: Vec<f64> = set
                        .iter()
                        .map(|b| b_payload[*b as usize])
                        .filter(|v| (*v > b_median) == want_high)
                        .collect();
                    if selected.is_empty() {
                        // The gated half is empty: fall back to the full route
                        // rather than manufacturing a missing cell.
                        return Some(
                            set.iter().map(|b| b_payload[*b as usize]).sum::<f64>()
                                / set.len() as f64,
                        );
                    }
                    Some(selected.iter().sum::<f64>() / selected.len() as f64)
                })
                .collect();
            let defined: Vec<f64> = gated_raw.iter().filter_map(|v| *v).collect();
            if defined.len() >= 4 {
                let fill = quantile_of(&defined, 0.5);
                let latent: Vec<f64> = gated_raw.iter().map(|v| v.unwrap_or(fill)).collect();
                if is_constant(&latent) {
                    continue;
                }
                let (y, link) = apply_observation_link(&latent, &mut rng);
                push_f64(
                    &mut db.tables[a_idx],
                    &format!("routeqgreg_{link}_{b_col}_{a_name}"),
                    y,
                );
                if let Some(k2) = median_k2(&latent) {
                    push_i64(
                        &mut db.tables[a_idx],
                        &format!("routeqg_k2_{b_col}_{a_name}"),
                        k2,
                    );
                }
            }
        }
        scans += 1;
    }
}

/// A bridge row has exactly two FKs to different tables.
fn add_comembership_peer_label_labels(db: &mut GeneratedDb, seed: u64) {
    use std::collections::{BTreeMap, BTreeSet};

    let table_count = db.tables.len();
    for bridge_idx in 0..table_count {
        let fks: Vec<(String, String)> = db.tables[bridge_idx]
            .fkey_col_to_pkey_table
            .iter()
            .map(|(col, parent)| (col.clone(), parent.clone()))
            .collect();
        if fks.len() != 2 || fks[0].1 == fks[1].1 {
            continue;
        }
        let Some(a_idx) = db.tables.iter().position(|t| t.name == fks[0].1) else {
            continue;
        };
        let Some(b_idx) = db.tables.iter().position(|t| t.name == fks[1].1) else {
            continue;
        };
        if a_idx == bridge_idx || b_idx == bridge_idx || a_idx == b_idx {
            continue;
        }
        let (Some(ColumnData::I64(a_fk)), Some(ColumnData::I64(b_fk))) = (
            db.tables[bridge_idx].column(&fks[0].0).cloned(),
            db.tables[bridge_idx].column(&fks[1].0).cloned(),
        ) else {
            continue;
        };

        // Both endpoint orientations are useful co-membership views.  The
        // bridge itself is never treated as a member table, and a direct FK
        // between endpoint tables is irrelevant to this construction.
        for (orientation, (member_idx, hub_idx, member_fk, hub_fk)) in
            [(a_idx, b_idx, &a_fk, &b_fk), (b_idx, a_idx, &b_fk, &a_fk)]
                .into_iter()
                .enumerate()
        {
            let n_member = db.tables[member_idx].num_rows();
            let n_hub = db.tables[hub_idx].num_rows();
            if n_member < 8 || n_hub < 2 || member_fk.len() != hub_fk.len() {
                continue;
            }
            let Some(member_time_ns) = timestamp_column(&db.tables[member_idx]) else {
                continue;
            };
            if member_time_ns.len() != n_member {
                continue;
            }
            let member_time: Vec<f64> = member_time_ns.iter().map(|t| *t as f64).collect();
            let mut rng = substream(
                seed.wrapping_add(211_337 * (bridge_idx as u64 + 1))
                    .wrapping_add(orientation as u64),
                "labels_v2:comembership_peer_label",
            );
            let candidates = base_numeric_columns(&db.tables[member_idx]);
            let Some((payload_col, payload)) =
                pick_payload(&db.tables[member_idx], &candidates, &mut rng)
            else {
                continue;
            };

            let mut hub_members: Vec<BTreeSet<usize>> =
                (0..n_hub).map(|_| BTreeSet::new()).collect();
            for (&member, &hub) in member_fk.iter().zip(hub_fk) {
                if member >= 0 && (member as usize) < n_member && hub >= 0 && (hub as usize) < n_hub
                {
                    hub_members[hub as usize].insert(member as usize);
                }
            }
            let degrees: Vec<usize> = hub_members
                .iter()
                .map(BTreeSet::len)
                .filter(|degree| *degree >= 2)
                .collect();
            if degrees.is_empty() {
                continue;
            }

            // Draw a hub-rarity regime per bridge orientation.  The all-hubs,
            // upper-quartile and median-degree regimes expose different
            // topology scales without changing the column-name contract.
            let mut sorted_degrees = degrees;
            sorted_degrees.sort_unstable();
            let rarity = rng.choose_index(3);
            let max_degree = match rarity {
                0 => usize::MAX,
                1 => sorted_degrees[((sorted_degrees.len() - 1) * 3) / 4],
                _ => sorted_degrees[(sorted_degrees.len() - 1) / 2],
            }
            .max(2);

            let mut peer_weights: Vec<BTreeMap<usize, usize>> =
                (0..n_member).map(|_| BTreeMap::new()).collect();
            for members in hub_members
                .iter()
                .filter(|members| members.len() >= 2 && members.len() <= max_degree)
            {
                for &member in members {
                    for &peer in members {
                        if member != peer {
                            *peer_weights[member].entry(peer).or_default() += 1;
                        }
                    }
                }
            }
            let with_peer = peer_weights
                .iter()
                .filter(|peers| !peers.is_empty())
                .count();
            if with_peer < 4 || with_peer < n_member / 8 {
                continue;
            }

            let aggregate = rng.choose_index(3); // mean / max / count-weighted mean
            let median = quantile_of(&payload, 0.5);
            let mut order: Vec<usize> = (0..n_member).collect();
            order.sort_by(|a, b| {
                member_time[*a]
                    .total_cmp(&member_time[*b])
                    .then_with(|| a.cmp(b))
            });

            let noise = rng.f64_range(0.02, 0.15);
            let payload_spread =
                (quantile_of(&payload, 0.75) - quantile_of(&payload, 0.25)).max(1e-9);
            let salt = seed
                .wrapping_add(211_337 * (bridge_idx as u64 + 1))
                .wrapping_add(orientation as u64);
            let mut emitted = vec![median; n_member];
            let mut latent = vec![median; n_member];
            for &row in &order {
                let eligible: Vec<(f64, usize)> = peer_weights[row]
                    .iter()
                    .filter(|(peer, _)| member_time[**peer] < member_time[row])
                    .map(|(peer, weight)| (emitted[*peer], *weight))
                    .collect();
                latent[row] = if eligible.is_empty() {
                    median
                } else {
                    match aggregate {
                        0 => {
                            eligible.iter().map(|(value, _)| *value).sum::<f64>()
                                / eligible.len() as f64
                        }
                        1 => eligible
                            .iter()
                            .map(|(value, _)| *value)
                            .fold(f64::NEG_INFINITY, f64::max),
                        _ => {
                            let weight_sum: usize =
                                eligible.iter().map(|(_, weight)| *weight).sum();
                            eligible
                                .iter()
                                .map(|(value, weight)| *value * *weight as f64)
                                .sum::<f64>()
                                / weight_sum as f64
                        }
                    }
                };
                let mut row_rng =
                    child_substream(salt, "labels_v2:comembership_peer_label:forward", row);
                emitted[row] = latent[row] + noise * payload_spread * row_rng.standard_normal();
            }
            if is_constant(&latent) || !all_finite(&latent) {
                continue;
            }
            let noised = iqr_noise(
                &emitted,
                noise,
                salt,
                "labels_v2:comembership_peer_label:row",
            );
            if is_constant(&noised) || !all_finite(&noised) {
                continue;
            }
            let (y, link) = apply_observation_link(&noised, &mut rng);
            if !all_finite(&y) {
                continue;
            }
            let table_name = db.tables[member_idx].name.clone();
            push_f64(
                &mut db.tables[member_idx],
                &format!("comembpeer_r_{link}_{payload_col}_{table_name}"),
                y,
            );
            if let Some(k2) = median_k2(&latent) {
                push_i64(
                    &mut db.tables[member_idx],
                    &format!("comembpeer_k2_{payload_col}_{table_name}"),
                    k2,
                );
            }
        }
    }
}

/// Random 1-hidden-layer network with drawn width, signs and threshold:
/// the task-level function g shared by every entity of a table.
struct DrawnFunction {
    w1: Vec<Vec<f64>>, // hidden x input
    b1: Vec<f64>,
    w2: Vec<f64>,
}

impl DrawnFunction {
    fn draw(n_in: usize, rng: &mut crate::rng::GenRng) -> Self {
        let width = rng.usize_inclusive(2, 6);
        let w1 = (0..width)
            .map(|_| (0..n_in).map(|_| rng.standard_normal()).collect())
            .collect();
        let b1 = (0..width).map(|_| rng.standard_normal() * 0.5).collect();
        let w2 = (0..width).map(|_| rng.standard_normal()).collect();
        Self { w1, b1, w2 }
    }

    fn eval(&self, x: &[f64]) -> f64 {
        self.w1
            .iter()
            .zip(&self.b1)
            .zip(&self.w2)
            .map(|((row, b), w2)| {
                let pre: f64 = row.iter().zip(x).map(|(w, v)| w * v).sum::<f64>() + b;
                w2 * pre.tanh()
            })
            .sum()
    }
}

fn f64_column(table: &TableData, name: &str) -> Option<Vec<f64>> {
    let filled: Vec<f64> = match table.column(name)? {
        ColumnData::F64(values) => values.iter().map(|v| v.unwrap_or(0.0)).collect(),
        ColumnData::I64(values) => values.iter().map(|v| *v as f64).collect(),
        _ => return None,
    };
    // Usable as a standardized input only if every cell is finite, the column
    // varies, and its scale cannot overflow the variance (epoch-like integers
    // are the case that bites).
    let usable = filled.iter().all(|v| v.is_finite() && v.abs() < 1.0e9) && !is_constant(&filled);
    usable.then_some(filled)
}

fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite())
}

/// Base (pre-target) numeric columns of a table, in publication order: the
/// generator's own feature columns, never another family's target.
fn base_numeric_columns(table: &TableData) -> Vec<String> {
    table
        .column_names()
        .filter(|name| {
            *name != table.pkey_col
                && !table.fkey_col_to_pkey_table.contains_key(*name)
                && !crate::scm::PUBLISHED_TARGET_PREFIXES
                    .iter()
                    .any(|prefix| name.starts_with(prefix))
                && f64_column(table, name).is_some()
        })
        .map(str::to_owned)
        .collect()
}

/// Bag statistics (mean, log-count, max) of one child column per parent row.
fn child_bag_stats(child_fk: &[i64], payload: &[f64], n_parent: usize) -> Vec<[f64; 3]> {
    let mut sum = vec![0.0; n_parent];
    let mut count = vec![0usize; n_parent];
    let mut max = vec![f64::NEG_INFINITY; n_parent];
    for (row, &fk) in child_fk.iter().enumerate() {
        if fk >= 0 && (fk as usize) < n_parent {
            let p = fk as usize;
            sum[p] += payload[row];
            count[p] += 1;
            if payload[row] > max[p] {
                max[p] = payload[row];
            }
        }
    }
    (0..n_parent)
        .map(|p| {
            if count[p] == 0 {
                [0.0, 0.0, 0.0]
            } else {
                [sum[p] / count[p] as f64, (count[p] as f64).ln_1p(), max[p]]
            }
        })
        .collect()
}

fn add_cross_entity_functional_labels(db: &mut GeneratedDb, seed: u64) {
    let table_count = db.tables.len();
    let mut scans = 0usize;
    for table_idx in 0..table_count {
        if scans >= 2 {
            break;
        }
        let Some((fk_col, parent_name)) = db.tables[table_idx]
            .fkey_col_to_pkey_table
            .iter()
            .next()
            .map(|(c, p)| (c.clone(), p.clone()))
        else {
            continue;
        };
        let Some(parent_idx) = db.tables.iter().position(|t| t.name == parent_name) else {
            continue;
        };
        if parent_idx == table_idx {
            continue;
        }
        let Some(ColumnData::I64(child_fk)) = db.tables[table_idx].column(&fk_col).cloned() else {
            continue;
        };
        let n_parent = db.tables[parent_idx].num_rows();
        // A cross-entity function needs enough labelled entities to be learnt
        // from in context: this is the episode floor, not a gate on the DB.
        if n_parent < 64 {
            continue;
        }
        let salt = seed + 271_829 * (table_idx as u64 + 1);
        let mut rng = substream(salt, "labels_v2:xfun");
        let parent_cols = base_numeric_columns(&db.tables[parent_idx]);
        let child_cols = base_numeric_columns(&db.tables[table_idx]);
        if parent_cols.is_empty() && child_cols.is_empty() {
            continue;
        }
        // Inputs: up to 3 drawn own-row columns of the entity + bag stats of
        // one drawn child column. Each standardized so g sees unit scales.
        let n_own = parent_cols.len().min(rng.usize_inclusive(1, 3));
        let mut own_idx: Vec<usize> = (0..parent_cols.len()).collect();
        rng.shuffle(&mut own_idx);
        let mut inputs: Vec<Vec<f64>> = Vec::new();
        for &i in own_idx.iter().take(n_own) {
            let mut col =
                f64_column(&db.tables[parent_idx], &parent_cols[i]).expect("filtered above");
            standardize(&mut col);
            if all_finite(&col) {
                inputs.push(col);
            }
        }
        if !child_cols.is_empty() {
            let c = rng.choose_index(child_cols.len());
            let payload =
                f64_column(&db.tables[table_idx], &child_cols[c]).expect("filtered above");
            let stats = child_bag_stats(&child_fk, &payload, n_parent);
            for k in 0..3 {
                let mut col: Vec<f64> = stats.iter().map(|s| s[k]).collect();
                if is_constant(&col) {
                    continue;
                }
                standardize(&mut col);
                if all_finite(&col) {
                    inputs.push(col);
                }
            }
        }
        if inputs.is_empty() {
            continue;
        }
        let g = DrawnFunction::draw(inputs.len(), &mut rng);
        let noise = rng.f64_range(0.05_f64.ln(), 0.5_f64.ln()).exp();
        let latent: Vec<f64> = (0..n_parent)
            .map(|p| {
                let x: Vec<f64> = inputs.iter().map(|col| col[p]).collect();
                let mut r = child_substream(salt, "labels_v2:xfun:noise", p);
                g.eval(&x) + noise * r.standard_normal()
            })
            .collect();
        if is_constant(&latent) || !all_finite(&latent) {
            continue;
        }
        let parent_table_name = db.tables[parent_idx].name.clone();
        let (y, link) = apply_observation_link(&latent, &mut rng);
        if !all_finite(&y) {
            continue;
        }
        push_f64(
            &mut db.tables[parent_idx],
            &format!("xfunreg_{link}_{fk_col}_{parent_table_name}"),
            y,
        );
        if let Some(k2) = median_k2(&latent) {
            push_i64(
                &mut db.tables[parent_idx],
                &format!("xfun_k2_{fk_col}_{parent_table_name}"),
                k2,
            );
        }
        scans += 1;
    }
}

/// The drawn link h from entity latent to label; never the identity.
fn drawn_state_link(rng: &mut crate::rng::GenRng) -> (Box<dyn Fn(f64) -> f64>, &'static str) {
    match rng.choose_index(3) {
        0 => {
            // Saturating monotone at a drawn scale and shift.
            let scale = rng.f64_range(0.5, 3.0);
            let shift = rng.f64_range(-1.0, 1.0);
            (Box::new(move |t| ((t - shift) * scale).tanh()), "sat")
        }
        1 => {
            // Non-monotone bump: the label is high near a drawn centre.
            let centre = rng.f64_range(-1.0, 1.0);
            let width = rng.f64_range(0.3, 1.5);
            (
                Box::new(move |t| (-((t - centre) / width).powi(2)).exp()),
                "bump",
            )
        }
        _ => {
            // Band: 1 inside a drawn interval, 0 outside (smoothed edges).
            let lo = rng.f64_range(-1.5, 0.0);
            let hi = lo + rng.f64_range(0.5, 2.0);
            let edge = rng.f64_range(0.05, 0.3);
            (
                Box::new(move |t| {
                    let a = ((t - lo) / edge).tanh();
                    let b = ((hi - t) / edge).tanh();
                    0.5 * (a + b)
                }),
                "band",
            )
        }
    }
}

fn add_latent_state_link_labels(db: &mut GeneratedDb, seed: u64) {
    let table_count = db.tables.len();
    let mut scans = 0usize;
    for table_idx in 0..table_count {
        if scans >= 2 {
            break;
        }
        let Some((fk_col, parent_name)) = db.tables[table_idx]
            .fkey_col_to_pkey_table
            .iter()
            .next()
            .map(|(c, p)| (c.clone(), p.clone()))
        else {
            continue;
        };
        let Some(parent_idx) = db.tables.iter().position(|t| t.name == parent_name) else {
            continue;
        };
        if parent_idx == table_idx {
            continue;
        }
        let Some(ColumnData::I64(child_fk)) = db.tables[table_idx].column(&fk_col).cloned() else {
            continue;
        };
        let n_parent = db.tables[parent_idx].num_rows();
        let n_child = child_fk.len();
        if n_parent < 64 || n_child < 2 * n_parent {
            continue;
        }
        let salt = seed + 314_159 * (table_idx as u64 + 1);
        let mut rng = substream(salt, "labels_v2:xstate");
        let sigma = rng.f64_range(0.2_f64.ln(), 1.5_f64.ln()).exp();
        let theta: Vec<f64> = (0..n_parent)
            .map(|p| child_substream(salt, "labels_v2:xstate:theta", p).standard_normal())
            .collect();
        let payload: Vec<f64> = child_fk
            .iter()
            .enumerate()
            .map(|(row, &fk)| {
                let mut r = child_substream(salt, "labels_v2:xstate:event", row);
                let mean = if fk >= 0 && (fk as usize) < n_parent {
                    theta[fk as usize]
                } else {
                    0.0
                };
                mean + sigma * r.standard_normal()
            })
            .collect();
        let (link, link_tag) = drawn_state_link(&mut rng);
        let noise = rng.f64_range(0.02_f64.ln(), 0.3_f64.ln()).exp();
        let latent: Vec<f64> = theta
            .iter()
            .enumerate()
            .map(|(p, t)| {
                let mut r = child_substream(salt, "labels_v2:xstate:noise", p);
                link(*t) + noise * r.standard_normal()
            })
            .collect();
        if is_constant(&latent) || !all_finite(&latent) {
            continue;
        }
        let mut payload = payload;
        contaminate_payload(&mut payload, salt, &mut rng);
        let missing = mnar_mask(&payload, salt, &mut rng);
        let cells: Vec<Option<f64>> = payload
            .iter()
            .enumerate()
            .map(|(row, v)| match &missing {
                Some(mask) if mask[row] => None,
                _ => Some(*v),
            })
            .collect();
        let child_name = db.tables[table_idx].name.clone();
        let parent_table_name = db.tables[parent_idx].name.clone();
        push_f64_opt(
            &mut db.tables[table_idx],
            &format!("evs_{child_name}"),
            cells,
        );
        let (y, obs_link) = apply_observation_link(&latent, &mut rng);
        push_f64(
            &mut db.tables[parent_idx],
            &format!("xstatereg_{link_tag}_{obs_link}_{fk_col}_{parent_table_name}"),
            y,
        );
        if let Some(k2) = median_k2(&latent) {
            push_i64(
                &mut db.tables[parent_idx],
                &format!("xstate_k2_{link_tag}_{fk_col}_{parent_table_name}"),
                k2,
            );
        }
        scans += 1;
    }
}

/// One `xagg_*` column: what it is (table/column/fk/child_column/stat/window) and how to
/// standardize it (mean/std of the raw values this database published).
#[derive(Debug, Clone, Serialize)]
pub struct ExactAggregateColumn {
    /// The PARENT table this column lives on.
    pub table: String,
    /// The published column name (`xagg_{fk}_{child_column}_{stat}_{window}`,
    /// or `xagg_{fk}_count_{window}` for `stat == "count"`).
    pub column: String,
    /// The child table's FK column that defines the group.
    pub fk: String,
    pub child_column: String,
    /// One of `count`, `sum`, `mean`, `max`.
    pub stat: String,
    /// One of `7d`, `30d`, `all`.
    pub window: String,
    /// Population mean of the raw published values, over this database's
    /// parent rows.
    pub mean: f64,
    /// Population std of the raw published values; `0.0` on a degenerate column.
    pub std: f64,
}

/// Cap on `xagg_*` columns PER PARENT TABLE.
pub const XAGG_MAX_COLUMNS_PER_TABLE: usize = 24;

const XAGG_NS_PER_DAY: i64 = 86_400_000_000_000;

/// A table's own `time_col`, as raw nanoseconds, when it has one and the
/// column is physically a timestamp.
fn timestamp_column(table: &TableData) -> Option<Vec<i64>> {
    let name = table.time_col.as_ref()?;
    match table.column(name)? {
        ColumnData::TimestampNs(values) => Some(values.clone()),
        _ => None,
    }
}

struct XaggCandidate {
    child_idx: usize,
    fk_col: String,
    /// `None` for the `count` stat, which does not read a payload column.
    col: Option<String>,
    stat: &'static str,
    window: &'static str,
}

/// Exact `(count, sum, mean, max)` of one child column per parent row, over child rows admitted
/// under a lookback window ending at the PARENT row's own timestamp.
fn child_bag_stats_windowed(
    child_fk: &[i64],
    payload: &[f64],
    child_time: Option<&[i64]>,
    parent_time: Option<&[i64]>,
    window_ns: Option<i64>,
    n_parent: usize,
) -> Vec<[f64; 4]> {
    let mut count = vec![0.0_f64; n_parent];
    let mut sum = vec![0.0_f64; n_parent];
    let mut max = vec![f64::NEG_INFINITY; n_parent];
    for (row, &fk) in child_fk.iter().enumerate() {
        if fk < 0 || (fk as usize) >= n_parent {
            continue;
        }
        let p = fk as usize;
        if let Some(w) = window_ns {
            let (Some(child_time), Some(parent_time)) = (child_time, parent_time) else {
                continue;
            };
            let c = child_time[row];
            let t = parent_time[p];
            if c > t || t - c > w {
                continue;
            }
        }
        count[p] += 1.0;
        sum[p] += payload[row];
        if payload[row] > max[p] {
            max[p] = payload[row];
        }
    }
    (0..n_parent)
        .map(|p| {
            if count[p] == 0.0 {
                [0.0, 0.0, 0.0, 0.0]
            } else {
                [count[p], sum[p], sum[p] / count[p], max[p]]
            }
        })
        .collect()
}

/// Exact population aggregates of a child table's own numeric columns, computed over rows
/// admitted under a window and pushed onto the PARENT row.
pub fn add_exact_aggregate_sidecars(db: &mut GeneratedDb, seed: u64) {
    let table_count = db.tables.len();
    for parent_idx in 0..table_count {
        let parent_name = db.tables[parent_idx].name.clone();
        let n_parent = db.tables[parent_idx].num_rows();
        if n_parent == 0 {
            continue;
        }
        let parent_time = timestamp_column(&db.tables[parent_idx]);

        let mut children: Vec<(usize, String)> = Vec::new();
        for child_idx in 0..table_count {
            if child_idx == parent_idx {
                continue;
            }
            let mut fk_cols: Vec<String> = db.tables[child_idx]
                .fkey_col_to_pkey_table
                .iter()
                .filter(|(_, p)| **p == parent_name)
                .map(|(c, _)| c.clone())
                .collect();
            fk_cols.sort();
            for fk_col in fk_cols {
                children.push((child_idx, fk_col));
            }
        }
        if children.is_empty() {
            continue;
        }

        let mut candidates: Vec<XaggCandidate> = Vec::new();
        for (child_idx, fk_col) in &children {
            let child_time = timestamp_column(&db.tables[*child_idx]);
            let windows: &[&str] = if parent_time.is_some() && child_time.is_some() {
                &["7d", "30d", "all"]
            } else {
                &["all"]
            };
            for &window in windows {
                candidates.push(XaggCandidate {
                    child_idx: *child_idx,
                    fk_col: fk_col.clone(),
                    col: None,
                    stat: "count",
                    window,
                });
            }
            for col in base_numeric_columns(&db.tables[*child_idx]) {
                for &window in windows {
                    for stat in ["sum", "mean", "max"] {
                        candidates.push(XaggCandidate {
                            child_idx: *child_idx,
                            fk_col: fk_col.clone(),
                            col: Some(col.clone()),
                            stat,
                            window,
                        });
                    }
                }
            }
        }
        if candidates.is_empty() {
            continue;
        }

        let mut selected: Vec<usize> = (0..candidates.len()).collect();
        if candidates.len() > XAGG_MAX_COLUMNS_PER_TABLE {
            let mut rng = substream(
                seed + 197_611 * (parent_idx as u64 + 1),
                "labels_v2:xagg:select",
            );
            rng.shuffle(&mut selected);
            selected.truncate(XAGG_MAX_COLUMNS_PER_TABLE);
            // Canonical emission order, independent of the shuffle: only
            // WHICH candidates survive the cap depends on the seed, not the
            // order they are pushed in.
            selected.sort_unstable();
        }

        for candidate_idx in selected {
            let candidate = &candidates[candidate_idx];
            let child_time = timestamp_column(&db.tables[candidate.child_idx]);
            let Some(ColumnData::I64(child_fk)) = db.tables[candidate.child_idx]
                .column(&candidate.fk_col)
                .cloned()
            else {
                continue;
            };
            let window_ns = match candidate.window {
                "7d" => Some(7 * XAGG_NS_PER_DAY),
                "30d" => Some(30 * XAGG_NS_PER_DAY),
                _ => None,
            };
            let payload = match &candidate.col {
                Some(col) => match f64_column(&db.tables[candidate.child_idx], col) {
                    Some(values) => values,
                    None => continue,
                },
                None => vec![0.0; child_fk.len()],
            };
            let stats = child_bag_stats_windowed(
                &child_fk,
                &payload,
                child_time.as_deref(),
                parent_time.as_deref(),
                window_ns,
                n_parent,
            );
            let values: Vec<f64> = match candidate.stat {
                "count" => stats.iter().map(|s| s[0]).collect(),
                "sum" => stats.iter().map(|s| s[1]).collect(),
                "mean" => stats.iter().map(|s| s[2]).collect(),
                "max" => stats.iter().map(|s| s[3]).collect(),
                _ => unreachable!("stat is drawn from a fixed literal set above"),
            };
            if !all_finite(&values) {
                continue;
            }
            let mean = values.iter().sum::<f64>() / n_parent as f64;
            let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n_parent as f64;
            let std = var.sqrt();

            let base = match &candidate.col {
                Some(col) => format!(
                    "xagg_{}_{col}_{}_{}",
                    candidate.fk_col, candidate.stat, candidate.window
                ),
                None => format!("xagg_{}_count_{}", candidate.fk_col, candidate.window),
            };
            let column = push_f64(&mut db.tables[parent_idx], &base, values);
            db.exact_aggregates.push(ExactAggregateColumn {
                table: parent_name.clone(),
                column,
                fk: candidate.fk_col.clone(),
                child_column: candidate.col.clone().unwrap_or_default(),
                stat: candidate.stat.to_string(),
                window: candidate.window.to_string(),
                mean,
                std,
            });
        }
    }
}

/// Additive relation-null episode columns.
pub fn add_relation_null_labels(db: &mut GeneratedDb, seed: u64) {
    for table in &mut db.tables {
        let n = table.num_rows();
        let mut n_rng = substream(seed, &format!("relnull:neighbour:{}", table.name));
        // At two hops the model may encounter another row of this same table.
        // Give each row a distinct named stream too: its own X is independent
        // of every other row's X, as well as of all evx/readout streams.
        let mut features: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::with_capacity(n));
        for row in 0..n {
            let mut x_rng = child_substream(seed, &format!("relnull:focal:{}", table.name), row);
            for values in &mut features {
                values.push(x_rng.standard_normal());
            }
        }
        for (j, values) in features.into_iter().enumerate() {
            push_f64(table, &format!("relnull_x_{j}"), values);
            push_f64(
                table,
                &format!("relnull_evx_{j}"),
                (0..n).map(|_| n_rng.standard_normal()).collect(),
            );
        }
        add_relation_null_readouts(table, seed);
    }
}

/// Recompute readouts from the focal columns alone.
pub fn add_relation_null_readouts(table: &mut TableData, seed: u64) {
    if table.num_rows() < 8 {
        return;
    }
    let names: Vec<String> = (0..4).map(|j| format!("relnull_x_{j}")).collect();
    let mut y_rng = substream(seed, &format!("relnull:readout:{}", table.name));
    let (_, y, _) = crate::labels::own_feature_latent(table, &names, &mut y_rng, true)
        .expect("fresh Gaussian columns have rows");
    let classes = crate::labels::cut_feature_classes(&y, &[0.0]);
    // Population cut (roughly 5--20% positives), never a sample quantile.
    let cut = y_rng.f64_range(0.85, 1.65);
    let imbalanced = crate::labels::cut_feature_classes(&y, &[cut]);
    push_f64(table, "relnull_r", y);
    push_i64(table, "relnull_k2", classes);
    push_i64(table, "relnull_imb", imbalanced);
}

// DFS-prior: explicit, bounded aggregates of the existing FK graph.
// Relation keys are (child table name, FK column name), sorted lexically.
fn aggcol_relations(db: &GeneratedDb, parent: usize) -> Vec<(usize, String)> {
    let mut relations = Vec::new();
    for (child, table) in db.tables.iter().enumerate() {
        for (fk, target) in &table.fkey_col_to_pkey_table {
            if target == &db.tables[parent].name {
                relations.push((child, fk.clone()));
            }
        }
    }
    relations.sort_by(|a, b| (&db.tables[a.0].name, &a.1).cmp(&(&db.tables[b.0].name, &b.1)));
    relations
}

/// Resolve actual PK values (not row offsets), then reuse the recent-pointer
/// machinery on a joined parent/child panel. Integer timestamp ranks preserve
/// nanosecond ties even at epoch magnitudes that cannot be represented in f64.
fn aggcol_groups(parent: &TableData, child: &TableData, fk: &str) -> Vec<Vec<usize>> {
    let n = parent.num_rows();
    let Some(ColumnData::I64(keys)) = parent.column(&parent.pkey_col) else {
        return vec![vec![]; n];
    };
    let Some(ColumnData::I64(foreign)) = child.column(fk) else {
        return vec![vec![]; n];
    };
    let lookup: std::collections::BTreeMap<i64, usize> = keys
        .iter()
        .copied()
        .enumerate()
        .map(|(row, key)| (key, row))
        .collect();
    let destinations: Vec<Option<usize>> =
        foreign.iter().map(|key| lookup.get(key).copied()).collect();
    aggcol_admissible(parent, child, &destinations)
}

fn aggcol_admissible(
    parent: &TableData,
    child: &TableData,
    destinations: &[Option<usize>],
) -> Vec<Vec<usize>> {
    let n = parent.num_rows();
    if let (Some(pt), Some(ct)) = (timestamp_column(parent), timestamp_column(child)) {
        let mut clocks = pt.clone();
        clocks.extend(&ct);
        let mut unique = clocks.clone();
        unique.sort_unstable();
        unique.dedup();
        let times: Vec<f64> = clocks
            .iter()
            .map(|t| unique.binary_search(t).unwrap() as f64)
            .collect();
        let mut entities: Vec<i64> = (0..n as i64).collect();
        // Invalid pointers get distinct groups, never a focal row's group.
        entities.extend(
            destinations
                .iter()
                .enumerate()
                .map(|(r, p)| p.map(|p| p as i64).unwrap_or((n + r) as i64)),
        );
        let prior = prior_same_entity_rows(&entities, &times);
        return prior[..n]
            .iter()
            .map(|rows| rows.iter().filter_map(|&r| r.checked_sub(n)).collect())
            .collect();
    }
    let mut groups = vec![vec![]; n];
    for (row, dest) in destinations.iter().enumerate() {
        if let Some(p) = dest {
            groups[*p].push(row);
        }
    }
    groups
}

/// Deterministic numeric payload selection, excluding clocks, keys, entity IDs
/// and all published families. Missing cells use the pointer's median fill.
fn aggcol_payload(table: &TableData) -> Option<Vec<f64>> {
    let mut candidates = base_numeric_columns(table);
    candidates.retain(|name| name != ENTITY_ID_COL && Some(name) != table.time_col.as_ref());
    candidates.sort();
    for name in candidates {
        let values = table.column(&name)?.as_numeric_lossy()?;
        let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        if finite.is_empty() {
            continue;
        }
        let median = quantile_of(&finite, 0.5);
        return Some(
            values
                .iter()
                .map(|v| if v.is_finite() { *v } else { median })
                .collect(),
        );
    }
    None
}

fn aggcol_bag_columns(
    groups: &[Vec<usize>],
    child: &TableData,
    suffix: &str,
    depth: &str,
) -> Vec<(String, Vec<f64>)> {
    let mut columns = vec![(
        format!("aggcol_cnt{depth}_{suffix}"),
        groups.iter().map(|g| g.len() as f64).collect(),
    )];
    if let Some(payload) = aggcol_payload(child) {
        let fallback = quantile_of(&payload, 0.5);
        columns.push((
            format!("aggcol_mean{depth}_{suffix}"),
            groups
                .iter()
                .map(|g| {
                    if g.is_empty() {
                        fallback
                    } else {
                        g.iter().map(|&r| payload[r]).sum::<f64>() / g.len() as f64
                    }
                })
                .collect(),
        ));
    }
    columns
}

/// Append one focal DFS-prior episode; the focal table has the most incoming
/// child tables, first table on ties.
pub fn add_aggcol_labels(db: &mut GeneratedDb, seed: u64) {
    let Some(focal) = (0..db.tables.len()).max_by_key(|&p| {
        (
            db.tables
                .iter()
                .filter(|t| {
                    t.fkey_col_to_pkey_table
                        .values()
                        .any(|v| v == &db.tables[p].name)
                })
                .count(),
            std::cmp::Reverse(p),
        )
    }) else {
        return;
    };
    let n = db.tables[focal].num_rows();
    if n < 8 {
        return;
    }
    let relations = aggcol_relations(db, focal);
    if relations.is_empty() {
        return;
    }
    let mut one = Vec::new();
    let mut two = Vec::new();
    for (k, (c, fk)) in relations.iter().enumerate() {
        let parent = &db.tables[focal];
        let child = &db.tables[*c];
        let groups = aggcol_groups(parent, child, fk);
        one.extend(aggcol_bag_columns(&groups, child, &k.to_string(), ""));
        if let (Some(pt), Some(ct)) = (timestamp_column(parent), timestamp_column(child)) {
            let gaps: Vec<Option<f64>> = groups
                .iter()
                .enumerate()
                .map(|(p, g)| {
                    g.iter()
                        .map(|&r| ct[r])
                        .max()
                        .map(|t| (pt[p] as i128 - t as i128) as f64)
                })
                .collect();
            let present: Vec<f64> = gaps.iter().flatten().copied().collect();
            let fallback = quantile_of(&present, 0.5);
            one.push((
                format!("aggcol_rec_{k}"),
                gaps.iter().map(|v| v.unwrap_or(fallback)).collect(),
            ));
        }
        if !two.is_empty() || *c == focal {
            continue;
        }
        for (g, gfk) in aggcol_relations(db, *c) {
            if g == focal || g == *c {
                continue;
            }
            let grandchild = &db.tables[g];
            let inner = aggcol_groups(child, grandchild, &gfk);
            let mut descendants = vec![vec![]; n];
            for (p, rows) in groups.iter().enumerate() {
                for &r in rows {
                    descendants[p].extend(&inner[r]);
                }
                descendants[p].sort_unstable();
                descendants[p].dedup();
            }
            // Also require descendants to precede the focal clock when the
            // intermediate table has no clock.
            let mut destinations = vec![None; grandchild.num_rows()];
            for (p, rows) in descendants.iter().enumerate() {
                for &r in rows {
                    destinations[r] = Some(p);
                }
            }
            descendants = aggcol_admissible(parent, grandchild, &destinations);
            two = aggcol_bag_columns(&descendants, grandchild, &k.to_string(), "2");
            break;
        }
    }
    one.truncate(6 - two.len());
    one.extend(two);
    let table = &mut db.tables[focal];
    let mut inputs = Vec::new();
    for (name, values) in one {
        push_f64(table, &name, values.clone());
        let mut standardized = values;
        if is_constant(&standardized) {
            standardized.fill(0.0);
        } else {
            standardize(&mut standardized);
        }
        inputs.push(standardized);
    }
    let aggregate_width = inputs.len();
    let mut own: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::with_capacity(n));
    for row in 0..n {
        let mut rng = child_substream(seed, &format!("aggcol:focal:{}", table.name), row);
        for col in &mut own {
            col.push(rng.standard_normal());
        }
    }
    for (j, values) in own.into_iter().enumerate() {
        push_f64(table, &format!("aggcol_x_{j}"), values.clone());
        inputs.push(values);
    }
    let mut weights_rng = substream(seed, &format!("aggcol:weights:{}", table.name));
    let mut weights: Vec<f64> = inputs
        .iter()
        .map(|_| weights_rng.f64_range(-1.5, 1.5))
        .collect();
    if weights[..aggregate_width].iter().all(|w| *w == 0.0) {
        weights[0] = 1.0;
    }
    // Population variance: retain aggregate covariance over the finite focal
    // population and the analytic variance of the independent N(0,1) own X.
    // Summing squared aggregate weights would incorrectly assume independence.
    let aggregate: Vec<f64> = (0..n)
        .map(|r| {
            (0..aggregate_width)
                .map(|j| weights[j] * inputs[j][r])
                .sum()
        })
        .collect();
    let mean = aggregate.iter().sum::<f64>() / n as f64;
    let variance = aggregate.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64
        + weights[aggregate_width..]
            .iter()
            .map(|w| w * w)
            .sum::<f64>();
    let sd = variance.sqrt().max(1e-12);
    let mut readout = substream(seed, &format!("aggcol:readout:{}", table.name));
    let noise = readout.f64_range(0.02, 0.25);
    let y: Vec<f64> = (0..n)
        .map(|r| {
            inputs
                .iter()
                .zip(&weights)
                .map(|(x, w)| w * x[r])
                .sum::<f64>()
                / sd
                + noise * readout.standard_normal()
        })
        .collect();
    let k2 = crate::labels::cut_feature_classes(&y, &[0.0]);
    let cut = readout.f64_range(0.85, 1.65);
    let imb = crate::labels::cut_feature_classes(&y, &[cut]);
    push_f64(table, "aggcol_r", y);
    push_i64(table, "aggcol_k2", k2);
    push_i64(table, "aggcol_imb", imb);
}

/// The three aggcol readouts, in publication order, with their task shape.
pub const AGGCOL_TASKS: [(&str, &str, usize); 3] = [
    ("aggcol_r", "regression", 0),
    ("aggcol_k2", "clf", 2),
    ("aggcol_imb", "clf", 2),
];

/// Declare the aggcol readouts of an already generated database as tasks.
pub fn declare_aggcol_tasks(db: &mut GeneratedDb) -> Result<usize> {
    let is_target = |name: &str| AGGCOL_TASKS.iter().any(|(stem, _, _)| *stem == name);
    let Some(focal) = db
        .tables
        .iter()
        .position(|t| t.column("aggcol_r").is_some())
    else {
        return Ok(0);
    };
    let mut hidden: Vec<(String, String)> = Vec::new();
    for table in &db.tables {
        for (name, _) in &table.columns {
            let predictor = name.starts_with("aggcol_") && !is_target(name);
            if FeatureContract::is_published_target(name) && !predictor {
                hidden.push((table.name.clone(), name.clone()));
            }
        }
    }
    let table_name = db.tables[focal].name.clone();
    let num_rows = db.tables[focal].num_rows();
    let mut declared = 0;
    for (stem, task_type, n_classes) in AGGCOL_TASKS {
        anyhow::ensure!(
            db.tables[focal].column(stem).is_some(),
            "aggcol focal table {table_name} lacks {stem}"
        );
        let drops: Vec<serde_json::Value> = hidden
            .iter()
            .filter(|(t, c)| !(t == &table_name && c == stem))
            .map(|(t, c)| serde_json::json!([t, c]))
            .collect();
        let mut public_task = serde_json::json!({"drop_columns": drops});
        if n_classes > 0 {
            public_task["num_classes"] = serde_json::json!(n_classes);
        }
        let mut params = std::collections::BTreeMap::new();
        params.insert("public_task".to_string(), public_task);
        params.insert(
            "declared_by".to_string(),
            serde_json::json!("worldgen --declare-aggcol-tasks"),
        );
        db.program_metadata.push(crate::labels::LabelMetadata {
            family: "aggcol".into(),
            target_table: table_name.clone(),
            source_tables: vec![table_name.clone()],
            target_type: if n_classes > 0 {
                "classification".into()
            } else {
                "continuous".into()
            },
            n_classes,
            hops: 0,
            uses_time: false,
            params,
            seed: db.seed,
            target_col: stem.into(),
            task_type: task_type.into(),
            num_rows,
        });
        declared += 1;
    }
    Ok(declared)
}

/// S-J readouts only apply to the opt-in schema.
pub fn add_sj_labels(db: &mut GeneratedDb, seed: u64) {
    let Some(focal) = db.tables.iter().position(|t| t.name == "sj_entity_2") else {
        return;
    };
    let n = db.tables[focal].num_rows();
    if n < 8 {
        return;
    }
    let numeric =
        |t: &TableData, c: &str| t.column(c).and_then(ColumnData::as_numeric_lossy).unwrap();
    let own = numeric(&db.tables[focal], "sj_x_0");
    let mut satellite = Vec::new();
    for t in &db.tables {
        if t.name.starts_with("sj_sat_2_") {
            for (name, data) in &t.columns {
                if name.starts_with("feature_") {
                    satellite.push(data.as_numeric_lossy().unwrap());
                }
            }
        }
    }
    if satellite.is_empty() {
        return;
    }
    // These two mandatory junctions give one hub identity per member. Repeated
    // bridge rows carry tag multiplicity; sibling sets are de-duplicated.
    let mut hubs = Vec::new();
    let mut hub_x = Vec::new();
    let mut mixture = Vec::new();
    for h in 0..2 {
        let Some(j) = db
            .tables
            .iter()
            .find(|t| t.name == format!("sj_junction_{h}_2"))
        else {
            return;
        };
        let hub = db
            .tables
            .iter()
            .find(|t| t.name == format!("sj_hub_{h}"))
            .unwrap();
        let ha = numeric(j, "fk_a");
        let fb = numeric(j, "fk_b");
        let tag = numeric(j, "tag");
        let hx = numeric(hub, "sj_x_0");
        let mut group: Vec<i64> = (0..n).map(|r| -(r as i64) - 1).collect();
        let mut tags = vec![vec![]; n];
        let keys = numeric(&db.tables[focal], &db.tables[focal].pkey_col);
        let lookup: std::collections::BTreeMap<i64, usize> = keys
            .iter()
            .enumerate()
            .map(|(r, k)| (*k as i64, r))
            .collect();
        let hkeys = numeric(hub, &hub.pkey_col);
        let hlookup: std::collections::BTreeMap<i64, usize> = hkeys
            .iter()
            .enumerate()
            .map(|(r, k)| (*k as i64, r))
            .collect();
        for r in 0..ha.len() {
            if let (Some(&p), Some(&g)) =
                (lookup.get(&(fb[r] as i64)), hlookup.get(&(ha[r] as i64)))
            {
                group[p] = g as i64;
                tags[p].push(tag[r] as usize);
            }
        }
        hub_x.push(
            group
                .iter()
                .map(|&g| if g < 0 { 0.0 } else { hx[g as usize] })
                .collect::<Vec<_>>(),
        );
        hubs.push(group);
        if h == 0 {
            let c = substream(seed, "sj:junmix:category").choose_index(4);
            mixture = tags
                .iter()
                .map(|bag| {
                    if bag.is_empty() {
                        return [0.0; 3];
                    }
                    let mut counts = [0.0; 4];
                    for &v in bag {
                        counts[v] += 1.0;
                    }
                    let proportions = counts.map(|v| v / bag.len() as f64);
                    [
                        proportions[c],
                        -proportions
                            .iter()
                            .filter(|&&p| p > 0.0)
                            .map(|p| p * p.ln())
                            .sum::<f64>(),
                        bag.len() as f64,
                    ]
                })
                .collect::<Vec<_>>();
        }
    }
    let times = timestamp_column(&db.tables[focal]).unwrap();
    for family in [
        "satfeat", "satnull", "junmix", "juncount", "hubhist", "hubnull", "xhub", "xhub_add",
    ] {
        let mut rng = substream(seed, &format!("sj:weights:{family}"));
        let weights: Vec<f64> = (0..satellite.len().max(4))
            .map(|_| rng.f64_range(-1.5, 1.5))
            .collect();
        let mut readout = substream(seed, &format!("sj:readout:{family}"));
        let sigma = readout.f64_range(0.02, 0.25);
        let noise: Vec<f64> = (0..n).map(|_| sigma * readout.standard_normal()).collect();
        let mut y = vec![0.0; n];
        match family {
            "satfeat" => {
                let sd = weights[..satellite.len()]
                    .iter()
                    .map(|w| w * w)
                    .sum::<f64>()
                    .sqrt()
                    .max(1e-12);
                for r in 0..n {
                    y[r] = satellite
                        .iter()
                        .zip(&weights)
                        .map(|(x, w)| x[r] * w)
                        .sum::<f64>()
                        / sd;
                }
            }
            "satnull" => {
                for r in 0..n {
                    y[r] = weights[0].signum() * own[r];
                }
            }
            "junmix" | "juncount" => {
                for r in 0..n {
                    y[r] = if family == "juncount" {
                        mixture[r][2]
                    } else {
                        (0..3).map(|j| weights[j] * mixture[r][j]).sum()
                    };
                }
                standardize(&mut y);
            }
            "hubhist" | "hubnull" => {
                let mut membership = hubs[0].clone();
                if family == "hubnull" {
                    // Latent pre-shuffle membership.
                    substream(seed, "sj:hubnull:membership").shuffle(&mut membership);
                }
                y = sj_history_draw(&own, &membership, &times, seed, family).labels;
            }
            "xhub" => {
                for r in 0..n {
                    y[r] = weights[0].signum() * hub_x[0][r] * hub_x[1][r];
                }
            }
            _ => {
                let sd = (weights[0].powi(2) + weights[1].powi(2)).sqrt().max(1e-12);
                for r in 0..n {
                    y[r] = (weights[0] * hub_x[0][r] + weights[1] * hub_x[1][r]) / sd;
                }
            }
        }
        if family != "hubhist" && family != "hubnull" {
            for r in 0..n {
                y[r] += noise[r];
            }
        }
        let binary = crate::labels::cut_feature_classes(&y, &[0.0]);
        push_f64(&mut db.tables[focal], &format!("{family}_r"), y);
        push_i64(&mut db.tables[focal], &format!("{family}_k2"), binary);
        use sha2::{Digest, Sha256};
        let source_sha = format!("{:x}", Sha256::digest(include_bytes!("labels_v2.rs")));
        for suffix in ["r", "k2"] {
            db.program_metadata.push(crate::labels::LabelMetadata {
                family: family.into(),
                target_table: db.tables[focal].name.clone(),
                source_tables: if family.starts_with("hub") {
                    vec![
                        "sj_entity_2".into(),
                        "sj_junction_0_2".into(),
                        "sj_hub_0".into(),
                    ]
                } else {
                    db.tables
                        .iter()
                        .filter(|t| t.name.starts_with("sj_"))
                        .map(|t| t.name.clone())
                        .collect()
                },
                target_type: if suffix == "r" {
                    "regression"
                } else {
                    "classification"
                }
                .into(),
                n_classes: if suffix == "r" { 0 } else { 2 },
                hops: if family.starts_with("hub") {
                    4
                } else if family.starts_with("sat") {
                    1
                } else {
                    2
                },
                uses_time: family.starts_with("hub"),
                params: [
                    (
                        "history_scale_constant".into(),
                        serde_json::json!(SJ_HISTORY_SCALE),
                    ),
                    (
                        "mechanism_version".into(),
                        serde_json::json!("local-history-v2"),
                    ),
                    ("source_sha256".into(), serde_json::json!(source_sha)),
                    (
                        "schema_sha256".into(),
                        serde_json::json!(format!("{:x}", Sha256::digest(include_bytes!("sj.rs")))),
                    ),
                ]
                .into(),
                seed,
                target_col: format!("{family}_{suffix}"),
                task_type: if suffix == "r" {
                    "regression"
                } else {
                    "binary_classification"
                }
                .into(),
                num_rows: n,
            });
        }
    }
}

/// Own-hub causal recursion. Fixed scale is a program parameter, never an
/// outcome statistic. Alpha weights the N(0,1) own coordinate.
pub const SJ_HISTORY_SCALE: f64 = 0.25;
pub const SJ_RATE_NOISE_SCALE: f64 = 0.25;
#[derive(Debug)]
pub struct SjHistoryDraw {
    pub labels: Vec<f64>,
    pub own_component: Vec<f64>,
    pub relation_component: Vec<f64>,
    pub own_variance_share: f64,
    pub successes: Vec<f64>,
    pub trials: Vec<i64>,
}
pub fn sj_history_draw(
    own: &[f64],
    membership: &[i64],
    times: &[i64],
    seed: u64,
    family: &str,
) -> SjHistoryDraw {
    assert!(
        !family.starts_with("hubrate"),
        "rates must read their own activity stream"
    );
    use std::collections::BTreeMap;
    let n = own.len();
    let mut weights = substream(seed, &format!("sj:variance-share:{family}"));
    let alpha = weights.f64_range(0.001, 0.005);
    let sign = if weights.bernoulli(0.5) { 1.0 } else { -1.0 };
    let mut rng = substream(seed, &format!("sj:readout:{family}"));
    let sigma = rng.f64_range(0.02, 0.25);
    let noise: Vec<_> = (0..n).map(|_| sigma * rng.standard_normal()).collect();
    let mut history = BTreeMap::<i64, (f64, usize)>::new();
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by_key(|&r| (times[r], r));
    let mut out = SjHistoryDraw {
        labels: vec![0.0; n],
        own_component: vec![0.0; n],
        relation_component: vec![0.0; n],
        own_variance_share: alpha,
        successes: vec![0.0; n],
        trials: vec![0; n],
    };
    let mut pos = 0;
    while pos < n {
        let end = order[pos..]
            .iter()
            .take_while(|&&r| times[r] == times[order[pos]])
            .count()
            + pos;
        for &r in &order[pos..end] {
            let (successes, trials) = history.get(&membership[r]).copied().unwrap_or_default();
            let raw = if trials == 0 {
                0.0
            } else {
                successes / trials as f64
            };
            out.successes[r] = successes;
            out.trials[r] = trials as i64;
            let has_history = trials > 0;
            out.own_component[r] = if has_history {
                alpha.sqrt() * sign * own[r]
            } else {
                0.0
            };
            out.relation_component[r] = if has_history {
                (1.0 - alpha).sqrt() * raw / SJ_HISTORY_SCALE
            } else {
                0.0
            };
            out.labels[r] =
                ((out.own_component[r] + out.relation_component[r]) / 2.0).tanh() + noise[r];
        }
        for &r in &order[pos..end] {
            let entry = history.entry(membership[r]).or_default();
            entry.0 += out.labels[r];
            entry.1 += 1;
        }
        pos = end;
    }
    out
}

/// Rate reducer has no hub, sibling-label, or own-feature argument: the only
/// observations it can consume are the query entity's strictly-prior events.
pub fn sj_stream_rate_draw(
    activity: &TableData,
    keys: &[i64],
    times: &[i64],
    seed: u64,
    family: &str,
) -> SjHistoryDraw {
    let n = keys.len();
    let mut out = SjHistoryDraw {
        labels: vec![0.0; n],
        own_component: vec![0.0; n],
        relation_component: vec![0.0; n],
        own_variance_share: 0.0,
        successes: vec![0.0; n],
        trials: vec![0; n],
    };
    let lookup: std::collections::BTreeMap<_, _> =
        keys.iter().enumerate().map(|(r, &k)| (k, r)).collect();
    let ColumnData::I64(entity) = activity.column("entity").unwrap() else {
        panic!()
    };
    let ColumnData::I64(success) = activity.column("sj_success").unwrap() else {
        panic!()
    };
    let ts = timestamp_column(activity).unwrap();
    for ((&key, &y), &time) in entity.iter().zip(success).zip(&ts) {
        if let Some(&r) = lookup.get(&key) {
            let age = times[r] as i128 - time as i128;
            if age > 0 && age <= 86_400_000_000_000 {
                out.trials[r] += 1;
                out.successes[r] += y as f64;
            }
        }
    }
    let mut rng = substream(seed, &format!("sj:readout:{family}"));
    let sigma = rng.f64_range(0.02, 0.25);
    for r in 0..n {
        let raw = (out.successes[r] + 0.5) / (out.trials[r] as f64 + 1.0);
        out.relation_component[r] = raw - 0.5;
        let scale = if out.trials[r] > 0 {
            SJ_RATE_NOISE_SCALE
        } else {
            1.0
        };
        out.labels[r] = (raw + sigma * scale * rng.standard_normal()).clamp(0.0, 1.0);
    }
    out
}

pub fn add_sj_extension_labels(db: &mut GeneratedDb, seed: u64) {
    let Some(focal) = db.tables.iter().position(|t| t.name == "sj_entity_2") else {
        return;
    };
    let Some(stream) = db.table("sj_stream_2") else {
        return;
    };
    let n = db.tables[focal].num_rows();
    if n < 8 {
        return;
    }
    let times = timestamp_column(&db.tables[focal]).unwrap();
    let own = db.tables[focal]
        .column("sj_x_0")
        .unwrap()
        .as_numeric_lossy()
        .unwrap();
    let ColumnData::I64(keys) = db.tables[focal].column(&db.tables[focal].pkey_col).unwrap() else {
        return;
    };
    let keys = keys.clone();
    let lookup: std::collections::BTreeMap<_, _> =
        keys.iter().enumerate().map(|(r, &k)| (k, r)).collect();
    let ColumnData::I64(fk) = stream.column("entity").unwrap() else {
        return;
    };
    let event_times = timestamp_column(stream).unwrap();
    let window = 30.0;
    let day = 86_400_000_000_000.0;
    let mut ages = vec![vec![]; n];
    for (&key, &time) in fk.iter().zip(&event_times) {
        if let Some(&r) = lookup.get(&key) {
            if time < times[r] {
                ages[r].push((times[r] as i128 - time as i128) as f64 / day);
            }
        }
    }
    let counts: Vec<Vec<f64>> = [1.0, 7.0, 30.0]
        .iter()
        .map(|&w| {
            ages.iter()
                .map(|bag| bag.iter().filter(|&&a| a <= w).count() as f64)
                .collect()
        })
        .collect();
    let recency: Vec<f64> = ages
        .iter()
        .map(|bag| bag.iter().copied().reduce(f64::min).unwrap_or(0.0))
        .collect();
    let warm: Vec<bool> = ages.iter().map(|bag| !bag.is_empty()).collect();
    let raw_evidence = [
        counts[0].clone(),
        counts[1].clone(),
        counts[2].clone(),
        recency,
    ];
    let mut evidence = raw_evidence.clone();
    for x in &mut evidence {
        sj_standardize_warm(x, &warm);
    }
    let j = db.table("sj_junction_0_2").unwrap();
    let ColumnData::I64(a) = j.column("fk_a").unwrap() else {
        return;
    };
    let ColumnData::I64(b) = j.column("fk_b").unwrap() else {
        return;
    };
    let mut membership: Vec<i64> = (0..n).map(|r| -(r as i64) - 1).collect();
    for (&hub, &key) in a.iter().zip(b) {
        if let Some(&r) = lookup.get(&key) {
            membership[r] = hub;
        }
    }
    push_f64(
        &mut db.tables[focal],
        "sj_stream_window_days",
        vec![window; n],
    );
    let mut rate_activities = Vec::new();
    for family in ["streamrec", "streamnull", "hubrate", "hubrate_null"] {
        let y = match family {
            "streamrec" => {
                let mut rng = substream(seed, "sj:weights:streamrec");
                let w: Vec<_> = (0..4).map(|_| rng.f64_range(-1.5, 1.5)).collect();
                let mut y: Vec<_> = (0..n)
                    .map(|r| (0..4).map(|j| w[j] * evidence[j][r]).sum())
                    .collect();
                sj_standardize_warm(&mut y, &warm);
                let mut noise = substream(seed, "sj:readout:streamrec");
                let sigma = noise.f64_range(0.02, 0.25);
                for x in &mut y {
                    *x += sigma * noise.standard_normal();
                }
                y
            }
            "streamnull" => crate::sj::streamnull_draw(seed, &own),
            _ => {
                let mut groups = membership.clone();
                if family == "hubrate_null" {
                    substream(seed, "sj:hubrate_null:membership").shuffle(&mut groups);
                }
                // One exposure event per strictly-prior donor, available in the focal prediction's
                // explicit one-day window.
                let mut activity = sj_empty_table(&format!("sj_{family}_activity"), 0);
                let mut entities = Vec::new();
                let mut outcomes = Vec::new();
                let mut dates = Vec::new();
                for r in 0..n {
                    let mut events = substream(seed, &format!("sj:events:{family}:{}", keys[r]));
                    let propensity = events.f64_range(0.1, 0.9);
                    for p in 0..n {
                        if groups[p] == groups[r] && times[p] < times[r] {
                            entities.push(keys[r]);
                            outcomes.push(i64::from(events.bernoulli(propensity)));
                            dates.push(times[r].saturating_sub(1));
                        }
                    }
                }
                activity.columns[0].1 = ColumnData::I64((0..entities.len() as i64).collect());
                activity
                    .fkey_col_to_pkey_table
                    .insert("entity".into(), "sj_entity_2".into());
                push_i64(&mut activity, "entity", entities);
                push_i64(&mut activity, "sj_success", outcomes);
                activity.time_col = Some("date".into());
                activity
                    .columns
                    .push(("date".into(), ColumnData::TimestampNs(dates)));
                let draw = sj_stream_rate_draw(&activity, &keys, &times, seed, family);
                if family == "hubrate_null" {
                    // Null control: shuffle exposed outcomes after drawing Y.
                    let (_, ColumnData::I64(values)) = activity
                        .columns
                        .iter_mut()
                        .find(|(name, _)| name == "sj_success")
                        .unwrap()
                    else {
                        unreachable!()
                    };
                    substream(seed, "sj:hubrate_null:activity-membership").shuffle(values);
                }
                rate_activities.push(activity);
                draw.labels
            }
        };
        let cut = if family.starts_with("hubrate") {
            0.5
        } else {
            0.0
        };
        let binary = crate::labels::cut_feature_classes(&y, &[cut]);
        push_f64(&mut db.tables[focal], &format!("{family}_r"), y);
        push_i64(&mut db.tables[focal], &format!("{family}_k2"), binary);
        use sha2::{Digest, Sha256};
        for suffix in ["r", "k2"] {
            let mut params: std::collections::BTreeMap<String, serde_json::Value> = [
                (
                    "source_sha256".into(),
                    serde_json::json!(format!(
                        "{:x}",
                        Sha256::digest(include_bytes!("labels_v2.rs"))
                    )),
                ),
                (
                    "schema_sha256".into(),
                    serde_json::json!(format!("{:x}", Sha256::digest(include_bytes!("sj.rs")))),
                ),
                ("binary_cut".into(), serde_json::json!(cut)),
            ]
            .into();
            if family.starts_with("stream") {
                params.insert("windows_days".into(), serde_json::json!([1, 7, 30]));
            }
            if family.starts_with("hubrate") {
                params.insert("own_variance_share".into(), serde_json::json!(0.0));
                params.insert("beta_prior".into(), serde_json::json!([0.5, 0.5]));
                params.insert("exposure_window_days".into(), serde_json::json!(1));
                params.insert(
                    "exposure_table".into(),
                    serde_json::json!(format!("sj_{family}_activity")),
                );
                params.insert(
                    "noise_scale_constant".into(),
                    serde_json::json!(SJ_RATE_NOISE_SCALE),
                );
                params.insert(
                    "mechanism_version".into(),
                    serde_json::json!("local-history-v2"),
                );
            }
            db.program_metadata.push(crate::labels::LabelMetadata {
                family: family.into(),
                target_table: db.tables[focal].name.clone(),
                source_tables: if family.starts_with("stream") {
                    vec!["sj_stream_2".into()]
                } else {
                    vec![format!("sj_{family}_activity")]
                },
                target_type: if suffix == "r" {
                    "regression"
                } else {
                    "classification"
                }
                .into(),
                n_classes: if suffix == "r" { 0 } else { 2 },
                hops: 1,
                uses_time: true,
                params,
                seed,
                target_col: format!("{family}_{suffix}"),
                task_type: if suffix == "r" {
                    "regression"
                } else {
                    "binary_classification"
                }
                .into(),
                num_rows: n,
            });
        }
    }
    db.tables.extend(rate_activities);
    let mut stats = sj_empty_table("sj_stream_evidence_2", n);
    stats
        .fkey_col_to_pkey_table
        .insert("entity".into(), "sj_entity_2".into());
    push_i64(
        &mut stats,
        "entity",
        db.tables[focal]
            .column(&db.tables[focal].pkey_col)
            .unwrap()
            .as_numeric_lossy()
            .unwrap()
            .iter()
            .map(|x| *x as i64)
            .collect(),
    );
    for (name, x) in ["count_1d", "count_7d", "count_30d", "days_since_last"]
        .iter()
        .zip(raw_evidence)
    {
        push_f64(&mut stats, name, x);
    }
    push_i64(
        &mut stats,
        "has_history",
        warm.iter().map(|&v| i64::from(v)).collect(),
    );
    db.tables.push(stats);
    add_sj_junction_history(db, seed);
}

fn sj_standardize_warm(x: &mut [f64], warm: &[bool]) {
    let values: Vec<_> = x
        .iter()
        .zip(warm)
        .filter_map(|(&v, &w)| w.then_some(v))
        .collect();
    let n = values.len().max(1) as f64;
    let mean = values.iter().sum::<f64>() / n;
    let sd = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
    for (v, &w) in x.iter_mut().zip(warm) {
        *v = if w && sd > 1e-12 {
            (*v - mean) / sd
        } else {
            0.0
        };
    }
}
fn sj_empty_table(name: &str, n: usize) -> TableData {
    TableData {
        name: name.into(),
        pkey_col: "row_idx".into(),
        fkey_col_to_pkey_table: Default::default(),
        time_col: None,
        columns: vec![("row_idx".into(), ColumnData::I64((0..n as i64).collect()))],
        hsbm_hierarchy: vec![],
        fk_hsbm: Default::default(),
        feature_contract: Default::default(),
        bag_summary: Default::default(),
        rank_field: Default::default(),
    }
}
/// Focal -> search activity -> click junction: the source LABEL is on hop two.
pub fn add_sj_junction_history(db: &mut GeneratedDb, seed: u64) {
    let Some(f) = db.tables.iter().position(|t| t.name == "sj_entity_2") else {
        return;
    };
    let n = db.tables[f].num_rows();
    let times = timestamp_column(&db.tables[f]).unwrap();
    let own = db.tables[f]
        .column("sj_x_0")
        .unwrap()
        .as_numeric_lossy()
        .unwrap();
    let keys = db.tables[f]
        .column(&db.tables[f].pkey_col)
        .unwrap()
        .as_numeric_lossy()
        .unwrap();
    let j = db.table("sj_junction_0_2").unwrap();
    let ColumnData::I64(a) = j.column("fk_a").unwrap() else {
        panic!()
    };
    let ColumnData::I64(b) = j.column("fk_b").unwrap() else {
        panic!()
    };
    let mut membership: Vec<i64> = (0..n).map(|r| -(r as i64) - 1).collect();
    let lookup: std::collections::BTreeMap<_, _> = keys
        .iter()
        .enumerate()
        .map(|(r, &k)| (k as i64, r))
        .collect();
    for (&hub, &key) in a.iter().zip(b) {
        membership[lookup[&key]] = hub;
    }
    let draw = sj_history_draw(&own, &membership, &times, seed, "hubhistj");
    let active: Vec<_> = (0..n).filter(|&r| draw.trials[r] > 0).collect();
    let mut search = sj_empty_table("sj_search_2", active.len());
    search
        .fkey_col_to_pkey_table
        .insert("entity".into(), "sj_entity_2".into());
    push_i64(
        &mut search,
        "entity",
        active.iter().map(|&r| keys[r] as i64).collect(),
    );
    search.time_col = Some("date".into());
    search.columns.push((
        "date".into(),
        ColumnData::TimestampNs(active.iter().map(|&r| times[r].saturating_sub(1)).collect()),
    ));
    let mut sf = vec![];
    let mut hf = vec![];
    let mut mf = vec![];
    let mut labels = vec![];
    let mut ct = vec![];
    for (s, &r) in active.iter().enumerate() {
        for p in 0..n {
            if membership[p] == membership[r] && times[p] < times[r] {
                sf.push(s as i64);
                hf.push(membership[r]);
                mf.push(keys[p] as i64);
                labels.push(draw.labels[p]);
                ct.push(times[p]);
            }
        }
    }
    let mut junction = sj_empty_table("sj_search_clicks_2", labels.len());
    junction.fkey_col_to_pkey_table = [
        ("search".into(), search.name.clone()),
        ("hub".into(), "sj_hub_0".into()),
        ("member".into(), "sj_entity_2".into()),
    ]
    .into();
    push_i64(&mut junction, "search", sf);
    push_i64(&mut junction, "hub", hf);
    push_i64(&mut junction, "member", mf);
    push_f64(&mut junction, "sj_click_label", labels);
    junction.time_col = Some("date".into());
    junction
        .columns
        .push(("date".into(), ColumnData::TimestampNs(ct)));
    for family in ["hubhistj", "hubhistj_null"] {
        let mut groups = membership.clone();
        if family == "hubhistj_null" {
            substream(seed, "sj:hubhistj_null:membership").shuffle(&mut groups);
        }
        let result = sj_history_draw(&own, &groups, &times, seed, family);
        let alpha = result.own_variance_share;
        let y = result.labels;
        let binary = crate::labels::cut_feature_classes(&y, &[0.0]);
        push_f64(&mut db.tables[f], &format!("{family}_r"), y);
        push_i64(&mut db.tables[f], &format!("{family}_k2"), binary);
        use sha2::{Digest, Sha256};
        for suffix in ["r", "k2"] {
            db.program_metadata.push(crate::labels::LabelMetadata {
                family: family.into(),
                target_table: db.tables[f].name.clone(),
                source_tables: vec![search.name.clone(), junction.name.clone()],
                target_type: if suffix == "r" {
                    "regression"
                } else {
                    "classification"
                }
                .into(),
                n_classes: if suffix == "r" { 0 } else { 2 },
                hops: 2,
                uses_time: true,
                params: [
                    ("source_column".into(), serde_json::json!("sj_click_label")),
                    (
                        "history_scale_constant".into(),
                        serde_json::json!(SJ_HISTORY_SCALE),
                    ),
                    (
                        "mechanism_version".into(),
                        serde_json::json!("local-history-v2"),
                    ),
                    ("own_variance_share".into(), serde_json::json!(alpha)),
                    (
                        "source_sha256".into(),
                        serde_json::json!(format!(
                            "{:x}",
                            Sha256::digest(include_bytes!("labels_v2.rs"))
                        )),
                    ),
                    (
                        "schema_sha256".into(),
                        serde_json::json!(format!("{:x}", Sha256::digest(include_bytes!("sj.rs")))),
                    ),
                ]
                .into(),
                seed,
                target_col: format!("{family}_{suffix}"),
                task_type: if suffix == "r" {
                    "regression"
                } else {
                    "binary_classification"
                }
                .into(),
                num_rows: n,
            });
        }
    }
    db.tables.push(search);
    db.tables.push(junction);
}

fn mpnn_message(x: &[f64; 4], w1: &[[f64; 4]; 4], b1: &[f64; 4], w2: &[f64; 4]) -> f64 {
    let mut hidden = [0.0f64; 4];
    for k in 0..4 {
        let mut s = b1[k];
        for j in 0..4 {
            s += w1[k][j] * x[j];
        }
        hidden[k] = s.tanh();
    }
    w2.iter().zip(hidden.iter()).map(|(w, h)| w * h).sum()
}

fn draw_mlp_weights(rng: &mut crate::rng::GenRng) -> ([[f64; 4]; 4], [f64; 4], [f64; 4]) {
    let mut w1 = [[0.0f64; 4]; 4];
    for row in w1.iter_mut() {
        for cell in row.iter_mut() {
            *cell = rng.normal(0.0, 0.8);
        }
    }
    let mut b1 = [0.0f64; 4];
    for cell in b1.iter_mut() {
        *cell = rng.normal(0.0, 0.2);
    }
    let mut w2 = [0.0f64; 4];
    for cell in w2.iter_mut() {
        *cell = rng.normal(0.0, 0.8);
    }
    (w1, b1, w2)
}

/// Per-row message: up to 3 randomly-picked own base-numeric columns plus
/// the strictly-prior same-entity history term, through one random MLP.
fn mpnn_row_messages(table: &TableData, rng: &mut crate::rng::GenRng) -> Option<Vec<f64>> {
    let base_cols = base_numeric_columns(table);
    if base_cols.is_empty() {
        return None;
    }
    let mut cols = base_cols;
    rng.shuffle(&mut cols);
    let feats: Vec<Vec<f64>> = cols
        .iter()
        .take(3)
        .filter_map(|name| f64_column(table, name))
        .collect();
    if feats.is_empty() {
        return None;
    }
    let n = table.num_rows();
    let history: Vec<f64> = match entity_and_time(table) {
        Some((entity, t_vec)) => {
            let prior = prior_same_entity_rows(&entity, &t_vec);
            let payload = &feats[0];
            prior
                .iter()
                .map(|rows| rows.last().map(|&r| payload[r]).unwrap_or(0.0))
                .collect()
        }
        None => vec![0.0; n],
    };
    let (w1, b1, w2) = draw_mlp_weights(rng);
    Some(
        (0..n)
            .map(|row| {
                let mut x = [0.0f64; 4];
                for (j, feat) in feats.iter().take(3).enumerate() {
                    x[j] = feat[row];
                }
                x[3] = history[row];
                mpnn_message(&x, &w1, &b1, &w2)
            })
            .collect(),
    )
}

/// Reduce `messages` over `fk` into per-parent [mean, log1p(count), max]
/// (`child_bag_stats`), then broadcast the mean/max back down to every
/// child row through the same `fk` column.
fn mpnn_hop_broadcast(fk: &[i64], messages: &[f64], n_parent: usize) -> (Vec<f64>, Vec<f64>) {
    let stats = child_bag_stats(fk, messages, n_parent);
    let mean = fk
        .iter()
        .map(|&f| {
            if f >= 0 && (f as usize) < n_parent {
                stats[f as usize][0]
            } else {
                0.0
            }
        })
        .collect();
    let max = fk
        .iter()
        .map(|&f| {
            if f >= 0 && (f as usize) < n_parent {
                stats[f as usize][2]
            } else {
                0.0
            }
        })
        .collect();
    (mean, max)
}

pub fn add_mpnn_readout_labels(db: &mut GeneratedDb, seed: u64) {
    let table_count = db.tables.len();
    for table_idx in 0..table_count {
        let Some((fk_col, parent_name)) = db.tables[table_idx]
            .fkey_col_to_pkey_table
            .iter()
            .next()
            .map(|(c, p)| (c.clone(), p.clone()))
        else {
            continue;
        };
        let Some(parent_idx) = db.tables.iter().position(|t| t.name == parent_name) else {
            continue;
        };
        if parent_idx == table_idx {
            continue;
        }
        if db.tables[table_idx].num_rows() < 8 {
            continue;
        }
        let n_rows = db.tables[table_idx].num_rows();

        let mut rng = substream(seed + 262_147 * (table_idx as u64 + 1), "labels_v2:mpnn");
        let Some(own_h) = mpnn_row_messages(&db.tables[table_idx], &mut rng) else {
            continue;
        };
        let Some(ColumnData::I64(fk_values)) = db.tables[table_idx].column(&fk_col).cloned() else {
            continue;
        };
        let n_parent = db.tables[parent_idx].num_rows();
        let (hop1_mean, hop1_max) = mpnn_hop_broadcast(&fk_values, &own_h, n_parent);

        let hop2_mean = {
            let gp_link = db.tables[parent_idx]
                .fkey_col_to_pkey_table
                .iter()
                .next()
                .map(|(c, p)| (c.clone(), p.clone()));
            let mut out = vec![0.0; n_rows];
            if let Some((gp_fk_col, gp_name)) = gp_link {
                if let Some(gp_idx) = db.tables.iter().position(|t| t.name == gp_name) {
                    if gp_idx != parent_idx && gp_idx != table_idx {
                        let mut prng = substream(
                            seed + 393_221 * (table_idx as u64 + 1),
                            "labels_v2:mpnn:hop2",
                        );
                        if let Some(parent_h) = mpnn_row_messages(&db.tables[parent_idx], &mut prng)
                        {
                            if let Some(ColumnData::I64(parent_fk)) =
                                db.tables[parent_idx].column(&gp_fk_col).cloned()
                            {
                                let n_gp = db.tables[gp_idx].num_rows();
                                let (hop2_at_parent, _) =
                                    mpnn_hop_broadcast(&parent_fk, &parent_h, n_gp);
                                out = fk_values
                                    .iter()
                                    .map(|&f| {
                                        if f >= 0 && (f as usize) < hop2_at_parent.len() {
                                            hop2_at_parent[f as usize]
                                        } else {
                                            0.0
                                        }
                                    })
                                    .collect();
                            }
                        }
                    }
                }
            }
            out
        };

        let use_attention = rng.bernoulli(0.5);
        let variant = if use_attention { "attn" } else { "mlp" };
        let latent: Vec<f64> = if use_attention {
            let query = [
                rng.normal(0.0, 1.0),
                rng.normal(0.0, 1.0),
                rng.normal(0.0, 1.0),
            ];
            let keys: [[f64; 3]; 3] = std::array::from_fn(|_| {
                [
                    rng.normal(0.0, 1.0),
                    rng.normal(0.0, 1.0),
                    rng.normal(0.0, 1.0),
                ]
            });
            let scale = [
                rng.normal(0.0, 1.0),
                rng.normal(0.0, 1.0),
                rng.normal(0.0, 1.0),
            ];
            let logits: Vec<f64> = (0..3)
                .map(|k| {
                    query
                        .iter()
                        .zip(&keys[k])
                        .map(|(q, kk)| q * kk)
                        .sum::<f64>()
                })
                .collect();
            let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = logits.iter().map(|l| (l - m).exp()).collect();
            let z: f64 = exps.iter().sum::<f64>().max(1e-12);
            let weights: Vec<f64> = exps.iter().map(|e| e / z).collect();
            (0..n_rows)
                .map(|row| {
                    let vals = [own_h[row], hop1_mean[row], hop2_mean[row]];
                    let combined: f64 = (0..3).map(|k| weights[k] * vals[k] * scale[k]).sum();
                    combined.tanh()
                })
                .collect()
        } else {
            let (rw1, rb1, rw2) = draw_mlp_weights(&mut rng);
            (0..n_rows)
                .map(|row| {
                    let x = [own_h[row], hop1_mean[row], hop1_max[row], hop2_mean[row]];
                    mpnn_message(&x, &rw1, &rb1, &rw2)
                })
                .collect()
        };

        if is_constant(&latent) {
            continue;
        }
        let noise = rng.f64_range(0.02, 0.15);
        let noised = iqr_noise(
            &latent,
            noise,
            seed + 262_147 * (table_idx as u64 + 1),
            "labels_v2:mpnn:row",
        );
        if is_constant(&noised) {
            continue;
        }
        let (y, link) = apply_observation_link(&noised, &mut rng);
        let table_name = db.tables[table_idx].name.clone();
        push_f64(
            &mut db.tables[table_idx],
            &format!("mpnn_r_{variant}_{link}_{table_name}"),
            y,
        );
        if let Some(k2) = median_k2(&latent) {
            push_i64(
                &mut db.tables[table_idx],
                &format!("mpnn_k2_{variant}_{table_name}"),
                k2,
            );
        }
    }
}

/// Population membership of one incoming FK relation, per parent row.
fn poplaw_population_groups(
    fk: &[i64],
    child_time: Option<&[i64]>,
    parent_time: Option<&[i64]>,
    n_parent: usize,
) -> Vec<Vec<usize>> {
    let mut groups = vec![Vec::new(); n_parent];
    for (row, &f) in fk.iter().enumerate() {
        if f < 0 || (f as usize) >= n_parent {
            continue;
        }
        let p = f as usize;
        if let (Some(ct), Some(pt)) = (child_time, parent_time) {
            if ct[row] > pt[p] {
                continue;
            }
        }
        groups[p].push(row);
    }
    groups
}

fn poplaw_largest_relation(db: &GeneratedDb, parent_idx: usize) -> Option<(usize, String)> {
    aggcol_relations(db, parent_idx)
        .into_iter()
        .filter(|(child_idx, _)| *child_idx != parent_idx)
        .max_by_key(|(child_idx, fk)| match db.tables[*child_idx].column(fk) {
            Some(ColumnData::I64(values)) => values.len(),
            _ => 0,
        })
}

fn publish_population_law(
    table: &mut TableData,
    family: &str,
    latent: &[f64],
    seed: u64,
    rng: &mut crate::rng::GenRng,
) {
    if is_constant(latent) {
        return;
    }
    let noise = rng.f64_range(0.02, 0.15);
    let noised = iqr_noise(latent, noise, seed, &format!("labels_v2:{family}:row"));
    if is_constant(&noised) {
        return;
    }
    let (y, link) = apply_observation_link(&noised, rng);
    let table_name = table.name.clone();
    push_f64(table, &format!("{family}_r_{link}_{table_name}"), y);
    if let Some(k2) = median_k2(latent) {
        push_i64(table, &format!("{family}_k2_{table_name}"), k2);
    }
}

pub fn add_population_law_labels(db: &mut GeneratedDb, seed: u64) {
    let table_count = db.tables.len();
    for parent_idx in 0..table_count {
        let Some((child_idx, fk_col)) = poplaw_largest_relation(db, parent_idx) else {
            continue;
        };
        let n_parent = db.tables[parent_idx].num_rows();
        if n_parent < 4 {
            continue;
        }
        let Some(ColumnData::I64(fk_values)) = db.tables[child_idx].column(&fk_col).cloned() else {
            continue;
        };
        let child_time = timestamp_column(&db.tables[child_idx]);
        let parent_time = timestamp_column(&db.tables[parent_idx]);
        let groups = poplaw_population_groups(
            &fk_values,
            child_time.as_deref(),
            parent_time.as_deref(),
            n_parent,
        );
        if groups.iter().filter(|g| g.len() >= 2).count() < 2 {
            continue;
        }

        let base_seed = seed + 917_513 * (parent_idx as u64 + 1);
        let mut rng = substream(base_seed, "labels_v2:poplaw");
        let candidates = base_numeric_columns(&db.tables[child_idx]);
        let Some((_, payload)) = pick_payload(&db.tables[child_idx], &candidates, &mut rng) else {
            continue;
        };
        let n_child = payload.len();

        let p_rate = rng.f64_range(1e-3_f64.ln(), 3e-2_f64.ln()).exp();
        let event: Vec<bool> = (0..n_child)
            .map(|row| child_substream(base_seed, "labels_v2:poplaw:a", row).bernoulli(p_rate))
            .collect();
        let alpha = if rng.bernoulli(0.5) { 1.0 } else { 0.0 };
        let law_a: Vec<f64> = groups
            .iter()
            .map(|g| {
                let n = g.len() as f64;
                let c = g.iter().filter(|&&r| event[r]).count() as f64;
                (1.0 + c).ln() - alpha * (1.0 + n).ln()
            })
            .collect();
        publish_population_law(
            &mut db.tables[parent_idx],
            "poplaw_a",
            &law_a,
            base_seed,
            &mut rng,
        );

        let m = rng.usize_inclusive(2, 8);
        let beta = if rng.bernoulli(0.5) { 0.5 } else { -0.5 };
        let subgroup: Vec<usize> = (0..n_child)
            .map(|row| {
                child_substream(base_seed, "labels_v2:poplaw:b", row).usize_inclusive(0, m - 1)
            })
            .collect();
        let law_b: Vec<f64> = groups
            .iter()
            .map(|g| {
                if g.len() < 2 {
                    return 0.0;
                }
                let mut by_h: std::collections::BTreeMap<usize, Vec<f64>> =
                    std::collections::BTreeMap::new();
                for &row in g {
                    by_h.entry(subgroup[row]).or_default().push(payload[row]);
                }
                let terms: Vec<f64> = by_h
                    .values()
                    .map(|xs| {
                        let n = xs.len() as f64;
                        let mean = xs.iter().sum::<f64>() / n;
                        let var = xs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
                        mean + beta * var.sqrt()
                    })
                    .collect();
                terms.iter().sum::<f64>() / terms.len() as f64
            })
            .collect();
        publish_population_law(
            &mut db.tables[parent_idx],
            "poplaw_b",
            &law_b,
            base_seed,
            &mut rng,
        );

        let h_scale = rng.f64_range(1.0, 60.0); // day-scale for the age distribution
        let q_rate = rng.f64_range(1e-3_f64.ln(), 3e-2_f64.ln()).exp();
        let qualifies: Vec<bool> = (0..n_child)
            .map(|row| child_substream(base_seed, "labels_v2:poplaw:c", row).bernoulli(q_rate))
            .collect();
        const NS_PER_DAY: f64 = 86_400.0e9;
        let age_days: Vec<Option<f64>> = match (&child_time, &parent_time) {
            (Some(ct), Some(pt)) => (0..n_child)
                .map(|row| {
                    let f = fk_values[row];
                    if !qualifies[row] || f < 0 || (f as usize) >= n_parent {
                        None
                    } else {
                        Some(((pt[f as usize] - ct[row]).max(0) as f64) / NS_PER_DAY)
                    }
                })
                .collect(),
            _ => vec![None; n_child],
        };
        let gamma = rng.f64_range(0.2, 1.0);
        let eta = rng.f64_range(0.2, 1.0);
        let law_c: Vec<f64> = groups
            .iter()
            .map(|g| {
                let n = g.len() as f64;
                let c = g.iter().filter(|&&r| qualifies[r]).count() as f64;
                let min_age = g
                    .iter()
                    .filter_map(|&r| age_days[r])
                    .fold(f64::INFINITY, f64::min);
                let a_r = if min_age.is_finite() {
                    min_age
                } else {
                    10.0 * h_scale
                };
                -(1.0 + a_r).ln() + gamma * (1.0 + c).ln() - eta * (1.0 + n).ln()
            })
            .collect();
        publish_population_law(
            &mut db.tables[parent_idx],
            "poplaw_c",
            &law_c,
            base_seed,
            &mut rng,
        );

        let parent_name = db.tables[parent_idx].name.clone();
        push_f64_opt(
            &mut db.tables[child_idx],
            &format!("evc_poplaw_c_{parent_name}"),
            age_days,
        );
    }
}

pub fn add_nuisance_text_columns(db: &mut GeneratedDb, seed: u64) {
    let shared_pool = nuisance_shared_pool();
    let table_count = db.tables.len();
    for table_idx in 0..table_count {
        let n_rows = db.tables[table_idx].num_rows();
        if n_rows == 0 || n_rows > ENTITY_TABLE_ROW_CEILING {
            continue;
        }
        let base_seed = seed ^ (0x6e75_6973 << 32) ^ (table_idx as u64 + 1);
        let mut table_rng = substream(base_seed, "labels_v2:nuisance_text");
        let n_cols = table_rng.usize_inclusive(6, 14);
        for col_idx in 0..n_cols {
            let mut col_rng = child_substream(base_seed, "labels_v2:nuisance_text:col", col_idx);
            let use_pool = col_rng.bernoulli(0.5);
            let values: Vec<Option<String>> = if use_pool {
                (0..n_rows)
                    .map(|row| {
                        let mut rrng = child_substream(
                            base_seed,
                            &format!("labels_v2:nuisance_text:pick:{col_idx}"),
                            row,
                        );
                        let idx = rrng.choose_index(shared_pool.len());
                        Some(shared_pool[idx].clone())
                    })
                    .collect()
            } else {
                (0..n_rows)
                    .map(|row| {
                        let mut rrng = child_substream(
                            base_seed,
                            &format!("labels_v2:nuisance_text:unique:{col_idx}"),
                            row,
                        );
                        Some(nuisance_token(&mut rrng))
                    })
                    .collect()
            };
            let col_name =
                db.tables[table_idx].next_column_name(&format!("nuisance_text_{col_idx}"));
            db.tables[table_idx].push_column(col_name, ColumnData::Utf8(values));
        }
    }
}

const ENTITY_TABLE_ROW_CEILING: usize = 1000;

fn nuisance_shared_pool() -> &'static [String] {
    use std::sync::OnceLock;
    static POOL: OnceLock<Vec<String>> = OnceLock::new();
    POOL.get_or_init(|| {
        let mut rng = substream(0x6e75_6973_706f_6f6c, "labels_v2:nuisance_text:shared_pool");
        let pool_size = rng.usize_inclusive(10_000, 20_000);
        (0..pool_size).map(|_| nuisance_token(&mut rng)).collect()
    })
}

/// A 16-hex-character token from one `u64` draw, never derived from any SCM latent or label.
fn nuisance_token(rng: &mut crate::rng::GenRng) -> String {
    use rand::Rng;
    format!("{:016x}", rng.random::<u64>())
}
