use crate::scm::GeneratedDb;
use anyhow::Result;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum Tier {
    L0,
    L1,
    L2,
    L3,
}

#[derive(Debug, Serialize)]
pub struct Family {
    pub family: &'static str,
    pub tier: Tier,
    pub aliases: &'static [&'static str],
    pub prefixes: &'static [&'static str],
    pub role: &'static str,
    pub reason: &'static str,
}
macro_rules! family {
    ($name:literal, $tier:ident, [$($alias:literal),*], [$($prefix:literal),*], $why:literal) => {
        Family { family: $name, tier: Tier::$tier, aliases: &[$($alias),*],
            prefixes: &[$($prefix),*], role: "label", reason: $why }
    };
}

pub static FAMILIES: &[Family] = &[
    family!("feature", L0, ["featreg", "featclf", "featimb", "tabicl_v2_like_discrete"], ["featreg", "featclf", "featimb"], "Row-local feature map; no relational evidence."),
    family!("noisyimb", L0, ["noisy-imb"], ["noisyimb_"], "Row-local noise and class-balance control."),
    family!("relnull", L0, [], ["relnull_r", "relnull_k2", "relnull_imb"], "Independent own-feature readout; graph is nuisance."),
    family!("aggcol", L0, [], ["aggcol_r", "aggcol_k2", "aggcol_imb"], "Aggregate values are already published on the focal row."),
    family!("satnull", L0, [], ["satnull_r", "satnull_k2"], "Own-feature null for satellite evidence."),
    family!("streamnull", L0, [], ["streamnull_r", "streamnull_k2"], "Own-feature null for stream evidence."),
    family!("parent_lookup", L1, ["plookup"], ["plookup_"], "One parent attribute lookup."),
    family!("grandparent_lookup", L2, ["plookup2"], ["plookup2_"], "Two FK hops; unlike the former stage naming, tier counts this dependence."),
    family!("recent_lookup", L1, ["plookuprec"], ["plookuprec_"], "Strictly prior same-entity event lookup."),
    family!("transformed_pointer_h1", L1, ["plookupt_h1"], ["plookupt_r_h1_", "plookupt_k2_h1_"], "Nonlinear one-parent lookup plus own feature."),
    family!("transformed_pointer_h2", L2, ["plookupt_h2"], ["plookupt_r_h2_", "plookupt_k2_h2_"], "Nonlinear two-parent-chain lookup plus own feature."),
    family!("transformed_pointer_recent", L1, ["plookupt", "plookupt_recent"], ["plookupt_"], "Nonlinear prior-event lookup plus own feature."),
    family!("child_sum", L1, [], [], "Direct one-hop sum."),
    family!("child_mean", L1, [], [], "Direct one-hop mean."),
    family!("child_count", L1, [], [], "Direct one-hop child count."),
    family!("child_min", L1, [], [], "Direct one-hop minimum."),
    family!("child_max", L1, [], [], "Direct one-hop maximum."),
    family!("child_last", L1, [], [], "Latest one-hop child payload."),
    family!("child_std", L1, [], [], "Direct one-hop dispersion."),
    family!("child_n_unique", L1, [], [], "Direct one-hop distinct-value count."),
    family!("child_median", L1, [], [], "Direct one-hop median."),
    family!("child_mode", L1, [], [], "Direct one-hop mode."),
    family!("recent_child_count", L1, [], [], "Time-windowed one-hop count."),
    family!("grandchild_count", L2, [], [], "Two-hop descendant count."),
    family!("cooccurrence_count", L2, [], [], "Joint-predicate count in a child population."),
    family!("cooccurrence_sum", L2, [], [], "Joint-predicate sum in a related population."),
    family!("recent_cooccurrence_sum", L2, [], [], "Time-legal co-occurrence population sum."),
    family!("recent_cooccurrence_delta", L2, [], [], "Difference of temporal co-occurrence populations."),
    family!("exclusive_vs_joint_gap", L2, [], [], "Compare exclusive and joint populations."),
    family!("joint_to_exclusive_ratio", L2, [], [], "Ratio across joint and exclusive populations."),
    family!("twohop_gated_sum", L3, [], [], "Query gate selects two-hop contributions."),
    family!("exists_grandchild_gate_sum", L3, [], [], "Grandchild existence gates child contributions."),
    family!("bagcbeta", L2, [], ["bagcbeta"], "Beta-Bernoulli latent estimated from a population sample."),
    family!("bagcpois", L2, [], ["bagcpois"], "Gamma-Poisson latent estimated from a population sample."),
    family!("bagcnorm", L2, [], ["bagcnorm"], "Normal-Normal latent estimated from a population sample."),
    family!("bagnsmax", L2, [], ["bagnsmax"], "Population extreme; incomplete bags miss rare values."),
    family!("bagnsq90", L2, [], ["bagnsq90"], "Population tail quantile."),
    family!("bagnsexc", L2, [], ["bagnsexc"], "Population exceedance statistic."),
    family!("hazard", L1, ["haznext", "hazgapreg"], ["haznext_", "hazgapreg_"], "Own event history determines event window or censored gap."),
    family!("qsel", L3, [], ["qsel"], "Query features choose which neighbours contribute."),
    family!("mident", L3, [], ["mident_"], "Support identifies a hidden mixture of bag and pointer mechanisms."),
    family!("routecnt", L2, [], ["routecnt"], "Deduplicated two-hop endpoint population count."),
    family!("routemean", L2, [], ["routemean"], "Deduplicated two-hop endpoint population mean."),
    family!("routeqg", L3, [], ["routeqg"], "Query-conditioned two-hop endpoint selection."),
    family!("comembpeer", L3, [], ["comembpeer_"], "Time-legal co-membership peer label evidence."),
    family!("xfun", L3, [], ["xfun"], "Cross-entity functional response needs peer evidence."),
    family!("xstate", L3, [], ["xstate"], "Shared latent state is inferred through related populations."),
    family!("xroute", L3, ["xroute_k2"], ["xroute_lid_", "xroute_lzi_", "xroute_lha_", "xroute_k2_"], "Own, foreign and bridge routes have a drawn hidden regime."),
    family!("group_rank", L2, ["grankreg_", "mech_group_rank", "group-rank"], ["grankreg_", "grankclf_"], "Compare the focal value to its FK-group population."),
    family!("count", L2, ["cntreg_", "count-ratio", "mech_count_ratio"], ["cntreg_", "cntclf_"], "Published population-count mechanism with count-shaped readout."),
    family!("ratio", L2, ["ratioreg_"], ["ratioreg_", "ratioclf_"], "Population share with an explicit denominator."),
    family!("order_stat", L2, ["osreg_", "order-stat", "mech_order_stat"], ["osreg_", "osclf_"], "Population quantile, top-k, range or value-selected payload."),
    family!("satfeat", L1, [], ["satfeat_r", "satfeat_k2"], "Features reside in one-hop satellite rows."),
    family!("junmix", L2, [], ["junmix_r", "junmix_k2"], "Junction category proportions and entropy."),
    family!("juncount", L2, [], ["juncount_r", "juncount_k2"], "Junction multiplicity statistic."),
    family!("hubhist", L3, [], ["hubhist_r", "hubhist_k2"], "Strictly prior co-member histories reached through a hub."),
    family!("hubnull", L3, [], ["hubnull_r", "hubnull_k2"], "Hidden shuffled co-membership control; observed graph cannot recover it."),
    family!("xhub", L2, [], ["xhub_r", "xhub_k2"], "Interaction between two joined hub attributes."),
    family!("xhub_add", L2, [], ["xhub_add_r", "xhub_add_k2"], "Additive readout of two joined hub attributes."),
    family!("streamrec", L1, [], ["streamrec_r", "streamrec_k2"], "Own prior stream counts and recency."),
    family!("remote_evidence_v1", L3, [], ["remote_evidence_v1_"], "Raw multi-window evidence with delayed availability and local-sufficient controls; exact stratum/hops in receipt."),
    family!("remote_evidence_long_v1", L3, [], ["remote_evidence_long_v1_"], "Opt-in 2/4/6-hop native raw evidence with bounded branches, legal temporal cohorts and receipt-only witnesses."),
    family!("hubrate", L1, [], ["hubrate_r", "hubrate_k2"], "Own exposure table supplies strictly prior outcomes."),
    family!("hubrate_null", L1, [], ["hubrate_null_r", "hubrate_null_k2"], "Same exposure mechanism with shuffled observed outcomes; null control."),
    family!("hubhistj", L3, [], ["hubhistj_r", "hubhistj_k2"], "Peer history requires junction and peer subtree traversal."),
    family!("hubhistj_null", L3, [], ["hubhistj_null_r", "hubhistj_null_k2"], "Shuffled peer-history control retains its latent dependency tier."),
    family!("condrel", L3, [], ["condrel_r", "condrel_k2"], "Parent/relational predicate selects the readout regime."),
    family!("condrel_permuted", L3, [], ["condrel_permuted_r", "condrel_permuted_k2"], "Counterfactual predicate regime control."),
    family!("xrep", L2, [], ["xrep_y_"], "Same relation-derived problem in entity, satellite and bag views."),
    family!("mpnn_mlp", L2, ["mpnn"], ["mpnn_r_mlp_", "mpnn_k2_mlp_"], "Random MLP over own and one-/two-hop messages."),
    family!("mpnn_attn", L2, [], ["mpnn_r_attn_", "mpnn_k2_attn_"], "Drawn attention weights combine one-/two-hop messages."),
    family!("population_law_a", L2, ["poplaw_a"], ["poplaw_a"], "Population count law."),
    family!("population_law_b", L2, ["poplaw_b"], ["poplaw_b"], "Population rate law."),
    family!("population_law_c", L2, ["poplaw_c"], ["poplaw_c"], "Population recency, qualifying count and size law."),
    family!("population_law_long_range", L3, ["poplaw_lr2"], ["poplaw_lr2_"], "Long-range two-hop coverage-limited population law."),
    family!("p2_label_complexity", L3, ["label-complexity", "long-context-scm"], ["p2_y_label_complexity"], "High-dimensional map with parent-keyed regime-specific parameters."),
    family!("p2_long_range", L3, ["long-range-aggregates"], ["p2_y_long_range"], "Two-/three-hop independent innovations require increasing coverage."),
    family!("p2_long_history", L1, ["long-history"], ["p2_y_long_history"], "Long own history supplies trend, seasonality and time since event."),
    family!("p2_peer_subtree", L3, ["peer-subtree-labels"], ["p2_y_peer_subtree"], "Peer covariates are available only inside peer subtrees."),
    family!("p2_peer_event", L3, ["peer-event-labels"], ["p2_y_peer_event"], "Query-conditioned own histories and other entities' rows meet through a shared event parent."),
    family!("p2_v4_long_temporal_state", L3, ["p2v4-long-temporal-state"], ["p2v4_y_long_temporal_state"], "AR/regime/changepoint forecasting from a long strict-past own history."),
    family!("p2_v4_shared_parent_posterior", L3, ["p2v4-shared-parent-posterior"], ["p2v4_y_shared_parent_posterior"], "Sibling evidence identifies an unobserved shared-parent effect."),
    family!("p2_v4_coparticipation_adjustment", L3, ["p2v4-coparticipation-adjustment"], ["p2v4_y_coparticipation_adjustment"], "Past co-participants identify event effects in x=a_i+z_e+epsilon."),
    family!("p2_v4_hierarchical_long_tail", L3, ["p2v4-hierarchical-long-tail"], ["p2v4_y_hierarchical_long_tail"], "Heavy-tailed parent-child-grandchild aggregation."),
    family!("p2_v4_peer_subtree_evidence", L3, ["p2v4-peer-subtree-evidence"], ["p2v4_y_peer_subtree_evidence"], "Peer descendants, not peer roots, identify the label mechanism."),
    family!("p2_v4_query_conditioned_selection", L3, ["p2v4-query-conditioned-selection"], ["p2v4_y_query_conditioned_selection"], "Query attributes select which evidence branch contributes."),
    family!("p2_v5_own_label_history", L1, ["p2v5-own-label-history"], ["p2v5_y_own_label_history"], "The label is an autoregression on the entity's own strictly-past labels."),
    family!("p2_v5_peer_specificity", L3, ["p2v5-peer-specificity"], ["p2v5_y_peer_specificity"], "Only rare hubs carry the latent; budget must be weighted by hub specificity."),
    family!("p2_v5_rate_denominator", L2, ["p2v5-rate-denominator"], ["p2v5_y_rate_denominator"], "Two summed quantities on separate tables; only their ratio identifies the label."),
    family!("p2_v5_cold_start", L1, ["p2v5-cold-start"], ["p2v5_y_cold_start"], "A fixed fraction of entities has no evidence; the answer is an attribute-conditional base rate."),
    family!("p2_v5_breadth_channels", L3, ["p2v5-breadth-channels"], ["p2v5_y_breadth_channels"], "Twenty-four typed child tables each contribute an individually insufficient term."),
    family!("p2_v5_gated_local", L2, ["p2v5-gated-local"], ["p2v5_y_gated_local"], "A query-row quality gate decides whether remote evidence is signal or decoy."),
    family!("hsbm", L2, ["homo"], ["hsbm_"], "Cluster-dependent homophily prior."),
    family!("scm_imb", L2, ["imb"], ["scm_imb"], "Relational SCM score with imbalanced readout."),
    family!("hop", L2, [], ["hop_", "oplabel_"], "Operator library; concrete one-hop arms refine to L1."),
    family!("orth6_path_homo", L0, [], ["orth6_path_homo_"], "Legacy implementation reads focal FK parity; no neighbour fetch is needed."),
    family!("orth6_path_hetero", L2, [], ["orth6_path_hetero_"], "Relation-conditioned heterophily."),
    family!("orth6_recent_count", L1, [], ["orth6_recent_count_"], "Own recent child count."),
    family!("orth6_longterm_avg", L1, [], ["orth6_longterm_avg_"], "Own long-term child average."),
    family!("orth6_latent_compat", L2, [], ["orth6_latent_compat_"], "Latent compatibility across related rows."),
    family!("orth6_text_match", L1, [], ["orth6_text_match_"], "Parent-child text match."),
    family!("paired_alpha", L2, ["paired-alpha"], ["clfα", "regα"], "Relational base diffused over FK siblings."),
    family!("pathhomo", L2, ["path-homo", "paired20-path-homo"], ["pathhomo_", "paired30_pathhomo_", "paired20_pathhomo_"], "Path-conditioned group dependence."),
    family!("aggident", L3, ["agg-ident"], ["aggident_"], "Support identifies the hidden sibling aggregation operator."),
    family!("proto", L2, ["proto-multi"], ["proto_k"], "Feature prototype assignment mixed with relation-block identity."),
    family!("tempk", L2, ["temporal-kernel"], ["tempk"], "Temporal library includes own history and related population terms."),
    family!("tempk_coupled", L3, [], ["tempkcpl", "tempkdual", "tempk2lcpl", "tempk2ldual", "tempk2fcpl", "tempk2fdual"], "Temporal and peer channels are coupled by latent readout weights."),
    family!("bipartiteC", L2, ["bipartite-c", "paired20-bipartite-c"], ["bipartiteC_", "paired30_bipC_", "paired20_bipC_"], "Compatibility of two endpoint group identities."),
    family!("vector_latent", L2, ["vector-latent"], ["vec_"], "Joint depth, aggregation and homophily latent."),
    family!("paired_homo", L2, ["paired20-homo", "paired-homo-kway10", "paired-homo-imb"], ["paired30_homo", "paired20_homo", "phw10_", "phi_p"], "Multiple readouts of relational homophily."),
    family!("paired_agg", L2, ["paired20-agg"], ["paired30_agg_", "paired20_agg_"], "Depth/operator sweep; depth-one arms refine to L1."),
    family!("paired_recency", L1, ["paired20-recency"], ["paired30_recency_", "paired20_recency_"], "Own recency-weighted child history."),
    family!("label_mrf", L3, ["label-mrf"], ["mrflab_"], "Exogenous correlated label field requires peer labels."),
    family!("support_required_multiclass", L3, [], [], "Support labels bind latent class identities."),
    family!("relational_operator_discrete", L2, [], [], "Mixture envelope; concrete operator tier is recorded per target."),
    family!("relational_fl_discrete", L3, [], [], "Feature/label relational support binding."),
    family!("scm_base", L2, [], ["scm_"], "Relational SCM mixture envelope."),
    family!("relational_regression", L2, [], ["regression_target_"], "Relational regression operator mixture."),
    Family { family: "nuisance_text", tier: Tier::L0, aliases: &["nuisance-text"], prefixes: &[], role: "observation_modifier", reason: "Independent text; inherits the attached task tier and creates no label." },
    Family { family: "p2_nuisance_scale", tier: Tier::L0, aliases: &["nuisance-scale"], prefixes: &[], role: "observation_modifier", reason: "Independent rows/columns/text; attached task retains its tier." },
    Family { family: "exact_aggregates", tier: Tier::L2, aliases: &["exact-aggregates", "xagg"], prefixes: &[], role: "evidence_sidecar", reason: "Population aggregate-bank evidence, not a target family." },
];

pub fn family(name: &str) -> Option<&'static Family> {
    let normalized = name.replace(['_', '-'], "").to_ascii_lowercase();
    FAMILIES.iter().find(|f| {
        std::iter::once(&f.family)
            .chain(f.aliases.iter())
            .any(|s| s.replace(['_', '-'], "").to_ascii_lowercase() == normalized)
    })
}

pub fn column_family(name: &str) -> Option<&'static Family> {
    // `aggcol_r` is a target, but `aggcol_rec_*` is an observed recency
    // statistic. Prefix-only classification would destroy this evidence.
    if name.starts_with("aggcol_rec_") {
        return None;
    }
    FAMILIES
        .iter()
        .flat_map(|f| f.prefixes.iter().map(move |p| (f, p)))
        .filter(|(_, p)| name.starts_with(**p))
        .max_by_key(|(_, p)| p.len())
        .map(|(f, _)| f)
}

#[derive(Debug, Clone, Serialize)]
pub struct TargetTier {
    pub table: String,
    pub target: String,
    pub family: String,
    pub legacy_family: Option<String>,
    pub tier: Tier,
    pub assignment: &'static str,
}

pub fn targets(db: &GeneratedDb) -> Result<Vec<TargetTier>> {
    let programs: BTreeMap<_, _> = db
        .program_metadata
        .iter()
        .map(|m| ((m.target_table.as_str(), m.target_col.as_str()), m))
        .collect();
    let mut result = Vec::new();
    for table in &db.tables {
        for (name, _) in &table.columns {
            let metadata = programs.get(&(table.name.as_str(), name.as_str())).copied();
            let definition =
                column_family(name).or_else(|| metadata.and_then(|m| family(&m.family)));
            // Unknown declared programs are surfaced explicitly, never guessed
            // feature-only from a zero hop count (opaque manifests use zero).
            if definition.is_none() && metadata.is_none() {
                continue;
            }
            let (canonical, mut tier, assignment) = match definition {
                Some(f) => (f.family.to_string(), f.tier, "registry"),
                None => (
                    metadata.unwrap().family.clone(),
                    Tier::L3,
                    "unresolved_conservative",
                ),
            };
            if matches!(canonical.as_str(), "hop" | "paired_agg" | "scm_base") {
                if name.contains("_d1_") {
                    tier = Tier::L1;
                }
                if let Some(m) = metadata {
                    if let Some(f) = family(&m.family) {
                        tier = f.tier;
                    } else if m.hops == 1 {
                        tier = Tier::L1;
                    }
                }
            }
            result.push(TargetTier {
                table: table.name.clone(),
                target: name.clone(),
                family: canonical,
                legacy_family: metadata.map(|m| m.family.clone()),
                tier,
                assignment,
            });
        }
    }
    Ok(result)
}

pub fn manifest(db: &GeneratedDb) -> Result<Value> {
    let targets = targets(db)?;
    let task_tables: BTreeSet<_> = targets.iter().map(|t| &t.table).collect();
    let per_entity: BTreeMap<_, _> = db
        .tables
        .iter()
        .map(|parent| {
            let children: Vec<_> = db
                .tables
                .iter()
                .filter(|child| {
                    child
                        .fkey_col_to_pkey_table
                        .values()
                        .any(|p| p == &parent.name)
                })
                .map(|t| t.name.clone())
                .collect();
            (parent.name.clone(), children)
        })
        .collect();
    let mut key_coverage = BTreeMap::new();
    for child in &db.tables {
        for (fk, parent) in &child.fkey_col_to_pkey_table {
            let Some(crate::scm::ColumnData::I64(values)) = child.column(fk) else {
                continue;
            };
            let mut counts = std::collections::HashMap::<i64, usize>::new();
            for &key in values.iter().filter(|&&key| key >= 0) {
                *counts.entry(key).or_default() += 1;
            }
            let mut degrees: Vec<_> = counts.values().copied().collect();
            degrees.sort_unstable();
            key_coverage.insert(format!("{}.{}", child.name, fk), json!({
                "parent_table":parent, "active_parent_keys":counts.len(),
                "available_parent_keys":db.table(parent).map(|t| t.num_rows()),
                "max_fanout":degrees.last().copied().unwrap_or(0),
                "p95_nonempty_fanout":degrees.get(degrees.len().saturating_sub(1) * 95 / 100).copied().unwrap_or(0)}));
        }
    }
    Ok(json!({"schema_version":1, "contract":"dependency_tiers_v1",
        "naming_convention":"L{tier}_{ctx}", "families":FAMILIES, "targets":targets,
        "schema":{"tables":db.tables.len(), "task_tables":task_tables,
            "depth":crate::manifest::corpus_scale_coverage(db, None).depth,
            "multi_parent_tables":db.tables.iter().filter(|t|
                t.fkey_col_to_pkey_table.values().collect::<BTreeSet<_>>().len() >= 2).count(),
            "child_table_types_per_entity":per_entity,
            "fk_key_coverage":key_coverage,
            "primary_key_cardinality":db.tables.iter().map(|t| (t.name.clone(), t.num_rows())).collect::<BTreeMap<_,_>>(),
            "non_target_width":db.tables.iter().map(|t| (t.name.clone(), t.columns.iter().filter(|(name, _)|
                !targets.iter().any(|y| y.table == t.name && y.target == *name)).count())).collect::<BTreeMap<_,_>>()},
        "note":"Tier describes dependence, not context utility; observation modifiers inherit the attached task tier."}))
}

/// Locked manifest records retain their original family names and hashes.
pub fn manifest_for_record(
    db: &GeneratedDb,
    record: &crate::manifest::MixtureRecord,
) -> Result<Value> {
    let mut value = manifest(db)?;
    let definition = family(&record.family)
        .ok_or_else(|| anyhow::anyhow!("unregistered mixture family {}", record.family))?;
    value["mixture_family"] = json!({"family":record.family,"tier":definition.tier,
        "assignment":"family_envelope; concrete targets refine operator mixtures"});
    for target in value["targets"].as_array_mut().unwrap() {
        if target["assignment"] == "unresolved_conservative" {
            target["family"] = json!(record.family);
            target["tier"] = json!(definition.tier);
            target["assignment"] = json!("locked_manifest_family");
        }
    }
    Ok(value)
}
