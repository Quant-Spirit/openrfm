// Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
//! Turns the normalised tables into one node per row (its cells and foreign-key edges) and the parent-to-child edges.
use crate::archive_v1::{Adj, Edge, Node, SemType, TableType, pack_discrete_label};
use crate::pre::{ColStat, Table, TableMap};
use indicatif::ProgressBar;
use polars::prelude::*;
use std::collections::HashMap;

/// The nodes and edges of a database, its text vocabulary (text -> id, in order of first use) and the column ids.
pub(crate) struct Graph {
    pub(crate) nodes: Vec<Node>,
    pub(crate) p2f: Adj,
    pub(crate) text_to_idx: HashMap<String, i32>,
    pub(crate) column_index: Vec<(String, i32)>,
}

fn text_id(text_to_idx: &mut HashMap<String, i32>, text: &str) -> i32 {
    let next = text_to_idx.len() as i32;
    *text_to_idx.entry(text.to_string()).or_insert(next)
}

fn timestamp_seconds(table: &Table, row: usize) -> Option<i32> {
    let name = table.tcol_name.as_ref()?;
    table.df.column(name).unwrap().datetime().unwrap().phys.get(row).map(|v| (v / 1_000_000_000) as i32)
}

pub(crate) fn build_graph(table_map: &TableMap, num_rows: i32, pbar: &ProgressBar) -> Graph {
    let mut graph = Graph {
        nodes: (0..num_rows).map(|_| Node::default()).collect(),
        p2f: Adj { adj: vec![Vec::new(); num_rows as usize] },
        text_to_idx: HashMap::new(),
        column_index: Vec::new(),
    };
    for ((_, table_type), table) in table_map {
        let table_name_idx = text_id(&mut graph.text_to_idx, &table.table_name);
        for col_idx in 0..table.df.width() {
            let series = table.df.select_at_idx(col_idx).unwrap().clone().rechunk();
            let col_name = format!("{} of {}", series.name(), table.table_name);
            let next = graph.text_to_idx.len() as i32;
            let col_name_idx = *graph.text_to_idx.entry(col_name.clone()).or_insert_with(|| {
                graph.column_index.push((col_name.clone(), next));
                next
            });
            let raw_name = series.name().to_string();
            // Foreign keys first: a column that is both a key and a foreign key still gets its edges.
            if let Some(ptable_name) = table.fcol_name_to_ptable_name.get(&raw_name) {
                foreign_key_edges(&mut graph, table_map, table, table_type, table_name_idx, &series, ptable_name, pbar);
            } else if raw_name == table.pcol_name.as_deref().unwrap_or("") {
                pbar.inc(series.len() as u64);
            } else {
                value_cells(&mut graph, table, table_type, table_name_idx, col_name_idx, &series, &table.col_stats[col_idx], pbar);
            }
        }
    }
    graph
}

fn row_node<'a>(nodes: &'a mut [Node], table: &Table, table_type: &TableType, table_name_idx: i32, row: usize) -> &'a mut Node {
    let node_idx = table.node_idx_offset + row as i32;
    let node = &mut nodes[node_idx as usize];
    node.is_task_node = table_type != &TableType::Db;
    node.node_idx = node_idx;
    node.table_name_idx = table_name_idx;
    node
}

#[allow(clippy::too_many_arguments)]
fn foreign_key_edges(
    graph: &mut Graph,
    table_map: &TableMap,
    table: &Table,
    table_type: &TableType,
    table_name_idx: i32,
    series: &Column,
    ptable_name: &str,
    pbar: &ProgressBar,
) {
    let ptable = table_map.get(&(ptable_name.to_string(), TableType::Db)).unwrap_or_else(|| {
        dbg!(ptable_name.to_string());
        dbg!(table_map.keys());
        panic!()
    });
    if matches!(series.get(0).unwrap(), AnyValue::List(_)) {
        eprintln!(
            "Skipping List-type column '{}' (link prediction task not supported in preprocessing)",
            series.name()
        );
        pbar.inc(series.len() as u64);
        return;
    }
    for r in 0..series.len() {
        let val = series.get(r).unwrap();
        pbar.inc(1);
        if val == AnyValue::Null {
            continue;
        }
        let AnyValue::Int64(val) = val else {
            dbg!(val);
            dbg!(series.name());
            panic!()
        };
        let pnode_idx = ptable.node_idx_offset + val as i32;
        let timestamp = timestamp_seconds(table, r);
        let ptable_name_idx = text_id(&mut graph.text_to_idx, ptable_name);
        let node = row_node(&mut graph.nodes, table, table_type, table_name_idx, r);
        node.f2p_nbr_idxs.push(pnode_idx);
        node.timestamp = timestamp;
        node.f2p_edges.push(Edge {
            node_idx: pnode_idx,
            table_name_idx: ptable_name_idx,
            table_type: TableType::Db,
            timestamp: timestamp_seconds(ptable, val as usize),
        });
        let node_idx = node.node_idx;
        graph.p2f.adj[pnode_idx as usize].push(Edge {
            node_idx,
            table_name_idx,
            table_type: table_type.clone(),
            timestamp,
        });
    }
}

/// The value of a numeric cell, or None for a NaN (a missing cell) or a non-numeric value.
fn number(val: &AnyValue) -> Option<f64> {
    let value = match *val {
        AnyValue::UInt8(v) => v as f64,
        AnyValue::UInt16(v) => v as f64,
        AnyValue::UInt32(v) => v as f64,
        AnyValue::UInt64(v) => v as f64,
        AnyValue::Int8(v) => v as f64,
        AnyValue::Int16(v) => v as f64,
        AnyValue::Int32(v) => v as f64,
        AnyValue::Int64(v) => v as f64,
        AnyValue::Float32(v) => v as f64,
        AnyValue::Float64(v) => v,
        _ => return None,
    };
    (!value.is_nan()).then_some(value)
}

/// One cell: its semantic type, the value in the channel of that type (zero elsewhere) and the class/text code.
fn push_cell(node: &mut Node, sem: SemType, value: f32, text: i32, class: i32, col_name_idx: i32) {
    node.boolean_values.push(if matches!(sem, SemType::Boolean) { value } else { 0.0 });
    node.number_values.push(if matches!(sem, SemType::Number) { value } else { 0.0 });
    node.text_values.push(text);
    node.datetime_values.push(if matches!(sem, SemType::DateTime) { value } else { 0.0 });
    node.sem_types.push(sem);
    node.col_name_idxs.push(col_name_idx);
    node.class_value_idx.push(class);
}

#[allow(clippy::too_many_arguments)]
fn value_cells(
    graph: &mut Graph,
    table: &Table,
    table_type: &TableType,
    table_name_idx: i32,
    col_name_idx: i32,
    series: &Column,
    stat: &ColStat,
    pbar: &ProgressBar,
) {
    let std = if stat.std == 0.0 { 1.0 } else { stat.std };
    let normalized = |value: f64| ((value - stat.mean) / std) as f32;
    let discrete_num_classes = table.discrete_target_num_classes.get(series.name().as_str()).copied();
    for r in 0..series.len() {
        pbar.inc(1);
        let val = series.get(r).unwrap();
        let text = match &val {
            AnyValue::String(text) => Some(text_id(&mut graph.text_to_idx, text)),
            _ => None,
        };
        let node = row_node(&mut graph.nodes, table, table_type, table_name_idx, r);
        match val {
            AnyValue::Null => {}
            AnyValue::Boolean(v) => {
                push_cell(node, SemType::Boolean, normalized(if v { 1.0 } else { 0.0 }), 0, -1, col_name_idx)
            }
            AnyValue::UInt32(v) if discrete_num_classes.is_some() => {
                let classes = discrete_num_classes.expect("discrete target column must have validated cardinality");
                let code = pack_discrete_label(classes, v as i32).expect("validated discrete label must pack");
                push_cell(node, SemType::Discrete, 0.0, 0, code, col_name_idx)
            }
            AnyValue::Datetime(v, unit, _) => {
                assert_eq!(unit, TimeUnit::Nanoseconds);
                push_cell(node, SemType::DateTime, normalized(v as f64), 0, -1, col_name_idx)
            }
            AnyValue::String(_) => {
                let text = text.unwrap();
                push_cell(node, SemType::Text, 0.0, text, text, col_name_idx)
            }
            ref other => match number(other) {
                Some(value) => push_cell(node, SemType::Number, normalized(value), 0, -1, col_name_idx),
                None if matches!(other, AnyValue::Float32(_) | AnyValue::Float64(_)) => {}
                None => {
                    dbg!(&table.table_name);
                    dbg!(series.name());
                    dbg!(other);
                    panic!()
                }
            },
        }
    }
}
