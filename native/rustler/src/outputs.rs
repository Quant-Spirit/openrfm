// Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
//! The JSON metadata of an archive (column index, table info, column statistics) and the format-v1 node files.
use crate::archive_v1::{Adj, CsrAdj, Node, Offsets, TableInfo, TableType};
use crate::pre::{PBAR_TEMPLATE, TableMap};
use crate::text_output::output_path;
use indicatif::{ProgressBar, ProgressStyle};
use polars::prelude::*;
use rkyv::rancor::Error;
use serde::Serialize;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Seek, Write};

/// Serialized through `serde_json::Value`, whose maps are sorted: the file does not depend on hash-map order.
fn write_json<T: Serialize>(pre_path: &str, stem: &str, value: &T) {
    let mut writer = BufWriter::new(fs::File::create(output_path(pre_path, stem, "json")).unwrap());
    serde_json::to_writer(&mut writer, &serde_json::to_value(value).unwrap()).unwrap();
    writer.flush().unwrap();
}

pub(crate) fn write_column_index(pre_path: &str, column_index: Vec<(String, i32)>) {
    let column_index: HashMap<String, i32> = column_index.into_iter().collect();
    write_json(pre_path, "column_index", &column_index);
}

fn infer_column_types(df: &DataFrame) -> HashMap<String, String> {
    df.get_columns()
        .iter()
        .map(|col| {
            let ty = match col.dtype() {
                DataType::Boolean => "boolean",
                dt if dt.is_categorical() || matches!(dt, DataType::List(_)) => "categorical",
                DataType::Datetime(_, _) => "timestamp",
                DataType::String => "text_embedded",
                _ => "numerical",
            };
            (col.name().to_string(), ty.to_string())
        })
        .collect()
}

pub(crate) fn write_table_info(pre_path: &str, table_map: &TableMap) {
    let table_info: HashMap<String, TableInfo> = table_map
        .iter()
        .map(|(table_key, table)| {
            (
                format!("{}:{:?}", table_key.0, table_key.1),
                TableInfo {
                    node_idx_offset: table.node_idx_offset,
                    num_nodes: table.df.height() as i32,
                    pkey_col: table.pcol_name.clone(),
                    fkey_col_to_pkey_table: table.fcol_name_to_ptable_name.clone(),
                    column_types: infer_column_types(&table.df),
                    column_aliases: HashMap::new(),
                    column_display_overrides: HashMap::new(),
                },
            )
        })
        .collect();
    write_json(pre_path, "table_info", &table_info);
}

/// The statistics of the Db and Train tables (Val and Test task tables use their Train table's).
pub(crate) fn write_col_stats(pre_path: &str, table_map: &TableMap, db_name: &str, time_cutoff_ns: Option<i64>,
                              is_no_temporal_split: bool) {
    let mut tables: HashMap<String, Vec<serde_json::Value>> = HashMap::new();
    for ((table_name, table_type), table) in table_map {
        if table_type == &TableType::Val || table_type == &TableType::Test {
            continue;
        }
        let column_names = table.df.get_column_names();
        let stats = table
            .col_stats
            .iter()
            .enumerate()
            .map(|(i, stat)| {
                let column = column_names.get(i).map(|s| s.to_string()).unwrap_or_else(|| format!("col_{}", i));
                json!({"column": column, "mean": stat.mean, "std": stat.std})
            })
            .collect();
        tables.insert(format!("{}:{:?}", table_name, table_type), stats);
    }
    let metadata = json!({
        "time_cutoff_ns": time_cutoff_ns,
        "is_no_temporal_split": is_no_temporal_split,
        "db_name": db_name,
        "tables": tables
    });
    let mut writer = BufWriter::new(fs::File::create(output_path(pre_path, "col_stats", "json")).unwrap());
    serde_json::to_writer_pretty(&mut writer, &metadata).unwrap();
    writer.flush().unwrap();
}

fn write_rkyv(pre_path: &str, stem: &str, bytes: &[u8]) {
    let mut writer = BufWriter::new(fs::File::create(output_path(pre_path, stem, "rkyv")).unwrap());
    writer.write_all(bytes).unwrap();
}

/// The format-v1 archive: rkyv nodes with their byte offsets, the parent-to-child edges and their CSR form.
pub(crate) fn write_v1(pre_path: &str, nodes: Vec<Node>, p2f: &Adj) {
    let pbar = ProgressBar::new(nodes.len() as u64).with_style(ProgressStyle::default_bar().template(PBAR_TEMPLATE).unwrap());
    println!("writing out nodes...");
    let mut offsets = vec![0];
    let mut writer = BufWriter::new(fs::File::create(output_path(pre_path, "nodes", "rkyv")).unwrap());
    for node in nodes {
        writer.write_all(&rkyv::to_bytes::<Error>(&node).unwrap()).unwrap();
        offsets.push(writer.stream_position().unwrap() as i64);
        pbar.inc(1);
    }
    pbar.finish();
    println!("writing out offsets...");
    write_rkyv(pre_path, "offsets", &rkyv::to_bytes::<Error>(&Offsets { offsets }).unwrap());
    println!("writing out p2f_adj...");
    write_rkyv(pre_path, "p2f_adj", &rkyv::to_bytes::<Error>(p2f).unwrap());
    println!("converting p2f_adj to CSR format...");
    let num_nodes = p2f.adj.len();
    let total_edges: usize = p2f.adj.iter().map(|edges| edges.len()).sum();
    let mut csr = CsrAdj {
        rowptr: Vec::with_capacity(num_nodes + 1),
        col_idx: Vec::with_capacity(total_edges),
        table_name_idxs: Vec::with_capacity(total_edges),
        table_types: Vec::with_capacity(total_edges),
        timestamps: Vec::with_capacity(total_edges),
    };
    let mut offset: i64 = 0;
    for edges in &p2f.adj {
        csr.rowptr.push(offset);
        for edge in edges {
            csr.col_idx.push(edge.node_idx);
            csr.table_name_idxs.push(edge.table_name_idx);
            csr.table_types.push(edge.table_type.clone());
            csr.timestamps.push(edge.timestamp.unwrap_or(-1));
        }
        offset += edges.len() as i64;
    }
    csr.rowptr.push(offset);
    println!("writing out p2f_adj_csr ({} nodes, {} edges)...", num_nodes, total_edges);
    write_rkyv(pre_path, "p2f_adj_csr", &rkyv::to_bytes::<Error>(&csr).unwrap());
}
