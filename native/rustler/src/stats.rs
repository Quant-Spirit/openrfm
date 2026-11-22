// Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
//! The time cutoff of the training rows and the per-column normalisation statistics fitted on them.
use crate::archive_v1::TableType;
use crate::pre::{ColStat, Table, TableMap};
use polars::prelude::*;
use std::collections::HashMap;

/// The configured validation start, or the earliest TEST timestamp of any task when that is earlier.
pub(crate) fn training_cutoff(table_map: &TableMap, val_time_ns: Option<i64>) -> Option<i64> {
    println!("finding earliest test cutoff across all tasks...");
    let mut earliest: Option<i64> = val_time_ns;
    for ((_, table_type), table) in table_map {
        if table_type != &TableType::Test {
            continue;
        }
        if let Some(Ok(dt)) = table
            .tcol_name
            .as_ref()
            .and_then(|name| table.df.column(name).ok())
            .map(|series| series.datetime())
        {
            for ts in dt.phys.iter().flatten() {
                earliest = Some(earliest.map_or(ts, |curr| curr.min(ts)));
            }
        }
    }
    match earliest {
        Some(earliest) if val_time_ns.map_or(true, |v| earliest < v) => {
            eprintln!(
                "INFO: Using earliest test cutoff across all tasks: {} ns (was {:?})",
                earliest, val_time_ns
            );
            Some(earliest)
        }
        _ => val_time_ns,
    }
}

/// Fit each Db and Train table's column statistics on its rows before `val_time_ns`; Val and Test task tables
/// take those of their Train table.
pub(crate) fn fit_column_stats(table_map: &mut TableMap, val_time_ns: Option<i64>) {
    let mut col_stats_map: HashMap<String, Vec<ColStat>> = HashMap::new();
    for ((table_name, table_type), table) in table_map.iter() {
        if table_type == &TableType::Val || table_type == &TableType::Test {
            continue;
        }
        let mask = train_mask(table, val_time_ns);
        let stats = table.df.iter().map(|col| column_stat(&col.rechunk(), &mask)).collect();
        col_stats_map.insert(table_name.clone(), stats);
    }
    for ((table_name, _), table) in table_map.iter_mut() {
        if let Some(stats) = col_stats_map.get(table_name) {
            table.col_stats = stats.clone();
        } else {
            eprintln!(
                "Warning: No train stats found for table {}. Using default stats.",
                table_name
            );
            table.col_stats = vec![ColStat { mean: 0.0, std: 1.0 }; table.df.width()];
        }
    }
}

fn train_mask(table: &Table, val_time_ns: Option<i64>) -> BooleanChunked {
    let tcol = table.tcol_name.as_ref().and_then(|name| table.df.column(name).ok());
    let mask_values: Vec<bool> = if let (Some(val_time), Some(tcol)) = (val_time_ns, tcol) {
        let tcol_dt = tcol.datetime().unwrap();
        (0..tcol_dt.len())
            .map(|i| tcol_dt.phys.get(i).is_some_and(|ts| ts < val_time))
            .collect()
    } else {
        vec![true; table.df.height()]
    };
    Series::new("mask".into(), mask_values)
        .cast(&DataType::Boolean)
        .unwrap()
        .bool()
        .unwrap()
        .clone()
}

fn column_stat(col: &Series, mask: &BooleanChunked) -> ColStat {
    let train = |drop_nan: bool| {
        let values = col.cast(&DataType::Float64).unwrap().filter(mask).unwrap();
        let values = values.filter(&values.is_not_null()).unwrap();
        if drop_nan { values.filter(&values.is_not_nan().unwrap()).unwrap() } else { values }
    };
    match col.dtype() {
        DataType::Boolean => {
            let values = train(false);
            let std = values.std(1).unwrap_or(0.0);
            ColStat { mean: values.mean().unwrap_or(0.0), std: if std == 0.0 { 1.0 } else { std } }
        }
        DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::Float64
        | DataType::Float32
        | DataType::Datetime(_, _) => {
            if let DataType::Datetime(unit, _) = col.dtype() {
                assert!(*unit == TimeUnit::Nanoseconds);
            }
            let values = train(true);
            let std = values.std(1).unwrap_or(1.0);
            ColStat { mean: values.mean().unwrap_or(0.0), std: if std == 0.0 { 1.0 } else { std } }
        }
        _ => ColStat { mean: 0.0, std: 0.0 },
    }
}
