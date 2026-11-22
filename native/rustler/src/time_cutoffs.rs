// Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;

use anyhow::{Context, Result};
use serde_json::Value;

pub struct TimeCutoffs {
    pub cutoffs: HashMap<String, i64>,
    pub no_temporal_split: HashSet<String>,
}

fn parse_date_string_to_ns(date_str: &str) -> Result<i64> {
    let cleaned = date_str.trim().trim_matches('\'').trim_matches('"');

    // "YYYY-MM-DD", "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DDTHH:MM:SS", else an integer timestamp.
    let formats = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"];

    for format in &formats {
        if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(
            &if cleaned.len() == 10 {
                format!("{} 00:00:00", cleaned)
            } else {
                cleaned.to_string()
            },
            if *format == "%Y-%m-%d" {
                "%Y-%m-%d %H:%M:%S"
            } else {
                format
            },
        ) {
            return Ok(dt.and_utc().timestamp_nanos_opt().unwrap_or(0));
        }
    }

    if let Ok(v) = cleaned.parse::<i64>() {
        return Ok(normalize_timestamp(v));
    }

    anyhow::bail!("Failed to parse date string: {}", date_str)
}

fn normalize_timestamp(v: i64) -> i64 {
    if v < 1_000_000_000_000 {
        v * 1_000_000_000
    } else {
        v
    }
}

pub fn load_time_cutoffs(path_opt: &Option<String>) -> Result<TimeCutoffs> {
    if let Some(path) = path_opt {
        let raw = fs::read_to_string(path).with_context(|| format!("read {path}"))?;
        let json_value: HashMap<String, Value> = serde_json::from_str(&raw)?;

        let mut cutoffs = HashMap::new();
        let mut no_temporal_split = HashSet::new();

        for (key, value) in json_value {
            let timestamp_ns = match value {
                Value::Number(n) => {
                    if let Some(v) = n.as_i64() {
                        Some(normalize_timestamp(v))
                    } else {
                        anyhow::bail!("Invalid number for key {}: {}", key, n)
                    }
                }
                Value::String(s) => {
                    // An empty string marks a database without a temporal split.
                    if s.trim().is_empty() {
                        no_temporal_split.insert(key.clone());
                        None
                    } else {
                        Some(parse_date_string_to_ns(&s)?)
                    }
                }
                _ => anyhow::bail!("Invalid value type for key {}", key),
            };
            if let Some(ts) = timestamp_ns {
                cutoffs.insert(key, ts);
            }
        }

        Ok(TimeCutoffs {
            cutoffs,
            no_temporal_split,
        })
    } else {
        Ok(TimeCutoffs {
            cutoffs: HashMap::new(),
            no_temporal_split: HashSet::new(),
        })
    }
}
