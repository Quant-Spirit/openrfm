// Derived from stanford-star/relational-transformer (Apache-2.0); modified by the OpenRFM Authors. See NOTICE.
use crate::archive_v1::{Adj, Node, SemType, TableType, unpack_discrete_label};
use anyhow::{Result, ensure};
use half::bf16;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

fn hash(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut digest = Sha256::new();
    let mut block = [0u8; 65536];
    loop {
        let n = reader.read(&mut block)?;
        if n == 0 {
            break;
        }
        digest.update(&block[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn int_array<I>(root: &Path, name: &str, values: I) -> Result<Value>
where
    I: Iterator<Item = i64> + Clone,
{
    let mut lo = i64::MAX;
    let mut hi = i64::MIN;
    let mut count = 0usize;
    for value in values.clone() {
        lo = lo.min(value);
        hi = hi.max(value);
        count += 1;
    }
    if count == 0 {
        lo = 0;
        hi = 0;
    }
    let (dtype, width) = if lo >= 0 && hi <= 255 {
        ("uint8", 1)
    } else if lo >= 0 && hi <= 65535 {
        ("uint16", 2)
    } else if lo >= i32::MIN as i64 && hi <= i32::MAX as i64 {
        ("int32", 4)
    } else {
        ("int64", 8)
    };
    let path = root.join(format!("{name}.bin"));
    let mut writer = BufWriter::new(File::create(&path)?);
    for value in values {
        writer.write_all(&value.to_le_bytes()[..width])?;
    }
    writer.flush()?;
    Ok(
        json!({"file":format!("{name}.bin"),"dtype":dtype,"shape":[count],
        "nbytes":count*width,"sha256":hash(&path)?,"min":lo,"max":hi}),
    )
}

fn float_array<I: Iterator<Item = f32>>(root: &Path, name: &str, values: I) -> Result<Value> {
    let path = root.join(format!("{name}.bin"));
    let mut writer = BufWriter::new(File::create(&path)?);
    let mut count = 0usize;
    for value in values {
        let encoded = bf16::from_f32(value);
        ensure!(
            !value.is_finite() || encoded.is_finite(),
            "finite payload overflows bf16"
        );
        writer.write_all(&encoded.to_bits().to_le_bytes())?;
        count += 1;
    }
    writer.flush()?;
    Ok(
        json!({"file":format!("{name}.bin"),"dtype":"bf16","shape":[count],
        "nbytes":count*2,"sha256":hash(&path)?}),
    )
}

fn f32_array<I: Iterator<Item = f32>>(root: &Path, name: &str, values: I) -> Result<Value> {
    let path = root.join(format!("{name}.bin"));
    let mut writer = BufWriter::new(File::create(&path)?);
    let mut count = 0usize;
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
        count += 1;
    }
    writer.flush()?;
    Ok(
        json!({"file":format!("{name}.bin"),"dtype":"float32","shape":[count],
        "nbytes":count*4,"sha256":hash(&path)?}),
    )
}

fn table_type(t: &TableType) -> i64 {
    match t {
        TableType::Db => 0,
        TableType::Train => 1,
        TableType::Val => 2,
        TableType::Test => 3,
    }
}

fn sem_type(t: &SemType) -> i64 {
    match t {
        SemType::Number => 0,
        SemType::Text => 1,
        SemType::DateTime => 2,
        SemType::Boolean => 3,
        SemType::Discrete => 4,
    }
}

/// Writes `<root>/v2`; `fp32_datetimes` keeps cell datetimes in fp32 instead of bf16.
pub fn write(
    root: &Path,
    nodes: &[Node],
    p2f: &Adj,
    model: &str,
    tag: &str,
    fp32_datetimes: bool,
) -> Result<()> {
    ensure!(
        !nodes.is_empty() && p2f.adj.len() == nodes.len(),
        "invalid v2 row partition"
    );
    for node in nodes {
        let n = node.col_name_idxs.len();
        ensure!(
            [
                node.sem_types.len(),
                node.number_values.len(),
                node.datetime_values.len(),
                node.boolean_values.len(),
                node.text_values.len(),
                node.class_value_idx.len()
            ]
            .iter()
            .all(|&len| len == n),
            "unaligned v2 cell payloads"
        );
        for (typ, &code) in node.sem_types.iter().zip(&node.class_value_idx) {
            if matches!(typ, SemType::Discrete) {
                unpack_discrete_label(code).map_err(anyhow::Error::msg)?;
            }
        }
    }
    let output = root.join("v2");
    ensure!(
        !output.exists(),
        "v2 archive already exists; refusing overwrite"
    );
    let staging = root.join(".v2-writing");
    ensure!(
        !staging.exists(),
        "interrupted .v2-writing must be inspected before retry"
    );
    fs::create_dir(&staging)?;
    let mut metadata = Map::new();
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "json") {
            let name = path.file_name().unwrap().to_str().unwrap();
            fs::copy(&path, staging.join(name))?;
            metadata.insert(name.to_string(), json!(hash(&path)?));
        }
    }
    let mut arrays = Map::new();
    macro_rules! ints {
        ($name:literal, $values:expr) => {
            arrays.insert($name.into(), int_array(&staging, $name, $values)?);
        };
    }
    macro_rules! floats {
        ($name:literal, $field:ident) => {
            arrays.insert(
                $name.into(),
                float_array(
                    &staging,
                    $name,
                    nodes.iter().flat_map(|n| n.$field.iter().copied()),
                )?,
            );
        };
    }
    ints!(
        "node_timestamps",
        nodes.iter().map(|n| n.timestamp.unwrap_or(i32::MIN) as i64)
    );
    ints!(
        "node_table_name_idxs",
        nodes.iter().map(|n| n.table_name_idx as i64)
    );
    // scan has O(1) auxiliary memory and is cloneable for the width pass.
    ints!(
        "f2p_rowptr",
        std::iter::once(0i64).chain(nodes.iter().scan(0i64, |s, n| {
            *s += n.f2p_edges.len() as i64;
            Some(*s)
        }))
    );
    ints!(
        "f2p_col_idx",
        nodes
            .iter()
            .flat_map(|n| n.f2p_edges.iter().map(|e| e.node_idx as i64))
    );
    ints!(
        "p2f_rowptr",
        std::iter::once(0i64).chain(p2f.adj.iter().scan(0i64, |s, e| {
            *s += e.len() as i64;
            Some(*s)
        }))
    );
    ints!(
        "p2f_col_idx",
        p2f.adj.iter().flatten().map(|e| e.node_idx as i64)
    );
    ints!(
        "p2f_table_name_idxs",
        p2f.adj.iter().flatten().map(|e| e.table_name_idx as i64)
    );
    ints!(
        "p2f_table_types",
        p2f.adj.iter().flatten().map(|e| table_type(&e.table_type))
    );
    ints!(
        "p2f_timestamps",
        p2f.adj
            .iter()
            .flatten()
            .map(|e| e.timestamp.unwrap_or(i32::MIN) as i64)
    );
    ints!(
        "cell_nodes",
        nodes
            .iter()
            .enumerate()
            .flat_map(|(i, n)| std::iter::repeat(i as i64).take(n.col_name_idxs.len()))
    );
    ints!(
        "cell_columns",
        nodes
            .iter()
            .flat_map(|n| n.col_name_idxs.iter().map(|&v| v as i64))
    );
    ints!(
        "cell_types",
        nodes.iter().flat_map(|n| n.sem_types.iter().map(sem_type))
    );
    ints!(
        "cell_text_ids",
        nodes
            .iter()
            .flat_map(|n| n.text_values.iter().map(|&v| v as i64))
    );
    ints!(
        "cell_codes",
        nodes.iter().flat_map(
            |n| n.sem_types.iter().zip(&n.class_value_idx).map(|(t, &v)| {
                if matches!(t, SemType::Discrete) {
                    unpack_discrete_label(v).unwrap().1 as i64
                } else {
                    v as i64
                }
            })
        )
    );
    ints!(
        "cell_num_classes",
        nodes.iter().flat_map(
            |n| n.sem_types.iter().zip(&n.class_value_idx).map(|(t, &v)| {
                if matches!(t, SemType::Discrete) {
                    unpack_discrete_label(v).unwrap().0 as i64
                } else {
                    -1
                }
            })
        )
    );
    floats!("cell_numbers", number_values);
    if fp32_datetimes {
        arrays.insert(
            "cell_datetimes".into(),
            f32_array(
                &staging,
                "cell_datetimes",
                nodes.iter().flat_map(|n| n.datetime_values.iter().copied()),
            )?,
        );
    } else {
        floats!("cell_datetimes", datetime_values);
    }
    floats!("cell_booleans", boolean_values);
    let text_map: std::collections::HashMap<String, i32> =
        serde_json::from_reader(File::open(root.join("text_map.json"))?)?;
    let embedding = root.join(format!("text_emb_{tag}.bin"));
    let bytes = embedding.metadata()?.len();
    ensure!(
        !text_map.is_empty() && bytes > 0 && bytes % (2 * text_map.len() as u64) == 0,
        "v2 requires completed bf16 embeddings; deferred/empty encoding is not publishable"
    );
    let dimension = bytes / (2 * text_map.len() as u64);
    // Already bf16 and a private inode (see `text_output::write_text_embeddings`): move it into the bundle.
    fs::rename(&embedding, staging.join("embedding_bits.bin"))?;
    arrays.insert(
        "embedding_bits".into(),
        json!({"file":"embedding_bits.bin","dtype":"bf16",
        "shape":[bytes/2],"nbytes":bytes,"sha256":hash(&staging.join("embedding_bits.bin"))?}),
    );
    let manifest = json!({"archive_format":"v2","byte_order":"little","num_nodes":nodes.len(),
        "embedding_model":model,"d_text":dimension,"arrays":arrays,"metadata_sha256":metadata});
    let mut writer = BufWriter::new(File::create(staging.join("archive_manifest.json"))?);
    serde_json::to_writer_pretty(&mut writer, &manifest)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    fs::rename(staging, output)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widths_and_bf16_bytes_match_manifest() {
        let temp = tempfile::tempdir().unwrap();
        for (values, dtype) in [
            (vec![0, 255], "uint8"),
            (vec![256, 65535], "uint16"),
            (vec![-1, 200], "int32"),
            (vec![0, 1i64 << 31], "int64"),
        ] {
            let record = int_array(temp.path(), "index", values.into_iter()).unwrap();
            assert_eq!(record["dtype"], dtype);
            assert_eq!(
                record["sha256"],
                hash(&temp.path().join("index.bin")).unwrap()
            );
        }
        let values = [1.00390625f32, -0.33333];
        let record = float_array(temp.path(), "numbers", values.into_iter()).unwrap();
        assert_eq!(record["dtype"], "bf16");
        assert_eq!(record["nbytes"], 4);
        let expected: Vec<u8> = values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect();
        assert_eq!(fs::read(temp.path().join("numbers.bin")).unwrap(), expected);
    }

    #[test]
    fn direct_writer_publishes_complete_v2_without_rkyv() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        fs::write(
            root.join("table_info.json"),
            r#"{"t:Db":{"node_idx_offset":0,"num_nodes":1}}"#,
        )
        .unwrap();
        fs::write(root.join("text_map.json"), r#"{"t":0}"#).unwrap();
        fs::write(root.join("column_index.json"), r#"{"x":0}"#).unwrap();
        fs::write(root.join("text_emb_test.bin"), [0u8; 8]).unwrap();
        let node = Node {
            col_name_idxs: vec![0],
            sem_types: vec![SemType::Number],
            number_values: vec![1.00390625],
            datetime_values: vec![0.0],
            boolean_values: vec![0.0],
            text_values: vec![-1],
            class_value_idx: vec![-1],
            ..Node::default()
        };
        write(root, &[node], &Adj { adj: vec![vec![]] }, "test", "test", false).unwrap();
        let output = root.join("v2");
        let manifest: Value =
            serde_json::from_reader(File::open(output.join("archive_manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["archive_format"], "v2");
        assert_eq!(manifest["arrays"].as_object().unwrap().len(), 19);
        assert_eq!(manifest["arrays"]["cell_text_ids"]["dtype"], "int32");
        assert_eq!(manifest["arrays"]["cell_numbers"]["dtype"], "bf16");
        assert!(!root.join("nodes.rkyv").exists());
        assert!(!root.join("text_emb_test.bin").exists());
        assert_eq!(
            fs::read(output.join("embedding_bits.bin")).unwrap(),
            vec![0u8; 8]
        );
    }
}
