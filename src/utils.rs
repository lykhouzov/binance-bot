use crate::candlestick::CandleStickData;
use serde::{Deserialize, Deserializer};
use std::str::FromStr;
pub fn from_string<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: FromStr,
    <T as FromStr>::Err: std::fmt::Display,
{
    let s = String::deserialize(deserializer)?;
    <T as FromStr>::from_str(&s).map_err(serde::de::Error::custom)
}

pub fn load_data<P: AsRef<std::path::Path>>(path: P) -> Vec<CandleStickData> {
    let mut rdr = csv::Reader::from_path(path).unwrap();
    rdr.deserialize()
        .map(|x| {
            let v: CandleStickData = x.unwrap();
            v
        })
        .collect()
}

pub fn argmax<const M: usize>(x: &[f32; M]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _v)| i)
        .unwrap()
}