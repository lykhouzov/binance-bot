use std::{fmt::Display, str::FromStr};

use serde::Deserialize;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct KlineResponse {
    stream: String,
    data: KlineData,
}
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct KlineData {
    #[serde(rename = "e")]
    kind: String,
    #[serde(rename = "E")]
    time: i64,
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "k")]
    kandle: Kandle,
}
#[derive(Default, Debug, serde::Serialize, serde::Deserialize)]
pub struct Kandle {
    #[serde(rename = "t")]
    open_time: i64,
    #[serde(rename = "T")]
    close_time: i64,
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "i")]
    interval: String,
    #[serde(rename = "f")]
    fisrt_trade_id: i64,
    #[serde(rename = "L")]
    last_trade_id: i64,
    #[serde(rename = "o", deserialize_with = "from_str")]
    open: f64,
    #[serde(rename = "c", deserialize_with = "from_str")]
    close: f64,
    #[serde(rename = "h", deserialize_with = "from_str")]
    high: f64,
    #[serde(rename = "l", deserialize_with = "from_str")]
    low: f64,
    #[serde(rename = "v", deserialize_with = "from_str")]
    volume: f64,
    #[serde(rename = "n")]
    num_trades: i64,
    #[serde(rename = "x")]
    is_closed: bool,
    #[serde(rename = "q", deserialize_with = "from_str")]
    quote_asset_volume: f64,
    #[serde(rename = "V", deserialize_with = "from_str")]
    taker_buy_base_volume: f64,
    #[serde(rename = "Q", deserialize_with = "from_str")]
    taker_buy_quote_volume: f64,
}
fn from_str<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    T: FromStr,
    T::Err: Display,
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    T::from_str(&s).map_err(serde::de::Error::custom)
}
