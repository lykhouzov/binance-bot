pub mod utils;
pub mod candlestick;

use serde::Deserialize;

use utils::from_string;

#[derive(Debug, Deserialize)]
pub struct Kline {
    pub open_time: u64,
    #[serde(deserialize_with = "from_string")]
    pub open: f64,
    #[serde(deserialize_with = "from_string")]
    pub high: f64,
    #[serde(deserialize_with = "from_string")]
    pub low: f64,
    #[serde(deserialize_with = "from_string")]
    pub close: f64,
    #[serde(deserialize_with = "from_string")]
    pub volume: f64,
    pub close_time: u64,
    _qav: String,
    _nt: u32,
    _tbb: String,
    _tbq: String,
    _ignore: String,
}
