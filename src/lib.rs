#![feature(generic_const_exprs)]

#[allow(incomplete_features)]


pub mod agent;
pub mod candlestick;
pub mod dataset;
#[cfg(feature = "surrealdb")]
pub mod db;
pub mod environment;
// pub mod nn;
pub mod utils;
pub mod ws;

use std::str::FromStr;

use serde::{Deserialize, Serialize};

use utils::from_string;
// {
//     "e": "kline",     // Event type
//     "E": 123456789,   // Event time
//     "s": "BNBBTC",    // Symbol
//     "k": {
//       "t": 123400000, // Kline start time
//       "T": 123460000, // Kline close time
//       "s": "BNBBTC",  // Symbol
//       "i": "1m",      // Interval
//       "f": 100,       // First trade ID
//       "L": 200,       // Last trade ID
//       "o": "0.0010",  // Open price
//       "c": "0.0020",  // Close price
//       "h": "0.0025",  // High price
//       "l": "0.0015",  // Low price
//       "v": "1000",    // Base asset volume
//       "n": 100,       // Number of trades
//       "x": false,     // Is this kline closed?
//       "q": "1.0000",  // Quote asset volume
//       "V": "500",     // Taker buy base asset volume
//       "Q": "0.500",   // Taker buy quote asset volume
//       "B": "123456"   // Ignore
//     }
//   }
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct Kline<T, V>
where
    V: FromStr + for<'d> serde::Deserialize<'d>,
    <V as FromStr>::Err: std::fmt::Display + std::fmt::Debug,
{
    pub open_time: T,
    #[serde(deserialize_with = "from_string")]
    pub open: V,
    #[serde(deserialize_with = "from_string")]
    pub high: V,
    #[serde(deserialize_with = "from_string")]
    pub low: V,
    #[serde(deserialize_with = "from_string")]
    pub close: V,
    #[serde(deserialize_with = "from_string")]
    pub volume: V,
    pub close_time: T,
    #[serde(deserialize_with = "from_string")]
    pub quote_asset_volume: V,
    // #[serde(deserialize_with = "from_string")]
    pub number_of_trades: T,
    #[serde(deserialize_with = "from_string")]
    pub taker_buy_base_asset_volume: V,
    #[serde(deserialize_with = "from_string")]
    pub taker_buy_quote_asset_volume: V,
    #[serde(skip_serializing, default)]
    _ignore: Option<String>,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum Interval {
    S1,
    M1,
    M3,
    M5,
    M15,
    M30,
    H1,
    H2,
    H4,
    H6,
    H8,
    H12,
    #[default]
    D1,
    D3,
    W1,
    Month,
}

impl std::fmt::Display for Interval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::S1 => "1s",
            Self::M1 => "1m",
            Self::M3 => "3m",
            Self::M5 => "5m",
            Self::M15 => "15m",
            Self::M30 => "30m",
            Self::H1 => "1h",
            Self::H2 => "2h",
            Self::H4 => "4h",
            Self::H6 => "6h",
            Self::H8 => "8h",
            Self::H12 => "12h",
            Self::D1 => "1d",
            Self::D3 => "3d",
            Self::W1 => "1w",
            Self::Month => "1M",
        };
        s.fmt(f)
    }
}
impl std::str::FromStr for Interval {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "1s" => Ok(Interval::S1),
            "1m" => Ok(Interval::M1),
            "3m" => Ok(Interval::M3),
            "5m" => Ok(Interval::M5),
            "15m" => Ok(Interval::M15),
            "30m" => Ok(Interval::M30),
            "1h" => Ok(Interval::H1),
            "2h" => Ok(Interval::H2),
            "4h" => Ok(Interval::H4),
            "6h" => Ok(Interval::H6),
            "8h" => Ok(Interval::H8),
            "12h" => Ok(Interval::H12),
            "1d" => Ok(Interval::D1),
            "3d" => Ok(Interval::D3),
            "1w" => Ok(Interval::W1),
            "1M" => Ok(Interval::Month),
            _ => Err(format!("Unknown time interval: {s}")),
        }
    }
}
