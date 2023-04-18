use crate::utils::from_string;
#[derive(serde::Deserialize, Debug, PartialEq, Default)]
pub struct Indicators{
    #[serde(deserialize_with = "from_string")]
    pub sma_15: f32,
    #[serde(deserialize_with = "from_string")]
    pub sma_30: f32,
    #[serde(deserialize_with = "from_string")]
    pub sma_60: f32,
    #[serde(deserialize_with = "from_string")]
    pub sma_120: f32,
    #[serde(deserialize_with = "from_string")]
    pub sma_240: f32,

    #[serde(deserialize_with = "from_string")]
    pub volume_sma_15: f32,
    #[serde(deserialize_with = "from_string")]
    pub volume_sma_30: f32,
    #[serde(deserialize_with = "from_string")]
    pub volume_sma_60: f32,
    #[serde(deserialize_with = "from_string")]
    pub volume_sma_120: f32,
    #[serde(deserialize_with = "from_string")]
    pub volume_sma_240: f32,
}
#[derive(serde::Deserialize, Debug, PartialEq, Default)]
pub struct CandleStickData {
    pub open_time: u64,
    #[serde(deserialize_with = "from_string")]
    pub open: f32,
    #[serde(deserialize_with = "from_string")]
    pub high: f32,
    #[serde(deserialize_with = "from_string")]
    pub low: f32,
    #[serde(deserialize_with = "from_string")]
    pub close: f32,

    

    #[serde(deserialize_with = "from_string")]
    pub volume: f32,
    #[serde(flatten)]
    pub indicators: Option<Indicators>
    
}
impl Into<Vec<f32>> for &CandleStickData {
    fn into(self) -> Vec<f32> {
        vec![
            self.open,
            self.high,
            self.low,
            self.close,
            // self.sma_15,
            // self.sma_30,
            // self.sma_60,
            // self.sma_120,
            // self.sma_240,
            // self.volume,
            // self.volume_sma_15,
            // self.volume_sma_30,
            // self.volume_sma_60,
            // self.volume_sma_120,
            // self.volume_sma_240,
        ]
    }
}
