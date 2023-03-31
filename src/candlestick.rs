use crate::utils::from_string;

#[derive(serde::Deserialize, Debug, PartialEq)]
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
}
impl Into<Vec<f32>> for &CandleStickData {
    fn into(self) -> Vec<f32> {
        vec![self.open, self.high, self.low, self.close]
    }
}
