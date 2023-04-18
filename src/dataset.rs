use csv::StringRecord;
use yata::{
    core::{PeriodType, ValueType},
    methods::{SMA, WMA},
    prelude::Method,
};

use crate::Kline;
pub trait TaTrait:
    Method<Params = PeriodType, Input = ValueType, Output = ValueType> + std::fmt::Debug
{
}
impl TaTrait for WMA {}
impl TaTrait for SMA {}
pub type Ta = Box<dyn TaTrait>;
pub type Tas = Vec<Ta>;

#[derive(Debug)]
pub struct Dataset {
    skip_index: PeriodType,
    // data: Vec<Kline>,
    indicators: Tas,
    // iter: csv::DeserializeRecordsIter<'a, std::fs::File, Kline>,
    rdr: csv::Reader<std::fs::File>,
    current_index: isize,
}

impl Dataset {
    pub fn skip_index(&self) -> PeriodType {
        self.skip_index
    }
    pub fn current_step(&self) -> isize {
        self.current_index - self.skip_index as isize
    }
    pub fn with_wma(mut self, window: PeriodType, price: &ValueType) -> Self {
        self.skip_index = self.skip_index.max(window);
        let wma = WMA::new(window, price).unwrap();
        self.indicators.push(Box::new(wma) as Ta);
        self
    }
    pub fn with_sma(mut self, window: PeriodType, price: &ValueType) -> Self {
        self.skip_index = self.skip_index.max(window);
        let wma = SMA::new(window, price).unwrap();
        self.indicators.push(Box::new(wma) as Ta);
        self
    }
}
impl<P> From<P> for Dataset
where
    P: AsRef<std::path::Path>,
{
    fn from(value: P) -> Self {
        let rdr = csv::Reader::from_path(value).unwrap();
        Self {
            rdr,
            skip_index: 0,
            current_index: 0,
            indicators: Default::default(),
        }
    }
}
impl Iterator for Dataset {
    type Item = (isize, Kline<PeriodType, ValueType>, Vec<ValueType>);

    fn next(&mut self) -> Option<Self::Item> {
        let mut record = StringRecord::new();
        if self.rdr.read_record(&mut record).unwrap() {
            self.current_index += 1;
            let kline: Kline<PeriodType, ValueType> = record.deserialize(None).unwrap();
            let tas: Vec<ValueType> = self
                .indicators
                .iter_mut()
                .map(|ta| {
                    let price: ValueType = kline.close.clone();
                    let out = ta.next(&price);
                    out
                })
                .collect();
            Some((self.current_index - self.skip_index as isize, kline, tas))
        } else {
            None
        }
    }
}
