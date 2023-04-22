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

pub type DatasetRecord = (Kline<PeriodType, ValueType>, Vec<ValueType>);

#[derive(Debug)]
pub struct Dataset {
    skip_index: PeriodType,
    // data: Vec<Kline>,
    // indicators: Tas,
    // iter: csv::DeserializeRecordsIter<'a, std::fs::File, Kline>,
    // rdr: csv::Reader<std::fs::File>,
    records: Vec<DatasetRecord>,
    current_index: usize,
}

impl Dataset {
    pub fn skip_index(&self) -> PeriodType {
        self.skip_index
    }
    pub fn current_step(&self) -> usize {
        self.current_index - self.skip_index as usize
    }
    pub fn windows(&self, window_size: usize) -> std::slice::Windows<DatasetRecord> {
        self.records[self.skip_index as usize..].windows(window_size)
    }
    pub fn iter(&self) -> std::slice::Iter<DatasetRecord> {
        self.records[self.skip_index as usize..].iter()
    }
    // pub fn with_wma(mut self, window: PeriodType) -> Self {
    //     self.skip_index = self.skip_index.max(window);
    //     let kline = self.records.get(0).unwrap();
    //     let wma = WMA::new(window, &kline.close).unwrap();
    //     self.indicators.push(Box::new(wma) as Ta);
    //     self
    // }
    // pub fn with_sma(mut self, window: PeriodType) -> Self {
    //     self.skip_index = self.skip_index.max(window);
    //     let kline = self.records.get(0).unwrap();
    //     let wma = SMA::new(window, &kline.close).unwrap();
    //     self.indicators.push(Box::new(wma) as Ta);
    //     self
    // }
}
// impl<P> From<P> for Dataset
// where
//     P: AsRef<std::path::Path>,
// {
//     fn from(value: P) -> Self {
//         let mut rdr = csv::Reader::from_path(value).unwrap();
//         let records: Vec<Kline<PeriodType, ValueType>> = rdr
//             .records()
//             .map(|x| x.unwrap().deserialize(None).unwrap())
//             .collect();
//         Self {
//             records,
//             skip_index: 0,
//             current_index: 0,
//             indicators: Default::default(),
//         }
//     }
// }
pub struct DatasetBuilder {
    indicators: Vec<(&'static str, PeriodType)>,
    rdr: csv::Reader<std::fs::File>,
    skip_index: PeriodType,
}
impl<P> From<P> for DatasetBuilder
where
    P: AsRef<std::path::Path>,
{
    fn from(value: P) -> Self {
        let rdr = csv::Reader::from_path(value).unwrap();

        Self {
            rdr,
            skip_index: 0,
            indicators: Default::default(),
        }
    }
}
impl DatasetBuilder {
    pub fn build(mut self) -> Dataset {
        let mut record = StringRecord::new();
        let mut indicators: Tas = self
            .indicators
            .iter()
            .map(|(name, window)| match *name {
                "WMA" => Box::new(WMA::new(*window, &0.0).unwrap()) as Ta,
                "SMA" => Box::new(SMA::new(*window, &0.0).unwrap()) as Ta,
                m => todo!("Method {} Is not supported yet", m),
            })
            .collect();
        let mut records = Vec::new();
        while let Ok(read) = self.rdr.read_record(&mut record) {
            if !read {
                break;
            }
            let kline: Kline<PeriodType, ValueType> = record.deserialize(None).unwrap();

            let tas: Vec<ValueType> = indicators
                .iter_mut()
                .map(|ta| {
                    // let price: ValueType = kline.close.clone();
                    let out = ta.next(&kline.close);
                    out
                })
                .collect();
            records.push((kline, tas));
        }
        Dataset {
            skip_index: self.skip_index,
            // indicators: indicators.unwrap(),
            records,
            current_index: 0,
        }
    }
    pub fn with_wma(mut self, window: PeriodType) -> Self {
        self.skip_index = self.skip_index.max(window);
        self.indicators.push(("WMA", window));
        self
    }
    pub fn with_sma(mut self, window: PeriodType) -> Self {
        self.skip_index = self.skip_index.max(window);
        self.indicators.push(("SMA", window));
        self
    }
}

impl<'a> IntoIterator for &'a Dataset {
    type Item = &'a DatasetRecord;

    type IntoIter = std::slice::Iter<'a, DatasetRecord>;

    fn into_iter(self) -> Self::IntoIter {
        let slice = &self.records[self.skip_index as usize..];
        slice.iter()
    }
}

// impl<'a> Iterator for DatasetIterator<'a> {
//     type Item = DatasetRecord;

//     fn next(&mut self) -> Option<Self::Item> {
//         if let Some(kline) = self.records.get(self.current_index) {
//             let tas: Vec<ValueType> = self
//                 .indicators
//                 .iter_mut()
//                 .map(|ta| {
//                     let out = ta.next(&kline.close);
//                     out
//                 })
//                 .collect();
//             Some((
//                 self.current_index as isize - self.skip_index as isize,
//                 kline.clone(),
//                 tas,
//             ))
//         } else {
//             None
//         }
//         let mut record = StringRecord::new();
//         if self.rdr.read_record(&mut record).unwrap() {
//             self.current_index += 1;
//             let kline: Kline<PeriodType, ValueType> = record.deserialize(None).unwrap();
//             let tas: Vec<ValueType> = self
//                 .indicators
//                 .iter_mut()
//                 .map(|ta| {
//                     let price: ValueType = kline.close.clone();
//                     let out = ta.next(&price);
//                     out
//                 })
//                 .collect();
//             Some((self.current_index - self.skip_index as isize, kline, tas))
//         } else {
//             None
//         }
//     }
// }
