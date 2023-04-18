use std::error::Error;

use binance::{utils::logger_init, Kline};
use yata::core::{PeriodType, ValueType};
// Vec<Kline<PeriodType, ValueType>>
fn main() -> Result<(), Box<dyn Error>> {
    logger_init();
    let tick_data_path = "data/tick/ETHEUR/1s.csv";
    let main_data_path = "data/tick/ETHEUR/1m.csv";
    let mut rdr_tick = csv::Reader::from_path(tick_data_path).unwrap();
    let mut rdr_main = csv::Reader::from_path(main_data_path).unwrap();
    let mut tick_data_iter = rdr_tick.records();

    // let main_data_iter = rdr_main.records();
    for r in rdr_main.records() {
        let kline: Kline<PeriodType, ValueType> = r?.deserialize(None)?;
        let mut tick: Kline<PeriodType, ValueType> = Default::default();
        for _ in 0..60 {
            if let Some(tick_data) = tick_data_iter.next() {
                tick = tick_data?.deserialize(None)?;
            } else {
                break;
            }
        }
        if kline.close != tick.close {
            log::error!("Close time is not equal");
            println!("ERROR");
        }
    }
    Ok(())
}
