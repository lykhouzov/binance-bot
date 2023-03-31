use binance::Kline;
use reqwest::Error;

use csv::Writer;

// use yata::{core::Method, methods::SMA};

const BINANCE_URL: &'static str = "https://api.binance.com";
const SYMBOL: &'static str = "BTCEUR";
#[tokio::main]
pub async fn main() -> std::result::Result<(), Error> {
    let client = reqwest::Client::new();
    let mut time: u64 = 1672527600_000;
    let mut wtr = Writer::from_path(SYMBOL).unwrap();
    wtr.write_record(&["open_time", "open", "high", "low", "close"])
        .unwrap();
    loop {
        let url = get_url(time, 1000);
        let res = client.get(url).send().await?;
        let data: Vec<Kline> = res.json().await?;
        if data.len() == 0 {
            break;
        }
        if let Some(kline) = data.last() {
            time = kline.close_time;
        }
        for kline in data.iter() {
            let record = [
                kline.open_time.to_string(),
                kline.open.to_string(),
                kline.high.to_string(),
                kline.low.to_string(),
                kline.close.to_string(),
                // sma_5.next(&kline.close).to_string(),
                // sma_15.next(&kline.close).to_string(),
                // sma_30.next(&kline.close).to_string(),
                // sma_60.next(&kline.close).to_string(),
            ];
            // println!("{}", &record.join(", "));
            wtr.write_record(&record).unwrap();
        }
    }

    Ok(())
}

fn get_url(start_time: u64, limit: usize) -> String {
    format!(
        "{}/api/v3/klines?interval=1m&startTime={}&symbol={}&limit={}",
        BINANCE_URL, start_time, SYMBOL, limit
    )
}
