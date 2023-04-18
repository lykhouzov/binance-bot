#[cfg(feature = "surrealdb")]
use surrealdb::{
    dbs::{Auth, Session},
    kvs::Datastore,
    sql,
};
const BINANCE_URL: &'static str = "https://api.binance.com";
const SYMBOL: &'static str = "BTCEUR";
#[cfg(feature = "surrealdb")]
mod prelude {
    pub use binance::Kline;
    pub const DS_PATH: &'static str = "file:data/surrealdb";
    pub const NS: &'static str = "alxly";
    pub const DB: &'static str = "kline";
}
#[cfg(feature = "surrealdb")]
use prelude::*;

#[cfg(feature = "surrealdb")]
#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = &Datastore::new(DS_PATH).await?;
    let mut sess = Session::for_db(NS, DB);
    sess.au = Arc::new(Auth::Db("alxly".into(), "alxly".into()));
    let client = reqwest::Client::new();
    let mut time: usize = 1672527600_000;

    loop {
        let url = get_url(time, 1000);
        let res = client.get(url).send().await?;
        let data: Vec<Kline> = res.json().await?;
        if data.len() == 0 {
            break;
        }
        if let Some(kline) = data.last() {
            time = kline.close_time as usize;
        }
        for kline in data.iter() {
            let txt = format!(
                "INSERT IGNORE INTO {} (id, open_time,open,high,low,close,volume,close_time,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume) VALUES
            ($id,$open_time,$open,$high,$low,$close,$volume,$close_time,$quote_asset_volume,$number_of_trades,$taker_buy_base_asset_volume,$taker_buy_quote_asset_volume)
            ",
                SYMBOL.to_lowercase()
            );

            let vars: BTreeMap<String, sql::Value> = [
                ("id".into(), kline.open_time.into()),
                ("open_time".into(), kline.open_time.into()),
                ("open".into(), kline.open.into()),
                ("high".into(), kline.high.into()),
                ("low".into(), kline.low.into()),
                ("close".into(), kline.close.into()),
                ("volume".into(), kline.volume.into()),
                ("close_time".into(), kline.close_time.into()),
                ("quote_asset_volume".into(), kline.quote_asset_volume.into()),
                ("number_of_trades".into(), kline.number_of_trades.into()),
                (
                    "taker_buy_base_asset_volume".into(),
                    kline.taker_buy_base_asset_volume.into(),
                ),
                (
                    "taker_buy_quote_asset_volume".into(),
                    kline.taker_buy_quote_asset_volume.into(),
                ),
            ]
            .into();
            let responses = db.execute(txt.as_str(), &sess, Some(vars), false).await?;
            for resp in responses {
                if let Err(e) = resp.output() {
                    println!("Error when insert data {}", e);
                }
            }
        }
    }

    Ok(())
}
#[tokio::main]
async fn main() {
    panic!("feature 'surrealdb' is required");
}
#[allow(unused)]
fn get_url(start_time: usize, limit: usize) -> String {
    format!(
        "{}/api/v3/klines?interval=1m&startTime={}&symbol={}&limit={}",
        BINANCE_URL, start_time, SYMBOL, limit
    )
}
