#[cfg(feature = "surrealdb")]
use binance::db::{from_value, Db};

#[cfg(feature = "surrealdb")]
#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use binance::Kline;
    use std::time::Instant;
    let start_time = Instant::now();
    let db = Db::new("alxly", "alxly", "file:data/surrealdb").await?;
    let responses = db.execute("SELECT * FROM btceur", None, false).await?;
    for response in responses {
        println!("query execution: {:.5}", response.time.as_secs_f32());
        match response.output() {
            Ok(v) => {
                let values: Vec<Kline> = from_value(v)?;
                println!("{:#?}", values.len());
            }
            Err(e) => eprintln!("{}", e),
        }
    }
    println!("execution time {:.5}", start_time.elapsed().as_secs_f32());
    Ok(())
}
#[tokio::main]
pub async fn main() {
    panic!("'surrealdb' feature is required")
}
