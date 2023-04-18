use binance::utils::logger_init;
use binance::{Interval, Kline};
use chrono::{Duration, NaiveDate, NaiveDateTime, NaiveTime, ParseError, Utc};
use clap::builder::TypedValueParser as _;
use clap::{arg, command, Parser};
use csv::Writer;
use reqwest::Error;
use std::ops::Add;
use std::path::PathBuf;

const BINANCE_URL: &'static str = "https://api.binance.com";

#[tokio::main]
pub async fn main() -> std::result::Result<(), Error> {
    logger_init();
    let args = Args::parse();
    let filename = args.filename();
    if let Some(dir) = filename.parent() {
        if !dir.exists() {
            log::warn!("creating dir {:?}", dir);
            std::fs::create_dir_all(dir).unwrap();
        }
    }
    let client = reqwest::Client::new();
    let mut time = args.timestamp_from();

    let mut wtr = Writer::from_path(&filename)
        .expect(format!("Cannot create file in {:?}", filename).as_str());
    log::info!(
        "Start to download {} with interval {} starting from {:?}",
        args.symbol,
        args.interval,
        args.time_start
    );
    let time_end = if let Some(time_end) = args.time_end {
        time_end - Duration::milliseconds(1)
    } else {
        Utc::now().naive_utc()
    };
    log::info!("Download up to {}", &time_end);
    while time <= time_end {
        log::info!("Downloading data on {}", &time);
        let limit = if time + args.get_interval_duration() > time_end {
            args.get_limit_from_duration(time_end - time)
        } else {
            args.limit
        };

        let url = get_url(
            &args.symbol,
            time.timestamp_millis(),
            time_end.timestamp_millis(),
            args.interval,
            limit,
        );
        log::debug!("URL: {}", &url);
        let res = client.get(url).send().await?;
        let data: Vec<Kline<i64, f32>> = res.json().await?;
        if data.len() == 0 {
            log::info!("Now data to process");
            time = time.add(args.get_interval_duration());
            continue;
        }
        if time >= time_end {
            break;
        }
        if let Some(kline) = data.last() {
            time = NaiveDateTime::from_timestamp_millis(kline.close_time)
                .unwrap()
                .add(Duration::milliseconds(1));
        }
        for kline in data.iter() {
            wtr.serialize(kline).unwrap();
        }
    }

    Ok(())
}

fn get_url(symbol: &str, start_time: i64, time_end: i64, interval: Interval, limit: i64) -> String {
    let limit = if limit > 0 {
        format!("limit={}", limit)
    } else {
        "".to_string()
    };
    format!(
        "{}/api/v3/klines?interval={}&startTime={}&endTime={time_end}&symbol={}&{}",
        BINANCE_URL, interval, start_time, symbol, limit
    )
}

/// Simple program to download binance Kline data
#[derive(Debug, Parser)] // requires `derive` feature
#[command(name = "downloader")]
#[command(about = "Binance Kline data downloader", long_about = None)]
struct Args {
    /// Directory path where downloaded data save to
    #[arg(short, long, value_parser = clap::value_parser!(PathBuf), default_value="./")]
    dir: PathBuf,

    /// Download from date
    #[arg(short = 't', long, value_parser =  parse_datetime_from_str)]
    time_start: NaiveDateTime,
    /// Download from date
    #[arg(short = 'e', long, value_parser =  parse_datetime_from_str)]
    time_end: Option<NaiveDateTime>,
    /// Symbol
    #[arg(short, long)]
    symbol: String,
    /// Interval
    #[arg(short, long, default_value_t = Interval::default(), value_parser = clap::builder::PossibleValuesParser::new(["1s","1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w","1M",]).map(|s| s.parse::<Interval>().unwrap()))]
    interval: Interval,
    #[arg(short, long, value_parser = clap::value_parser!(i64), default_value="1000")]
    limit: i64,
}
fn parse_datetime_from_str(arg: &str) -> Result<NaiveDateTime, ParseError> {
    match arg.parse::<NaiveDateTime>() {
        Ok(d) => Ok(d),
        Err(_) => match arg.parse::<NaiveDate>() {
            Ok(d) => Ok(d.and_time(NaiveTime::from_hms_opt(0, 0, 0).unwrap())),
            Err(e) => Err(e),
        },
    }
}

impl Args {
    pub fn timestamp_from(&self) -> NaiveDateTime {
        self.time_start
    }
    pub fn filename(&self) -> PathBuf {
        let mut pb = self.dir.to_path_buf();
        pb.push(self.symbol.as_str());
        pb.push(format!("{}.csv", self.interval));
        pb
    }

    pub fn get_interval_duration(&self) -> Duration {
        match self.interval {
            Interval::M1 | Interval::M3 | Interval::M5 | Interval::M15 | Interval::M30 => {
                Duration::minutes(self.limit * self.get_interval_coef())
            }
            Interval::H1
            | Interval::H2
            | Interval::H4
            | Interval::H6
            | Interval::H8
            | Interval::H12 => Duration::hours(self.limit * self.get_interval_coef()),
            Interval::D1 | Interval::D3 => Duration::days(self.limit * self.get_interval_coef()),
            Interval::Month => Duration::weeks(self.limit * 4),
            Interval::S1 => Duration::seconds(self.limit * self.get_interval_coef()),
            Interval::W1 => Duration::weeks(self.limit * self.get_interval_coef()),
        }
    }
    pub fn get_interval_coef(&self) -> i64 {
        match self.interval {
            Interval::M1 => 1,
            Interval::M3 => 3,
            Interval::M5 => 5,
            Interval::M15 => 15,
            Interval::M30 => 30,
            Interval::H1 => 1,
            Interval::H2 => 2,
            Interval::H4 => 4,
            Interval::H6 => 6,
            Interval::H8 => 8,
            Interval::H12 => 12,
            Interval::D1 => 1,
            Interval::D3 => 3,
            Interval::Month => 4,
            Interval::S1 => 1,
            Interval::W1 => 1,
        }
    }
    pub fn get_limit_from_duration(&self, d: Duration) -> i64 {
        match self.interval {
            Interval::M1 | Interval::M3 | Interval::M5 | Interval::M15 | Interval::M30 => {
                d.num_minutes() / self.get_interval_coef()
            }
            Interval::H1
            | Interval::H2
            | Interval::H4
            | Interval::H6
            | Interval::H8
            | Interval::H12 => d.num_hours() / self.get_interval_coef(),
            Interval::D1 | Interval::D3 => d.num_days() / self.get_interval_coef(),
            Interval::Month => d.num_weeks() / self.get_interval_coef(),
            Interval::S1 => d.num_seconds() / self.get_interval_coef(),
            Interval::W1 => d.num_weeks() / self.get_interval_coef(),
        }
    }
}
