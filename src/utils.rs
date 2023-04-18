use crate::{
    agent::consts::{BUFFER_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, SAVE_STEP_IMAGE, WINDOW_SIZE, BatchTensor},
    candlestick::CandleStickData,
    Kline,
};
use log4rs::{append::console::ConsoleAppender, filter::threshold::ThresholdFilter};
use plotters::{
    prelude::{BitMapBackend, CandleStick, ChartBuilder, IntoDrawingArea},
    series::LineSeries,
    style::{
        full_palette::{BLUEGREY, LIGHTBLUE, PURPLE, YELLOW_900},
        Color, BLUE, GREEN, RED, WHITE,
    },
};
use serde::{
    de::{self, Visitor},
    Deserialize, Deserializer,
};
use std::{
    error::Error,
    fmt::{self, Debug},
    marker::PhantomData,
    str::FromStr,
};
pub fn from_string<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: FromStr + Deserialize<'de>,
    <T as FromStr>::Err: std::fmt::Display + Debug,
{
    struct StringOrNumber<T>(PhantomData<fn() -> T>);

    impl<'de, T> Visitor<'de> for StringOrNumber<T>
    where
        T: Deserialize<'de> + FromStr,
        <T as FromStr>::Err: std::fmt::Display + Debug,
    {
        type Value = T;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("string or f64")
        }

        fn visit_str<E>(self, value: &str) -> Result<T, E>
        where
            E: de::Error,
        {
            Ok(FromStr::from_str(value).unwrap())
        }
        fn visit_f32<E>(self, v: f32) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Deserialize::deserialize(de::value::F32Deserializer::new(v))
        }
        fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Deserialize::deserialize(de::value::F64Deserializer::new(v))
        }
        fn visit_i32<E>(self, v: i32) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Deserialize::deserialize(de::value::I32Deserializer::new(v))
        }
        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Deserialize::deserialize(de::value::I64Deserializer::new(v))
        }
    }
    // let s = String::deserialize(deserializer)?;
    // <T as FromStr>::from_str(&s).map_err(serde::de::Error::custom)
    deserializer.deserialize_any(StringOrNumber(PhantomData))
}

pub fn load_data<P: AsRef<std::path::Path>>(path: P, limit: Option<usize>) -> Vec<CandleStickData> {
    let mut rdr = csv::Reader::from_path(path).unwrap();
    let iter = rdr.deserialize();
    let mapper = |x: Result<CandleStickData, csv::Error>| {
        let v: CandleStickData = x.unwrap();
        v
    };
    if let Some(l) = limit {
        iter.take(l).map(mapper).collect()
    } else {
        iter.map(mapper).collect()
    }
}

pub fn argmax(x: &[f32]) -> i64 {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _v)| i)
        .unwrap() as i64
}

// pub trait NormalizeState {
//     fn normalize(&self) -> tch::Tensor;
// }
// impl NormalizeState for tch::Tensor {
//     fn normalize(&self) -> tch::Tensor {
//         let (max, _) = self.max_dim(0, true);

//         let (min, _) = self.min_dim(0, true);

//         let out = (self - &min) / (&max - &min);

//         out.nan_to_num(1.0, 1.0, 0.0)
//     }
// }

pub fn to_array<const N: usize, T>(v: Vec<T>) -> [T; N] {
    v.try_into()
        .unwrap_or_else(|v: Vec<T>| panic!("Expected a Vec of length {} but it was {}", N, v.len()))
}

pub fn logger_init() {
    let _ = dotenv::dotenv().ok();
    env_logger::init();
}
pub fn init_logger(file_path: &str) -> Result<(), log::SetLoggerError> {
    use log4rs::{
        append::file::FileAppender,
        config::{Appender, Config, Root},
        encode::pattern::PatternEncoder,
    };
    // Logging to log file.
    let logfile = FileAppender::builder()
        // Pattern: https://docs.rs/log4rs/*/log4rs/encode/pattern/index.html
        .encoder(Box::new(PatternEncoder::new("{d(%Y-%m-%d %H:%M:%S)} {l} - {m}\n")))
        .build(file_path)
        .unwrap();
    let console = ConsoleAppender::builder()
        .encoder(Box::new(PatternEncoder::new("{d(%Y-%m-%d %H:%M:%S)} {l} - {m}\n")))
        .build();
    // Log Trace level output to file where trace is the default level
    // and the programmatically specified level to stderr.
    let config = Config::builder()
        .appender(Appender::builder().build("logfile", Box::new(logfile)))
        .appender(Appender::builder().filter(Box::new(ThresholdFilter::new(log::LevelFilter::Info))).build("concole", Box::new(console)))

        .build(
            Root::builder()
                .appender("logfile")
                .appender("concole")
                .build(log::LevelFilter::Debug),
        )
        .unwrap();

    // Use this to change log levels at runtime.
    // This means you can change the default log level to trace
    // if you are trying to debug an issue and need more logs on then turn it off
    // once you are done.
    let _handle = log4rs::init_config(config)?;
    Ok(())
}
pub fn get_img_vec(
    data: &[CandleStickData],
    last: &CandleStickData,
    step: usize,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let (min, max) = data.iter().fold((f32::MAX, f32::MIN), |acc, x| {
        let (min, max) = (acc.0.min(x.low), acc.1.max(x.high));
        if let Some(indicators) = &x.indicators {
            (
                min.min(indicators.sma_15).min(indicators.sma_30),
                max.max(indicators.sma_15).max(indicators.sma_30),
            )
        } else {
            (min, max)
        }
    });
    let min = min.min(last.open);
    let max = max.max(last.open);
    log::trace!("Calculated (min,max) for the range: ({},{})", min, max);
    let x_range = 0..data.len() as i32;
    let y_range = min..max;
    let mut buffer = [0; BUFFER_SIZE];
    {
        let root =
            BitMapBackend::with_buffer(&mut buffer, (IMAGE_WIDTH as u32, IMAGE_HEIGHT as u32))
                .into_drawing_area();

        let mut chart = ChartBuilder::on(&root)
            .margin_left(1)
            .margin_right(0)
            .build_cartesian_2d(x_range, y_range)?;
        chart
            .configure_mesh()
            .disable_x_mesh()
            .disable_y_mesh()
            .draw()?;
        chart.draw_series(data.iter().enumerate().map(|(i, x)| {
            CandleStick::new(
                i as i32,
                x.open,
                x.high,
                x.low,
                x.close,
                GREEN.filled(),
                RED.filled(),
                3,
            )
        }))?;
        chart
            .draw_series(LineSeries::new(
                data.iter()
                    .enumerate()
                    .map(|(x, c)| {
                        if let Some(idicators) = &c.indicators {
                            Some((x as i32, idicators.sma_30))
                        } else {
                            None
                        }
                    })
                    .filter(Option::is_some)
                    .map(|x| x.unwrap()),
                &LIGHTBLUE,
            ))?
            .label("SMA30");
        chart
            .draw_series(LineSeries::new(
                data.iter()
                    .enumerate()
                    .map(|(x, c)| {
                        if let Some(idicators) = &c.indicators {
                            Some((x as i32, idicators.sma_60))
                        } else {
                            None
                        }
                    })
                    .filter(Option::is_some)
                    .map(|x| x.unwrap()),
                &PURPLE,
            ))?
            .label("SMA60");
        chart
            .draw_series(LineSeries::new(
                data.iter()
                    .enumerate()
                    .map(|(x, c)| {
                        if let Some(idicators) = &c.indicators {
                            Some((x as i32, idicators.sma_240))
                        } else {
                            None
                        }
                    })
                    .filter(Option::is_some)
                    .map(|x| x.unwrap()),
                &PURPLE,
            ))?
            .label("SMA240");
        chart
            .draw_series(LineSeries::new(
                [last.open; WINDOW_SIZE]
                    .iter()
                    .enumerate()
                    .map(|(x, y)| (x as i32, *y)),
                WHITE.stroke_width(1),
            ))?
            .label("Current Price");

        // To avoid the IO failure being ignored silently, we manually call the present function
        root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    }
    if let Some(img_buf) =
        image::RgbImage::from_raw(IMAGE_WIDTH as u32, IMAGE_HEIGHT as u32, buffer.to_vec())
    {
        // let filename_rgb = format!("data/stock/{}_rgb.png", step);
        // img_buf.save(filename_rgb);
        let gray: image::GrayImage = image::DynamicImage::ImageRgb8(img_buf).into_luma8();
        if SAVE_STEP_IMAGE {
            let filename = format!("data/stock/{}.png", step);
            gray.save(filename).expect("Cannot save image");
        }
        Ok(gray.to_vec())
    } else {
        Err("Cannot get RgbImage image from buffer".into())
    }
}

pub fn get_img_state_vec(
    data: &[Kline<u64, f64>],
    indicators: &[Vec<f64>],
    step: usize,
    save_step_image: bool,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let (min, max) = data.iter().fold((f32::MAX, f32::MIN), |acc, x| {
        (acc.0.min(x.low as f32), acc.1.max(x.high as f32))
    });
    let (i_min, i_max) = indicators
        .iter()
        .flatten()
        .fold((f32::MAX, f32::MIN), |acc, x| {
            (acc.0.min(*x as f32), acc.1.max(*x as f32))
        });
    let min = min.min(i_min);
    let max = max.max(i_max);
    log::trace!("Calculated (min,max) for the range: ({},{})", min, max);
    let x_range = 0..data.len() as i32;
    let y_range = min..max;
    let mut buffer = [0; BUFFER_SIZE];
    {
        let root =
            BitMapBackend::with_buffer(&mut buffer, (IMAGE_WIDTH as u32, IMAGE_HEIGHT as u32))
                .into_drawing_area();

        let mut chart = ChartBuilder::on(&root)
            .margin_left(1)
            .margin_right(0)
            .build_cartesian_2d(x_range, y_range)?;
        chart
            .configure_mesh()
            .disable_x_mesh()
            .disable_y_mesh()
            .draw()?;
        chart.draw_series(data.iter().enumerate().map(|(i, x)| {
            CandleStick::new(
                i as i32,
                x.open as f32,
                x.high as f32,
                x.low as f32,
                x.close as f32,
                GREEN.filled(),
                RED.filled(),
                3,
            )
        }))?;
        let colors = [LIGHTBLUE, PURPLE, YELLOW_900, BLUE, BLUEGREY];
        let colors = colors.iter().cycle();
        let indicators_num = indicators.get(0).unwrap().len();
        for (num, color) in (0..indicators_num - 1).zip(colors) {
            chart
                .draw_series(LineSeries::new(
                    (0..data.len()).into_iter().map(|x| {
                        let y = indicators[x][num] as f32;
                        (x as i32, y)
                    }),
                    color,
                ))?
                .label(format!("Indicatro {}", num));
        }
        let last = data.last().unwrap();
        chart
            .draw_series(LineSeries::new(
                (0..data.len())
                    .into_iter()
                    .map(|x| (x as i32, last.close as f32)),
                WHITE.stroke_width(1),
            ))?
            .label("Current Price");

        // To avoid the IO failure being ignored silently, we manually call the present function
        root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    }
    if let Some(img_buf) =
        image::RgbImage::from_raw(IMAGE_WIDTH as u32, IMAGE_HEIGHT as u32, buffer.to_vec())
    {
        // let filename_rgb = format!("data/stock/{}_rgb.png", step);
        // img_buf.save(filename_rgb);
        let gray: image::GrayImage = image::DynamicImage::ImageRgb8(img_buf).into_luma8();
        if save_step_image {
            let filename = format!("data/stock/{}.png", step);
            gray.save(filename).expect("Cannot save image");
        }
        Ok(gray.to_vec())
    } else {
        Err("Cannot get RgbImage image from buffer".into())
    }
}


pub fn normalize(x: BatchTensor<f32>) -> BatchTensor<f32> {
    use dfdx::prelude::*;
    let mean = x.clone().mean();
    let std = x.clone().stddev(1e-8);
    (x - mean.broadcast()) / std.broadcast()
}