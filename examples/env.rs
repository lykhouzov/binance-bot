use std::error::Error;

use binance::utils::load_data;
use plotters::{
    backend::{PixelFormat, RGBPixel},
    prelude::*,
    style::full_palette::{GREY_300, GREY_400},
};
const OUT_FILE_NAME: &'static str = "data/stock.png";
const WINDOW_SIZE: usize = 120;
const IMG_SIZE: u32 = 3 * WINDOW_SIZE as u32;
const BUFFER_SIZE: usize = RGBPixel::PIXEL_SIZE * (IMG_SIZE * IMG_SIZE) as usize;
fn main() -> Result<(), Box<dyn Error>> {
    let data = load_data("data/BTCEUR", Some(WINDOW_SIZE + 5));
    let window = data.windows(WINDOW_SIZE);
    let data = data[WINDOW_SIZE..].iter();
    for ((step, data), last) in window.enumerate().zip(data) {
        let (min, max) = data.iter().fold((f32::MAX, f32::MIN), |acc, x| {
            let vec: Vec<f32> = x.into();
            let (min, max) = vec.iter().fold((f32::MAX, f32::MIN), |acc, x| {
                (acc.0.min(*x), acc.1.max(*x))
            });
            (
                acc.0.min(min),
                // .min(x.low)
                // .min(x.sma_15)
                // .min(x.sma_30)
                // .min(x.sma_60)
                // .min(x.sma_120)
                // .min(x.sma_240),
                acc.1.max(max), // .max(x.high)
                                // .max(x.sma_15)
                                // .max(x.sma_30)
                                // .max(x.sma_60)
                                // .max(x.sma_120)
                                // .max(x.sma_240),
            )
        });
        let min = min.min(last.open);
        let max = max.max(last.open);
        let x_range = 0..data.len() as i32;
        let y_range = min..max;

        let mut buffer = [0; BUFFER_SIZE];
        {
            let root =
                BitMapBackend::with_buffer(&mut buffer, (IMG_SIZE, IMG_SIZE)).into_drawing_area();

            let mut chart = ChartBuilder::on(&root)
                .margin(0)
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
                .draw_series(
                    LineSeries::new(
                        data.iter()
                            .enumerate()
                            .map(|(x, c)| {
                                if let Some(idicators) = &c.indicators {
                                    Some((x as i32, idicators.sma_15))
                                } else {
                                    None
                                }
                            })
                            .filter(Option::is_some)
                            .map(|x| x.unwrap()),
                        &GREY_300,
                    )
                    .point_size(1),
                )?
                .label("SMA5");
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
                    &GREY_400,
                ))?
                .label("SMA30");
            chart
                .draw_series(LineSeries::new(
                    [last.open; WINDOW_SIZE]
                        .iter()
                        .enumerate()
                        .map(|(x, y)| (x as i32, *y)),
                    &WHITE,
                ))?
                .label("Current Price");

            // To avoid the IO failure being ignored silently, we manually call the present function
            root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
        }
        if let Some(img_buf) = image::RgbImage::from_raw(IMG_SIZE, IMG_SIZE, buffer.to_vec()) {
            let filename = format!("data/stock/{}.png", step);
            // let filename_rgb = format!("data/stock/{}_rgb.png", step);
            // img_buf.save(filename_rgb);
            let gray: image::GrayImage = image::DynamicImage::ImageRgb8(img_buf).into_luma8();
            gray.save(filename).expect("Cannot save image");
        }
    }
    println!("Result has been saved to {}", OUT_FILE_NAME);
    Ok(())
}
