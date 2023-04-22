use binance::dataset::DatasetBuilder;

fn main() {
    let value = "data/ETHEUR/1m.csv";

    let ds = &mut DatasetBuilder::from(value).with_wma(7).with_wma(22).build();
    let mut iter = ds.into_iter();
    for _ in 0..ds.skip_index() {
        let _ = iter.next().unwrap();
    }
    let mut i = 0;
    for line in iter {
        println!("{:?}", line);
        i += 1;
    }
    println!("i = {}", i);
    // let line = ds.next();
}
