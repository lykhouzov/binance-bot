use binance::dataset::Dataset;

fn main() {
    let value = "data/ETHEUR/1m.csv";

    let mut ds = Dataset::from(value).with_wma(7, &0.0).with_wma(22, &0.0);
    for _ in 0..ds.skip_index() {
        let _ = ds.next().unwrap();
    }
    let mut i = 0;
    for line in ds {
        println!("{:?}", line);
        i += 1;
    }
    println!("i = {}", i);
    // let line = ds.next();
}
