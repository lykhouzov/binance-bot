use std::{collections::VecDeque, time::Instant};

use binance::agent::consts::MEMORY_SIZE;
fn main() {
    let time = Instant::now();
    let mut vd = VecDeque::with_capacity(MEMORY_SIZE);
    vd.push_back(0);
    vd.push_back(1);
    vd.push_back(2);
    let _ = vd.pop_front();
    vd.push_back(3);
    let _ = vd.pop_front();
    vd.push_back(4);
    let _ = vd.pop_front();
    vd.push_back(5);
    let _ = vd.pop_front();
    let v= vd.as_slices();
    println!("{:?}", v);
    println!("{}", time.elapsed().as_secs_f32());
}
