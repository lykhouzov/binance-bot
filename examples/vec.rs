use std::sync::{Arc, Mutex};

use rayon::prelude::*;

fn main() {
    // let mut vec = Vec::with_capacity(4);
    // for i in 0..10 {
    //     if vec.len()>=4{
    //         vec.remove(0);
    //     }
    //     vec.push(i);
    //     println!("{:?}; len={}, capacity={}", vec, vec.len(), vec.capacity());
    // }
    let output = Arc::new(Mutex::new((vec![0; 2], vec![0.0; 2], vec![""; 2])));
    let data = vec![(1, 1.0, "a"), (2, 2.9, "b")];
    let result = data
        .into_par_iter()
        .enumerate()
        .fold_with(
            Arc::clone(&output),
            |mut acc: Arc<Mutex<(Vec<i32>, Vec<f64>, Vec<&str>)>>,
             (idx, x): (usize, (i32, f64, &str))| {
                println!("{:?}", acc);
                println!("{:?}", x.0);
                println!("{:?}", x.1);
                let d = Arc::clone(&acc);
                let mut s = d.lock().unwrap();
                s.0[idx] = x.0;
                s.1[idx] = x.1;
                s.2[idx] = x.2;
                println!("{:?}", acc);
                acc
            },
        )
        .for_each(|_x| {});
    let guard: std::sync::MutexGuard<(Vec<i32>, Vec<f64>, Vec<&str>)> = output.lock().unwrap();
    println!("{:#?}", result);
    println!("{:#?}", guard);
}
