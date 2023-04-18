use chrono::Duration;
fn main(){
    let duration = Duration::hours(1) + Duration::minutes(2)+ Duration::milliseconds(2012);
    let h = duration.num_hours();
    let duration = duration-Duration::hours(h);
    let m = duration.num_minutes();
    let duration = duration-Duration::minutes(m);
    let s = duration.num_milliseconds() as f32/ 1000.0;
    println!("{h:0>2}:{m:0>2}:{s:0>6.3}")
}