// use serde::{Deserialize, Deserializer};
// use std::env;
// use std::fmt::Display;
// use std::str::FromStr;
// use tungstenite::{connect, Message};
fn main() {
    // dotenv::dotenv().ok();
    // env_logger::init();
    // let ws_url = url::Url::parse(&env::var("STREAM_API_URL").unwrap().as_ref()).unwrap();
    // // url.set_path("stream/btcusdt@kline_1m");
    // println!("{:?}", ws_url.to_string());
    // let (mut socket, response) = connect(ws_url).expect("Can't connect");

    // println!("Connected to the server");
    // println!("Response HTTP code: {}", response.status());
    // println!("Response contains the following headers:");
    // for (ref header, _value) in response.headers() {
    //     println!("* {}", header);
    // }
    // socket
    //     .write_message(Message::Text(
    //         "{\"method\":\"UNSUBSCRIBE\",\"params\":[\"btcusdt@kline_1m\"],\"id\": 1 }".into(),
    //     ))
    //     .unwrap();
    // {
    //     let msg = socket.read_message().expect("Error reading message");
    //     println!("{:?}", &msg);
    // }
    // socket
    //     .write_message(Message::Text(
    //         "{\"method\":\"SUBSCRIBE\",\"params\":[\"btcusdt@kline_1m\"],\"id\": 1 }".into(),
    //     ))
    //     .unwrap();
    // {
    //     let msg = socket.read_message().expect("Error reading message");
    //     println!("{:?}", &msg);
    // }
    // loop {
    //     let msg = socket.read_message().expect("Error reading message");
    //     match msg {
    //         Message::Ping(p) => {
    //             socket.write_message(Message::Pong(p)).unwrap();
    //         }
    //         Message::Text(t) => {
    //             println!("{:?}", &t);
    //             let data: KlineResponse = serde_json::from_slice(t.as_bytes().as_ref()).unwrap();
    //             println!("Received: {:#?}", data);
    //         }
    //         _ => {
    //             println!("UNHANDLED MESSAGE {:?}", msg)
    //         }
    //     }
    // }
    // soc
}