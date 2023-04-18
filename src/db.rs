use std::{collections::BTreeMap, error::Error, sync::Arc};

use serde::{de::DeserializeOwned, Serialize};
use serde_json::{Value as JsonValue, Map};
use surrealdb::{
    dbs::{Auth, Session},
    kvs::Datastore,
    sql::{serde::serialize_internal, Value}, error::Api,
};

pub struct Db {
    sess: Session,
    db: Datastore,
}

impl Db {
    pub async fn new(user: &str, password: &str, path: &str) -> Result<Self, Box<dyn Error>> {
        let db = Datastore::new(path).await?;
        let mut sess = Session::for_db("alxly", "kline");
        sess.au = Arc::new(Auth::Db(user.into(), password.into()));
        Ok(Self { sess, db })
    }

    pub async fn execute(
        &self,
        txt: &str,
        vars: Option<BTreeMap<String, Value>>,
        strict: bool,
    ) -> Result<Vec<surrealdb::dbs::Response>, surrealdb::error::Db> {
        self.db.execute(txt, &self.sess, vars, strict).await
    }
}

pub fn from_value<T>(value: Value) -> Result<T, Api>
where
    T: DeserializeOwned,
{
    let json = match serialize_internal(|| into_json(value.clone())) {
        Ok(json) => json,
        Err(error) => {
            return Err(Api::FromValue {
                value,
                error: error.to_string(),
            })
        }
    };
    serde_json::from_value(json).map_err(|error| Api::FromValue {
        value,
        error: error.to_string(),
    })
}

fn into_json(value: Value) -> serde_json::Result<JsonValue> {
    use surrealdb::sql;
    use surrealdb::sql::Number;
    use serde_json::Error;

    #[derive(Serialize)]
    struct Array(Vec<JsonValue>);

    impl TryFrom<sql::Array> for Array {
        type Error = Error;

        fn try_from(arr: sql::Array) -> Result<Self, Self::Error> {
            let mut vec = Vec::with_capacity(arr.0.len());
            for value in arr.0 {
                vec.push(into_json(value)?);
            }
            Ok(Self(vec))
        }
    }

    #[derive(Serialize)]
    struct Object(Map<String, JsonValue>);

    impl TryFrom<sql::Object> for Object {
        type Error = Error;

        fn try_from(obj: sql::Object) -> Result<Self, Self::Error> {
            let mut map = Map::with_capacity(obj.0.len());
            for (key, value) in obj.0 {
                map.insert(key.to_owned(), into_json(value)?);
            }
            Ok(Self(map))
        }
    }

    #[derive(Serialize)]
    enum Id {
        Number(i64),
        String(String),
        Array(Array),
        Object(Object),
    }

    impl TryFrom<sql::Id> for Id {
        type Error = Error;

        fn try_from(id: sql::Id) -> Result<Self, Self::Error> {
            use sql::Id::*;
            Ok(match id {
                Number(n) => Id::Number(n),
                String(s) => Id::String(s),
                Array(arr) => Id::Array(arr.try_into()?),
                Object(obj) => Id::Object(obj.try_into()?),
            })
        }
    }

    #[derive(Serialize)]
    struct Thing {
        tb: String,
        id: Id,
    }

    impl TryFrom<sql::Thing> for Thing {
        type Error = Error;

        fn try_from(thing: sql::Thing) -> Result<Self, Self::Error> {
            Ok(Self {
                tb: thing.tb,
                id: thing.id.try_into()?,
            })
        }
    }

    match value {
        Value::None | Value::Null => Ok(JsonValue::Null),
        Value::False => Ok(false.into()),
        Value::True => Ok(true.into()),
        Value::Number(Number::Int(n)) => Ok(n.into()),
        Value::Number(Number::Float(n)) => Ok(n.into()),
        Value::Number(Number::Decimal(n)) => serde_json::to_value(n),
        Value::Strand(strand) => Ok(strand.0.into()),
        Value::Duration(d) => serde_json::to_value(d),
        Value::Datetime(d) => serde_json::to_value(d),
        Value::Uuid(uuid) => serde_json::to_value(uuid),
        Value::Array(arr) => Ok(JsonValue::Array(Array::try_from(arr)?.0)),
        Value::Object(obj) => Ok(JsonValue::Object(Object::try_from(obj)?.0)),
        Value::Geometry(geometry) => serde_json::to_value(geometry),
        Value::Bytes(bytes) => serde_json::to_value(bytes),
        Value::Param(param) => serde_json::to_value(param),
        Value::Idiom(idiom) => serde_json::to_value(idiom),
        Value::Table(table) => serde_json::to_value(table),
        Value::Thing(thing) => serde_json::to_value(thing),
        Value::Model(model) => serde_json::to_value(model),
        Value::Regex(regex) => serde_json::to_value(regex),
        Value::Block(block) => serde_json::to_value(block),
        Value::Range(range) => serde_json::to_value(range),
        Value::Edges(edges) => serde_json::to_value(edges),
        Value::Future(future) => serde_json::to_value(future),
        Value::Constant(constant) => serde_json::to_value(constant),
        Value::Function(function) => serde_json::to_value(function),
        Value::Subquery(subquery) => serde_json::to_value(subquery),
        Value::Expression(expression) => serde_json::to_value(expression),
    }
}
