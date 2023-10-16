// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Udaf module contains functions and structs supporting user-defined aggregate functions.

use crate::Expr;
use crate::{
    AccumulatorFactoryFunction, ReturnTypeFunction, Signature, StateTypeFunction,
};
use std::fmt::{self, Debug, Formatter};
use std::sync::Arc;

/// Logical representation of a user-defined [aggregate function] (UDAF).
///
/// An aggregate function combines the values from multiple input rows
/// into a single output "aggregate" (summary) row. It is different
/// from a scalar function because it is stateful across batches. User
/// defined aggregate functions can be used as normal SQL aggregate
/// functions (`GROUP BY` clause) as well as window functions (`OVER`
/// clause).
///
/// `AggregateUDF` provides DataFusion the information needed to plan
/// and call aggregate functions, including name, type information,
/// and a factory function to create [`Accumulator`], which peform the
/// actual aggregation.
///
/// For more information, please see [the examples].
///
/// [the examples]: https://github.com/apache/arrow-datafusion/tree/main/datafusion-examples#single-process
/// [aggregate function]: https://en.wikipedia.org/wiki/Aggregate_function
/// [`Accumulator`]: crate::Accumulator
#[derive(Clone)]
pub struct AggregateUDF {
    /// name
    pub name: String,
    /// Signature (input arguments)
    pub signature: Signature,
    /// Return type
    pub return_type: ReturnTypeFunction,
    /// actual implementation
    pub accumulator: AccumulatorFactoryFunction,
    /// the accumulator's state's description as a function of the return type
    pub state_type: StateTypeFunction,
    pub order_sensitive: bool,
    pub support_concurrency: bool,
}

impl Debug for AggregateUDF {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("AggregateUDF")
            .field("name", &self.name)
            .field("signature", &self.signature)
            .field("fun", &"<FUNC>")
            .finish()
    }
}

impl PartialEq for AggregateUDF {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.signature == other.signature
    }
}

impl Eq for AggregateUDF {}

impl std::hash::Hash for AggregateUDF {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.signature.hash(state);
    }
}

impl AggregateUDF {
    /// Create a new AggregateUDF
    pub fn new(
        name: &str,
        signature: &Signature,
        return_type: &ReturnTypeFunction,
        accumulator: &AccumulatorFactoryFunction,
        state_type: &StateTypeFunction,
    ) -> Self {
        Self::new_with_preference(
            name,
            signature,
            return_type,
            accumulator,
            state_type,
            false,
            true,
        )
    }

    /// Create a new AggregateUDF
    pub fn new_with_preference(
        name: &str,
        signature: &Signature,
        return_type: &ReturnTypeFunction,
        accumulator: &AccumulatorFactoryFunction,
        state_type: &StateTypeFunction,
        order_sensitive: bool,
        support_concurrency: bool,
    ) -> Self {
        Self {
            name: name.to_owned(),
            signature: signature.clone(),
            return_type: return_type.clone(),
            accumulator: accumulator.clone(),
            state_type: state_type.clone(),
            order_sensitive,
            support_concurrency,
        }
    }

    /// creates an [`Expr`] that calls the aggregate function.
    ///
    /// This utility allows using the UDAF without requiring access to
    /// the registry, such as with the DataFrame API.
    pub fn call(&self, args: Vec<Expr>) -> Expr {
        Expr::AggregateUDF(crate::expr::AggregateUDF {
            fun: Arc::new(self.clone()),
            args,
            filter: None,
            order_by: None,
        })
    }
}

pub fn check_agg_arg_count(
    agg_fun: &str,
    input_types: &[arrow::datatypes::DataType],
    signature: &crate::TypeSignature,
) -> Result<(), datafusion_common::DataFusionError> {
    use crate::TypeSignature;
    use datafusion_common::DataFusionError;
    match signature {
        TypeSignature::Uniform(agg_count, _) | TypeSignature::Any(agg_count) => {
            if input_types.len() != *agg_count {
                return Err(DataFusionError::Plan(format!(
                    "The function {:?} expects {:?} arguments, but {:?} were provided",
                    agg_fun,
                    agg_count,
                    input_types.len()
                )));
            }
        }
        TypeSignature::Exact(types) => {
            if types.len() != input_types.len() {
                return Err(DataFusionError::Plan(format!(
                    "The function {:?} expects {:?} arguments, but {:?} were provided",
                    agg_fun,
                    types.len(),
                    input_types.len()
                )));
            }
        }
        TypeSignature::OneOf(variants) => {
            let ok = variants
                .iter()
                .any(|v| check_agg_arg_count(agg_fun, input_types, v).is_ok());
            if !ok {
                return Err(DataFusionError::Plan(format!(
                    "The function {:?} does not accept {:?} function arguments.",
                    agg_fun,
                    input_types.len()
                )));
            }
        }
        TypeSignature::VariadicAny => {
            if input_types.is_empty() {
                return Err(DataFusionError::Plan(format!(
                    "The function {agg_fun:?} expects at least one argument"
                )));
            }
        }
        _ => {
            return Err(DataFusionError::Internal(format!(
                "Aggregate functions do not support this {signature:?}"
            )));
        }
    }
    Ok(())
}
