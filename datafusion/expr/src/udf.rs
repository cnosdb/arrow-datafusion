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

//! Udf module contains foundational types that are used to represent UDFs in DataFusion.

use crate::{Expr, ReturnTypeFunction, ScalarFunctionImplementation, Signature};
use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;

/// Logical representation of a UDF.
#[derive(Clone)]
pub struct ScalarUDF {
    /// name
    pub name: String,
    /// signature
    pub signature: Signature,
    /// Return type
    pub return_type: ReturnTypeFunction,
    /// actual implementation
    ///
    /// The fn param is the wrapped function but be aware that the function will
    /// be passed with the slice / vec of columnar values (either scalar or array)
    /// with the exception of zero param function, where a singular element vec
    /// will be passed. In that case the single element is a null array to indicate
    /// the batch's row count (so that the generative zero-argument function can know
    /// the result array size).
    pub fun: ScalarFunctionImplementation,
}

impl Debug for ScalarUDF {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("ScalarUDF")
            .field("name", &self.name)
            .field("signature", &self.signature)
            .field("fun", &"<FUNC>")
            .finish()
    }
}

impl PartialEq for ScalarUDF {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.signature == other.signature
    }
}

impl Eq for ScalarUDF {}

impl std::hash::Hash for ScalarUDF {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.signature.hash(state);
    }
}

impl ScalarUDF {
    /// Create a new ScalarUDF
    pub fn new(
        name: &str,
        signature: &Signature,
        return_type: &ReturnTypeFunction,
        fun: &ScalarFunctionImplementation,
    ) -> Self {
        Self {
            name: name.to_owned(),
            signature: signature.clone(),
            return_type: return_type.clone(),
            fun: fun.clone(),
        }
    }

    /// creates a logical expression with a call of the UDF
    /// This utility allows using the UDF without requiring access to the registry.
    pub fn call(&self, args: Vec<Expr>) -> Expr {
        Expr::ScalarUDF(crate::expr::ScalarUDF::new(Arc::new(self.clone()), args))
    }
}

pub fn check_scalar_arg_count(
    func_name: &str,
    input_types: &[arrow::datatypes::DataType],
    signature: &crate::TypeSignature,
) -> Result<(), datafusion_common::DataFusionError> {
    use crate::TypeSignature;
    use datafusion_common::DataFusionError;
    match signature {
        TypeSignature::Uniform(func_count, _) | TypeSignature::Any(func_count) => {
            if input_types.len() != *func_count {
                return Err(DataFusionError::Plan(format!(
                    "The function {:?} expects {:?} arguments, but {:?} were provided",
                    func_name,
                    func_count,
                    input_types.len()
                )));
            }
        }
        TypeSignature::Exact(types) => {
            if types.len() != input_types.len() {
                return Err(DataFusionError::Plan(format!(
                    "The function {:?} expects {:?} arguments, but {:?} were provided",
                    func_name,
                    types.len(),
                    input_types.len()
                )));
            }
        }
        TypeSignature::OneOf(variants) => {
            let ok = variants
                .iter()
                .any(|v| check_scalar_arg_count(func_name, input_types, v).is_ok());
            if !ok {
                return Err(DataFusionError::Plan(format!(
                    "The function {:?} does not accept {:?} function arguments.",
                    func_name,
                    input_types.len()
                )));
            }
        }
        TypeSignature::VariadicAny
        | TypeSignature::Variadic(_)
        | TypeSignature::VariadicEqual => {
            if input_types.is_empty() {
                return Err(DataFusionError::Plan(format!(
                    "Scalar function {func_name} expects at least one argument"
                )));
            }
        }
    }
    Ok(())
}
