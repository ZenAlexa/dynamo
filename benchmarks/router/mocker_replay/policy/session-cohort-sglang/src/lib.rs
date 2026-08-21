// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic per-session worker cohorts with SGLang-style cache/load selection.

mod picker;

use std::sync::Arc;

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{KvRouterConfig, WorkerSelectionPolicy};
use picker::SessionCohortPicker;

const DEFAULT_COHORT_SIZE: usize = 4;
const DEFAULT_CACHE_THRESHOLD: f64 = 0.3;
const DEFAULT_LOAD_BALANCE_ABS_THRESHOLD: usize = 284;
const DEFAULT_LOAD_BALANCE_REL_THRESHOLD: f64 = 1.5;
const DEFAULT_HASH_SEED: u64 = 0x5345_5353_494f_4e31;

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Parameters {
    cohort_size: usize,
    cache_threshold: f64,
    load_balance_abs_threshold: usize,
    load_balance_rel_threshold: f64,
    hash_seed: u64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            cohort_size: DEFAULT_COHORT_SIZE,
            cache_threshold: DEFAULT_CACHE_THRESHOLD,
            load_balance_abs_threshold: DEFAULT_LOAD_BALANCE_ABS_THRESHOLD,
            load_balance_rel_threshold: DEFAULT_LOAD_BALANCE_REL_THRESHOLD,
            hash_seed: DEFAULT_HASH_SEED,
        }
    }
}

fn validate(parameters: Parameters) -> Result<Parameters, WorkerSelectionPolicyProviderError> {
    if parameters.cohort_size == 0 {
        return Err(WorkerSelectionPolicyProviderError::new(
            "cohort_size must be greater than zero",
        ));
    }
    if !parameters.cache_threshold.is_finite() || !(0.0..=1.0).contains(&parameters.cache_threshold)
    {
        return Err(WorkerSelectionPolicyProviderError::new(
            "cache_threshold must be a finite number from 0 through 1",
        ));
    }
    if !parameters.load_balance_rel_threshold.is_finite()
        || parameters.load_balance_rel_threshold < 1.0
    {
        return Err(WorkerSelectionPolicyProviderError::new(
            "load_balance_rel_threshold must be a finite number at least 1",
        ));
    }
    Ok(parameters)
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters = validate(parameters.deserialize()?)?;
    Ok(Arc::new(
        move |config: &KvRouterConfig, worker_type, _partition| {
            WorkerSelectionPolicy::new(
                config.clone(),
                worker_type.as_str(),
                Vec::new(),
                Box::new(SessionCohortPicker::new(parameters)),
            )
        },
    ))
}

/// Register the `session-cohort-sglang` policy type.
pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register("session-cohort-sglang", Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_parameters() {
        assert!(validate(Parameters::default()).is_ok());

        let mut invalid = Parameters::default();
        invalid.cohort_size = 0;
        assert!(validate(invalid).is_err());

        invalid = Parameters::default();
        invalid.cache_threshold = f64::NAN;
        assert!(validate(invalid).is_err());

        invalid = Parameters::default();
        invalid.cache_threshold = 1.01;
        assert!(validate(invalid).is_err());

        invalid = Parameters::default();
        invalid.load_balance_rel_threshold = 0.99;
        assert!(validate(invalid).is_err());
    }
}
