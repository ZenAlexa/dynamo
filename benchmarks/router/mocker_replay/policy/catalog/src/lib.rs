// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Catalog linked by the mocker replay custom-policy wheel build.

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyRegistry, WorkerSelectionPolicyRegistryError,
};

/// Register the policy types supplied by this demonstration catalog.
pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    session_cohort_sglang_policy::register(registry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_session_cohort_policy() {
        let mut registry = WorkerSelectionPolicyRegistry::default();
        register(&mut registry).unwrap();
        assert!(session_cohort_sglang_policy::register(&mut registry).is_err());
    }
}
