// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;

use dynamo_kv_router::protocols::WorkerWithDpRank;
use dynamo_kv_router::{
    WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicyError,
};
use xxhash_rust::xxh3::xxh3_64_with_seed;

use crate::Parameters;

#[derive(Clone, Copy, Debug)]
struct CandidateSignal {
    row: usize,
    worker: WorkerWithDpRank,
    overlap_blocks: f64,
    active_requests: usize,
    rendezvous_score: u64,
}

impl CandidateSignal {
    #[cfg(test)]
    fn test(worker_id: u64, active_requests: usize, overlap_blocks: f64) -> Self {
        Self {
            row: worker_id as usize,
            worker: WorkerWithDpRank::from_worker_id(worker_id),
            overlap_blocks,
            active_requests,
            rendezvous_score: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SelectionReason {
    LoadImbalance,
    CacheHit,
    ColdLoad,
}

impl SelectionReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::LoadImbalance => "load_imbalance",
            Self::CacheHit => "cache_hit",
            Self::ColdLoad => "cold_load",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Decision {
    selected: CandidateSignal,
    reason: SelectionReason,
    match_rate: f64,
    min_active_requests: usize,
    max_active_requests: usize,
}

pub(crate) struct SessionCohortPicker {
    parameters: Parameters,
    cohort: Vec<CandidateSignal>,
}

impl SessionCohortPicker {
    pub(crate) fn new(parameters: Parameters) -> Self {
        Self {
            parameters,
            cohort: Vec::new(),
        }
    }

    fn rendezvous_score(&self, affinity_key: &str, worker: WorkerWithDpRank) -> u64 {
        let mut worker_bytes = [0_u8; 12];
        worker_bytes[..8].copy_from_slice(&worker.worker_id.to_le_bytes());
        worker_bytes[8..].copy_from_slice(&worker.dp_rank.to_le_bytes());
        let worker_seed = xxh3_64_with_seed(&worker_bytes, self.parameters.hash_seed);
        xxh3_64_with_seed(affinity_key.as_bytes(), worker_seed)
    }

    /// Compare cohort membership rank. Greater means a better rendezvous candidate.
    fn cohort_rank(left: &CandidateSignal, right: &CandidateSignal) -> Ordering {
        left.rendezvous_score
            .cmp(&right.rendezvous_score)
            .then_with(|| right.worker.cmp(&left.worker))
    }

    fn consider_for_cohort(&mut self, affinity_key: &str, mut signal: CandidateSignal) {
        signal.rendezvous_score = self.rendezvous_score(affinity_key, signal.worker);
        if self.cohort.len() < self.parameters.cohort_size {
            self.cohort.push(signal);
            return;
        }

        let Some((worst_row, worst)) = self
            .cohort
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| Self::cohort_rank(left, right))
        else {
            self.cohort.push(signal);
            return;
        };
        if Self::cohort_rank(&signal, worst).is_gt() {
            self.cohort[worst_row] = signal;
        }
    }

    fn build_fixed_cohort(
        &mut self,
        affinity_key: &str,
        mut candidates: Vec<CandidateSignal>,
        cohort_count: usize,
    ) {
        candidates.sort_unstable_by_key(|candidate| candidate.worker);
        let cohort_index = xxh3_64_with_seed(affinity_key.as_bytes(), self.parameters.hash_seed)
            as usize
            % cohort_count;
        let base_size = candidates.len() / cohort_count;
        let larger_cohorts = candidates.len() % cohort_count;
        let start = cohort_index * base_size + cohort_index.min(larger_cohorts);
        let size = base_size + usize::from(cohort_index < larger_cohorts);
        let hash_seed = self.parameters.hash_seed;
        self.cohort.extend(
            candidates
                .into_iter()
                .skip(start)
                .take(size)
                .map(|mut candidate| {
                    let mut worker_bytes = [0_u8; 12];
                    worker_bytes[..8].copy_from_slice(&candidate.worker.worker_id.to_le_bytes());
                    worker_bytes[8..].copy_from_slice(&candidate.worker.dp_rank.to_le_bytes());
                    let worker_seed = xxh3_64_with_seed(&worker_bytes, hash_seed);
                    candidate.rendezvous_score =
                        xxh3_64_with_seed(affinity_key.as_bytes(), worker_seed);
                    candidate
                }),
        );
    }

    fn least_loaded(&self) -> Option<CandidateSignal> {
        self.cohort
            .iter()
            .min_by(|left, right| {
                left.active_requests
                    .cmp(&right.active_requests)
                    .then_with(|| right.rendezvous_score.cmp(&left.rendezvous_score))
                    .then_with(|| left.worker.cmp(&right.worker))
            })
            .copied()
    }

    fn most_cached(&self) -> Option<CandidateSignal> {
        self.cohort
            .iter()
            .min_by(|left, right| {
                right
                    .overlap_blocks
                    .total_cmp(&left.overlap_blocks)
                    .then_with(|| right.rendezvous_score.cmp(&left.rendezvous_score))
                    .then_with(|| left.worker.cmp(&right.worker))
            })
            .copied()
    }

    fn choose<I>(
        &mut self,
        affinity_key: &str,
        request_blocks: u64,
        candidates: I,
    ) -> Option<Decision>
    where
        I: IntoIterator<Item = CandidateSignal>,
    {
        self.cohort.clear();
        if let Some(cohort_count) = self.parameters.fixed_cohort_count {
            self.build_fixed_cohort(affinity_key, candidates.into_iter().collect(), cohort_count);
        } else {
            for candidate in candidates {
                self.consider_for_cohort(affinity_key, candidate);
            }
        }
        let min_active_requests = self
            .cohort
            .iter()
            .map(|candidate| candidate.active_requests)
            .min()?;
        let max_active_requests = self
            .cohort
            .iter()
            .map(|candidate| candidate.active_requests)
            .max()?;
        let imbalanced = max_active_requests.saturating_sub(min_active_requests)
            > self.parameters.load_balance_abs_threshold
            && max_active_requests as f64
                > min_active_requests as f64 * self.parameters.load_balance_rel_threshold;

        let most_cached = self.most_cached()?;
        let match_rate = if request_blocks == 0 {
            0.0
        } else {
            most_cached.overlap_blocks / request_blocks as f64
        };
        let (selected, reason) = if imbalanced {
            (self.least_loaded()?, SelectionReason::LoadImbalance)
        } else if match_rate > self.parameters.cache_threshold {
            (most_cached, SelectionReason::CacheHit)
        } else {
            (self.least_loaded()?, SelectionReason::ColdLoad)
        };

        Some(Decision {
            selected,
            reason,
            match_rate,
            min_active_requests,
            max_active_requests,
        })
    }
}

impl WorkerPicker for SessionCohortPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        let cache = input
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
        let load = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        if candidates.len() != cache.len() || candidates.len() != load.len() {
            return Err(WorkerSelectionPolicyError::failed(
                "worker input columns have different lengths",
            ));
        }

        // WorkerSelectionContext deliberately exposes no request ID. This policy is
        // session-scoped, so avoid reaching into private request state and fail clearly
        // rather than silently making a non-affine request cohort.
        let session = context.session_context().ok_or_else(|| {
            WorkerSelectionPolicyError::failed(
                "session-cohort-sglang requires session context; send X-Dynamo-Session-ID",
            )
        })?;
        let (affinity_key, affinity_source) = match session.parent_session_id() {
            Some(parent) => (parent, "parent_session"),
            None => (session.session_id(), "session"),
        };
        let affinity_hash = xxh3_64_with_seed(affinity_key.as_bytes(), self.parameters.hash_seed);
        let signals = candidates.iter().zip(cache).zip(load).enumerate().map(
            |(row, ((candidate, cache), load))| CandidateSignal {
                row,
                worker: candidate.worker(),
                overlap_blocks: cache.device_overlap_blocks(),
                active_requests: load.active_requests(),
                rendezvous_score: 0,
            },
        );
        let decision = self
            .choose(affinity_key, context.request_blocks(), signals)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))?;

        tracing::debug!(
            source = "router",
            affinity_key,
            event_type = "session_cohort_selection",
            affinity_source,
            affinity_hash,
            configured_cohort_size = self.parameters.cohort_size,
            fixed_cohort_count = self.parameters.fixed_cohort_count,
            cohort = ?self.cohort,
            worker_id = decision.selected.worker.worker_id,
            dp_rank = decision.selected.worker.dp_rank,
            active_requests = decision.selected.active_requests,
            device_overlap_blocks = decision.selected.overlap_blocks,
            min_active_requests = decision.min_active_requests,
            max_active_requests = decision.max_active_requests,
            match_rate = decision.match_rate,
            reason = decision.reason.as_str(),
            "Selected worker from deterministic session cohort"
        );
        Ok(decision.selected.row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameters(cohort_size: usize) -> Parameters {
        Parameters {
            cohort_size,
            fixed_cohort_count: None,
            cache_threshold: 0.3,
            load_balance_abs_threshold: 284,
            load_balance_rel_threshold: 1.5,
            hash_seed: 7,
        }
    }

    fn worker_ids(picker: &SessionCohortPicker) -> Vec<u64> {
        let mut workers: Vec<_> = picker
            .cohort
            .iter()
            .map(|candidate| candidate.worker.worker_id)
            .collect();
        workers.sort_unstable();
        workers
    }

    #[test]
    fn cohort_and_ties_are_independent_of_candidate_order() {
        let candidates: Vec<_> = (0..24)
            .map(|worker| CandidateSignal::test(worker, 0, 0.0))
            .collect();
        let mut reversed = candidates.clone();
        reversed.reverse();

        let mut first = SessionCohortPicker::new(parameters(4));
        let first_decision = first.choose("session-a", 100, candidates).unwrap();
        let first_workers = worker_ids(&first);
        let mut second = SessionCohortPicker::new(parameters(4));
        let second_decision = second.choose("session-a", 100, reversed).unwrap();

        assert_eq!(first_workers, worker_ids(&second));
        assert_eq!(
            first_decision.selected.worker,
            second_decision.selected.worker
        );
    }

    #[test]
    fn severe_load_imbalance_overrides_cache_affinity() {
        let mut params = parameters(2);
        params.load_balance_abs_threshold = 1;
        let mut picker = SessionCohortPicker::new(params);
        let decision = picker
            .choose(
                "session-a",
                100,
                [
                    CandidateSignal::test(1, 0, 0.0),
                    CandidateSignal::test(2, 3, 100.0),
                ],
            )
            .unwrap();

        assert_eq!(decision.selected.worker.worker_id, 1);
        assert_eq!(decision.reason, SelectionReason::LoadImbalance);
    }

    #[test]
    fn hot_cache_wins_when_load_is_balanced() {
        let mut picker = SessionCohortPicker::new(parameters(2));
        let decision = picker
            .choose(
                "session-a",
                100,
                [
                    CandidateSignal::test(1, 0, 10.0),
                    CandidateSignal::test(2, 10, 80.0),
                ],
            )
            .unwrap();

        assert_eq!(decision.selected.worker.worker_id, 2);
        assert_eq!(decision.reason, SelectionReason::CacheHit);
    }

    #[test]
    fn cold_request_uses_least_loaded_cohort_worker() {
        let mut picker = SessionCohortPicker::new(parameters(2));
        let decision = picker
            .choose(
                "session-a",
                100,
                [
                    CandidateSignal::test(1, 0, 10.0),
                    CandidateSignal::test(2, 10, 20.0),
                ],
            )
            .unwrap();

        assert_eq!(decision.selected.worker.worker_id, 1);
        assert_eq!(decision.reason, SelectionReason::ColdLoad);
    }

    #[test]
    fn workers_outside_the_cohort_cannot_win() {
        let base: Vec<_> = (0..6)
            .map(|worker| CandidateSignal::test(worker, 0, 0.0))
            .collect();
        let mut picker = SessionCohortPicker::new(parameters(2));
        picker.choose("session-a", 100, base.clone()).unwrap();
        let cohort = worker_ids(&picker);
        let excluded = (0..6).find(|worker| !cohort.contains(worker)).unwrap();
        let candidates = base.into_iter().map(|mut candidate| {
            if candidate.worker.worker_id == excluded {
                candidate.overlap_blocks = 100.0;
            }
            candidate
        });

        let decision = picker.choose("session-a", 100, candidates).unwrap();
        assert_ne!(decision.selected.worker.worker_id, excluded);
        assert_eq!(cohort, worker_ids(&picker));
    }

    #[test]
    fn fixed_cohorts_are_six_disjoint_groups_of_four() {
        let candidates: Vec<_> = (0..24)
            .map(|worker| CandidateSignal::test(worker, 0, 0.0))
            .collect();
        let mut params = parameters(4);
        params.fixed_cohort_count = Some(6);
        let mut picker = SessionCohortPicker::new(params);
        let mut observed = std::collections::BTreeSet::new();

        for session in 0..1_000 {
            picker
                .choose(&format!("session-{session}"), 100, candidates.clone())
                .unwrap();
            let cohort = worker_ids(&picker);
            assert_eq!(cohort.len(), 4);
            observed.insert(cohort);
        }

        assert_eq!(observed.len(), 6);
        let workers = observed.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(workers, (0..24).collect::<Vec<_>>());
    }
}
