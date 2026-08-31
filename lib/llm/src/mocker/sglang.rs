// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native SGLang `/generate` response shaping for the mocker engine.

use dynamo_mocker::sglang::{LogprobOptions, ResponseMetadata};
use dynamo_runtime::error::{DynamoError, ErrorType};
use serde_json::{Map, Value, json};

use crate::protocols::common::FinishReason;
use crate::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};

const PAYLOAD_KEY: &str = "sglang_tito";

#[derive(Debug)]
pub(super) struct ResponseAdapter {
    metadata: ResponseMetadata,
}

impl ResponseAdapter {
    pub(super) fn from_request(
        request: &PreprocessedRequest,
        fallback_request_id: &str,
    ) -> Result<Option<Self>, DynamoError> {
        let Some(payload) = request
            .extra_args
            .as_ref()
            .and_then(Value::as_object)
            .and_then(|extra| extra.get(PAYLOAD_KEY))
        else {
            return Ok(None);
        };
        let payload = payload
            .as_object()
            .ok_or_else(|| invalid_argument("extra_args.sglang_tito must be a JSON object"))?;

        let request_id = optional_string(payload, "rid")?
            .unwrap_or(fallback_request_id)
            .to_string();
        let logprob_options = LogprobOptions::new(
            optional_bool(payload, "return_logprob")?.unwrap_or(false),
            optional_i64(payload, "top_logprobs_num")?.unwrap_or(0),
            optional_i64(payload, "logprob_start_len")?.unwrap_or(-1),
        )
        .map_err(invalid_argument)?;
        Ok(Some(Self {
            metadata: ResponseMetadata::new(request_id, &request.token_ids, logprob_options),
        }))
    }

    pub(super) fn replay_key(&self) -> &str {
        self.metadata.request_id()
    }

    pub(super) fn adapt(&self, output: &mut LLMEngineOutput, completion_tokens: usize) {
        let response = self.metadata.response(
            &output.token_ids,
            completion_tokens,
            output.finish_reason.as_ref().map(native_finish_reason),
        );
        output.engine_data = Some(json!({"sglang_response": response}));
    }
}

fn optional_string<'a>(
    payload: &'a Map<String, Value>,
    field: &str,
) -> Result<Option<&'a str>, DynamoError> {
    match payload.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value)),
        Some(_) => Err(invalid_argument(format!(
            "extra_args.sglang_tito.{field} must be a string"
        ))),
    }
}

fn optional_bool(payload: &Map<String, Value>, field: &str) -> Result<Option<bool>, DynamoError> {
    match payload.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(invalid_argument(format!(
            "extra_args.sglang_tito.{field} must be a boolean"
        ))),
    }
}

fn optional_i64(payload: &Map<String, Value>, field: &str) -> Result<Option<i64>, DynamoError> {
    match payload.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(value)) => value.as_i64().map(Some).ok_or_else(|| {
            invalid_argument(format!("extra_args.sglang_tito.{field} must be an integer"))
        }),
        Some(_) => Err(invalid_argument(format!(
            "extra_args.sglang_tito.{field} must be an integer"
        ))),
    }
}

fn native_finish_reason(reason: &FinishReason) -> Value {
    match reason {
        FinishReason::Length => json!({"type": "length"}),
        FinishReason::EoS | FinishReason::Stop => json!({"type": "stop"}),
        FinishReason::Cancelled => json!({
            "type": "abort",
            "message": "request was cancelled",
        }),
        FinishReason::Error(message) => json!({
            "type": "abort",
            "message": message,
        }),
        FinishReason::ContentFilter => json!({
            "type": "abort",
            "message": "generation stopped by content filter",
        }),
    }
}

fn invalid_argument(message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::InvalidArgument)
        .message(message.into())
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};

    fn request(extra_args: Option<Value>) -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("mock".to_string())
            .token_ids(vec![11, 12, 13])
            .stop_conditions(StopConditions {
                max_tokens: Some(2),
                ..Default::default()
            })
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .extra_args(extra_args)
            .build()
            .unwrap()
    }

    #[test]
    fn recognizes_only_native_sglang_envelopes() {
        assert!(
            ResponseAdapter::from_request(&request(None), "fallback")
                .unwrap()
                .is_none()
        );

        let adapter = ResponseAdapter::from_request(
            &request(Some(json!({
                "sglang_tito": {
                    "rid": "resolved-id",
                    "future_field": {"opaque": true},
                }
            }))),
            "fallback",
        )
        .unwrap()
        .unwrap();
        assert_eq!(adapter.replay_key(), "resolved-id");
    }

    #[test]
    fn emits_incremental_and_terminal_native_responses() {
        let adapter = ResponseAdapter::from_request(
            &request(Some(json!({"sglang_tito": {"rid": "request-1"}}))),
            "fallback",
        )
        .unwrap()
        .unwrap();
        let mut token = LLMEngineOutput {
            token_ids: vec![101],
            ..Default::default()
        };
        adapter.adapt(&mut token, 1);
        let response = &token.engine_data.as_ref().unwrap()["sglang_response"];
        assert_eq!(response["output_ids"], json!([101]));
        assert_eq!(response["meta_info"]["id"], "request-1");
        assert_eq!(response["meta_info"]["prompt_tokens"], 3);
        assert_eq!(response["meta_info"]["completion_tokens"], 1);
        assert!(response["meta_info"]["finish_reason"].is_null());

        let mut terminal = LLMEngineOutput::length();
        adapter.adapt(&mut terminal, 1);
        let response = &terminal.engine_data.as_ref().unwrap()["sglang_response"];
        assert_eq!(response["output_ids"], json!([]));
        assert_eq!(response["meta_info"]["completion_tokens"], 1);
        assert_eq!(
            response["meta_info"]["finish_reason"],
            json!({"type": "length"})
        );
    }

    #[test]
    fn shapes_requested_logprobs_like_sglang() {
        let adapter = ResponseAdapter::from_request(
            &request(Some(json!({
                "sglang_tito": {
                    "rid": "logprobs",
                    "return_logprob": true,
                    "top_logprobs_num": 2,
                    "logprob_start_len": 0,
                }
            }))),
            "fallback",
        )
        .unwrap()
        .unwrap();
        let mut token = LLMEngineOutput {
            token_ids: vec![107],
            ..Default::default()
        };
        adapter.adapt(&mut token, 1);
        let meta = &token.engine_data.as_ref().unwrap()["sglang_response"]["meta_info"];
        assert_eq!(meta["output_token_logprobs"][0][1], 107);
        assert_eq!(meta["output_top_logprobs"][0].as_array().unwrap().len(), 2);
        assert!(meta.get("input_token_logprobs").is_none());

        let mut terminal = LLMEngineOutput::length();
        adapter.adapt(&mut terminal, 1);
        let meta = &terminal.engine_data.as_ref().unwrap()["sglang_response"]["meta_info"];
        assert_eq!(meta["input_token_logprobs"].as_array().unwrap().len(), 3);
        assert!(meta["input_token_logprobs"][0][0].is_null());
        assert_eq!(meta["input_top_logprobs"][0], Value::Null);
    }

    #[test]
    fn maps_cancellation_and_errors_to_abort() {
        let adapter = ResponseAdapter::from_request(
            &request(Some(json!({"sglang_tito": {"rid": "terminal"}}))),
            "fallback",
        )
        .unwrap()
        .unwrap();

        for (mut output, expected_message) in [
            (
                LLMEngineOutput::error("backend failed".to_string()),
                "backend failed",
            ),
            (LLMEngineOutput::cancelled(), "request was cancelled"),
        ] {
            adapter.adapt(&mut output, 0);
            let finish = &output.engine_data.as_ref().unwrap()["sglang_response"]["meta_info"]["finish_reason"];
            assert_eq!(finish["type"], "abort");
            assert_eq!(finish["message"], expected_message);
        }
    }
}
