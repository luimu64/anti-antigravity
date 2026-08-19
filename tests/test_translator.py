import pytest
import json
import asyncio
from app.translator import OpenAITranslator, ChatCompletionRequest

def test_resolve_model():
    assert OpenAITranslator.resolve_model("gpt-4o") == "gemini-3.7-flash-high"
    assert OpenAITranslator.resolve_model("claude-3-7-sonnet") == "claude-sonnet-4-6"
    assert OpenAITranslator.resolve_model("claude-opus-4-6-thinking") == "claude-opus-4-6-thinking"
    assert OpenAITranslator.resolve_model("gemini-3.7-flash-high") == "gemini-3.7-flash-high"

def test_openai_to_internal_simple_messages():
    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "System instructions here"},
            {"role": "user", "content": "Hello assistant"},
            {"role": "assistant", "content": "Hello user"}
        ],
        temperature=0.7,
        max_tokens=2048
    )
    internal_model, contents, sys_inst, gen_cfg, tools = OpenAITranslator.openai_to_internal_request(req)
    
    assert internal_model == "gemini-3.7-flash-high"
    assert sys_inst == {"parts": [{"text": "System instructions here"}]}
    assert len(contents) == 2
    assert contents[0] == {"role": "user", "parts": [{"text": "Hello assistant"}]}
    assert contents[1] == {"role": "model", "parts": [{"text": "Hello user"}]}
    assert gen_cfg["temperature"] == 0.7
    assert gen_cfg["maxOutputTokens"] == 2048

def test_openai_to_internal_tools():
    req = ChatCompletionRequest(
        model="claude-sonnet-4-6",
        messages=[
            {"role": "user", "content": "What is the weather?"}
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"]
                    }
                }
            }
        ]
    )
    internal_model, contents, sys_inst, gen_cfg, tools = OpenAITranslator.openai_to_internal_request(req)
    assert tools is not None
    assert len(tools) == 1
    assert "functionDeclarations" in tools[0]
    decl = tools[0]["functionDeclarations"][0]
    assert decl["name"] == "get_weather"
    assert decl["description"] == "Get current weather"

def test_openai_to_internal_advanced_spec_features():
    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[
            {"role": "developer", "content": "Developer instructions"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Listen to this and answer"},
                    {"type": "input_audio", "input_audio": {"data": "UklGRg...", "format": "mp3"}}
                ]
            }
        ],
        stop=["\n\n", "User:"],
        presence_penalty=0.5,
        frequency_penalty=0.3,
        seed=42,
        n=1,
        response_format={"type": "json_schema", "json_schema": {"schema": {"type": "object", "properties": {"name": {"type": "string"}}}}},
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather"
                }
            }
        ]
    )
    internal_model, contents, sys_inst, gen_cfg, tools = OpenAITranslator.openai_to_internal_request(req)
    
    assert internal_model == "gemini-3.7-flash-high"
    assert sys_inst == {"parts": [{"text": "Developer instructions"}]}
    assert len(contents) == 1
    assert contents[0]["parts"][0] == {"text": "Listen to this and answer"}
    assert contents[0]["parts"][1]["inlineData"]["mimeType"] == "audio/mp3"
    assert contents[0]["parts"][1]["inlineData"]["data"] == "UklGRg..."
    
    # Check generation config
    assert gen_cfg["stopSequences"] == ["\n\n", "User:"]
    assert gen_cfg["presencePenalty"] == 0.5
    assert gen_cfg["frequencyPenalty"] == 0.3
    assert gen_cfg["seed"] == 42
    assert gen_cfg["responseMimeType"] == "application/json"
    assert gen_cfg["responseSchema"] == {"type": "object", "properties": {"name": {"type": "string"}}}
    
    # Check tools & toolConfig
    assert len(tools) == 2
    assert "functionDeclarations" in tools[0]
    assert tools[1]["functionCallingConfig"]["mode"] == "ANY"
    assert tools[1]["functionCallingConfig"]["allowedFunctionNames"] == ["get_weather"]

def test_internal_to_openai_response_with_caching_and_reasoning():
    internal_result = {
        "responseId": "test-resp-123",
        "text": "The answer is 42.",
        "thoughts": "Thinking deeply...",
        "finishReason": "STOP",
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 50,
            "totalTokenCount": 150,
            "cachedContentTokenCount": 80,
            "thoughtsTokenCount": 30
        }
    }
    openai_resp = OpenAITranslator.internal_to_openai_response(internal_result, "gpt-4o")
    assert openai_resp["id"] == "test-resp-123"
    assert openai_resp["model"] == "gpt-4o"
    assert openai_resp["choices"][0]["message"]["content"] == "The answer is 42."
    assert openai_resp["choices"][0]["message"]["reasoning_content"] == "Thinking deeply..."
    assert openai_resp["choices"][0]["finish_reason"] == "stop"
    assert openai_resp["usage"]["prompt_tokens"] == 100
    assert openai_resp["usage"]["completion_tokens"] == 50
    assert openai_resp["usage"]["total_tokens"] == 150
    assert openai_resp["usage"]["prompt_tokens_details"]["cached_tokens"] == 80
    assert openai_resp["usage"]["completion_tokens_details"]["reasoning_tokens"] == 30

@pytest.mark.asyncio
async def test_streaming_chunks():
    async def mock_events():
        yield {
            "response": {
                "candidates": [{"content": {"role": "model", "parts": [{"text": "Hello"}]}}],
                "responseId": "resp_001"
            }
        }
        yield {
            "response": {
                "candidates": [{"content": {"role": "model", "parts": [{"text": " world!"}]}, "finishReason": "STOP"}],
                "responseId": "resp_001"
            }
        }

    chunks = []
    async for chunk_str in OpenAITranslator.internal_stream_to_openai_chunks(mock_events(), "gemini-3.7-flash-high"):
        chunks.append(chunk_str)

    assert len(chunks) >= 3
    assert chunks[-1] == "data: [DONE]\n\n"
    first_chunk = json.loads(chunks[0].replace("data: ", ""))
    assert first_chunk["choices"][0]["delta"]["content"] == "Hello"

@pytest.mark.asyncio
async def test_streaming_chunks_with_include_usage_and_caching():
    async def mock_unwrapped_events():
        # Test unwrapped Google Cloud Code SSE events
        yield {
            "candidates": [{"content": {"role": "model", "parts": [{"text": "Testing", "thought": False}]}}],
            "responseId": "resp_002"
        }
        yield {
            "candidates": [{"content": {"role": "model", "parts": [{"text": " usage", "thought": False}]}}],
            "finishReason": "STOP",
            "usageMetadata": {
                "promptTokenCount": 200,
                "candidatesTokenCount": 45,
                "totalTokenCount": 245,
                "cachedContentTokenCount": 150,
                "thoughtsTokenCount": 20
            }
        }

    chunks = []
    async for chunk_str in OpenAITranslator.internal_stream_to_openai_chunks(
        mock_unwrapped_events(),
        "gemini-3.7-flash-high",
        include_usage=True
    ):
        chunks.append(chunk_str)

    assert chunks[-1] == "data: [DONE]\n\n"
    usage_chunk_str = chunks[-2]
    assert usage_chunk_str.startswith("data: ")
    usage_chunk = json.loads(usage_chunk_str.replace("data: ", ""))
    assert usage_chunk["choices"] == []
    assert "usage" in usage_chunk
    usage = usage_chunk["usage"]
    assert usage["prompt_tokens"] == 200
    assert usage["completion_tokens"] == 45
    assert usage["total_tokens"] == 245
    assert usage["prompt_tokens_details"]["cached_tokens"] == 150
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 20

def test_audio_content_parts_standard_and_legacy():
    # Test standard "audio" type and legacy "input_audio"
    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": {"data": "AUDIO_MP3_DATA", "format": "mp3"}},
                    {"type": "audio", "audio": {"data": "AUDIO_WAV_DEFAULT"}},
                    {"type": "input_audio", "input_audio": {"data": "LEGACY_WAV_DATA", "format": "wav"}}
                ]
            }
        ]
    )
    _, contents, _, _, _ = OpenAITranslator.openai_to_internal_request(req)
    assert len(contents) == 1
    parts = contents[0]["parts"]
    assert len(parts) == 3
    assert parts[0]["inlineData"]["mimeType"] == "audio/mp3"
    assert parts[0]["inlineData"]["data"] == "AUDIO_MP3_DATA"
    assert parts[1]["inlineData"]["mimeType"] == "audio/wav"
    assert parts[1]["inlineData"]["data"] == "AUDIO_WAV_DEFAULT"
    assert parts[2]["inlineData"]["mimeType"] == "audio/wav"
    assert parts[2]["inlineData"]["data"] == "LEGACY_WAV_DATA"

def test_multi_candidate_non_streaming():
    internal_result = {
        "responseId": "resp_multi_001",
        "candidates": [
            {
                "index": 0,
                "text": "Candidate 1 text response",
                "thoughts": "Thinking candidate 1...",
                "finishReason": "STOP",
                "toolCalls": []
            },
            {
                "index": 1,
                "text": "Candidate 2 alternative response",
                "thoughts": "Thinking candidate 2...",
                "finishReason": "STOP",
                "toolCalls": []
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 50,
            "candidatesTokenCount": 60,
            "totalTokenCount": 110
        }
    }
    openai_resp = OpenAITranslator.internal_to_openai_response(internal_result, "gpt-4o")
    assert len(openai_resp["choices"]) == 2
    assert openai_resp["choices"][0]["index"] == 0
    assert openai_resp["choices"][0]["message"]["content"] == "Candidate 1 text response"
    assert openai_resp["choices"][0]["message"]["reasoning_content"] == "Thinking candidate 1..."
    assert openai_resp["choices"][1]["index"] == 1
    assert openai_resp["choices"][1]["message"]["content"] == "Candidate 2 alternative response"
    assert openai_resp["choices"][1]["message"]["reasoning_content"] == "Thinking candidate 2..."

@pytest.mark.asyncio
async def test_multi_candidate_streaming():
    async def mock_multi_candidate_stream():
        # Stream chunks for candidate 0 and candidate 1
        yield {
            "candidates": [
                {"index": 0, "content": {"role": "model", "parts": [{"text": "Hello from 0"}]}},
                {"index": 1, "content": {"role": "model", "parts": [{"text": "Hello from 1"}]}}
            ]
        }
        yield {
            "candidates": [
                {"index": 0, "content": {"role": "model", "parts": []}, "finishReason": "STOP"},
                {"index": 1, "content": {"role": "model", "parts": []}, "finishReason": "STOP"}
            ]
        }

    chunks = []
    async for chunk_str in OpenAITranslator.internal_stream_to_openai_chunks(
        mock_multi_candidate_stream(),
        "gemini-3.7-flash-high"
    ):
        chunks.append(chunk_str)

    assert chunks[-1] == "data: [DONE]\n\n"
    # Parse chunks
    parsed_chunks = [json.loads(c.replace("data: ", "")) for c in chunks if c.startswith("data: {")]
    
    # Verify candidate indices in choices
    cand_0_chunks = [c for c in parsed_chunks if c["choices"] and c["choices"][0]["index"] == 0]
    cand_1_chunks = [c for c in parsed_chunks if c["choices"] and c["choices"][0]["index"] == 1]
    assert len(cand_0_chunks) >= 2
    assert len(cand_1_chunks) >= 2
    assert cand_0_chunks[0]["choices"][0]["delta"]["content"] == "Hello from 0"
    assert cand_1_chunks[0]["choices"][0]["delta"]["content"] == "Hello from 1"
    assert cand_0_chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert cand_1_chunks[-1]["choices"][0]["finish_reason"] == "stop"

def test_file_document_attachments():
    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze these documents"},
                    {"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBERi0xLjQK..."}},
                    {"type": "file", "data": "BASE64_DOC_DATA", "mime_type": "text/plain"},
                    {"type": "file", "url": "https://example.com/doc.pdf"}
                ]
            }
        ]
    )
    _, contents, _, _, _ = OpenAITranslator.openai_to_internal_request(req)
    assert len(contents) == 1
    parts = contents[0]["parts"]
    assert len(parts) == 4
    assert parts[0]["text"] == "Analyze these documents"
    assert parts[1]["inlineData"]["mimeType"] == "application/pdf"
    assert parts[1]["inlineData"]["data"] == "JVBERi0xLjQK..."
    assert parts[2]["inlineData"]["mimeType"] == "text/plain"
    assert parts[2]["inlineData"]["data"] == "BASE64_DOC_DATA"
    assert parts[3]["text"] == "[File URL: https://example.com/doc.pdf]"

def test_internal_to_openai_text_completion():
    internal_result = {
        "responseId": "cmpl-test-123",
        "text": "Completed text.",
        "finishReason": "STOP",
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15
        }
    }
    resp = OpenAITranslator.internal_to_openai_text_completion(internal_result, "gemini-3.7-flash-high")
    assert resp["object"] == "text_completion"
    assert resp["id"] == "cmpl-test-123"
    assert resp["model"] == "gemini-3.7-flash-high"
    assert "system_fingerprint" in resp
    assert resp["service_tier"] == "default"
    assert len(resp["choices"]) == 1
    assert resp["choices"][0]["text"] == "Completed text."
    assert resp["choices"][0]["logprobs"] is None
    assert resp["choices"][0]["finish_reason"] == "stop"
    assert resp["usage"]["total_tokens"] == 15

@pytest.mark.asyncio
async def test_streaming_text_chunks():
    async def mock_events():
        yield {
            "response": {
                "candidates": [{"content": {"role": "model", "parts": [{"text": "Hello text completion"}]}}],
                "responseId": "resp_txt_001"
            }
        }
        yield {
            "response": {
                "candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "STOP"}],
                "responseId": "resp_txt_001"
            }
        }

    chunks = []
    async for chunk_str in OpenAITranslator.internal_stream_to_openai_text_chunks(mock_events(), "gemini-3.7-flash-high"):
        chunks.append(chunk_str)

    assert len(chunks) >= 3
    assert chunks[-1] == "data: [DONE]\n\n"
    first_chunk = json.loads(chunks[0].replace("data: ", ""))
    assert first_chunk["object"] == "text_completion"
    assert first_chunk["choices"][0]["text"] == "Hello text completion"
    assert first_chunk["choices"][0]["logprobs"] is None
    assert first_chunk["choices"][0]["finish_reason"] is None
    finish_chunk = json.loads(chunks[1].replace("data: ", ""))
    assert finish_chunk["choices"][0]["finish_reason"] == "stop"


