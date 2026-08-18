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
