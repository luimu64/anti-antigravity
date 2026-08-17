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

def test_internal_to_openai_response():
    internal_result = {
        "responseId": "test-resp-123",
        "text": "The answer is 42.",
        "thoughts": "Thinking deeply...",
        "finishReason": "STOP",
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
            "thoughtsTokenCount": 8
        }
    }
    openai_resp = OpenAITranslator.internal_to_openai_response(internal_result, "gpt-4o")
    assert openai_resp["id"] == "test-resp-123"
    assert openai_resp["model"] == "gpt-4o"
    assert openai_resp["choices"][0]["message"]["content"] == "The answer is 42."
    assert openai_resp["choices"][0]["message"]["reasoning_content"] == "Thinking deeply..."
    assert openai_resp["choices"][0]["finish_reason"] == "stop"
    assert openai_resp["usage"]["total_tokens"] == 15

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
