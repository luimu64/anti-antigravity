import time

from openai import OpenAI

from app.keys import api_key_manager


def generate_large_haystack_prompt(target_tokens=500_000):
    """
    Generate approximately target_tokens of text with a hidden needle.
    1 token ~ 3.8 - 4.0 characters.
    500k tokens ~ 1.95 - 2.0 million characters.
    """
    needle = "THE SECRET KEYWORD IS: >> HYPER-GRAVITY-PULSAR-7734 <<"

    paragraph_template = (
        "In the year 2184, astrophysics research stations across the outer solar system "
        "recorded anomalous gravitational waves originating from deep interstellar space. "
        "The telemetry logs contained detailed multi-spectral sensor sweeps, orbital decay "
        "parameters, photon flux measurements, and plasma density fluctuations. "
        "Engineers analyzed subsystem diagnostic logs numbered {idx:06d}, confirming that "
        "quantum entanglement transceivers maintained coherence across 4.2 astronomical units. "
    )

    # Each formatted paragraph is ~420 characters (~105 tokens)
    para_len_chars = len(paragraph_template.format(idx=0))
    approx_tokens_per_para = para_len_chars / 3.9
    num_paras = int(target_tokens / approx_tokens_per_para) + 50

    print(f"Generating haystack with ~{num_paras} paragraphs...")

    paras = []
    needle_inserted_at = int(num_paras * 0.65)  # place needle at 65% depth

    for i in range(num_paras):
        if i == needle_inserted_at:
            paras.append(
                f"\n--- CONFIDENTIAL ARCHIVE ENTRY ---\n{needle}\n---------------------------------\n"
            )
        paras.append(paragraph_template.format(idx=i))

    haystack = "\n".join(paras)

    user_prompt = (
        f"{haystack}\n\n"
        "QUESTION: Search the above telemetry archive and tell me what is the secret keyword? "
        "Provide only the secret keyword inside the markers."
    )

    return user_prompt, needle


def main():
    base_url = "http://127.0.0.1:8000/v1"
    key = api_key_manager.get_first_active_key() or "test-key"
    client = OpenAI(base_url=base_url, api_key=key, timeout=300.0)

    print("==================================================")
    print("🚀 LARGE PROMPT TEST (500k Tokens)")
    print("==================================================")

    # 1. Generate ~500k token prompt
    start_gen = time.time()
    prompt, needle = generate_large_haystack_prompt(target_tokens=500_000)
    gen_time = time.time() - start_gen

    char_count = len(prompt)
    mb_size = char_count / (1024 * 1024)
    est_tokens = char_count / 4.0
    print(f"Prompt Size: {char_count:,} characters ({mb_size:.2f} MB)")
    print(f"Estimated Tokens: ~{int(est_tokens):,} tokens")
    print(f"Prompt Generation Time: {gen_time:.2f}s")
    print("--------------------------------------------------")

    print(
        "Sending request to /v1/chat/completions (model: gemini-3.7-flash-high, stream=True)..."
    )
    start_req = time.time()
    ttft = None
    full_response = []
    thinking_chunks = []

    try:
        stream = client.chat.completions.create(
            model="gemini-3.7-flash-high",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        for chunk in stream:
            if ttft is None:
                ttft = time.time() - start_req
                print(f"⚡ Time to First Token (TTFT): {ttft:.2f}s")
                print("Streaming response: ", end="", flush=True)

            delta = chunk.choices[0].delta if chunk.choices else None
            if delta:
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    thinking_chunks.append(delta.reasoning_content)
                if delta.content:
                    print(delta.content, end="", flush=True)
                    full_response.append(delta.content)

        total_time = time.time() - start_req
        print("\n--------------------------------------------------")
        print(f"✅ Streaming completed in {total_time:.2f}s")
        ans_text = "".join(full_response)
        print(f"Model Answer: {ans_text}")
        if thinking_chunks:
            print(
                f"Thinking content received: ~{len(''.join(thinking_chunks))} chars of reasoning"
            )

        # Verify correctness
        if "HYPER-GRAVITY-PULSAR-7734" in ans_text:
            print("🎯 NEEDLE FOUND SUCCESSFULLY! 100% ACCURACY AT 500K CONTEXT!")
        else:
            print("⚠️ Needle check: Expected 'HYPER-GRAVITY-PULSAR-7734' in answer.")

    except Exception as e:
        print(f"❌ Streaming request failed: {e}")
        import traceback

        traceback.print_exc()

    # 3. Test non-streaming to verify exact usage token counts
    print("\n--------------------------------------------------")
    print("Testing non-streaming request to inspect usage metadata...")
    try:
        start_non_stream = time.time()
        resp = client.chat.completions.create(
            model="gemini-3.7-flash-high",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        non_stream_time = time.time() - start_non_stream
        print(f"Non-streaming completed in {non_stream_time:.2f}s")
        print(f"Answer: {resp.choices[0].message.content}")
        print(
            f"Usage: prompt_tokens={resp.usage.prompt_tokens:,}, completion_tokens={resp.usage.completion_tokens}, total_tokens={resp.usage.total_tokens:,}"
        )
        if resp.usage.completion_tokens_details:
            print(
                f"Reasoning tokens: {resp.usage.completion_tokens_details.reasoning_tokens}"
            )
    except Exception as e:
        print(f"❌ Non-streaming request failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
