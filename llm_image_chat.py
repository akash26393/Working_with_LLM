"""
This was simple code to provide image to vision ollam model and ask model to describe it.
But later change to code which also checks which model takes lesser time.
"""

import time
import ollama

models = [
    "ministral-3:14b-cloud",
    "qwen3-vl:235b-cloud",
    "qwen3.5:397b-cloud",
    "gemini-3-flash-preview:cloud",
    "gemma3:27b-cloud",
    "gemma3:12b-cloud",
]

messages = [
    {
        "role": "user",
        "content": "describe the image in one line",
        "images": ["image.jpg"],
    }
]

results = []

for model_name in models:
    print(f"\n--- Running model: {model_name} ---")
    start_time = time.perf_counter()
    try:
        response = ollama.chat(model=model_name, messages=messages)
        elapsed_seconds = time.perf_counter() - start_time
        print(response["message"]["content"])
        print(f"Time taken ({model_name}): {elapsed_seconds:.2f} seconds")
        results.append((model_name, elapsed_seconds, "success"))
    except Exception as error:
        elapsed_seconds = time.perf_counter() - start_time
        print(f"Error from {model_name}: {error}")
        print(f"Time taken before failure ({model_name}): {elapsed_seconds:.2f} seconds")
        results.append((model_name, elapsed_seconds, "failed"))

print("\n=== Summary ===")
for model_name, elapsed_seconds, status in results:
    print(f"{model_name}: {elapsed_seconds:.2f} seconds ({status})")