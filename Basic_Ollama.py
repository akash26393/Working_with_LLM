import ollama

response = ollama.chat(
    model='qwen3-coder:480b-cloud',
    messages=[
        {
            'role': 'user',
            'content': 'Write 7 day name',
        },
    ])

print(response)
print(response["message"]["content"])