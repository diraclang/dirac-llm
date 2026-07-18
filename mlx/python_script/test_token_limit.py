#!/usr/bin/env python3
"""
Test script to verify token limit handling in the chat server.
This creates a very large prompt to test truncation.
"""

import requests
import json

# Create a large conversation history
messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

# Add many messages to exceed the token limit
for i in range(100):
    messages.append({
        "role": "user",
        "content": f"This is user message {i}. " + "Here is some filler text. " * 200
    })
    messages.append({
        "role": "assistant", 
        "content": f"This is assistant response {i}. " + "Here is more filler text. " * 200
    })

# Add final question
messages.append({
    "role": "user",
    "content": "What is 2 + 2?"
})

print(f"Total messages: {len(messages)}")
print(f"Estimated tokens: {sum(len(m['content'].split()) for m in messages) * 1.3:.0f}")

# Send to chat endpoint
response = requests.post(
    "http://localhost:5001/chat",
    json={"messages": messages}
)

print(f"\nStatus: {response.status_code}")
print(f"Response: {response.json()}")
