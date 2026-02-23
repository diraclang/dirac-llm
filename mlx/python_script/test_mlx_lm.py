

# MLX-LM expects the model to be downloaded via CLI first.
# Example: python -m mlx_lm convert --hf-path mlx-community/Meta-Llama-3-8B-Instruct

from mlx_lm import load, generate

model_path = "./mlx_model"  # Path where the model is downloaded
model, tokenizer = load(model_path)

messages = [
	{"role": "user", "content": "Hello!"},
	{"role": "assistant", "content": "Hi, how can I help you?"},
	{"role": "user", "content": "Tell me a joke."}
]

# Use chat template if available
if hasattr(tokenizer, "apply_chat_template"):
	prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
else:
	# Fallback: simple concatenation
	prompt = "\n".join([m["content"] for m in messages])

response = generate(model, tokenizer, prompt, max_tokens=64)
print("Model output:", response)
