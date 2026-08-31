"""
Stateless HTTP chat endpoint for dirac integration (Qwen fine-tune).

Loads the fine-tuned Qwen model from:
    Priority 1: MLX_MODEL_PATH environment override
    Priority 2: llm_models/qwen

This mirrors stateless_chat_server_train.py (used for the Mistral fine-tune)
but is hardcoded to the Qwen model directory instead of toggling between
model_extended_A/B via the .current file.

If you need adapter-only inference, set MLX_ADAPTER_PATH explicitly. The server
does not auto-apply sibling adapters from fused model directories.

Just run: python stateless_chat_server_train_qwen.py
"""

import os
from pathlib import Path

try:
    from flask import Flask, request, jsonify
    from mlx_lm import generate, load
except ModuleNotFoundError as exc:
    missing_module = exc.name or "required dependency"
    raise SystemExit(
        "Missing Python dependency: "
        f"{missing_module}. Run 'bash setup.sh' and then start with "
        "'source .venv/bin/activate && python mlx/python_script/stateless_chat_server_train_qwen.py' "
        "or '.venv/bin/python mlx/python_script/stateless_chat_server_train_qwen.py'."
    ) from exc

app = Flask(__name__)

script_dir = Path(__file__).parent.resolve()
mlx_dir = script_dir.parent
llm_models_dir = mlx_dir / "llm_models"


def has_adapter_files(candidate: Path) -> bool:
    return (candidate / "adapters.safetensors").exists() or (candidate / "adapter_config.json").exists()


env_model_path = os.environ.get("MLX_MODEL_PATH")
env_adapter_path = os.environ.get("MLX_ADAPTER_PATH")

if env_model_path:
    model_path_base = Path(env_model_path).expanduser().resolve()
else:
    model_path_base = llm_models_dir / "qwen"

adapter_path = None

if env_adapter_path:
    selected_adapter = Path(env_adapter_path).expanduser().resolve()
    if not has_adapter_files(selected_adapter):
        raise SystemExit(
            f"\n❌ MLX_ADAPTER_PATH does not contain adapters: {selected_adapter}\n"
            "Expected adapters.safetensors or adapter_config.json in that directory.\n"
        )
    adapter_path = str(selected_adapter)

if (model_path_base / "config.json").exists():
    model_path = str(model_path_base)
    if adapter_path:
        print(f"Using model: {model_path} with explicit adapters: {adapter_path}")
    elif env_model_path:
        print(f"Using model: {model_path} (explicit override)")
    else:
        print(f"Using model: {model_path} (fused, no separate adapters)")
else:
    raise SystemExit(
        f"\n❌ No model found in {llm_models_dir}\n"
        f"Expected one of:\n"
        f"  - {model_path_base}/config.json\n"
    )

print(f"Loading model from {model_path}...")
model, tokenizer = load(model_path, adapter_path=adapter_path)

print("Model loaded successfully!")

@app.route("/chat", methods=["POST"])
def chat():
    """
    Stateless chat endpoint that accepts JSON with 'messages' array.
    The 'messages' contains the full conversation history as structured messages.
    Returns JSON with 'response' field.
    """
    data = request.get_json()

    # Accept either 'messages' array (new format) or 'message' string (legacy)
    if "messages" in data:
        messages = data["messages"]
    elif "message" in data:
        # Legacy format: parse flattened string
        user_input = data["message"].strip()
        if not user_input:
            return jsonify({"error": "No message provided."}), 400

        # Parse the concatenated message into structured messages
        messages = []
        lines = user_input.split('\n')

        for line in lines:
            if ':' in line:
                role, content = line.split(':', 1)
                role = role.strip().lower()
                content = content.strip()

                if role in ['user', 'assistant', 'system']:
                    messages.append({"role": role, "content": content})

        if not messages:
            messages = [{"role": "user", "content": user_input}]
    else:
        return jsonify({"error": "No messages or message provided."}), 400

    # Add system message if not present, otherwise move it to the front
    system_messages = [m for m in messages if m["role"] == "system"]
    non_system_messages = [m for m in messages if m["role"] != "system"]

    if not system_messages:
        # No system message, add default one
        messages = [{
            "role": "system",
            "content": "You are a helpful assistant that can answer general questions and write DIRAC code when asked."
        }] + non_system_messages
    else:
        # Consolidate all system messages into one at the front
        consolidated_system = "\n\n".join(m["content"] for m in system_messages)
        messages = [{
            "role": "system",
            "content": consolidated_system
        }] + non_system_messages

    # Ensure strict alternation: user/assistant/user/assistant after system message
    # The chat template requires this pattern
    consolidated = [messages[0]]  # Start with system message

    for msg in messages[1:]:
        if not consolidated:
            consolidated.append(msg.copy())
        else:
            last_msg = consolidated[-1]

            # If same role as previous, merge content
            if last_msg["role"] == msg["role"]:
                last_msg["content"] += "\n\n" + msg["content"]
            # If different role, add as new message
            else:
                consolidated.append(msg.copy())

    # Ensure first non-system message is 'user' (required by template)
    if len(consolidated) > 1 and consolidated[1]["role"] != "user":
        # If first non-system message is assistant, prepend a user message
        consolidated.insert(1, {
            "role": "user",
            "content": "Continue the conversation."
        })

    # Format prompt with chat template
    prompt = tokenizer.apply_chat_template(
        consolidated,
        tokenize=False,
        add_generation_prompt=True
    )

    # Check prompt token count and enforce context window limit
    MAX_CONTEXT_WINDOW = 32768  # Model's max_position_embeddings
    MAX_OUTPUT_TOKENS = 500
    MAX_INPUT_TOKENS = MAX_CONTEXT_WINDOW - MAX_OUTPUT_TOKENS - 100  # Reserve 100 tokens buffer

    prompt_tokens = tokenizer.encode(prompt)
    prompt_token_count = len(prompt_tokens)

    if prompt_token_count > MAX_INPUT_TOKENS:
        # Truncate from the beginning (keep most recent context)
        # Always keep system message (first message) and truncate from middle
        truncated_tokens = prompt_tokens[-MAX_INPUT_TOKENS:]
        prompt = tokenizer.decode(truncated_tokens)
        print(f"⚠️  Warning: Prompt truncated from {prompt_token_count} to {len(truncated_tokens)} tokens")

    # Generate response
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=MAX_OUTPUT_TOKENS,
        verbose=False
    )

    return jsonify({"response": response.strip()})

@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "ok",
        "model": model_path,
        "stateless": True,
        "context_window": 32768,
        "max_input_tokens": 32168,
        "max_output_tokens": 500
    })

if __name__ == "__main__":
    port = int(os.environ.get("CHAT_SERVER_PORT", "5001"))
    print(f"\nStarting stateless chat server (Qwen) on http://0.0.0.0:{port}")
    print("Endpoints:")
    print("  POST /chat - Send a message (stateless)")
    print("  GET /health - Health check")
    print("\nReady for dirac integration!\n")
    # MLX requires the generation call to happen on the main thread; Flask's default
    # worker threads trigger: "There is no Stream(cpu, 0) in current thread."
    app.run(host="0.0.0.0", port=port, threaded=False)
