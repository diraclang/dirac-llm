"""
Stateless HTTP chat endpoint for dirac integration.

Automatically loads model from llm_models directory:
  Priority 1: llm_models/adapters  (fine-tuned with LoRA)
  Priority 2: llm_models/model_extended_{A|B}  (base model, selected by .current file)

The .current file contains either "A" or "B" to toggle between base models.

Just run: python stateless_chat_server_train.py
"""

from pathlib import Path

try:
    from flask import Flask, request, jsonify
    from mlx_lm import generate, load
except ModuleNotFoundError as exc:
    missing_module = exc.name or "required dependency"
    raise SystemExit(
        "Missing Python dependency: "
        f"{missing_module}. Run 'bash setup.sh' and then start with "
        "'source .venv/bin/activate && python mlx/python_script/stateless_chat_server_train.py' "
        "or '.venv/bin/python mlx/python_script/stateless_chat_server_train.py'."
    ) from exc

app = Flask(__name__)

script_dir = Path(__file__).parent.resolve()
mlx_dir = script_dir.parent
llm_models_dir = mlx_dir / "llm_models"

# Read which base model to use (A or B) from .current file
current_file = llm_models_dir / ".current"
if current_file.exists():
    current_model = current_file.read_text().strip()
else:
    raise SystemExit(f"\n❌ {current_file} not found. Create it with 'A' or 'B'.\n")

# Check for model in llm_models directory
# Priority: adapters first, then model_extended_A or model_extended_B
model_path_adapters = llm_models_dir / "adapters"
model_path_base = llm_models_dir / f"model_extended_{current_model}"

if (model_path_adapters / "config.json").exists():
    model_path = str(model_path_adapters)
    print(f"Using model: {model_path} (adapters)")
elif (model_path_base / "config.json").exists():
    model_path = str(model_path_base)
    print(f"Using model: {model_path} (base model {current_model})")
else:
    raise SystemExit(
        f"\n❌ No model found in {llm_models_dir}\n"
        f"Expected one of:\n"
        f"  - {model_path_adapters}/config.json\n"
        f"  - {model_path_base}/config.json\n"
    )

print(f"Loading model from {model_path}...")
model, tokenizer = load(model_path)

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
    
    # Consolidate consecutive messages with the same role
    # The chat template requires alternating user/assistant roles
    consolidated = []
    for msg in messages:
        if consolidated and consolidated[-1]["role"] == msg["role"]:
            # Merge with previous message
            consolidated[-1]["content"] += "\n\n" + msg["content"]
        else:
            consolidated.append(msg.copy())
    
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
    print("\nStarting stateless chat server on http://0.0.0.0:5001")
    print("Endpoints:")
    print("  POST /chat - Send a message (stateless)")
    print("  GET /health - Health check")
    print("\nReady for dirac integration!\n")
    app.run(host="0.0.0.0", port=5001)
