"""
Stateless HTTP chat endpoint for dirac integration.
Accepts POST requests with a JSON body containing a 'message' field (concatenated conversation),
and returns the assistant's response.
Dirac handles conversation history - this server is stateless.
"""

import os
from pathlib import Path
import yaml

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


def _read_model_name_from_config(config_path: Path) -> str:
    default_model = "mlx-community/Mistral-7B-Instruct-v0.3"
    if not config_path.exists():
        return default_model

    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file) or {}

    return config.get("model", {}).get("name", default_model)


def _resolve_model_path() -> str:
    model_path = os.getenv("MLX_MODEL_PATH", "").strip()
    if model_path:
        return model_path

    config_path = mlx_dir / "config.yml"
    if config_path.exists():
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file) or {}

        model_outputs = config.get("output", {})
        models_dir = config.get("models_dir", "llm_models")
        dataset_name = os.getenv("MLX_MODEL_DATASET", "extended").strip() or "extended"

        preferred_local_output = mlx_dir / model_outputs.get(dataset_name, f"{models_dir}/model_{dataset_name}")
        if (preferred_local_output / "config.json").exists():
            return str(preferred_local_output)

        default_output = model_outputs.get("default")
        if default_output:
            default_output_path = mlx_dir / default_output
            if (default_output_path / "config.json").exists():
                return str(default_output_path)

    return _read_model_name_from_config(config_path)


model_path = _resolve_model_path()

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
    
    # Generate response
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=500,
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
        "stateless": True
    })

if __name__ == "__main__":
    print("\nStarting stateless chat server on http://0.0.0.0:5001")
    print("Endpoints:")
    print("  POST /chat - Send a message (stateless)")
    print("  GET /health - Health check")
    print("\nReady for dirac integration!\n")
    app.run(host="0.0.0.0", port=5001)
