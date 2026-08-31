# Qwen2.5 Fine-Tuning & Serving Guide

Reference for the switch from Mistral to Qwen2.5-7B-Instruct: where the files
live, how to fine-tune, and how to restart the inference server.

## 1. Where things live

| Purpose | Path |
|---|---|
| Base Qwen model (MLX format, used for training/inference) | `dirac-llm/mlx/Qwen2.5-7B-Instruct/` |
| Original HF checkpoint (source of the MLX conversion) | `dirac-llm/hf/Qwen2.5-7B-Instruct/` |
| Training script (Qwen-specific) | `dirac-llm/mlx/lora_train_balanced_qwen.sh` |
| Training dataset | `dirac-llm/mlx/dataset/dirac_data_extended/` |
| Trained models / adapters (A/B slots) | `dirac-llm/mlx/llm_models/model_extended_A/`, `.../model_extended_B/` |
| Trained model / adapters (single-slot, direct Qwen flow) | `dirac-llm/mlx/llm_models/qwen/` (+ `qwen/adapters/`) |
| Active slot marker | `dirac-llm/mlx/llm_models/.current` (contains `A` or `B`) |
| Server script — A/B slot aware (used by `train.sh`) | `dirac-llm/mlx/python_script/stateless_chat_server_train.py` |
| Server script — Qwen-only, no A/B toggle | `dirac-llm/mlx/python_script/stateless_chat_server_train_qwen.py` |
| Orchestration script (train + flip slot + restart server) | `dirac/train.sh` (defaults to Qwen) |
| Legacy Mistral orchestration script | `dirac/train-mistral.sh` |
| Server log | `dirac-llm/mlx/server.log` |
| Dirac LLM provider config (points at the server) | `dirac/config.yml` → `llmProvider: custom`, `customLLMUrl: http://localhost:5001` |

## 2. Recommended flow: `dirac/train.sh` (A/B slots, Qwen default)

This is the normal way to fine-tune and deploy. It trains into whichever slot
(`model_extended_A` or `model_extended_B`) is **not** currently active, then
flips `.current` and restarts the server pointing at the new slot. The old
slot is kept on disk for instant rollback.

```bash
cd dirac
./train.sh
```

What it does, step by step:
1. Reads `dirac-llm/mlx/llm_models/.current` to find the active slot (A or B).
2. Trains into the *other* slot via `lora_train_balanced_qwen.sh`, with
   `MODEL_OUTPUT_DIR` pointed at `model_extended_<new_slot>`.
3. Verifies `adapters.safetensors` was produced in the new slot's
   `adapters/` directory.
4. Kills whatever process is listening on port 5001 (the running server).
5. Writes the new slot letter into `.current`.
6. Restarts the server with:
   - `MLX_MODEL_PATH` = base Qwen model (fuse-independent, adapter-on-base)
   - `MLX_ADAPTER_PATH` = the new slot's `adapters/` directory
7. Logs to `dirac-llm/mlx/server.log`.

To override the base model or force a specific slot:
```bash
BASE_MODEL=/path/to/other/model ./train.sh
```

### Rollback
Flip `.current` back to the previous letter and restart the server manually
(see §4), since the old slot's model/adapters are still on disk.

## 3. Direct Qwen-only flow (no A/B toggle)

Use this only if you want a single persistent Qwen deployment without the
A/B rollback mechanism (e.g. quick local experiments).

```bash
cd dirac-llm
source .venv/bin/activate
cd mlx
./lora_train_balanced_qwen.sh
```

- Writes the fine-tuned adapters to `llm_models/qwen/adapters/`.
- Attempts `mlx_lm.fuse` to also produce a fully-fused model in
  `llm_models/qwen/`. If fusing fails, the script prints a warning but does
  **not** fail — training is still usable via adapter-on-base.
- Restart the matching server manually (see §4,
  `stateless_chat_server_train_qwen.py`).

## 4. Restarting the server manually

If you need to restart without re-training (e.g. after a crash or a config
change):

```bash
# Kill whatever is on port 5001
kill "$(lsof -ti :5001)"

cd dirac-llm
source .venv/bin/activate

# A/B slot-aware server (reads .current), adapter-on-base:
MLX_MODEL_PATH="mlx/Qwen2.5-7B-Instruct" \
MLX_ADAPTER_PATH="mlx/llm_models/model_extended_$(cat mlx/llm_models/.current)/adapters" \
nohup .venv/bin/python mlx/python_script/stateless_chat_server_train.py > mlx/server.log 2>&1 &

# OR the Qwen-only server (hardcoded to llm_models/qwen):
nohup .venv/bin/python mlx/python_script/stateless_chat_server_train_qwen.py > mlx/server.log 2>&1 &
```

Health check:
```bash
curl http://localhost:5001/health
```

Tail logs:
```bash
tail -f dirac-llm/mlx/server.log
```

## 5. Important gotchas

- **Adapter-on-base is the safe default.** `mlx_lm.fuse` can fail (observed
  with Qwen); both training scripts treat fuse failure as non-fatal and keep
  the adapter checkpoint usable via `MLX_ADAPTER_PATH` on top of the
  un-fused base model. Don't assume a fused model always exists — check for
  `adapters.safetensors` instead.
- **Relative paths**: `lora_train_balanced_qwen.sh` `cd`s into `mlx/` first,
  so the base model path is anchored at the script's own directory
  (`$SCRIPT_DIR/Qwen2.5-7B-Instruct`), not the caller's cwd.
- **Server does not auto-discover sibling adapters.** If you set
  `MLX_MODEL_PATH` to a fused model directory that happens to have an
  `adapters/` folder next to it, you must still pass `MLX_ADAPTER_PATH`
  explicitly — it is never inferred.
- **Port 5001** is hardcoded across scripts and matches
  `customLLMUrl: http://localhost:5001` in `dirac/config.yml`.
- **Context window**: the Qwen server enforces `MAX_CONTEXT_WINDOW = 32768`
  tokens and truncates from the front of the prompt (keeping the system
  message) if exceeded — see `stateless_chat_server_train_qwen.py`.
