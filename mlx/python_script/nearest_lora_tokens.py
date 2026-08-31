import argparse
from pathlib import Path
import re
import numpy as np
import safetensors.numpy as sn
from transformers import AutoTokenizer


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


def normalize_columns(mat: np.ndarray) -> np.ndarray:
    out = mat.copy()
    for j in range(out.shape[1]):
        norm = np.linalg.norm(out[:, j])
        if norm != 0.0:
            out[:, j] = out[:, j] / norm
    return out


def resolve_adapter_path(path_value: str) -> Path:
    p = Path(path_value).expanduser()
    if p.is_file():
        return p
    if p.is_dir():
        candidates = [
            p / "adapters.safetensors",
            p / "adapter.safetensors",
            *sorted(p.glob("*adapters*.safetensors")),
            *sorted(p.glob("*.safetensors")),
        ]
        for cand in candidates:
            if cand.exists() and cand.is_file():
                return cand
        raise FileNotFoundError(f"No adapter safetensors file found under directory: {p}")
    # support a direct file path that exists as a glob-like string
    # or a model directory without a specific adapter file.
    for candidate in [
        p.with_name("adapters.safetensors"),
        p / "adapters.safetensors",
        p / "adapter.safetensors",
    ]:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Adapter file not found: {path_value}")


parser = argparse.ArgumentParser(description="Compare LoRA directions to vocabulary embedding vectors.")
parser.add_argument("--model-dir", type=str, default="/Users/zhiwang/diraclang/dirac-llm/mlx/llm_models/model_extended_A", help="Path to the model directory containing tokenizer + weights.")
parser.add_argument("--adapter", type=str, default="/Users/zhiwang/diraclang/dirac-llm/mlx/llm_models/model_extended_A/adapters/adapters.safetensors", help="Path to the adapters.safetensors file or to a directory containing one.")
parser.add_argument("--layer", type=int, default=30, help="Layer index to inspect.")
parser.add_argument(
    "--proj",
    type=str,
    default="mlp.up_proj",
    help=(
        "Projection name to inspect. Examples: "
        "mlp.up_proj, mlp.gate_proj, mlp.down_proj, "
        "self_attn.q_proj, self_attn.k_proj, self_attn.v_proj, self_attn.o_proj"
    ),
)
parser.add_argument("--top-k", type=int, default=3, help="Number of dominant directions to inspect.")
parser.add_argument("--n-nearest", type=int, default=10, help="Number of nearest vocabulary tokens to print per direction.")
parser.add_argument("--vector-source", choices=["svd", "lora_a", "both"], default="svd", help="Vector source to probe against the embedding matrix.")
parser.add_argument("--token", type=str, default=None, help="Optional token string to probe directly; reports the LoRA update induced by that token embedding.")
args = parser.parse_args()

model_dir = Path(args.model_dir).expanduser()
adapter_path = resolve_adapter_path(args.adapter)

# Load embeddings from the base model
embed = None
for p in sorted(model_dir.glob("*.safetensors")):
    data = sn.load_file(str(p))
    if "model.embed_tokens.weight" in data:
        embed = data["model.embed_tokens.weight"].astype(np.float32)
        break
if embed is None:
    raise FileNotFoundError(f"Could not find model.embed_tokens.weight in {model_dir}")

embed_norm = normalize_rows(embed)
print(f"Loaded embedding matrix: shape={embed.shape}, vocab={embed.shape[0]}, dim={embed.shape[1]}")

tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
print(f"Loaded tokenizer: {type(tokenizer).__name__}, vocab_size={tokenizer.vocab_size}")

adapters = sn.load_file(str(adapter_path))
key_a = f"model.layers.{args.layer}.{args.proj}.lora_a"
key_b = f"model.layers.{args.layer}.{args.proj}.lora_b"
if key_a not in adapters or key_b not in adapters:
    raise KeyError(f"Could not find {key_a} and {key_b} in {adapter_path}")

A = adapters[key_a].astype(np.float32)
B = adapters[key_b].astype(np.float32)
D = A @ B
print(f"Matrix {args.layer}/{args.proj}: D shape={D.shape}")

if args.token is not None:
    token_ids = tokenizer.encode(args.token, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Token {args.token!r} was not found in the tokenizer vocabulary.")
    print(f"\nToken-driven LoRA probe for token={args.token!r}")
    for tid in token_ids:
        x = embed[tid].astype(np.float32)
        # MLX LoRA stores A as (in_dim, rank) and B as (rank, out_dim),
        # and the update is computed as (x @ A) @ B.
        a_x = x @ A
        delta = a_x @ B
        base_norm = float(np.linalg.norm(x))
        a_norm = float(np.linalg.norm(a_x))
        delta_norm = float(np.linalg.norm(delta))
        ratio = delta_norm / (a_norm + 1e-12)
        if delta.size > 0 and base_norm > 0:
            delta_dir = delta / (delta_norm + 1e-12)
            x_dir = x / (base_norm + 1e-12)
            cos_x_delta = float(np.dot(x_dir, delta_dir))
        else:
            cos_x_delta = 0.0
        print(
            f"  token_id={tid:>5}  token={tokenizer.decode([tid])!r}  "
            f"||x||={base_norm:.6f}  ||x @ A||={a_norm:.6f}  ||(x @ A) @ B||={delta_norm:.6f}  "
            f"ratio={ratio:.6f}  cos(x, delta)={cos_x_delta:.6f}"
        )
    raise SystemExit(0)

u, s, vt = np.linalg.svd(D, full_matrices=False)
print(f"Singular values: {np.array2string(s[: min(args.top_k, len(s))], precision=6)}")

candidates = {}

if args.vector_source in ("svd", "both"):
    # Use the side of the SVD that matches the embedding dimension.
    # For most MLP weights, the hidden dimension matches the embedding width (4096).
    if embed.shape[1] == D.shape[0]:
        vecs = u[:, : args.top_k]
        vec_kind = "u"
    elif embed.shape[1] == D.shape[1]:
        vecs = vt[: args.top_k, :].T
        vec_kind = "v"
    else:
        raise ValueError(f"Embedding dimension {embed.shape[1]} does not match D dimensions {D.shape}; cannot compare directly.")
    candidates["svd"] = {
        "vecs": vecs,
        "meta": [(f"sigma={float(s[j]):.6f}", vec_kind) for j in range(min(args.top_k, len(s)))],
    }

if args.vector_source in ("lora_a", "both"):
    if A.shape[0] == embed.shape[1]:
        candidates["lora_a"] = {
            "vecs": A[:, : args.top_k],
            "meta": [(f"lora_a_col={j}", "lora_a") for j in range(min(args.top_k, A.shape[1]))],
        }
    else:
        raise ValueError(f"Embedding dimension {embed.shape[1]} does not match A shape {A.shape}; cannot compare directly.")

for source_name, info in candidates.items():
    vecs = info["vecs"]
    vecs_norm = normalize_columns(vecs)
    scores = embed_norm @ vecs_norm
    print(f"\nNearest vocabulary tokens for source={source_name} on {args.proj} at layer {args.layer}")
    for j in range(vecs_norm.shape[1]):
        sims = scores[:, j]
        top_idx = np.argsort(sims)[::-1][: args.n_nearest]
        label, mode = info["meta"][j]
        print(f"\nDirection {j} ({label}, mode={mode})")
        for tid in top_idx:
            token = tokenizer.decode([int(tid)])
            print(f"  token_id={int(tid):>5}  sim={float(sims[tid]):.6f}  token={token!r}")
