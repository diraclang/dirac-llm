import argparse
from pathlib import Path
import re
import time
import numpy as np
import safetensors.numpy as sn

parser = argparse.ArgumentParser(description="Inspect and summarize LoRA adapter matrices using SVD.")
parser.add_argument("--adapter", type=str, default="/Users/zhiwang/diraclang/dirac-llm/mlx/llm_models/model_extended_A/adapters/adapters.safetensors", help="Path to an adapters.safetensors file or to an adapter directory containing it.")
parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of A/B pairs to process for a quick test run.")
parser.add_argument("--progress-every", type=int, default=20, help="Print progress every N matrices.")
parser.add_argument("--full-svd", action="store_true", help="Compute the exact full SVD for each matrix; slower but gives full U and V vectors.")
args = parser.parse_args()


def resolve_adapter_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_dir():
        candidates = [
            path / "adapters.safetensors",
            path / "adapter.safetensors",
            *sorted(path.glob("*.safetensors")),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No adapters.safetensors file found in adapter directory: {path}"
        )
    if path.is_file():
        return path
    raise FileNotFoundError(f"Adapter file not found: {path}")


file_path = resolve_adapter_path(args.adapter)
print(f"Loading adapter file: {file_path}")
data = sn.load_file(str(file_path))

# For each matrix pair (A, B), form the effective low-rank update D = B @ A.
# Then compute the SVD: D = U @ diag(sigma) @ V^T.
# This gives the dominant input/output directions and their strengths.


def parse_key(key: str):
    m = re.search(r"model\.layers\.(\d+)\.(.+)\.lora_a$", key)
    if not m:
        return None
    return int(m.group(1)), m.group(2)

entries = []

lora_a_keys = []
for key in data:
    if "lora_a" in key:
        parsed = parse_key(key)
        if parsed is not None:
            lora_a_keys.append(key)

if args.limit is not None:
    lora_a_keys = lora_a_keys[: args.limit]
    print(f"Quick test mode: processing only the first {len(lora_a_keys)} A/B pairs.")

print(f"Starting LoRA SVD scan for {len(lora_a_keys)} A/B pairs in {file_path}")
print("This can take a while because each matrix performs an SVD; progress will be reported below.")
start_time = time.perf_counter()

for idx, key_a in enumerate(lora_a_keys, start=1):
    arr_a = data[key_a]
    parsed = parse_key(key_a)
    if parsed is None:
        continue

    layer, proj = parsed
    key_b = key_a.replace(".lora_a", ".lora_b")
    if key_b not in data:
        continue

    a = np.asarray(arr_a)
    b = np.asarray(data[key_b])
    if a.ndim != 2 or b.ndim != 2:
        continue

    # In the MLX implementation, the stored matrices are:
    #   A: (in_dim, rank)
    #   B: (rank, out_dim)
    # and the effective update for a token vector x is x @ A @ B.
    # So the matrix form of the update is D = A @ B, not B @ A.
    d = a @ b

    if args.full_svd:
        u, s, vt = np.linalg.svd(d, full_matrices=False)
    else:
        # Faster: because the LoRA update has rank <= r, the nonzero singular values
        # of D = A @ B can be recovered from the much smaller r x r matrix
        # (A.T @ A) @ (B @ B.T). Its eigenvalues are sigma^2.
        gram = (a.T @ a) @ (b @ b.T)
        eigvals = np.linalg.eigvalsh(gram)
        eigvals = np.maximum(eigvals, 0.0)
        s = np.sqrt(eigvals[::-1])
        u = None
        vt = None

    # rank energies and matrix energy
    rank_strength = s ** 2
    total_energy = float(rank_strength.sum())
    mean_rank_energy = float(rank_strength.mean()) if rank_strength.size else 0.0
    max_rank_energy = float(rank_strength.max()) if rank_strength.size else 0.0
    top_rank = int(np.argmax(rank_strength)) if rank_strength.size else -1

    entries.append({
        "layer": layer,
        "proj": proj,
        "key_a": key_a,
        "key_b": key_b,
        "shape_a": a.shape,
        "shape_b": b.shape,
        "shape_d": d.shape,
        "rank": min(a.shape[1], b.shape[0]),
        "singular_values": s,
        "u": u,
        "v": vt,
        "total_energy": total_energy,
        "mean_rank_energy": mean_rank_energy,
        "max_rank_energy": max_rank_energy,
        "top_rank": top_rank,
    })

    elapsed = time.perf_counter() - start_time
    if idx % args.progress_every == 0 or idx == len(lora_a_keys):
        eta = (elapsed / idx) * (len(lora_a_keys) - idx) if idx > 0 else 0.0
        print(
            f"[progress] {idx:>3}/{len(lora_a_keys)} matrices processed | "
            f"layer={layer:>2} proj={proj:<25} | "
            f"elapsed={elapsed:7.1f}s | eta={eta:7.1f}s",
            flush=True,
        )

if not entries:
    raise RuntimeError(f"No complete LoRA A/B pairs found in {file_path}")

entries.sort(key=lambda e: e["total_energy"], reverse=True)

print(f"Adapter file: {file_path}")
print(f"Total LoRA pairs analyzed: {len(entries)}")
print("Top LoRA update matrices by SVD energy:")
print("-" * 140)
print(f"{'layer':>6} | {'projection':<35} | {'shape_d':>18} | {'rank':>4} | {'total_energy':>14} | {'mean_sigma^2':>14} | {'max_sigma^2':>14} | {'top_rank':>9}")
for e in entries[:20]:
    print(
        f"{e['layer']:>6} | {e['proj']:<35} | {str(e['shape_d']):>18} | {e['rank']:>4} | "
        f"{e['total_energy']:>14.6f} | {e['mean_rank_energy']:>14.6f} | {e['max_rank_energy']:>14.6f} | {e['top_rank']:>9}"
    )

print("\nExample: strongest matrix SVD summary")
strongest = entries[0]
print(f"Strongest matrix: layer {strongest['layer']}, projection {strongest['proj']}, shape_d={strongest['shape_d']}")
print(f"Nonzero rank = {min(strongest['shape_d'])} for this LoRA matrix; only the first {min(8, strongest['singular_values'].size)} singular values are shown.")
print("Singular values sigma_i (top entries only):")
for i, sigma in enumerate(strongest["singular_values"][: min(8, strongest["singular_values"].size)]):
    print(f"  sigma[{i}] = {float(sigma):.8f}, sigma^2 = {float(sigma**2):.8f}")
if strongest["u"] is not None and strongest["v"] is not None:
    print("\nLeading left singular vector u_0 (first 8 entries):")
    print(np.asarray(strongest["u"][:, 0])[:8])
    print("Leading right singular vector v_0 (first 8 entries):")
    print(np.asarray(strongest["v"][0, :])[:8])
else:
    print("\nFast mode: full U/V vectors were not computed; singular values are sufficient for rank strength analysis.")

print("\nLayer summary: total SVD energy per layer")
layer_summary = {}
layer_sigma_values = {}
for e in entries:
    layer_summary.setdefault(e["layer"], 0.0)
    layer_summary[e["layer"]] += e["total_energy"]
    layer_sigma_values.setdefault(e["layer"], [])
    layer_sigma_values[e["layer"]].extend(float(s) for s in e["singular_values"])

layer_rank = sorted(layer_summary.items(), key=lambda kv: kv[1], reverse=True)
for layer, total_energy in layer_rank:
    sigmas = sorted(layer_sigma_values[layer], reverse=True)
    print(f"layer {layer:>2}: total SVD energy = {total_energy:.6f} | sigma_count={len(sigmas)} | top_sigma={sigmas[0]:.6f} if {len(sigmas)>0} else 'n/a'")

print("\nLayer sigma spectrum (ordered by largest sigma within each layer):")
for layer, total_energy in layer_rank:
    sigmas = sorted(layer_sigma_values[layer], reverse=True)
    # Keep the output readable even when a layer has many singular values.
    preview = sigmas[: min(12, len(sigmas))]
    preview_str = ", ".join(f"{s:.6f}" for s in preview)
    if len(sigmas) > len(preview):
        preview_str += ", ..."
    print(f"layer {layer:>2}: {preview_str}")

print("\nProjection summary: total SVD energy per projection")
proj_summary = {}
for e in entries:
    proj_summary.setdefault(e["proj"], 0.0)
    proj_summary[e["proj"]] += e["total_energy"]

for proj, val in sorted(proj_summary.items(), key=lambda kv: kv[1], reverse=True):
    print(f"{proj:<35} : {val:.6f}")

