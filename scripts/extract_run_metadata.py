# scripts/extract_run_metadata.py
# Usage:
#   python3 scripts/extract_run_metadata.py \
#     --vae_ckpt experiments/checkpoints/vae_best_sp500_L50_lat16_real/best_vae.pt \
#     --diff_ckpt experiments/checkpoints/diffusion_latent_cond_sp500_logret_L50/best_diffusion.pt \
#     --out_dir experiments/results/run_metadata_sp500_L50 \
#     --dataset_name sp500_logret \
#     --seq_len 50
#
# Notes:
# - Works for your checkpoint styles:
#     VAE: {'model','cfg','epoch','global_step','best_val','args'}
#     Diff: often {'state_dict', ...} or raw state_dict
# - Produces:
#     out_dir/metadata.json   (paper-friendly single artifact)
#     out_dir/metadata.txt    (human-readable summary)
# - Does NOT assume Lightning; it tries to infer what it can robustly.

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


# ----------------------------
# Helpers
# ----------------------------
def _safe_jsonable(x: Any) -> Any:
    """Make a value JSON-serializable (best-effort)."""
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (list, tuple)):
        return [_safe_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _safe_jsonable(v) for k, v in x.items()}
    # Torch types
    if torch.is_tensor(x):
        return {
            "_type": "tensor",
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
        }
    # Fallback repr
    return {"_type": type(x).__name__, "repr": repr(x)}


def _load_ckpt(path: Path, device: str = "cpu") -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return torch.load(str(path), map_location=device)


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info() -> Dict[str, Any]:
    def run(cmd: list[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
            return out
        except Exception:
            return None

    return {
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": run(["git", "status", "--porcelain"]),
    }


def _torch_env() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def _count_params_from_state_dict(sd: Dict[str, torch.Tensor]) -> int:
    total = 0
    for v in sd.values():
        if torch.is_tensor(v):
            total += v.numel()
    return int(total)


def _infer_state_dict(obj: Any) -> Tuple[Optional[Dict[str, torch.Tensor]], str]:
    """
    Returns (state_dict, style)
      style in {"raw_state_dict", "wrapped_state_dict", "vae_style_model_key", "unknown"}
    """
    if isinstance(obj, dict):
        # very common diffusion style:
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
            if all(torch.is_tensor(v) for v in sd.values()):
                return sd, "wrapped_state_dict"

        # your VAE style:
        if "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
            if all(torch.is_tensor(v) for v in sd.values()):
                return sd, "vae_style_model_key"

        # raw:
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return obj, "raw_state_dict"

    return None, "unknown"


def _strip_common_prefixes(sd: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Optional[str]]:
    if not sd:
        return sd, None
    keys = list(sd.keys())
    for prefix in ("model.", "vae.", "net."):
        if all(k.startswith(prefix) for k in keys):
            return {k[len(prefix):]: v for k, v in sd.items()}, prefix
    for prefix in ("model.", "vae.", "net."):
        cnt = sum(1 for k in keys if k.startswith(prefix))
        if cnt >= int(0.8 * len(keys)):
            return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}, prefix
    return sd, None


def _guess_arch_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Minimal paper-friendly summary; you can extend if needed.
    keys = [
        "seq_len",
        "in_channels",
        "input_dim",
        "latent_dim",
        "hidden_dim",
        "num_layers",
        "dropout",
        "beta",
        "clamp_logvar",
        "logvar_min",
        "logvar_max",
        "use_mmd",
        "mmd_weight",
        "mmd_kernel",
        "mmd_imq_c",
        "mmd_rbf_sigma",
    ]
    out = {}
    for k in keys:
        if k in cfg:
            out[k] = cfg[k]
    return out


@dataclass
class CheckpointSummary:
    path: str
    exists: bool
    sha256: Optional[str]
    ckpt_type: str
    top_keys: list[str]
    epoch: Optional[int]
    global_step: Optional[int]
    best_val: Optional[float]
    cfg: Dict[str, Any]
    args: Dict[str, Any]
    state_dict_style: str
    stripped_prefix: Optional[str]
    num_params: Optional[int]
    dtype_breakdown: Dict[str, int]


def _dtype_breakdown(sd: Optional[Dict[str, torch.Tensor]]) -> Dict[str, int]:
    if not sd:
        return {}
    out: Dict[str, int] = {}
    for v in sd.values():
        if torch.is_tensor(v):
            k = str(v.dtype)
            out[k] = out.get(k, 0) + int(v.numel())
    return out


def summarize_checkpoint(path: Path, device: str = "cpu") -> CheckpointSummary:
    exists = path.exists()
    if not exists:
        return CheckpointSummary(
            path=str(path),
            exists=False,
            sha256=None,
            ckpt_type="missing",
            top_keys=[],
            epoch=None,
            global_step=None,
            best_val=None,
            cfg={},
            args={},
            state_dict_style="unknown",
            stripped_prefix=None,
            num_params=None,
            dtype_breakdown={},
        )

    obj = _load_ckpt(path, device=device)

    top_keys = list(obj.keys()) if isinstance(obj, dict) else []
    epoch = int(obj["epoch"]) if isinstance(obj, dict) and "epoch" in obj else None
    global_step = int(obj["global_step"]) if isinstance(obj, dict) and "global_step" in obj else None

    best_val = None
    if isinstance(obj, dict):
        for k in ("best_val", "best_val_loss", "best"):
            if k in obj:
                try:
                    best_val = float(obj[k])
                except Exception:
                    best_val = None
                break

    cfg = obj.get("cfg", {}) if isinstance(obj, dict) and isinstance(obj.get("cfg", {}), dict) else {}
    args = obj.get("args", {}) if isinstance(obj, dict) and isinstance(obj.get("args", {}), dict) else {}

    sd, style = _infer_state_dict(obj)
    stripped_prefix = None
    if sd is not None:
        sd, stripped_prefix = _strip_common_prefixes(sd)
        num_params = _count_params_from_state_dict(sd)
        dtypes = _dtype_breakdown(sd)
    else:
        num_params = None
        dtypes = {}

    ckpt_type = "dict" if isinstance(obj, dict) else type(obj).__name__

    return CheckpointSummary(
        path=str(path),
        exists=True,
        sha256=_sha256_file(path),
        ckpt_type=ckpt_type,
        top_keys=top_keys,
        epoch=epoch,
        global_step=global_step,
        best_val=best_val,
        cfg=_safe_jsonable(cfg) if cfg else {},
        args=_safe_jsonable(args) if args else {},
        state_dict_style=style,
        stripped_prefix=stripped_prefix,
        num_params=num_params,
        dtype_breakdown=dtypes,
    )


# ----------------------------
# Main
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--vae_ckpt", type=str, required=True)
    p.add_argument("--diff_ckpt", type=str, default=None)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default="sp500_logret")
    p.add_argument("--seq_len", type=int, default=50)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vae_path = Path(args.vae_ckpt)
    diff_path = Path(args.diff_ckpt) if args.diff_ckpt else None

    vae_sum = summarize_checkpoint(vae_path, device=args.device)
    diff_sum = summarize_checkpoint(diff_path, device=args.device) if diff_path else None

    paper_arch_vae = _guess_arch_from_cfg(vae_sum.cfg or {})
    paper_arch_diff = _guess_arch_from_cfg(diff_sum.cfg or {}) if diff_sum else {}

    meta = {
        "project": "compressing-chaos",
        "dataset": {"name": args.dataset_name, "seq_len": int(args.seq_len)},
        "environment": _torch_env(),
        "git": _git_info(),
        "checkpoints": {
            "vae": asdict(vae_sum),
            "diffusion": asdict(diff_sum) if diff_sum else None,
        },
        "paper_friendly": {
            "vae_arch_from_cfg": paper_arch_vae,
            "diff_arch_from_cfg": paper_arch_diff,
            "notes": [
                "If diffusion checkpoint does not include cfg/args/epoch, training hyperparams must be taken from the training script or config file used to create it.",
                "If you want per-epoch curves, ensure the training loop saves history arrays or logs to CSV/TensorBoard; current VAE ckpt only stores epoch/global_step/best_val/cfg/args.",
            ],
        },
    }

    json_path = out_dir / "metadata.json"
    with open(json_path, "w") as f:
        json.dump(_safe_jsonable(meta), f, indent=2)

    # Human-readable quick summary
    txt_path = out_dir / "metadata.txt"
    with open(txt_path, "w") as f:
        f.write("RUN METADATA SUMMARY\n")
        f.write("====================\n\n")
        f.write(f"Dataset: {args.dataset_name} | seq_len={args.seq_len}\n\n")

        f.write("VAE CHECKPOINT\n")
        f.write("-------------\n")
        f.write(f"path: {vae_sum.path}\n")
        f.write(f"sha256: {vae_sum.sha256}\n")
        f.write(f"epoch: {vae_sum.epoch} | global_step: {vae_sum.global_step} | best_val: {vae_sum.best_val}\n")
        f.write(f"ckpt keys: {vae_sum.top_keys}\n")
        f.write(f"state_dict_style: {vae_sum.state_dict_style} | stripped_prefix: {vae_sum.stripped_prefix}\n")
        f.write(f"num_params: {vae_sum.num_params}\n")
        f.write(f"cfg (paper): {paper_arch_vae}\n\n")

        if diff_sum:
            f.write("DIFFUSION CHECKPOINT\n")
            f.write("-------------------\n")
            f.write(f"path: {diff_sum.path}\n")
            f.write(f"sha256: {diff_sum.sha256}\n")
            f.write(f"epoch: {diff_sum.epoch} | global_step: {diff_sum.global_step} | best_val: {diff_sum.best_val}\n")
            f.write(f"ckpt keys: {diff_sum.top_keys}\n")
            f.write(f"state_dict_style: {diff_sum.state_dict_style} | stripped_prefix: {diff_sum.stripped_prefix}\n")
            f.write(f"num_params: {diff_sum.num_params}\n")
            if paper_arch_diff:
                f.write(f"cfg (paper): {paper_arch_diff}\n")
            else:
                f.write("cfg (paper): <not found in ckpt>\n")
            f.write("\n")

        f.write("ENVIRONMENT\n")
        f.write("-----------\n")
        for k, v in _torch_env().items():
            f.write(f"{k}: {v}\n")

        g = _git_info()
        f.write("\nGIT\n---\n")
        for k, v in g.items():
            f.write(f"{k}: {v}\n")

    print(f"Wrote:\n- {json_path}\n- {txt_path}")


if __name__ == "__main__":
    main()
