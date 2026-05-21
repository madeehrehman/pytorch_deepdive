"""Phase 0.1 — Verify PyTorch sees your RTX 4070."""

import torch


def main() -> None:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\nCUDA not detected. Install GPU build:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu124")
        return

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print(f"GPU:             {props.name}")
    print(f"Compute cap:     {props.major}.{props.minor}")
    print(f"VRAM:            {props.total_memory / 1024**3:.1f} GB")

    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x.T
    print(f"Matmul on GPU:   {y.shape}, device={y.device}")


if __name__ == "__main__":
    main()
