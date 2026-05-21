"""Phase 0.2 — Tensor devices, dtypes, and CPU↔GPU transfers."""

import torch

from shared.utils.device import get_device


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    # CPU tensor
    a = torch.ones(3, 3)
    print(f"a: {a.device}, dtype={a.dtype}")

    if device.type == "cuda":
        b = a.to(device, non_blocking=True)
        print(f"b on GPU: {b.device}")

        c = b.float().cpu()
        print(f"back on CPU: {c.device}")

    # dtype matters for memory and speed
    f32 = torch.zeros(1000, 1000, dtype=torch.float32)
    f16 = torch.zeros(1000, 1000, dtype=torch.float16)
    print(f"float32 size: {f32.element_size() * f32.numel() / 1024**2:.1f} MB")
    print(f"float16 size: {f16.element_size() * f16.numel() / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
