# PyTorch Deep Dive — Curriculum

**Goal:** Understand PyTorch from first principles through production patterns.  
**Hardware:** RTX 4070 Laptop (8 GB VRAM) — enough for most lessons; capstone projects use smaller models or mixed precision.  
**Pace:** ~1 module/week, 3–5 hours each.

---

## Phase 0 — Setup (Day 1)

| Step | Topic | Notebook |
|------|-------|----------|
| 0.1 | Verify CUDA, driver, VRAM | `00_setup/phase0_setup.ipynb` |
| 0.2 | Devices, dtypes, transfers | (same notebook) |

**Checkpoint:** `torch.cuda.is_available()` is True; you can move a tensor to GPU.

---

## Phase 1 — Foundations (Week 1)

Build mental model of tensors and automatic differentiation.

| Step | Topic | Key concepts |
|------|-------|--------------|
| 1.1 | Tensor creation & shape | `shape`, `dtype`, `device`, broadcasting |
| 1.2 | Indexing & views vs copies | `view`, `reshape`, `contiguous`, aliasing |
| 1.3 | Math ops | element-wise, matmul, `@`, reduction |
| 1.4 | Autograd | `requires_grad`, `.backward()`, `.grad`, graph |
| 1.5 | Detach & no_grad | inference vs training modes |

**Checkpoint:** Manually implement gradient descent on `y = wx + b` using only tensors + autograd.

---

## Phase 2 — Neural Networks (Week 2)

| Step | Topic | Key concepts |
|------|-------|--------------|
| 2.1 | `nn.Module` | parameters, `forward()`, state |
| 2.2 | Layers | Linear, Conv2d, BatchNorm, Dropout |
| 2.3 | Activations & init | ReLU, GELU, Xavier/Kaiming |
| 2.4 | Sequential vs custom modules | composition, submodules |
| 2.5 | Parameter inspection | `named_parameters()`, `state_dict()` |

**Checkpoint:** Build an MLP classifier from scratch (no `nn.Sequential` shortcut on first pass).

---

## Phase 3 — Training Loop (Week 3)

| Step | Topic | Key concepts |
|------|-------|--------------|
| 3.1 | Loss functions | MSE, CrossEntropy, `reduction` |
| 3.2 | Optimizers | SGD, Adam, param groups, lr |
| 3.3 | Full train loop | forward → loss → backward → step → zero_grad |
| 3.4 | Validation | `model.eval()`, `torch.no_grad()` |
| 3.5 | Checkpointing | save/load `state_dict` |

**Checkpoint:** Train on synthetic 2D classification; plot decision boundary.

---

## Phase 4 — Data Pipeline (Week 4)

| Step | Topic | Key concepts |
|------|-------|--------------|
| 4.1 | `Dataset` | `__len__`, `__getitem__` |
| 4.2 | `DataLoader` | batching, shuffle, `num_workers` |
| 4.3 | Transforms | `Compose`, normalization |
| 4.4 | Train/val split | reproducibility, seeds |
| 4.5 | Custom collate | variable-length sequences |

**Checkpoint:** Train MNIST end-to-end with your own Dataset wrapper.

---

## Phase 5 — GPU Deep Dive (Week 5)

Critical for your 4070 — understand *why* things break on GPU.

| Step | Topic | Key concepts |
|------|-------|--------------|
| 5.1 | `.to(device)` patterns | model + data + loss on same device |
| 5.2 | Memory profiling | `torch.cuda.memory_allocated`, OOM |
| 5.3 | Mixed precision (AMP) | `autocast`, `GradScaler` |
| 5.4 | Pin memory & non_blocking | DataLoader tuning |
| 5.5 | Benchmark CPU vs GPU | when GPU wins |

**Checkpoint:** Train same model with/without AMP; compare speed and memory.

---

## Phase 6 — Computer Vision (Week 6–7)

| Step | Topic | Key concepts |
|------|-------|--------------|
| 6.1 | Conv layers | kernel, stride, padding, receptive field |
| 6.2 | Pooling & CNN arch | LeNet-style → ResNet blocks |
| 6.3 | Transfer learning | freeze backbone, fine-tune head |
| 6.4 | Augmentation | random crop, flip, color jitter |
| 6.5 | torchvision models | pretrained weights, feature extraction |

**Checkpoint:** Fine-tune ResNet18 on a small custom image set (fits 8 GB).

---

## Phase 7 — NLP Intro (Week 8)

| Step | Topic | Key concepts |
|------|-------|--------------|
| 7.1 | Embeddings | `nn.Embedding`, lookup |
| 7.2 | RNN/LSTM | sequences, hidden state, packing |
| 7.3 | Attention intuition | query/key/value |
| 7.4 | Transformer blocks | self-attention, positional encoding |
| 7.5 | HuggingFace bridge | load pretrained, fine-tune head |

**Checkpoint:** Sentiment classifier with a small transformer (distilled model).

---

## Phase 8 — Advanced (Week 9–10)

| Step | Topic | Key concepts |
|------|-------|--------------|
| 8.1 | Custom autograd Function | forward/backward hooks |
| 8.2 | Hooks & debugging | register_forward_hook, anomaly detection |
| 8.3 | torch.compile | graph capture, when it helps |
| 8.4 | Export | TorchScript, ONNX basics |
| 8.5 | Reproducibility | seeds, deterministic algorithms |

**Checkpoint:** Export a trained model and run inference outside Python.

---

## Capstone Projects (`projects/`)

Pick one after Phase 6; revisit after Phase 8.

1. **Image classifier** — custom dataset, transfer learning, AMP, TensorBoard
2. **Sequence model** — char-RNN or time-series forecaster
3. **From-scratch transformer** — tiny GPT on a text corpus (8 GB friendly)

---

## Tutoring format

Each session we pick **one step**, you run the script, we discuss:
- What happens under the hood?
- What breaks if you change X?
- One exercise to solidify it

Say **"Start Phase 0"** when ready — open `00_setup/phase0_setup.ipynb`.
