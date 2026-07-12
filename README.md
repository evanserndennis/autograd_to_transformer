# LLM From Scratch: A Foundational Math Path

Goal: understand the calculus and linear algebra that actually make an LLM work, by building the
pieces yourself, from scalars up to a tiny GPT you can train on this laptop.

No deep learning framework is used until it's earned — every stage below builds directly on the one
before it. CPU-only, small enough to run in seconds to minutes.

## Stages

1. **`stage1_autograd.py` — Scalar autograd engine.**
   A `Value` class wraps a single number and remembers how it was computed. Calling `.backward()`
   walks the computational graph in reverse and applies the chain rule at every step to compute
   `d(output)/d(every input)`. This is the entire mathematical idea behind "training a neural net" —
   everything later is this same idea, vectorized and scaled up.
   *Math: derivatives, the chain rule, partial derivatives.*

2. **`stage2_mlp.py` — A neural net trained with your own engine.**
   Neurons, layers, and a training loop (forward pass -> loss -> backward() -> gradient descent
   update) built entirely on `Value`. Trained on a toy classification problem.
   *Math: gradient descent, loss functions, how backprop composes across layers.*

3. **`stage3_tensor_autograd.py` — Vectorized (matrix) autograd.**
   The same idea as stage 1, but operating on NumPy arrays instead of single numbers, so you get
   matrix multiplication, broadcasting, and their gradients. This is the step where "calculus" turns
   into "linear algebra" — a gradient of a matmul is itself a matmul.
   *Math: matrix calculus, Jacobians, why `dL/dW = x^T @ dL/dy`.*

4. **`stage4_attention.py` — Attention from scratch.**
   Scaled dot-product attention (Q, K, V), softmax, causal masking, multi-head attention — built
   using the stage 3 tensor engine, with a worked toy example you can step through by hand.
   *Math: dot products as similarity, softmax as a differentiable argmax, why attention is just
   weighted averaging with learned weights.*

5. **`stage5_mini_gpt.py` — A tiny character-level GPT.**
   Token + positional embeddings, transformer blocks (attention + MLP + layer norm + residual
   connections), a training loop, and text generation. Trained on a small text file, small enough
   to run on CPU.
   *Math: everything above, composed. Plus: why residual connections and layer norm matter for
   gradient flow through depth.*

6. **`notes_beyond_toy_models.md` — Where this connects to real LLMs.**
   Short notes (no code) on what's different at real scale: tokenization (BPE), positional encoding
   variants, optimizer choices (Adam), and where things like fine-tuning, RAG, and agent/tool-use sit
   on top of this foundation.

## Setup

```
pip install -r requirements.txt
pytest
```

## How to work through it

Each stage is a single, standalone, heavily-commented script — run it directly
(`python stage1_autograd.py`) and read top to bottom. Nothing is hidden behind a framework call;
if you can't answer "why does this line produce this gradient," that's the thing to slow down on
before moving to the next stage.

## Status

- [x] Stage 1: scalar autograd engine
- [ ] Stage 2: MLP
- [ ] Stage 3: tensor autograd
- [ ] Stage 4: attention
- [ ] Stage 5: mini GPT
- [ ] Notes on real-world LLMs
