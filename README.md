# SLiM

Small Lightweight Model for local training and inference. Pure C++17, no external dependencies. Leveraging multi-threading. Base model uses a subset of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories/tree/main)

## Architecture

Decoder-only transformer with full backpropagation:

- RMSNorm (faster than LayerNorm, better gradient flow)
- Rotary Position Embeddings (RoPE) for relative position encoding
- SwiGLU activation in feed-forward layers
- Multi-head attention with KV-cache for inference
- AdamW optimizer with warmup scheduling
- Confidence-based dynamic output length
- BPE subword tokenization

Default: 256-dim, 8 heads, 6 layers, 1024 hidden dim. TODO: Make configurable with specs

## Building

```bash
cd src
make
cd ..
mkdir base
curl -L -o ./base/TinyStories-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt
```

## Usage

**Pretrain on large corpus (builds BPE vocab):**

```bash
# Full corpus
./slim pretrain base/TinyStories-train.txt 10 base.bin

# First 1MB only (faster)
./slim pretrain base/TinyStories-train.txt 10 base.bin 1000000
```

**Finetune on specific text:**

```bash
./slim finetune base.bin mydata.txt 50 finetuned.bin
```
Example:

```bash
./slim finetune base.bin demo/bee-movie-script.txt 50 finetuned.bin
```

**Quick train from scratch:**

```bash
./slim train data.txt 100 model.bin
```
>Note: Likely gibberish
**Generate:**

```bash
./slim generate model.bin "once upon a time" 100
```

**Interactive:**

```bash
./slim chat model.bin
```

## Training Pipeline

1. Pretrain on TinyStories or similar corpus to learn general language patterns
2. Finetune on domain-specific text for focused generation
3. Generation uses confidence-based stopping (stops when model becomes uncertain)

## Design Principles

The model prioritizes training speed and coherent output over parameter count. Full backpropagation through all layers ensures the attention mechanism learns meaningful patterns rather than relying solely on embedding similarity.

Confidence-based generation prevents the model from producing low-quality tokens when it lacks certainty, resulting in shorter but more coherent outputs.

## License

MIT
