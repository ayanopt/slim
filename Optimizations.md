# Technical Optimizations

## Compiler Optimizations

```
-O3                 Aggressive optimization level
-march=native       CPU-specific instruction set
-ffast-math         Relaxed IEEE compliance for speed
-flto               Link-time optimization across translation units
-funroll-loops      Loop unrolling for reduced branch overhead
-pthread            POSIX threads support
```

## Numerical Optimizations

### Fast Exponential
Clamped input range $[-88, 88]$ prevents overflow/underflow. Uses standard expf which compilers optimize to hardware instructions.

### Fast Tanh Approximation
$$\tanh(x) \approx \frac{x(27 + x^2)}{27 + 9x^2}$$

Pade approximant accurate to $\sim 0.001$ in $[-3, 3]$, saturates outside. Avoids expensive exp calls in activation functions.

### SiLU Activation
$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

Smooth, non-monotonic activation. Backward pass computed analytically:
$$\frac{d}{dx} \text{SiLU}(x) = \sigma(x)(1 + x(1 - \sigma(x)))$$

## Memory Layout

### Row-Major Tensors
All matrices stored row-major for cache-friendly sequential access during matrix multiplication. Inner loop iterates over contiguous memory.

### Activation Caching
Forward pass stores intermediate activations (LayerCache) for backward pass reuse. Trades memory for compute by avoiding recomputation.

## Parallelization Strategy

### Thread Pool Pattern
Static thread count via hardware_concurrency(), cached to avoid repeated syscalls.

### Work Distribution
- Matrix multiply: parallelize over output rows
- Attention: parallelize over heads (independent computations)
- FFN: parallelize over sequence positions
- Gradient updates: parallelize over parameter dimensions

### Chunk Sizing
Minimum work threshold before spawning threads. Small workloads run single-threaded to avoid synchronization overhead.

## Gradient Flow

### RMSNorm vs LayerNorm
$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma$$
$$\text{RMS}(x) = \sqrt{\text{mean}(x^2) + \epsilon}$$

No mean subtraction, fewer operations, similar normalization effect. Gradient flows more directly through the scaling.

### Residual Connections
Pre-norm architecture: normalize before attention/FFN, add residual after. Gradient flows through both the transformed path and identity shortcut.

### Gradient Clipping
Per-parameter clipping to $[-1, 1]$ before optimizer step. Prevents exploding gradients without global norm computation.

## Optimizer: AdamW

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$
$$\theta_t = \theta_{t-1} - \alpha \left( \frac{m_t}{\sqrt{v_t} + \epsilon} + \lambda \cdot \theta_{t-1} \right)$$

Decoupled weight decay ($\lambda$ term applied to parameters, not gradients). Bias correction built into learning rate:
$$\alpha_t = \alpha \cdot \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}$$

### Warmup Schedule
Linear warmup over first $N$ steps prevents early training instability when gradients are large and Adam statistics are cold.

## Attention Mechanism

### Scaled Dot-Product
$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right) V$$

Scale factor prevents softmax saturation with large dimension.

### Causal Masking
Future positions masked with $-10^9$ before softmax. Ensures autoregressive property during training.

### KV-Cache (Inference)
Keys and values from previous positions cached and concatenated. Only new position's $Q$ computed against full $K, V$ history. Reduces inference from $O(n^2)$ to $O(n)$ per token.

## Rotary Position Embeddings (RoPE)

$$R_\theta(x, m) = \begin{bmatrix} x_1 \cos(m\theta) - x_2 \sin(m\theta) \\ x_1 \sin(m\theta) + x_2 \cos(m\theta) \\ \vdots \end{bmatrix}$$

Position information encoded as rotation in 2D subspaces. Relative position naturally emerges from dot product:
$$\langle R_\theta(q, m), R_\theta(k, n) \rangle \text{ depends on } (m - n)$$

Precomputed $\sin/\cos$ tables indexed by position. Backward pass applies inverse rotation (negate $\sin$ terms).

## BPE Tokenization

### Training
1. Initialize vocabulary with characters + end-of-word marker
2. Count adjacent pair frequencies across corpus
3. Merge most frequent pair into new token
4. Repeat for $N$ merges

### Encoding
Apply learned merges greedily left-to-right. Unknown characters fall back to UNK token.

### Decoding
Concatenate tokens, replace `</w>` markers with spaces.

## Confidence-Based Generation

Top token probability after softmax indicates model certainty. Generation stops when:
- Top probability falls below threshold (default $0.15$)
- Three consecutive low-confidence tokens
- EOS/PAD token sampled
- Max length reached

Prevents degenerate repetition and nonsense continuation.

## Backpropagation Implementation

### Forward Pass Storage
Each layer stores: input, normalized input, $Q/K/V$ projections, attention scores, attention output, FFN intermediates.

### Backward Pass Order
1. Compute loss gradient (cross-entropy: $\hat{y} - y$)
2. Backprop through output projection
3. For each layer (reverse order):
   - FFN backward ($W_2 \to$ activation $\to W_1/W_3 \to$ norm)
   - Attention backward ($W_o \to$ attention weights $\to Q/K/V$ projections $\to$ norm)
   - Add residual gradient
4. Accumulate embedding gradients

### Softmax Backward
$$\frac{\partial \mathcal{L}}{\partial x_i} = y_i \left( \frac{\partial \mathcal{L}}{\partial y_i} - \sum_j y_j \cdot \frac{\partial \mathcal{L}}{\partial y_j} \right)$$

Efficient formulation avoiding full Jacobian computation.

## References

- Pade, H. (1892). Sur la representation approchee d'une fonction par des fractions rationnelles.
- Elfwing et al. (2018). Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning. Neural Networks.
- Zhang & Sennrich (2019). Root Mean Square Layer Normalization. NeurIPS.
- Loshchilov & Hutter (2019). Decoupled Weight Decay Regularization. ICLR.
- Goyal et al. (2017). Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour. arXiv:1706.02677.
- Vaswani et al. (2017). Attention Is All You Need. NeurIPS.
- Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864.
- Sennrich et al. (2016). Neural Machine Translation of Rare Words with Subword Units. ACL.
