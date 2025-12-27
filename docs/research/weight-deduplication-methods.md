# Mathematical Methods for Neural Network Weight Deduplication

## Executive Summary

This document surveys mathematical methods applicable to Coral's weight deduplication system. We analyze the current implementation and identify opportunities for improvement across five key areas:

1. **Similarity Detection** - Finding similar weights efficiently
2. **Delta Encoding** - Compressing differences between similar weights
3. **Indexing & Search** - Fast lookup of candidate matches
4. **Tensor Decomposition** - Low-rank approximations for compression
5. **Entropy Coding** - Optimal bitstream encoding

---

## 1. Current Implementation Analysis

### 1.1 Similarity Computation

**Current Approach:** Hybrid cosine + magnitude similarity

```
weight_sim = 0.7 × (cosine_sim + 1)/2 + 0.3 × magnitude_sim
```

**Strengths:**
- Scale-aware (unlike pure cosine similarity)
- O(n) computation where n = number of elements
- Handles edge cases (zero vectors)

**Potential Improvements:**
- Consider **Mahalanobis distance** for distribution-aware similarity
- Explore **Earth Mover's Distance (EMD)** for weights with structural meaning
- **Spectral similarity** using eigenvalue comparison for layer-level matching

### 1.2 Indexing (LSH)

**Current Approach:** Random hyperplane LSH with configurable tables/hyperplanes

**Strengths:**
- O(1) average-case lookup
- Multi-dimensional support via dimension-specific indices

**Gaps Identified:**
- No adaptive hyperplane selection based on data distribution
- Could benefit from hierarchical LSH for multi-scale similarity

---

## 2. Advanced Similarity Detection Methods

### 2.1 SimHash (Recommended for Investigation)

**Concept:** Generate compact binary fingerprints where similar vectors produce similar fingerprints.

**Algorithm:**
1. Project vector onto k random directions
2. Create k-bit signature based on projection signs
3. Compare via Hamming distance

**Advantages over current LSH:**
- Fixed-size fingerprint (e.g., 64-128 bits) vs. variable bucket assignments
- Can be stored/compared with O(1) operations
- Hamming distance ≈ angular distance

**Implementation Reference:** [Google's SimHash for Web Deduplication](https://research.google.com/pubs/archive/33026.pdf)

### 2.2 MinHash with Jaccard Similarity

**Concept:** For sparse weight representations, treat non-zero positions as set elements.

**Formula:**
```
J(A,B) = |A ∩ B| / |A ∪ B|
P(h(A) = h(B)) = J(A,B)  # For MinHash function h
```

**Use Case:** Excellent for pruned networks where many weights are zero.

**Reference:** [MinHash LSH in Milvus](https://milvus.io/blog/minhash-lsh-in-milvus-the-secret-weapon-for-fighting-duplicates-in-llm-training-data.md)

### 2.3 Product Quantization for Similarity Search

**Concept:** Decompose vectors into subspaces, quantize each independently.

**Algorithm:**
1. Split D-dimensional vector into M subvectors of D/M dimensions
2. Train K centroids per subspace (K typically 256 for 8-bit codes)
3. Encode vector as M centroid indices
4. Distance ≈ sum of precomputed sub-distances

**Compression:** 512x reduction possible (e.g., 2048-dim float32 → 8 bytes)

**Advantages:**
- Asymmetric distance computation is fast
- Memory-efficient candidate storage
- Proven in billion-scale systems (FAISS)

**Reference:** [Product Quantization for Nearest Neighbor Search](https://www.pinecone.io/learn/series/faiss/product-quantization/)

---

## 3. Delta Encoding Improvements

### 3.1 Bitwise XOR Delta (Highly Recommended)

**Current State:** Coral uses arithmetic delta (weight - reference)

**Improved Approach:** Bitwise XOR for floating-point checkpoints

**Algorithm:**
1. Reinterpret floats as integers
2. Compute XOR: `delta_bits = weight_bits ^ reference_bits`
3. Split into exponent and mantissa streams
4. Compress each stream independently

**Results from Literature:**
- 62% compression for BF16
- 83% compression for FP8
- Exploits correlation in exponent bits across similar weights

**Reference:** [Lossless Compression for LLM Tensor Incremental Snapshots](https://arxiv.org/html/2505.09810v1)

### 3.2 Per-Axis Weight Deltas

**Concept:** Store only sign of difference + per-axis scaling factors

**Algorithm:**
```python
delta = weight - reference
signs = np.sign(delta)  # 1-bit per element
row_scales = np.mean(np.abs(delta), axis=1)  # FP16 per row
col_scales = np.mean(np.abs(delta), axis=0)  # FP16 per column
```

**Compression:** 5.24x smaller than FP16 for fine-tuned models

**Reference:** [Per-Axis Weight Deltas for Frequent Model Updates](https://arxiv.org/html/2512.19720)

### 3.3 Delta-DCT (Discrete Cosine Transform)

**Concept:** Transform delta to frequency domain, quantize, entropy code

**Algorithm:**
1. Reshape delta to 2D blocks (e.g., 8×8)
2. Apply DCT to each block
3. Quantize high-frequency components more aggressively
4. Entropy code the result

**Advantages:**
- Data-free (no training required)
- Exploits spatial correlation in weight matrices
- Inspired by JPEG compression

**Reference:** [Seeing Delta Parameters as JPEG Images](https://arxiv.org/html/2503.06676)

### 3.4 Quantization-Aware Delta Encoding

**Current State:** Coral has INT8/INT16 quantization strategies

**Improvement - Inshrinkerator Method:**
```python
# Observe that most weights stay in same quantization bin
prev_bin = quantize(prev_weight)
curr_bin = quantize(curr_weight)

# Only store bin changes (sparse)
changed_indices = np.where(prev_bin != curr_bin)
changes = curr_bin[changed_indices] - prev_bin[changed_indices]
```

**Compression:** 3-4x for sequential checkpoints

**Reference:** [Inshrinkerator: Compressing Deep Learning Training Checkpoints](https://arxiv.org/html/2306.11800)

---

## 4. Tensor Decomposition Methods

### 4.1 Truncated SVD for Weight Approximation

**Concept:** Approximate weight matrices with low-rank factors

**Formula:**
```
W ≈ U_r @ S_r @ V_r^T
```
Where r << min(m, n) retains top singular values.

**Storage:**
- Original: m × n values
- Decomposed: m×r + r + r×n = (m + n + 1)×r values
- Savings when r < mn/(m+n+1)

**Applications to Deduplication:**
- Store reference as full-rank, similar weights as rank-delta
- `delta_low_rank = SVD(weight - reference, k=small_rank)`

**Improvement - Fisher-Weighted SVD:**
Weight singular values by Fisher information (gradient sensitivity):
```
FWSVD: minimize ||W - U@S@V^T||_F weighted by Fisher matrix
```
This preserves task-critical components better.

**Reference:** [Language Model Compression with Weighted Low-Rank Factorization](https://openreview.net/forum?id=uPv9Y3gmAI5)

### 4.2 Tucker Decomposition for Convolutional Layers

**Concept:** Decompose 4D conv weights (out, in, h, w) into core + factors

**Formula:**
```
W ≈ G ×₁ U₁ ×₂ U₂ ×₃ U₃ ×₄ U₄
```
Where G is a small core tensor and U_i are factor matrices.

**Advantages over SVD:**
- Preserves tensor structure (no need to reshape to matrix)
- Can compress each mode independently
- Natural for conv layers

**Reference:** [Distribution-Aware Tensor Decomposition for CNN Compression](https://arxiv.org/html/2511.04494)

### 4.3 Tensor Train Decomposition

**Concept:** Chain of 3D tensors for extreme compression

**Formula:**
```
W[i₁,i₂,...,iₙ] = G₁[i₁] @ G₂[i₂] @ ... @ Gₙ[iₙ]
```

**Advantages:**
- Exponentially fewer parameters for high-dimensional tensors
- No curse of dimensionality
- Good for embedding layers

---

## 5. Entropy Coding Strategies

### 5.1 Current State: zlib Compression

Coral uses zlib (DEFLATE) for the COMPRESSED delta strategy.

### 5.2 Improvements

**Huffman Coding for Weight Distribution:**
Neural network weights follow predictable distributions (often Gaussian-like). Custom Huffman codes can exploit this.

**Deep Compression Pipeline:**
1. Prune weights (10x reduction)
2. Quantize to k centroids (3x reduction)
3. Huffman encode indices (1.3x reduction)
4. **Total: 35-49x compression**

**Reference:** [Deep Compression: Pruning, Quantization, Huffman Coding](https://arxiv.org/abs/1510.00149)

**Arithmetic Coding:**
More efficient than Huffman for non-power-of-2 symbol probabilities:
```
Bits per symbol ≈ -log₂(p) vs. ceil(-log₂(p)) for Huffman
```

**Exponent-Mantissa Separation (ZipNN):**
```python
# Separate float components
exponents = (float_bits >> 23) & 0xFF
mantissas = float_bits & 0x7FFFFF

# Compress each with optimal entropy coder
compressed_exp = huffman_encode(exponents)  # Low entropy
compressed_man = arithmetic_encode(mantissas)  # High entropy
```

**Reference:** [ZipNN: Lossless Compression for AI Models](https://arxiv.org/pdf/2411.05239)

---

## 6. Random Projection Methods

### 6.1 Johnson-Lindenstrauss Lemma

**Theorem:** n points in high-D space can be embedded in O(log n / ε²) dimensions while preserving pairwise distances within (1±ε).

**Application:** Reduce weight vectors to ~100-1000 dimensions for fast similarity search before exact computation.

**Formula:**
```
y = (1/√k) @ R @ x
```
Where R is a k×d random matrix with entries from N(0,1).

**Reference:** [Random Projection in Dimensionality Reduction](https://dl.acm.org/doi/10.1145/502512.502546)

### 6.2 Sparse Random Projection (Achlioptas)

**Improvement:** Use sparse projection matrix for faster computation

```python
# Instead of dense Gaussian:
R[i,j] = sqrt(3) × {+1 with prob 1/6, 0 with prob 2/3, -1 with prob 1/6}
```

**Advantages:**
- 3x faster matrix multiply
- Integer arithmetic possible
- Same JL guarantees

---

## 7. Learned Index Structures

### 7.1 Concept

Replace B-trees/hash tables with learned models that approximate the CDF of keys.

**Application to Weight Deduplication:**
- Train small neural network to predict bucket for weight fingerprint
- Can achieve 70% speedup and 10x memory savings over B-trees

**Reference:** [The Case for Learned Index Structures](https://arxiv.org/abs/1712.01208)

### 7.2 PGM-Index

Piecewise Geometric Model index with provable worst-case bounds:
- O(log log n) lookup time
- O(n) space
- Dynamic updates supported

---

## 8. Recommended Implementation Priorities

### High Priority (Significant Impact, Moderate Effort)

| Method | Expected Improvement | Implementation Complexity |
|--------|---------------------|--------------------------|
| XOR Delta Encoding | +15-25% compression | Medium |
| Exponent-Mantissa Separation | +10-20% compression | Low |
| Per-Axis Weight Deltas | +20-30% for fine-tuned models | Medium |
| SimHash Fingerprinting | Faster similarity lookup | Low |

### Medium Priority (Good Impact, Higher Effort)

| Method | Expected Improvement | Implementation Complexity |
|--------|---------------------|--------------------------|
| Product Quantization | 10-100x faster search | High |
| Delta-DCT | +10-15% compression | Medium |
| Quantization-Aware Delta | +15-20% for checkpoints | Medium |
| Truncated SVD Delta | Variable (rank-dependent) | Medium |

### Lower Priority (Specialized Use Cases)

| Method | Use Case | Notes |
|--------|----------|-------|
| Tucker Decomposition | Conv layers | Preserves tensor structure |
| MinHash | Pruned/sparse models | Requires sparsity |
| Learned Indexes | Very large deployments (>1M weights) | Research stage |

---

## 9. Mathematical Foundations Reference

### Key Formulas

**Cosine Similarity:**
```
cos(a,b) = (a·b) / (||a|| × ||b||)
```

**Jaccard Similarity:**
```
J(A,B) = |A ∩ B| / |A ∪ B|
```

**Johnson-Lindenstrauss Bound:**
```
k ≥ 8 × ln(n) / ε²
```

**SVD Approximation Error:**
```
||A - A_k||_F = √(σ_{k+1}² + ... + σ_r²)
```

**Entropy (bits per symbol):**
```
H(X) = -Σ p(x) × log₂(p(x))
```

---

## 10. Sources

### Similarity & Hashing
- [SimHash for Web Deduplication - Google Research](https://research.google.com/pubs/archive/33026.pdf)
- [MinHash LSH in Milvus](https://milvus.io/blog/minhash-lsh-in-milvus-the-secret-weapon-for-fighting-duplicates-in-llm-training-data.md)
- [Locality Sensitive Hashing Guide - Pinecone](https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/)
- [Random Projection in Dimensionality Reduction - ACM](https://dl.acm.org/doi/10.1145/502512.502546)

### Delta Encoding & Compression
- [Lossless Compression for LLM Tensor Incremental Snapshots](https://arxiv.org/html/2505.09810v1)
- [Per-Axis Weight Deltas for Frequent Model Updates](https://arxiv.org/html/2512.19720)
- [Seeing Delta Parameters as JPEG Images (Delta-DCT)](https://arxiv.org/html/2503.06676)
- [Inshrinkerator: Dynamic Quantization for Checkpoints](https://arxiv.org/html/2306.11800)
- [ZipNN: Lossless Compression for AI Models](https://arxiv.org/pdf/2411.05239)

### Tensor Decomposition
- [Distribution-Aware Tensor Decomposition for CNN Compression](https://arxiv.org/html/2511.04494)
- [Language Model Compression with Weighted Low-Rank Factorization](https://openreview.net/forum?id=uPv9Y3gmAI5)
- [Low-Rank Matrix Approximation for Neural Network Compression](https://arxiv.org/html/2504.20078)

### Entropy Coding
- [Deep Compression: Pruning, Quantization, Huffman Coding](https://arxiv.org/abs/1510.00149)
- [Survey of Model Compression Techniques - Frontiers](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1518965/full)

### Vector Search
- [Product Quantization for Nearest Neighbor Search - Pinecone](https://www.pinecone.io/learn/series/faiss/product-quantization/)
- [The Case for Learned Index Structures](https://arxiv.org/abs/1712.01208)

---

*Document generated: December 2024*
*Coral Version: 1.0*
