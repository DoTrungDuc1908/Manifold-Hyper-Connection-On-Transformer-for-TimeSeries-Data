# mHC-Time-Series  
## Manifold-Constrained Hyper-Connections for Transformer Time-Series Forecasting

This repository contains a research-oriented PyTorch implementation of **Hyper-Connections (HC)** and **Manifold-Constrained Hyper-Connections (mHC)** for Transformer-based time-series forecasting.

The project investigates whether manifold-constrained residual routing can improve:

- forecasting accuracy,
- optimization stability,
- information propagation,
- and depth scalability

across multiple Transformer architectures and benchmark datasets.

The implementation is inspired by:

- Hyper-Connections: https://arxiv.org/abs/2409.19606
- mHC: https://arxiv.org/abs/2512.24880

---

# Project Goal

Modern Transformer forecasting models rely heavily on residual propagation. Standard residual connections are stable but limited in representational flexibility.

HC extends the residual mechanism by introducing:

- multi-stream residual routing,
- learnable mixing operators,
- flexible information propagation across layers.

mHC further improves this idea by constraining the residual routing process through manifold-inspired normalization (Sinkhorn-based doubly stochastic routing), with the goal of:

- reducing unstable gradient amplification,
- improving deep-layer information flow,
- stabilizing optimization for deep Transformers.

---

# Supported Architectures

The repository currently supports:

| Baseline | HC Variant | mHC Variant |
|---|---|---|
| Vanilla Transformer | HC Transformer | mHC Transformer |
| Autoformer | HC Autoformer | mHC Autoformer |
| iTransformer | HC iTransformer | mHC iTransformer |
| PatchTST | HC PatchTST | mHC PatchTST |
