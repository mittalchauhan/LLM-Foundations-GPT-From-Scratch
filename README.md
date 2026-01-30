# 🤖 LLM-Foundations: GPT From Scratch

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Status](https://img.shields.io/badge/status-complete-brightgreen.svg)

A complete, from-scratch implementation of a Large Language Model (LLM) pipeline. This project moves from raw text tokenization to training a GPT-style architecture with custom attention mechanisms.

> **Inspiration:** Based on *Build a Large Language Model (From Scratch)* by Sebastian Raschka (2024).

---

## 💎 Project Highlights

### 🧱 Core Architecture
* **Transformer Blocks:** Implements custom self-attention, layer normalization, and residual connections.
* **GPT Class:** A modular stack of decoder blocks capable of handling variable sequence lengths.
* **Embeddings:** Dual-layer approach using both **Token** and **Positional** embeddings.

### ⚙️ Pipeline & Mechanics
* **Processing:** Uses `tiktoken` (BPE) for encoding and `GPTDatasetV1` for sliding-window data loading.
* **Decoding:** Features Greedy search, **Temperature Scaling**, and **Top-k Sampling** for creative text generation.
* **Optimization:** Driven by **AdamW** with cross-entropy loss and **Perplexity** tracking.

---

## 📊 Technical Gallery

To verify the model's health and mathematical foundation, I tracked the following visualizations:

### 📈 Convergence & Loss
| **Training Snapshot** | 
| :---: | 
| <img src="assets/loss_plot.png" width="550" /> | 
| *Training vs. Validation loss over 10 epochs.* |

---

### 🧪 Model Mechanics & Embeddings
We forced these images to match heights to ensure a clean, level interface.

| **GELU vs ReLU** | **Temperature Scaling** |
| :---: | :---: |
| <img src="assets/gelu_comparison.png" height="250" /> | <img src="assets/temperature_scaling.png" height="250" /> |
| *Activation smoothing.* | *Distribution shifts.* |

| **Vector Space A** | **Vector Space B** |
| :---: | :---: |
| <img src="assets/embedding_3d_1.png" height="250" /> | <img src="assets/embedding_3d_2.png" height="250" /> |
| *Token relationships.* | *Cluster projections.* |

---

## 🚀 Quick Start

```bash
# Install
pip install torch tiktoken matplotlib

# Basic Usage
from model import GPTModel
from train import train_model_simple

model = GPTModel(vocab_size=50257, block_size=128, n_layer=6, n_head=8, n_embd=256)
train_model_simple(model=model, data="data.txt", epochs=10, lr=3e-4)