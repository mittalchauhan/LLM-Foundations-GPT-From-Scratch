# 🤖 LLM-Foundations: GPT From Scratch

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Status](https://img.shields.io/badge/status-complete-brightgreen.svg)

This repository contains a **foundational implementation of a Large Language Model (LLM)**, combining principles from GPT and other Transformer architectures. It demonstrates a full pipeline from tokenization to model training, evaluation, and visualization from scratch.

> [!IMPORTANT]  
> Based on: **Sebastian Raschka - Build a Large Language Model (From Scratch)**, Manning (2024).

---

## Core Features & Mechanics

### Transformer Architecture
* **Foundational Blocks:** Implements Token/Positional embeddings, Multi-head self-attention, and Feed-forward layers.
* **GPT Model Class:** A sequential stack of Transformer blocks with linear output projection for next-token prediction.
* **Stability:** Integrated Layer Normalization and Residual Connections.

### Data Pipeline & Tokenization
* **Tiktoken:** Uses GPT-2 BPE tokenizer for efficient encoding.
* **Dataset (GPTDatasetV1):** Implements sliding window chunking for automatic input-target pair creation.

### Training & Inference
* **Optimization:** Standard training loop using **AdamW optimizer** with cross-entropy loss tracking.
* **Decoding:** Supports Greedy decoding, **Temperature Scaling**, and **Top-k Sampling** for varied text generation.
* **Evaluation:** Built-in utilities for **Perplexity** calculation and batch-level loss monitoring.

---

## Project Visualizations
To verify architectural correctness, I tracked training progress and mathematical behaviors.

### Model Performance & Latent Analysis
To verify the model's integrity, we monitor both the mathematical convergence (Loss) and the resulting spatial organization of the learned embeddings.

| **Training & Validation Loss** | **Embedding Topology (3D)** | **Cluster Projections (3D)** |
| :---: | :---: | :---: |
| <img src="assets/loss_plot.png" width="400" height="250" /> | <img src="assets/embedding_3d_1.png" width="400" height="250"/> | <img src="assets/embedding_3d_2.png" width="400" height="250"/> |
---

### Technical Deep-Dive
| **GELU vs ReLU Comparison** | **Temperature Scaling Impact** |
| :---: | :---: |
| <img src="assets/gelu_comparison.png" height="280" /> | <img src="assets/temperature_scaling.png" height="280" /> |

---

### Installation
```bash
git clone [https://github.com/](https://github.com/)<your-username>/LLM-Foundations-GPT-From-Scratch.git
cd LLM-Foundations-GPT-From-Scratch
pip install torch tiktoken matplotlib
### Installation

Install required dependencies:

```bash
pip install torch tiktoken matplotlib
```


### Usage

Clone the repository:

```bash
git clone <repo-url>
cd <repo-folder>
```

Load data and initialize the model:

```bash
from model import GPTModel
import tiktoken, torch

tokenizer = tiktoken.get_encoding("gpt2")
model = GPTModel(
    vocab_size=len(tokenizer),
    block_size=128,  # context length
    n_layer=6,       # number of transformer blocks
    n_head=8,        # number of attention heads
    n_embd=256       # embedding dimension
)
```

Train the Model

```bash
from train import train_model_simple

train_model_simple(
    model=model,
    data="data.txt",
    tokenizer=tokenizer,
    epochs=10,
    batch_size=16,
    lr=3e-4,
    device="cuda"  # or "cpu"
)
```

Quick Forward-Pass Test 

```bash
from utils import text_to_token_ids, token_ids_to_text, generate_text_simple, evaluate_model

input_text = "Once upon a time"
input_ids = text_to_token_ids(input_text, tokenizer)
```

Forward pass example

```bash
output_ids = generate_text_simple(model, input_ids, max_new_tokens=50)
output_text = token_ids_to_text(output_ids, tokenizer)
print(output_text)
```

Evaluate the Model
```bash
loss, perplexity = evaluate_model(model, val_loader)
print(f"Validation Loss: {loss:.4f}, Perplexity: {perplexity:.4f}")
```"# LLM-Foundations-GPT-From-Scratch" 
