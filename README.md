## Foundational Large Language Model: Transformer Pipeline

This repository contains a **foundational implementation of a Large Language Model (LLM)**, combining principles from GPT and other LLM architectures.
It demonstrates a full pipeline from tokenization to model training, text generation, evaluation, and visualization, implemented from scratch.

This project is based on the book: 
Sebastian Raschka - *Build a Large Language Model (From Scratch)*, Manning (2024). 
The code demonstrates foundational LLM concepts and training pipeline for educational purposes.

---

## Key Features

### 1. Foundational LLM Mechanics
- Implements the core Transformer building blocks for large language models:
  - **Token embeddings** and **positional embeddings**
  - **Multi-head self-attention** mechanism
  - **Feed-forward layers** within Transformer blocks
  - **Layer normalization** and residual connections
- Demonstrates essential LLM operations:
  - Attention computation
  - Forward pass propagation
  - Next-token prediction
- Designed as a **foundational framework** before adding higher-level architectures like GPT

### 2. GPTModel Class (Built on top of foundational LLM)
- Uses the Transformer building blocks to implement a **GPT-style model**:
  - Sequential stack of Transformer blocks
  - Linear output projection for next-token prediction
  - Supports variable batch sizes and sequence lengths
- Illustrates how **core LLM components integrate into a GPT-style architecture**

### 3. Text Tokenization
- Uses **GPT-2 tokenizer** (`tiktoken`) for converting text to token IDs
- Utility functions:
  - `text_to_token_ids(text, tokenizer)` → converts text to token tensors
  - `token_ids_to_text(token_ids, tokenizer)` → converts token tensors back to text

### 4. Text Generation
- Greedy decoding via `generate_text_simple`
  - Sequential token generation by selecting the most probable next token
- Supports advanced decoding strategies:
  - Temperature scaling
  - Probabilistic sampling with `torch.multinomial`
  - Top-k sampling for variety in generated text

### 5. Loss Calculation and Evaluation
- Cross-entropy loss between predicted logits and target token sequences
- Computes **perplexity** for interpretable performance metrics
- Evaluation utilities:
  - `calc_loss_batch` → batch-level loss
  - `calc_loss_loader` → average loss over a DataLoader
  - `evaluate_model` → training & validation loss monitoring

### 6. Dataset Preparation
- Prepares text into overlapping sequences using `GPTDatasetV1`
- Supports:
  - Sliding window text chunking
  - Automatic input-target pair creation for next-token prediction
- Compatible with PyTorch `DataLoader` for batch training

### 7. Training Loop
- `train_model_simple` function:
  - Standard LLM training loop using **AdamW optimizer**
  - Tracks training and validation loss across epochs
  - Generates text samples at the end of each epoch for qualitative evaluation
- Includes GPU/CPU device management and reproducibility utilities

### 8. Visualization
- Plots training and validation loss curves
- Dual-axis plots for epochs vs. tokens seen

### 9. Quick Testing
- Forward-pass test for validation without full training
- Ideal for low-resource environments or initial sanity checks

---

##  Project Visualizations

Below are the visual components generated during the development and training of the model. These plots verify the mathematical foundations and the training progress.

### 1. Training & Validation Loss
This plot shows how the model learned over time. The gap between the lines illustrates the training progress and the point where the model begins to overfit the small dataset.

<p align="center">
  <img src="assets/loss_plot.png" width="700px" />
</p>

---

### 2. Architectural & Decoding Mechanics
We compared activation functions and analyzed how decoding parameters like temperature affect the output distribution.

| **GELU vs ReLU Comparison** | **Temperature Scaling Impact** |
| :---: | :---: |
| <img src="assets/gelu_comparison.png" width="400px" /> | <img src="assets/temperature_scaling.png" width="400px" /> |
| *Smoother gradients for GPT blocks.* | *Shifting probability distribution ($\tau$).* |

---

### 3. Embedding Vector Space
These 3D plots represent the vector space of our token embeddings, showing how the model begins to organize relationships between different tokens in 3D space.

<p align="center">
  <img src="assets/embedding_3d_1.png" width="45%" />
  <img src="assets/embedding_3d_2.png" width="45%" />
  <br>
  <em>Figure: 3D projections of token vector relationships.</em>
</p>

---
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
    data="path/to/text/data.txt",
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
