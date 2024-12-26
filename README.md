# Building a Simple Transformer

## What is a Transformer?
A Transformer is a deep learning model that excels in natural language processing (NLP) tasks. It uses attention mechanisms to process sequences, allowing the model to focus on different parts of the input data simultaneously. Transformers have been instrumental in the development of state-of-the-art models like GPT and BERT.

### How Does It Work?
Transformers rely on self-attention mechanisms, which assign importance to tokens in a sequence based on their relevance to other tokens. They use positional encoding to capture the order of tokens, enabling the model to understand context and relationships effectively.

---

## Overview of the Project
The **Enwik8** dataset has been used inorder to build a character-level language model based on the GPT-style decoder architecture. 

The project will:
- Implement the architecture in an object-oriented style using PyTorch.
- Compute the **Bits Per Character (BPC)** on the test part of the Enwik8 dataset.

### Dataset Details
The **Enwik8** dataset contains 100 million characters:
- **90 million** characters for training.
- **5 million** characters for validation.
- **5 million** characters for testing.

---

## Classes Implemented in the Project

### 1. **`MyNLPDataSet`**
This class creates a PyTorch dataset for NLP tasks. It:
- Returns random sequences of length `seq_len + 1` from the dataset.
- Prepares data for GPU training.

---

### 2. **`Utils.py`**
The utility script contains the following elements:
- Stores the data path.
- Creates the test, training, and validation datasets.
- Builds the training dataloader.

---

### 3. **`PositionalEncoding`**
Positional encoding assigns each token a unique value based on its position in a sequence. This helps the transformer model understand the relative or absolute order of tokens. The positional encoding is calculated using sinusoidal patterns, combining sine and cosine waves.

---

### 4. **`MHSelfAttention`**
This class implements multi-headed self-attention using the **`einops`** library. Key points:
- A mask is applied during training to ensure causality.
- The mask is not used during inference.

---

### 5. **`TransformerBlock`**
The transformer block represents the decoder in this architecture. It uses `MHSelfAttention` as a core building block.

---

### 6. **`SimpleTransformer`**
This class holds:
- Stacks multiple transformer blocks.
- Adds a linear layer at the end to produce the logits.

---

### 7. **`TransformerAMMain`**
The main file that runs the project. It was tested using:
- **Visual Studio**: Version 17.12.3
- **Virtual Environment**: Set up using Anaconda 2.6.4
- **Python Version**: Use Python 3.9–3.11 (Do not use Python 2.12 or earlier for compatibility with PyTorch).

---

## Environment Setup

### Setting Up the Virtual Environment
Ensure the system has an NVIDIA GPU for CUDA compatibility. CUDA is NVIDIA's parallel computing platform, essential for leveraging GPU acceleration with PyTorch.

If you **do not have an NVIDIA GPU**, use PyTorch with CPU support by running:
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

Alternatively, you can use cloud platforms like **Google Colab**, **AWS**, or **Azure**, which provide GPU support.

---

## Commands to Set Up the Environment
1. Install Anaconda:
   ```bash
   conda install anaconda
   ```
2. Set up the Python environment:
   ```bash
   conda create -n myenv python=3.9
   conda activate myenv
   ```
3. Install PyTorch (with CUDA):
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
   ```

---

## Conclusion
In this project, we built a simple transformer-based language model using the **Enwik8** dataset. The modular object-oriented implementation in PyTorch facilitates understanding and experimentation with the architecture. With GPU support, this model is ready for efficient training and inference.

