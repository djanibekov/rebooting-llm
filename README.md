# SparQLe

SparQLe is a speech understanding and processing framework that integrates query-former (Qformer) architecture with large language models for speech-to-text applications. List all available pretrained weights for the model SparQLe from Hugging Face Model Hub.
- [SparQLe_best]([url](https://huggingface.co/amupd/SparQLe/blob/main/SparQLe_best.pt))
- [SparQLe_last]([url](https://huggingface.co/amupd/SparQLe/blob/main/SparQLe_last.pt))

---

## Repository Structure

The repository consists of two main components:

### 1. SparQLe-fairseq 

A fairseq-based implementation with the following structure:

```
SparQLe-fairseq/
├── criterions/           # Loss functions for training
├── data/                 # Dataset implementations for speech/text
├── models/
│   ├── [deprecated]/     # Legacy models and components
│   └── sparqle/          # Core SparQLe model implementations
└── tasks/                # Task definitions for fairseq
```

### 2. SparQLe-hf 

A Hugging Face compatible implementation with training scripts:

```
SparQLe-hf/
├── dataset.py            # Dataset utilities
├── encoders.py           # Encoding utilities 
├── librispeechdataset.py # LibriSpeech dataset handling 
├── qformer.py            # Query-former implementation 
└── train_*.py            # Various training scripts
```

---

## Features

- Speech-to-text conversion using Qformer architecture
- Integration with large language models
- Support for raw audio and feature-based inputs
- Fine-tuning capabilities for different downstream tasks
- Modular architecture for experimentation

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- fairseq
- transformers

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/SparQLe.git
cd SparQLe

# Install fairseq dependencies
cd SparQLe-fairseq
pip install -e .

# Install HF dependencies
cd ../SparQLe-hf
pip install -r requirements.txt
```

---

## Training

### Pretraining the Qformer (Stage 1)

```bash
cd SparQLe-hf
# Add your pretraining commands here...
```

### Fine-tuning with LLM (Stage 2)

```bash
cd SparQLe-hf
# Add your fine-tuning commands here...
```

---

## Inference

```python
# TODO: Add inference example
```

## Citation

If you use SparQLe in your research, please cite:

```bibtex
@misc{djanibekov2025sparqlespeechqueriestext,
      title={SparQLe: Speech Queries to Text Translation Through LLMs},
      author={Amirbek Djanibekov and Hanan Aldarmaki},
      year={2025},
      eprint={2502.09284},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.09284},
}
```

📄 Read the full paper on arXiv: [https://arxiv.org/abs/2502.09284](https://arxiv.org/abs/2502.09284)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- This work builds upon [fairseq](https://github.com/facebookresearch/fairseq) 💙
- The Qformer architecture is inspired by [BLIP-2](https://github.com/salesforce/BLIP-2) ✨

