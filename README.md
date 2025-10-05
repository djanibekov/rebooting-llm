
<a href=''> <a href='https://aclanthology.org/2025.iwslt-1.6/'><img src='https://img.shields.io/badge/paper-Paper-red'></a>
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

This repository contains implementation code for speech and large language model integration for speech to text translation task, SparQLe. SparQLe is a speech understanding and processing framework that integrates query-former (Qformer) architecture with large language models for speech-to-text applications.

## Overview
With the growing influence of Large Language Models (LLMs), there is increasing interest in integrating speech representations with them to enable more seamless multi-modal processing and speech understanding. This study introduces a novel approach that combines self-supervised speech representations with instruction-tuned LLMs for speech-to-text translation. The proposed approach leverages a modality adapter to align extracted speech features with instruction-tuned LLMs using English speech data. Our experiments demonstrate that this method effectively preserves the semantic content of the input speech and serves as an effective bridge between self-supervised speech models and instruction-tuned LLMs, offering a promising approach for various speech understanding applications.

List of available trained weights for the model SparQLe from Hugging Face Model Hub: [SparQLe - best](https://huggingface.co/amupd/SparQLe/blob/main/SparQLe_best.pt), [SparQLe - last](https://huggingface.co/amupd/SparQLe/blob/main/SparQLe_last.pt)

### Key Contributions

- **Modality Adapter**: Enables efficient text extraction from speech signals, bridging the gap between audio and language representations.
- **Multi-Model Support**: Compatible with various speech self-supervised learning (SSL) or encoder networks, allowing flexible integration across architectures.
- **Zero-Shot Translation for Low-Resource Languages**: Leverages LLM integration to translate speech inputs from input language (English) into languages supported by the model without requiring parallel data.
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

## Usage

### Prerequisites

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

### Inference 
```python
python "SparQLe-gradio/app.py"
```

## Citation

If you use SparQLe in your research, please cite:

```bibtex
@inproceedings{djanibekov-aldarmaki-2025-sparqle,
    title = "{S}par{QL}e: Speech Queries to Text Translation Through {LLM}s",
    author = "Djanibekov, Amirbek  and
      Aldarmaki, Hanan",
    editor = "Salesky, Elizabeth  and
      Federico, Marcello  and
      Anastasopoulos, Antonis",
    booktitle = "Proceedings of the 22nd International Conference on Spoken Language Translation (IWSLT 2025)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria (in-person and online)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.iwslt-1.6/",
    doi = "10.18653/v1/2025.iwslt-1.6",
    pages = "76--83",
    ISBN = "979-8-89176-272-5",
    abstract = "With the growing influence of Large Language Models (LLMs), there is increasing interest in integrating speech representations with them to enable more seamless multi-modal processing and speech understanding. This study introduces a novel approach that combines self-supervised speech representations with instruction-tuned LLMs for speech-to-text translation. The proposed approach leverages a modality adapter to align extracted speech features with instruction-tuned LLMs using English speech data. Our experiments demonstrate that this method effectively preserves the semantic content of the input speech and serves as an effective bridge between self-supervised speech models and instruction-tuned LLMs, offering a promising approach for various speech understanding applications."
}
```
---

## Acknowledgments

- This work builds upon [fairseq](https://github.com/facebookresearch/fairseq)
- The Qformer architecture is inspired by [BLIP-2](https://github.com/salesforce/BLIP-2)

