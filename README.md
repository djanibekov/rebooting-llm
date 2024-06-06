# Navigation

This repository uses fairseq framework for training our sytem.

* Learning rate: Tri-Stage, Decay
* Optimizer: Adam
* Hardware: one A100 GPU with 40GB memory


Models. 
    SPE - speech encoder (HuBERTbase, HuBERTlarge, SpeechTokenizer)

1. SPE-Qformerbase ```models/blip2/blip2_qformer_base_raw.py```
2. SPE-Qformerlarge ```models/blip2/blip2_qformer_large_raw.py```
3. SPE-Qformerbase-LLM ```models/blip2/blip2_qformer_gpt.py```
