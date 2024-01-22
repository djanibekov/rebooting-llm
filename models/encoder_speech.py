# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import logging
import torch
import json
from typing import List, Tuple
import torch.nn as nn

import numpy as np
from models.speechtokenizer import SpeechTokenizer
from collections import Counter

from fairseq.modules import (
    LayerNorm,
    FairseqDropout,
)
import torch.nn.functional as F
import torch.nn as nn

logger = logging.getLogger(__name__)

class Adapter(nn.Module):
    """
    Adapter for model finetuning, as described in:
    https://arxiv.org/pdf/1909.08478.pdf
    """

    def __init__(self, embed_dim, proj_dim, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.down_proj = nn.Linear(embed_dim, proj_dim)
        self.up_proj = nn.Linear(proj_dim, embed_dim)
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = F.relu(x)
        x = self.up_proj(x)
        x = self.dropout_module(x)
        x += residual
        return x

class SpeechEncoderPrenet(nn.Module):
    def __init__(
            self, 
            args,
            device='cuda'
        ):
        super().__init__()
        self.device = torch.device(device)
        self.register_buffer("version", torch.Tensor([3]))

        with open(args.speechtokenizer_configpath) as config_file:
            self.config = json.load(config_file)

        # logger.info(f"\n{self.speech_model}")

        print('Loading SpeechTokenizer')
        self.speech_model = SpeechTokenizer.load_from_checkpoint(
            args.speechtokenizer_configpath, args.speechtokenizer_ckptpath
        ).to(self.device)
        self.speech_model.eval()
        self.prune_modules()

        for name, param in self.speech_model.named_parameters():
            param.requires_grad = False
        print('Loading SpeechTokenizer Done')
        default_dtype = torch.get_default_dtype()
        self.adapter = Adapter(
            self.config['codebook_size'],
            64,
            args.dropout
        )

        
    def forward(self, source, padding_mask=None, **kwargs):
        # breakpoint()
        return self._forward(source, padding_mask)


    def _forward(self, source, padding_mask=None):
        # breakpoint()
        codes = self.speech_model.encode(source, 1) # codes: (n_q, B, T)
        # RVQ_1 = codes[:1, :, :] # Contain content info, can be considered as semantic tokens
        # RVQ_supplement = codes[1:, :, :] # Contain timbre info, complete info lost by the first quantizer
        if torch.unique(codes).numel() < 0.20 * codes.shape[-1]:
            logging.info('Non-unique tensor is detected')
            breakpoint()
            return None

        x = self.speech_model.quantizer.decode(codes)  # (B, C, T)
        x = x.transpose(1, 2)  # Changes shape from (B, C, T) to (B, T, C)
        # x = x.transpose(0, 1)  # Changes shape from (B, T, C) to (T, B, C)

        return x


    def prune_modules(self, modules_filter=None):
        if hasattr(self.speech_model, "decoder"): 
            del self.speech_model.decoder