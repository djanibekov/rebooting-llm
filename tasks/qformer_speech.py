# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import logging
import os.path as op
from argparse import Namespace
from collections import OrderedDict

import torch
from fairseq.data import (
    Dictionary, 
    encoders, 
    PrependTokenDataset,
    AppendTokenDataset, 
    data_utils, 
    StripTokenDataset,
    TokenBlockDataset,
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq import utils
from speechqformer_fairseq.data.speech_to_text_dataset import SpeechToTextDataset
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.hubert_pretraining import LabelEncoder 

import speechqformer_fairseq.criterions.speech_to_text_loss
import speechqformer_fairseq.criterions.speech_qformer_loss

from transformers import BertTokenizer

logger = logging.getLogger(__name__)

# logging.disable('WARNING')


@register_task("qformer_speech")
class QformerSpeechTask(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--sample-rate",
            default=100,
            type=float,
            help="target sample rate. audio files will be up/down sampled to this rate",
        )
        parser.add_argument(
            "--speechtokenizer-configpath",
            default=None,
            type=str,
            help="path to the config file",
        )
        parser.add_argument(
            "--speechtokenizer-ckptpath",
            default=None,
            type=str,
            help="path to the checkpoint file",
        )
        parser.add_argument(
            "--max-speech-sample-size",
            default=16000 * 30,
            type=int,
        )
        parser.add_argument(
            "--min-speech-sample-size",
            default=16000,
            type=int,
        )
       
    def __init__(self, args, dicts, config):
        super().__init__(args)
        self.dicts = dicts
        self.config = config
        self.seed = args.seed

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args, None, None)

    def build_criterion(self, args):
        from fairseq import criterions
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        speech_split, text_split = split.split('|')
        manifest = f"{self.args.data}/{speech_split}.tsv"
        paths = [f"{self.args.data}/{text_split}.txt"]
        self.datasets[split] = SpeechToTextDataset(
            manifest,
            sample_rate=self.args.sample_rate,
            label_paths=paths,
            max_keep_sample_size=self.args.max_speech_sample_size,
            min_keep_sample_size=self.args.min_speech_sample_size,
            normalize=False,
            store_labels=False,
        )
       
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)

        agg_loss, agg_sample_size, agg_logging_output = 0.0, 1.0, {}

        def forward_backward(model, samples, weight=1.0):
            nonlocal agg_loss, agg_logging_output
            if samples is None or len(samples) == 0:
                return
            loss, sample_size, logging_output = criterion(model, samples)
            if ignore_grad:
                loss *= 0
            else:
                loss *= weight
            loss = loss / sample_size
            optimizer.backward(loss)
            agg_loss += loss.detach().item()
            # # TODO make summing of the sample sizes configurable
            for k in logging_output:
                if k == 'ntokens' or k == 'nsentences':
                    if k not in agg_logging_output:
                        agg_logging_output[k] = 0
                    agg_logging_output[k] += logging_output[k]
            
            agg_logging_output.update(logging_output)
            agg_sample_size = agg_logging_output['sample_size']
        

        forward_backward(model, sample)
        agg_logging_output["loss"] = agg_loss

        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            agg_loss, agg_sample_size, agg_logging_output = 0.0, 1.0, {}
            agg_logging_output['sample_size'] = 1
            loss, sample_size, logging_output = criterion(model, sample)
            loss = loss / sample_size
            # agg_loss += loss.data.item() if isinstance(loss, torch.Tensor) else loss
            agg_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
            agg_logging_output.update(logging_output)
            agg_logging_output["loss"] = agg_loss
        return agg_loss, agg_sample_size, agg_logging_output

    @property
    def target_dictionary(self):
        return None

    @property
    def source_dictionary(self):
        return None

    def build_model(self, args):
        args.sample_rate = self.args.sample_rate
        return super(QformerSpeechTask, self).build_model(args)
