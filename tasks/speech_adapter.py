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
from speech_adapter.data.speech_to_text_dataset import SpeechToTextDataset
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.hubert_pretraining import LabelEncoder 

import speech_adapter.criterions.speech_to_text_loss

logger = logging.getLogger(__name__)

# logging.disable('WARNING')


@register_task("speech_adapter")
class SpeechAdapterTask(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--max-speech-sample-size",
            default=None,
            type=int,
            metavar="N",
            help="max speech sample size",
        )
        parser.add_argument(
            "--min-speech-sample-size",
            default=None,
            type=int,
            metavar="N",
            help="min speech sample size",
        )
        parser.add_argument(
            "--max-speech-positions",
            default=4000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-text-positions",
            default=450,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument(
            "--sample-rate",
            default=100,
            type=float,
            help="target sample rate. audio files will be up/down sampled to this rate",
        )
        parser.add_argument(
            "--random-crop",
            action="store_true",
            help="always crop from the beginning if false",
        )
        parser.add_argument(
            "--batch-ratio",
            default=None,
            type=str,
            help="ratio of bach size for each dataset",
        )
        parser.add_argument(
            "--speechtokenizer_configpath",
            default=None,
            type=str,
            help="path to the config file",
        )
        parser.add_argument(
            "--speechtokenizer_ckptpath",
            default=None,
            type=str,
            help="path to the checkpoint file",
        )
        parser.add_argument(
            "--llama-model",
            default=None,
            type=str,
            help="name of the llama version",
        )
       
    def __init__(self, args, dicts, config):
        super().__init__(args)
        self.dicts = dicts
        self.config = config
        self.seed = args.seed

    @classmethod
    def setup_task(cls, args, **kwargs):
        # load dictionaries and config
        # dicts = OrderedDict()
        # if args.t5_task == 'pretrain' and not hasattr(args, "shuffle_instance"):
        #     args.shuffle_instance = False

        # # Prepare config
        # config = None
        # logger.info('No config file for ' + args.t5_task)

        # if args.t5_task == "pretrain":
        #     dicts["hubert"] = [Dictionary.load(f"{args.hubert_label_dir}/dict.{label}.txt") for label in args.hubert_labels]
        #     dicts["text"] = Dictionary.load(op.join(args.data, "dict.txt"))
        # else:
        #     if config is None:
        #         dicts["text"] = Dictionary.load(op.join(args.data, "dict.txt"))
        #     else:
        #         dicts["text"] = Dictionary.load(op.join(args.data, config.vocab_filename))

        

        return cls(args, None, None)

    def build_criterion(self, args):
        from fairseq import criterions
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        speech_split, text_split = split.split('|')
        manifest = f"{self.args.data}/{speech_split}.tsv"
        # procs = [LabelEncoder(self.dicts["text"])]
        paths = [f"{self.args.data}/{text_split}.txt"]
        self.datasets[split] = SpeechToTextDataset(
            manifest,
            sample_rate=self.args.sample_rate,
            label_paths=paths,
            max_keep_sample_size=self.max_pos[0] if self.args.max_speech_sample_size is None else self.args.max_speech_sample_size,
            min_keep_sample_size=self.args.min_speech_sample_size,
            normalize=False,
            store_labels=False,
            tgt_dict=None,
            tokenizer=None,
        )
       
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)

        # Junyi: not use sample_size, but normalize the loss locally
        agg_loss, agg_sample_size, agg_logging_output = 0.0, 1.0, {}
        agg_logging_output['sample_size'] = 1

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
                    # continue
                # agg_logging_output[k] += logging_output[k]
                # agg_logging_output[task_name] += logging_output[k]
            agg_logging_output['model'] = logging_output

        forward_backward(model, sample)

        agg_logging_output["loss"] = agg_loss

        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            from collections import defaultdict

            agg_loss, agg_sample_size, agg_logging_output = 0.0, 1.0, defaultdict(float)
            agg_logging_output['sample_size'] = 1
            loss, sample_size, logging_output = criterion(model, sample)
            loss = loss / sample_size
            # agg_loss += loss.data.item() if isinstance(loss, torch.Tensor) else loss
            agg_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
            agg_logging_output['model'] = logging_output
            agg_logging_output["loss"] = agg_loss
        return agg_loss, agg_sample_size, agg_logging_output

    @property
    def target_dictionary(self):
        return None

    @property
    def source_dictionary(self):
        return None

    def build_model(self, args):
        try:
            args.input_feat_per_channel = self.config.input_feat_per_channel
            args.input_channels = self.config.input_channels
        except Exception as e:
            args.input_feat_per_channel = 80
            args.input_channels = 1
            logger.info(f"Cannot set input_feat_per_channel, input_channels, since: ")
            logger.warn(e)
            logger.info(f"Set to: {args.input_feat_per_channel} and {args.input_channels}")

        args.speech_odim = args.input_feat_per_channel * args.input_channels

        args.sample_rate = self.args.sample_rate
        return super(SpeechAdapterTask, self).build_model(args)
