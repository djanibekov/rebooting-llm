# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import itertools
import logging
import os
from typing import Any, List, Optional

import numpy as np
import torchaudio

import torch
import torch.nn.functional as F
from fairseq.data import data_utils, Dictionary
from fairseq.data.fairseq_dataset import FairseqDataset

from ..models.speechtokenizer.model import SpeechTokenizer
import pandas as pd

logger = logging.getLogger(__name__)


def load_label(label_path, inds, tot):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds, tot):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


class SpeechToTextDatasetTokenizer(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        label_paths: List[str],
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        shuffle: bool = True,
        normalize: bool = False,
        store_labels: bool = True,
        speechtokenizer_configpath=None,
        speechtokenizer_ckptpath=None
    ):
        self.speechcodes = pd.read_csv(manifest_path, sep='\t')[['speechcodes_str']]
        self.sample_rate = sample_rate
        self.shuffle = shuffle
        inds = list(range(len(self.speechcodes)))
        tot = len(inds)

        self.num_labels = len(label_paths)
        self.store_labels = store_labels
        if store_labels:
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
        else:
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]

        self.normalize = normalize
        logger.info(
            f"normalize={normalize}"
        )

    def get_label(self, index, label_idx):
        if self.store_labels:
            label = self.label_list[label_idx][index]
        else:
            with open(self.label_paths[label_idx]) as f:
                offset_s, offset_e = self.label_offsets_list[label_idx][index]
                f.seek(offset_s)
                label = f.read(offset_e - offset_s)

        # if self.label_processors is not None:
        #     label = self.label_processors[label_idx](label)
        return label

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]

    def __getitem__(self, index):
        wav = list(map(lambda x: int(x), self.speechcodes.iloc[index].to_dict()['speechcodes_str'].split(' ')))
        labels = self.get_labels(index)
        return {"id": index, "source": wav, "label_list": labels, "speech_id": index}

    def __len__(self):
        return len(self.speechcodes)

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        audios = [s["source"] for s in samples]
        speech_ids = [s["speech_id"] for s in samples]
        audio_sizes = [len(s) for s in audios]

        audio_size = max(audio_sizes)
        collated_audios, padding_mask = self.collater_audio(
            audios, audio_size
        )

        targets_by_label = [
            [s["label_list"][i] for s in samples] for i in range(self.num_labels)
        ]
        lengths_list, ntokens_list = self.collater_label(targets_by_label)

        net_input = {
            "source": torch.Tensor(collated_audios).to(int), 
            "padding_mask": padding_mask,
            "task_name": "s2t",
            "target": targets_by_label,
            "speech_id": torch.Tensor(speech_ids)
        }
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
            "target": targets_by_label,
            "task_name": "s2t",
            "ntokens": ntokens_list[0]
        }

        return batch

    def collater_audio(self, audios, audio_size):
        collated_audios = np.zeros((len(audios), audio_size))
        padding_mask = torch.BoolTensor(*collated_audios.shape).fill_(False)
        for i, audio in enumerate(audios):
            audio = torch.Tensor(audio)
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0)])
                padding_mask[i, diff:] = True
            else:
                raise Exception("Diff should not be larger than 0")
        return collated_audios, padding_mask

    def collater_seq_label(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        return lengths, ntokens

    def collater_label(self, targets_by_label):
        lengths_list, ntokens_list = [], []
        itr = zip(targets_by_label, [-100])
        for targets, pad in itr:
            lengths, ntokens = self.collater_seq_label(targets, pad)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return lengths_list, ntokens_list

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return len(self.speechcodes.iloc[index].to_dict()['speechcodes_str'].split(' '))

    @property
    def sizes(self):
        return np.array(self.speechcodes['speechcodes_str'].apply(lambda x: len(x)))

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.speechcodes['speechcodes_str'].apply(lambda x: len(x)))
        return np.lexsort(order)[::-1]

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav