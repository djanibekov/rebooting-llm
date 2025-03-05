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
        code_lengths = [len(line) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


class SpeechToTextDatasetTokenizer(FairseqDataset):
    def __init__(
        self,
        hubert_feats: List[str],
        label_paths: List[str],
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        shuffle: bool = True,
        normalize: bool = False,
        store_labels: bool = True,
        speechtokenizer_configpath=None,
        speechtokenizer_ckptpath=None
    ):
        with open(hubert_feats[0]) as f:
            self.clusters_lengths = [len(line) for line in f]
        inds, tot = range(len(self.clusters_lengths)), len(self.clusters_lengths)
        self.shuffle = shuffle

        self.samples_number = tot
        self.num_labels = len(label_paths)
        self.store_labels = store_labels
        if store_labels:
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
            self.feats_list = [load_label(p, inds, tot) for p in hubert_feats]
        else:
            self.label_paths = label_paths
            self.hubert_feats = hubert_feats

            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]
            self.feats_list = [
                load_label_offset(p, inds, tot) for p in hubert_feats
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

        return label

    def get_feat(self, index, label_idx):
        if self.store_labels:
            feat = self.feats_list[label_idx][index]
        else:
            with open(self.hubert_feats[label_idx]) as f:
                feat = f.readlines()[index].strip()
        return feat

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]

    def get_feats(self, index):
        return [self.get_feat(index, i) for i in range(self.num_labels)]

    def __getitem__(self, index):
        wav = self.get_feats(index)
        labels = self.get_labels(index)
        return {"id": index, "source_feats": wav, "label_list": labels, "speech_id": index}

    def __len__(self):
        return self.samples_number

    def collater(self, samples):
        samples = [s for s in samples if s["source_feats"] is not None]
        if len(samples) == 0:
            return {}

        audios = [s["source_feats"] for s in samples]
        speech_ids = [s["speech_id"] for s in samples]
        audio_sizes = [len(s) for s in audios]

        audio_size = max(audio_sizes)
        collated_audios = self._collate_frames([torch.Tensor([int(val) for val in audio[0].split()]).unique_consecutive().to(int) + 1 for audio in audios])

        targets_by_label = [
            [s["label_list"][i] for s in samples] for i in range(self.num_labels)
        ]
        lengths_list, ntokens_list = self.collater_label(targets_by_label)

        net_input = {
            "source_feats": collated_audios, 
            "padding_mask_feats": collated_audios == 0,
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

    def _collate_frames(self, frames: List[torch.Tensor]):
        max_len = max(frame.size(0) for frame in frames)
        out = frames[0].new_zeros((len(frames), max_len))
        for i, v in enumerate(frames):
            out[i, : v.size(0)] = v
        return out

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
        return self.clusters_lengths[index]

    @property
    def sizes(self):
        return np.array(self.clusters_lengths)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.clusters_lengths)
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