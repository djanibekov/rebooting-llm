import itertools
import logging
import os
from typing import Any, List, Optional, Tuple

import numpy as np
import torchaudio
import mmap

import torch
import torch.nn.functional as F
from fairseq.data import data_utils, Dictionary
from fairseq.data.fairseq_dataset import FairseqDataset

from torch import nn
import torchaudio as ta

logger = logging.getLogger(__name__)

class MelSpectrogramTorch(nn.Module):
    """Mel-Spectrogram using Torchaudio Implementation."""
    def __init__(
        self,
        preemp: bool = True,
        n_fft: int = 512,
        log: bool = False,
        win_length: int = 400,
        hop_length: int = 160,
        f_min: int = 20,
        f_max: int = 7600,
        n_mels: int = 80,
        window_fn: str = "hamming",
        mel_scale: str = "htk",
        normalize: Optional[str] = None,
    ):
        super().__init__()

        self.log = log
        self.n_mels = n_mels
        self.preemp = preemp
        self.normalize = normalize
        if window_fn == "hann":
            self.window_fn = torch.hann_window
        elif window_fn == "hamming":
            self.window_fn = torch.hamming_window

        if preemp:
            self.register_buffer(
                "flipped_filter",
                torch.FloatTensor([-0.97, 1.0]).unsqueeze(0).unsqueeze(0),
            )

        self.transform = ta.transforms.MelSpectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            window_fn=self.window_fn,
            mel_scale=mel_scale,
        )
    

    def forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input check
        assert (
            len(input.size()) == 2
        ), "The number of dimensions of input tensor must be 2!"
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                if self.preemp:
                    # reflect padding to match lengths of in/out
                    x = input.unsqueeze(1)
                    x = F.pad(x, (1, 0), "reflect")

                    # apply preemphasis
                    x = F.conv1d(x, self.flipped_filter).squeeze(1)
                else:
                    x = input

                # apply frame feature extraction
                x = self.transform(x)

                if self.log:
                    x = torch.log(x + 1e-6)
                if self.normalize is not None:
                    if self.normalize == "mn":
                        x = x - torch.mean(x, dim=-1, keepdim=True)
                    else:
                        raise NotImplementedError(
                            f"got {self.normalize}, not implemented"
                        )

        input_length = torch.Tensor([x.size(-1)]).repeat(x.size(0))
        return x.permute(0, 2, 1)


def load_audio(manifest_path, max_keep, min_keep):
    n_long, n_short = 0, 0
    names, inds, sizes, tasks = [], [], [], []
    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) >= 2, line
            # assert len(items) >= 3, line
            sz = int(items[1])
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                names.append(items[0])
                # tasks.append(items[2])
                tasks.append('transcription')
                inds.append(ind)
                sizes.append(sz)
    tot = ind + 1
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return root, names, inds, tot, sizes, tasks


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


class SpeechToTextDataset(FairseqDataset):
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
    ):
        self.audio_root, self.audio_names, inds, tot, self.wav_sizes, self.tasks = load_audio(
            manifest_path, max_keep_sample_size, min_keep_sample_size
        )
        self.sample_rate = sample_rate
        self.shuffle = shuffle

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
        self.spectrogram_transform = MelSpectrogramTorch()

    def get_audio(self, index):
        wav_path = os.path.join(self.audio_root, self.audio_names[index])
        wav, cur_sample_rate = torchaudio.load(wav_path)
        # wav = self.postprocess(wav, cur_sample_rate)
        return wav.squeeze()

    def get_label(self, index, label_idx):
        if self.store_labels:
            label = self.label_list[label_idx][index]
        else:
            with open(self.label_paths[label_idx], encoding='utf-8') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    offset_s, offset_e = self.label_offsets_list[label_idx][index]
                    label = mm[offset_s:offset_e].decode("utf-8")

        # if self.label_processors is not None:
        #     label = self.label_processors[label_idx](label)
        return label

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]

    def __getitem__(self, index):
        wav = self.get_audio(index)
        labels = self.get_labels(index)
        spectrogram = self.spectrogram_transform(wav.unsqueeze(0))
        return {"id": index, "source": wav, "label_list": labels, "speech_id": index, "melspectrogram": spectrogram, "task": self.tasks[index]}

    def __len__(self):
        return len(self.wav_sizes)

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        audios = [s["source"] for s in samples]
        tasks = [s["task"] for s in samples]
        speech_ids = [s["speech_id"] for s in samples]
        audio_sizes = [len(s) for s in audios]

        audio_size = max(audio_sizes)
        collated_audios, padding_mask = self.collater_audio(
            audios, audio_size
        )
        
        spectorgrams = [sample['melspectrogram'].squeeze() for sample in samples]
        max_audio_size = max([len(specs) for specs in spectorgrams])
        melsize = spectorgrams[0].shape[-1]
        collated_specs, melspectrogram_padding_mask = self.collater_spectrograms(spectorgrams, max_audio_size, melsize)

        targets_by_label = [
            [s["label_list"][i].lower().replace('\n', '') for s in samples] for i in range(self.num_labels)
        ]
        lengths_list, ntokens_list = self.collater_label(targets_by_label)

        net_input = {
            "source": collated_audios, 
            "melspectrogram": collated_specs,
            "padding_mask": padding_mask,
            "audio_sizes": audio_sizes,
            "melspectrogram_padding_mask": melspectrogram_padding_mask,
            "task_name": "s2t",
            "target": targets_by_label,
            "speech_id": torch.Tensor(speech_ids),
            "tasks": tasks
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
        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        padding_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(False)
        )
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
                padding_mask[i, diff:] = True
            else:
                raise Exception("Diff should not be larger than 0")
        return collated_audios, padding_mask
    
    def collater_spectrograms(self, specs, spec_size, melsize):
        collated_specs = []
        padding_mask = (
            torch.BoolTensor(torch.rand([len(specs), spec_size]).shape).fill_(False)
        )
        for i, spec in enumerate(specs):
            diff = spec.shape[0] - spec_size
            if diff == 0:
                collated_specs.append(spec)
            elif diff < 0:
                collated_specs.append(torch.cat([
                    spec, 
                    spec.new_full((-diff,melsize), 0.0)
                ]))
                padding_mask[i, diff:] = True
            else:
                raise Exception("Diff should not be larger than 0")
        return torch.stack(collated_specs, dim=0), padding_mask

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
        return self.wav_sizes[index]

    @property
    def sizes(self):
        return np.array(self.wav_sizes)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.wav_sizes)
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

