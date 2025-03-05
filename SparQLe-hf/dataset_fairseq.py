import os
import soundfile as sf
from pathlib import Path
from typing import Tuple, Union

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar, _load_waveform
import logging
import string

logger = logging.getLogger(__name__)

from tqdm import tqdm


def load_audio(manifest_path, max_keep, min_keep):
    n_long, n_short = 0, 0
    names, inds, sizes = [], [], []
    with open(manifest_path) as f:
        root = f.readline()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) >= 2, line
            sz = int(items[1])
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                names.append(items[0])
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
    return names, root


class CustomDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path] = None,
        manifest_file: str = None,
        text_file: str = None,
        maxkeep: int = None,
        minkeep: int = None
    ) -> None:
        
    
        self.translator = str.maketrans('', '', string.punctuation)

        assert maxkeep is not None, 'You should specify max keep number of frames'
        assert minkeep is not None, 'You should specify min keep number of frames'
        
        self._audiowalker, self.root = load_audio(manifest_file, maxkeep, minkeep)
        with open(text_file) as f:
            self._textwalker = f.readlines()

        self._walker = list(filter(lambda x: len(x[0]) > 10, zip(self._textwalker, self._audiowalker)))

    def get_path(self, n: int) -> Tuple[str, int, str, int, int, int]:
        fileid = self._walker[n][1]
        return os.path.join(self.root, f"{fileid}")
    
    def get_source_text(self, n: int):
        return self._walker[n][0]
    
    def text_normalization(self, text: str) -> str:
        return text.translate(self.translator).lower().replace('\n', '')


    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
        filepath = self.get_path(n)
        waveform, sampling_rate = sf.read(filepath)
        waveform = torch.Tensor(waveform)

        source_transcript = self.text_normalization(self.get_source_text(n))
        return waveform, sampling_rate, source_transcript, "", filepath


    def __len__(self) -> int:
        return len(self._walker)
    
    
if __name__ == "__main__":
    dataset = CustomDataset(
        manifest_file='/l/speech_lab/_SpeechT5PretrainDataset/FinetuneV3/ASR/_manifest/IWSLT_Translate/translate_train.tsv',
        text_file='/l/speech_lab/_SpeechT5PretrainDataset/FinetuneV3/ASR/_manifest/IWSLT_Translate/translate_train.txt',
        maxkeep=30 * 16000,
        minkeep=16000
    )
    print(f"Loaded test with {len(dataset)} examples")
    for data in dataset:
        breakpoint()
    
