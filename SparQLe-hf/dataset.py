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
    return names, inds, tot, sizes


class CustomDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        manifest_file: str = None,
        maxkeep: int = None,
        minkeep: int = None
    ) -> None:
        root = os.fspath(root)
        self.root = root
        self._ext_audio = ".wav"
        self._ext_text = ".txt"
        self._path = os.path.join(root)
        self.translator = str.maketrans('', '', string.punctuation)

        if manifest_file:
            assert maxkeep is not None, 'You should specify max keep number of frames'
            assert minkeep is not None, 'You should specify min keep number of frames'
            self._walker = load_audio(manifest_file, maxkeep, minkeep)[0]
        else:
            self._walker = sorted(str(p.stem) for p in Path(self._path).glob("*" + self._ext_audio))
            self._walker = list(filter(lambda x: sf.info(f'{root}{x}.wav').frames > 24000, self._walker))


    def get_path(self, n: int) -> Tuple[str, int, str, int, int, int]:
        fileid = self._walker[n]
        return os.path.join(self.root, f"{fileid}{self._ext_audio}")
    
    def get_source_text(self, n: int) -> Tuple[str, int, str, int, int, int]:
        fileid = self._walker[n]
        with open(os.path.join(self.root, f"{fileid}_source{self._ext_text}")) as f:
            return f.read()
    
    def get_target_text(self, n: int) -> Tuple[str, int, str, int, int, int]:
        fileid = self._walker[n]
        with open(os.path.join(self.root, f"{fileid}_target{self._ext_text}")) as f:
            return f.read()
        
    def text_normalization(self, text: str) -> str:
        return text.translate(self.translator).lower()


    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
        filepath = self.get_path(n)
        waveform, sampling_rate = sf.read(filepath)
        waveform = torch.Tensor(waveform)

        source_transcript = self.text_normalization(self.get_source_text(n))
        target_transcript = self.text_normalization(self.get_target_text(n))
        return waveform, sampling_rate, source_transcript, target_transcript, filepath


    def __len__(self) -> int:
        return len(self._walker)
    
    
if __name__ == "__main__":
    dataset = CustomDataset(
        root='/l/users/hanan.aldarmaki/training_code/SparQLe/.cache/IWSLT.OfflineTask/data/en-de/tst2022/segmented',
    )
    print(f"Loaded test with {len(dataset)} examples")
    for data in dataset:
        breakpoint()
    
