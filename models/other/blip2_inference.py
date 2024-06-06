import torch
import torch.nn as nn

from typing import Dict, List, Optional
from torch import Tensor

from tqdm import tqdm

class SpeechQFormerGenerator(nn.Module):

    def _validate_Qformer_loading(self, qformer_checkpoint_path):
        validator = {}
        checkpoint = torch.load(qformer_checkpoint_path)
        generator_values = checkpoint['model'].keys()
        for generator_ent in tqdm(generator_values):
            entities = generator_ent.split('.')[1:]

            if generator_ent.startswith('Qformer') and generator_ent not in [
                "Qformer.bert.embeddings.word_embeddings.weight",
                "Qformer.bert.embeddings.position_embeddings.weight",
                "Qformer.cls.predictions.bias",
                "Qformer.cls.predictions.transform.dense.weight",
                "Qformer.cls.predictions.transform.dense.bias",
                "Qformer.cls.predictions.transform.LayerNorm.weight",
                "Qformer.cls.predictions.transform.LayerNorm.bias",
                "Qformer.cls.predictions.decoder.weight",
                "Qformer.cls.predictions.decoder.bias",
            ]:
                network = self.model
                for ent in entities:
                    network = getattr(network, ent)
                validator[generator_ent] = (torch.eq(checkpoint['model'][generator_ent], network) == True).sum()
            else:
                continue
        return validator

    def __init__(
        self,
        models,
        tgt_dict,
        beam_size,
        max_len_a,
        max_len_b,
        min_len,
        normalize_scores,
        len_penalty,
        unk_penalty,
        temperature,
        match_source_len,
        no_repeat_ngram_size,
        **kwargs

    ):
        super().__init__()

        self.tgt_dict = tgt_dict
        self.model = models[0]

        self.model.eval()
        self.num_beams = beam_size
        self.max_length = 1000 * max_len_a + max_len_b
        self.min_length = min_len
        
        
        self.top_p = 0.9
        self.repetition_penalty = 1
        

    def cuda(self):
        self.model.cuda()
        return self

    @torch.no_grad()
    def generate(self, models, samples: Dict[str, Dict[str, Tensor]], **kwargs):
        """Generate translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            constraints (torch.LongTensor, optional): force decoder to include
                the list of constraints
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(samples['net_input'])
    
    def _generate(self, sample: Dict[str, Dict[str, Tensor]]):
        return self.model.generate(
            sample,
            False,
            1,
            200,
            10,
            self.top_p,
            self.repetition_penalty,
        )

