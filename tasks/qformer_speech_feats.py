
import logging

import torch
from data.speech_to_text_dataset_feats import SpeechToTextDatasetTokenizer
from fairseq.tasks import LegacyFairseqTask, register_task

from criterions import speech_to_text_loss
from criterions import speech_qformer_loss

from transformers import BertTokenizer

logger = logging.getLogger(__name__)

# logging.disable('WARNING')


@register_task("qformer_speech_feats")
class QformerSpeechTaskTokenized(LegacyFairseqTask):
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
        hubert_feats = [f"{self.args.data}/{speech_split}.km"]
        paths = [f"{self.args.data}/{text_split}.txt"]

        self.datasets[split] = SpeechToTextDatasetTokenizer(
            hubert_feats,
            label_paths=paths,
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
        return super(QformerSpeechTaskTokenized, self).build_model(args)

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        from .sequence_generator import SequenceGenerator
        extra_gen_cls_kwargs = {
            "ctc_weight": 0,
            **extra_gen_cls_kwargs
        }
        return super().build_generator(
            models, args, seq_gen_cls=SequenceGenerator, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    @property
    def target_dictionary(self):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="right")
        def pad():
            return tokenizer.pad_token_id
        def unk():
            return tokenizer.unk_token_id
        def eos():
            return tokenizer.eos_token_id
        
        setattr(tokenizer, 'pad', pad)
        setattr(tokenizer, 'unk', pad)
        setattr(tokenizer, 'eos', pad)
        return tokenizer

    @property
    def source_dictionary(self):
        return None
