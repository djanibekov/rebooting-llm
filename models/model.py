# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import logging
import contextlib
from ast import literal_eval
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)

from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .encoder_speech import SpeechEncoderPrenet
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.transformer import Embedding
from torch import Tensor

from transformers import LlamaTokenizer
from .llama import LlamaForCausalLM

from .q_former import BertConfig, BertLMHeadModel

from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_MAX_TEXT_POSITIONS = 450
DEFAULT_MAX_SPEECH_POSITIONS = 4000


@register_model("speech_adapter")
class T5TransformerModel(FairseqEncoderDecoderModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    def __init__(
            self, 
            args,
            encoder,
            decoder,
            speech_encoder_prenet,
            num_query_token=64,
            drop=False
        ):
        super().__init__(encoder, decoder)

        # self.encoder = encoder
        # self.decoder = decoder
        self.speech_encoder_prenet = speech_encoder_prenet
        self.speech_transformation = nn.Linear(1024, 768)
        self.drop = True
        # breakpoint()

        print('Loading Q-Former dropout:', self.drop)
        self.Qformer, self.audio_query_tokens = self.init_Qformer(
            num_query_token, self.speech_encoder_prenet.config['codebook_size'], drop=self.drop
        )
        self.device = self.audio_query_tokens.device
        # self.audio_query_tokens = nn.Parameter(
        #     torch.zeros(1, num_query_token, self.audio_query_tokens.shape[-1])
        # )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        # self.load_from_pretrained(url_or_filename=q_former_model)

        # if freeze_qformer:
        #     for name, param in self.Qformer.named_parameters():
        #         param.requires_grad = False
        #     self.Qformer = self.Qformer.eval()
        #     self.Qformer.train = disabled_train
        #     self.image_query_tokens.requires_grad = False
        #     self.audio_query_tokens.requires_grad = False
        #     for name, param in self.audio_trans.named_parameters():
        #         param.requires_grad = False
        #     logging.info("freeze Qformer")
        print('Loading Q-Former Done')


        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
            )

        for name, param in tqdm(self.llama_model.named_parameters()):
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(768, 4096)
        self.end_sym = '\n'


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--reduction-factor",
            type=int,
            help="reduction factor for decoder",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--encoder-sliding-window-attn",
            default=None,
            type=int,
            help="If not None but a even number, set sliding window attention to encoder's attn_mask, e.g., 4, 10, and 20",
        )

        # relative pos enc
        parser.add_argument(
            "--relative-position-embedding",
            action='store_true',
            help="whether to use relative position embedding",
        )
        parser.add_argument(
            "--num-buckets",
            type=int,
            default=320,
            help="num of buckets for relative position embedding",
        )
        parser.add_argument(
            "--max-distance",
            type=int,
            default=1280,
            help="max distance for relative position embedding",
        )
        parser.add_argument(
            "--encoder-max-relative-position",
            type=int,
            help="max distance for relative position embedding in encoder",
        )
        parser.add_argument(
            "--decoder-max-relative-position",
            type=int,
            help="max distance for relative position embedding in decoder",
        )

        # others
        parser.add_argument(
            "--bert-init",
            action='store_true',
            help="initilize as bert",
        )
        parser.add_argument(
            "--unb-enc-layer",
            type=int,
            default=-1,
            help="which layer's output is used as the input of decoder",
        )

    # Encoder, Decoder
    @classmethod
    def build_encoder(cls, args, dictionary=None, embed_tokens=None):
        return TransformerEncoder(args, dictionary, embed_tokens)

    @classmethod
    def build_decoder(cls, args):
        return TransformerDecoder(args)

    @classmethod
    def build_speech_encoder_prenet(cls, args):
        return SpeechEncoderPrenet(args)

    @classmethod
    def build_text_encoder_prenet(cls, args):
        return TextEncoderPrenet(args)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        speech_odim = args.speech_odim
        encoder = cls.build_encoder(args)      
        decoder = cls.build_decoder(args)

        speech_encoder_prenet = cls.build_speech_encoder_prenet(args)
        return cls(
            args,
            encoder,
            decoder, 
            speech_encoder_prenet
        )

    @classmethod
    def init_Qformer(cls, num_query_token, modality_width, cross_attention_freq=2, drop=True):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = modality_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        if not drop:
            encoder_config.hidden_dropout_prob = 0.0
            encoder_config.attention_probs_dropout_prob = 0.0
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def get_normalized_probs_for_ctc(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out_for_ctc"][0]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_logits(self, net_output, is_masked=True):
        if is_masked:
            logits_list = net_output["logit_m_list"]
        else:
            logits_list = net_output["logit_u_list"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, sample, net_output, is_masked=True):
        if "logit_m_list" in net_output:
            logits_list = self.get_logits(net_output, is_masked)
            targets_list = [
                x.new_zeros(x.size(0), dtype=torch.long) for x in logits_list
            ]
            return targets_list
        else:
            return sample["target"]

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        if "prob_perplexity" in net_output:
            extra_losses.append(
                (net_output["num_vars"] - net_output["prob_perplexity"])
                / net_output["num_vars"]
            )
            names.append("prob_perplexity")

        return extra_losses, names
        

    def forward(
            self, 
            source=None, 
            src_tokens=None,  # absent
            src_lengths=None,  # absent
            prev_output_tokens=None, 
            tgt_lengths=None,  # absent
            target_list=None,  # absent
            task_name=None,
            padding_mask=None, 
            only_hubert=False,   # absent
            only_ctc=False,      # absent
            feature_only=False,  # absent
            tgt_enc_layer=None,  # absent
            mask=True,            # absent
            target=None,
        ):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        self.llama_tokenizer.padding_side = "right"
        batch_size = len(source)
        device = source[0].device
        assert source is not None or src_tokens is not None
        # encoded_speech = []
        # processed_target = []
        # for i in range(batch_size):
        #     temp = self.speech_encoder_prenet(source[i].unsqueeze(0))
        #     if temp is not None:
        #         encoded_speech.append(temp)
        #         processed_target.append(target[0][i])

        
        
        # audio_size = max([s.shape[1] for s in encoded_speech])
        # collated_speech = encoded_speech[0].new_zeros(len(encoded_speech), audio_size, 1024)
        # padding_mask = (torch.BoolTensor(collated_speech.shape[:2]).fill_(False))
        # # breakpoint()
        # for i, audio in enumerate(encoded_speech):
        #     diff = audio.shape[1] - audio_size
        #     batch, timestamp, dim = audio.shape
        #     if diff == 0:
        #         collated_speech[i] = audio
        #     elif diff < 0:
        #         # breakpoint()
        #         collated_speech[i] = torch.cat([audio, audio.new_full((batch,-diff,dim), 0.0)], dim=1)
        #         padding_mask[i, -diff:] = True
        #     else:
        #         raise Exception("Diff should not be larger than 0")

        collated_speech = self.speech_encoder_prenet(source.unsqueeze(1))
        # encoded_speech = torch.cat(encoded_speech, dim=0)
        # breakpoint()
        # audio_atts = torch.ones(collated_speech.size()[:-1], dtype=torch.long).to(device)
        query_tokens = self.audio_query_tokens.expand(collated_speech.shape[0], -1, -1)
        query_output = self.Qformer.bert(query_embeds=query_tokens,encoder_hidden_states=collated_speech,encoder_attention_mask=None,return_dict=True)    
        
        aud_embeds = self.llama_proj(query_output.last_hidden_state)
        atts_llama = torch.ones(aud_embeds.size()[:-1], dtype=torch.long).to(collated_speech.device)
        to_regress_tokens = self.llama_tokenizer(target[0],return_tensors="pt",padding="longest",truncation=True,max_length=200,add_special_tokens=False).to(device)
        
        # breakpoint()

        targets = to_regress_tokens.input_ids.masked_fill(to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100)
        empty_targets = (torch.ones([atts_llama.shape[0], atts_llama.shape[1]+1],dtype=torch.long).to(device).fill_(-100))  # plus one for bos
        
        targets = torch.cat([empty_targets, targets], dim=1)
        
        batch_size = aud_embeds.shape[0]

        bos = torch.ones([batch_size, 1],dtype=to_regress_tokens.input_ids.dtype, device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)

        atts_bos = atts_llama[:, :1]
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)

        inputs_embeds = torch.cat([bos_embeds, aud_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_llama, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss
        # breakpoint()

        return {"loss": loss, 'output': outputs}

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()


    def forward_encoder_torchscript(self, net_input: Dict[str, Tensor]):
        """A TorchScript-compatible version of forward.

        Encoders which use additional arguments may want to override
        this method for TorchScript compatibility.
        """
        if torch.jit.is_scripting():
            return self.forward_encoder(
                source=net_input["source"],
                padding_mask=net_input["padding_mask"]
            )
        else:
            return self.forward_encoder_non_torchscript(net_input)

    @torch.jit.unused
    def forward_encoder_non_torchscript(self, net_input: Dict[str, Tensor]):
        encoder_input = {
            k: v for k, v in net_input.items() if k != "prev_output_tokens" and k != "task_name"
        }
        return self.forward_encoder(**encoder_input)

    def forward_encoder(self, source, padding_mask=None):
        # Encoder Prenet
        encoder_input, encoder_padding_mask = self.speech_encoder_prenet(source, padding_mask=padding_mask, mask=False)

        # Encoder
        encoder_output = self.encoder(encoder_input, encoder_padding_mask)

        return encoder_output

    def forward_decoder(self, tokens, encoder_out, incremental_state):
        # Decoder Prenet
        prev_output_tokens, tgt_mask, incremental_state = self.text_decoder_prenet(tokens, incremental_state)

        # Decoder
        decoder_output, extra = self.decoder(
            prev_output_tokens,
            tgt_mask,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
        )

        # Decoder Postnet
        return self.text_decoder_postnet(decoder_output), extra

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


@register_model_architecture(model_name="speech_adapter", arch_name="speech_adapter")
def base_architecture(args):
    # Transformer
    args.bert_init = getattr(args, "bert_init", False)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768 * 4)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.max_text_positions = getattr(args, "max_text_positions", DEFAULT_MAX_TEXT_POSITIONS)
    args.max_speech_positions = getattr(args, "max_speech_positions", DEFAULT_MAX_SPEECH_POSITIONS)

    # # Espnet related, including prenet, postnet
    # args.eprenet_conv_layers = getattr(args, "eprenet_conv_layers", 0)
    # args.eprenet_conv_filts = getattr(args, "eprenet_conv_filts", 0)
    # args.eprenet_conv_chans = getattr(args, "eprenet_conv_chans", 0)
    # args.use_batch_norm = getattr(args, "use_batch_norm", True)
    # args.eprenet_dropout_rate = getattr(args, "eprenet_dropout_rate", 0.0)
    # args.enc_use_scaled_pos_enc = getattr(args, "enc_use_scaled_pos_enc", True)
    # args.dec_use_scaled_pos_enc = getattr(args, "dec_use_scaled_pos_enc", True)
    # args.postnet_layers = getattr(args, "postnet_layers", 5)
    # args.postnet_chans = getattr(args, "postnet_chans", 256)
    # args.postnet_filts = getattr(args, "postnet_filts", 5)
    # args.postnet_dropout_rate = getattr(args, "postnet_dropout_rate", 0.5)
    # args.dprenet_dropout_rate = getattr(args, "dprenet_dropout_rate", 0.5)
    # args.dprenet_layers = getattr(args, "dprenet_layers", 2)
    # args.dprenet_units = getattr(args, "dprenet_units", 256)
    # args.initial_encoder_alpha = getattr(args, "initial_encoder_alpha", 1.0)
    # args.initial_decoder_alpha = getattr(args, "initial_decoder_alpha", 1.0)
    # args.spk_embed_integration_type = getattr(args, "spk_embed_integration_type", "pre")
    # args.spk_embed_dim = getattr(args, "spk_embed_dim", 512)
    # args.encoder_reduction_factor = getattr(args, "encoder_reduction_factor", 1)
    # args.reduction_factor = getattr(args, "reduction_factor", 2)
    # args.transformer_enc_positional_dropout_rate = getattr(args, "transformer_enc_positional_dropout_rate", 0.1)
    # args.transformer_dec_positional_dropout_rate = getattr(args, "transformer_dec_positional_dropout_rate", 0.1)
    # args.layer_norm_eps = getattr(args, "layer_norm_eps", 1e-5)
    # args.no_scale_embedding = getattr(args, "no_scale_embedding", True)

    # Relative pos embed
    args.relative_position_embedding = getattr(args, "relative_position_embedding", False)
    args.num_buckets = getattr(args, "num_buckets", 320)
    args.max_distance = getattr(args, "max_distance", 1280)
    args.encoder_max_relative_position = getattr(args, "encoder_max_relative_position", 160)
    args.decoder_max_relative_position = getattr(args, "decoder_max_relative_position", 160)

# @register_model_architecture("speech_adapter", "t5_transformer_base")
# def t5_transformer_base(args):
#     args.use_conv_pos = getattr(args, "use_conv_pos", True)
#     args.use_sinc_pos = getattr(args, "use_sinc_pos", True)
#     args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
#     args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
#     args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
#     args.layer_norm_first = getattr(args, "layer_norm_first", False)
#     args.relative_position_embedding = getattr(args, "relative_position_embedding", True)
#     args.dropout = getattr(args, "dropout", 0.1)
#     args.activation_dropout = getattr(args, "activation_dropout", 0.0)
#     args.attention_dropout = getattr(args, "attention_dropout", 0.1)
#     args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.05)
#     args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.05)
#     args.mask_prob = getattr(args, "mask_prob", 0.80)
#     base_architecture(args)

# @register_model_architecture("speech_adapter", "t5_transformer_large")
# def t5_transformer_large(args):
#     args.use_conv_pos = getattr(args, "use_conv_pos", True)
#     args.use_sinc_pos = getattr(args, "use_sinc_pos", True)
#     args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
#     args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
#     args.layer_norm_first = getattr(args, "layer_norm_first", True)
#     args.relative_position_embedding = getattr(args, "relative_position_embedding", True)
#     args.dropout = getattr(args, "dropout", 0.0)
#     args.activation_dropout = getattr(args, "activation_dropout", 0.0)
#     args.attention_dropout = getattr(args, "attention_dropout", 0.0)
#     args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)
#     args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
#     args.encoder_layers = getattr(args, "encoder_layers", 24)
#     args.decoder_layers = getattr(args, "decoder_layers", 6)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
#     args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
#     args.feature_grad_mult = getattr(args, "feature_grad_mult", 1.0)
#     args.extractor_mode = getattr(args, "extractor_mode", "layer_norm")
#     args.final_dim = getattr(args, "final_dim", 768)
#     args.mask_prob = getattr(args, "mask_prob", 0.80)
#     base_architecture(args)

# @register_model_architecture("speech_adapter", "t5_transformer_base_asr")
# def t5_transformer_base_asr(args):
#     args.use_conv_pos = getattr(args, "use_conv_pos", True)
#     args.use_sinc_pos = getattr(args, "use_sinc_pos", True)
#     args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
#     args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
#     args.layer_norm_first = getattr(args, "layer_norm_first", False)
#     args.relative_position_embedding = getattr(args, "relative_position_embedding", True)
#     args.dropout = getattr(args, "dropout", 0.1)
#     args.activation_dropout = getattr(args, "activation_dropout", 0.1)
#     args.attention_dropout = getattr(args, "attention_dropout", 0.1)
#     args.feature_grad_mult = getattr(args, "feature_grad_mult", 0.0)
#     args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.1)
#     args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.1)
#     args.mask_prob = getattr(args, "mask_prob", 0.75)
#     args.mask_selection = getattr(args, "mask_selection", "static")
#     args.mask_channel_length = getattr(args, "mask_channel_length", 64)
#     args.mask_channel_prob = getattr(args, "mask_channel_prob", 0.5)
#     args.mask_channel_selection = getattr(args, "mask_channel_selection", "static")
#     args.max_text_positions = getattr(args, "max_text_positions", 600)
#     base_architecture(args)