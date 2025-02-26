import os
import json
import logging
from packaging import version

import torch
import torch.nn as nn
from fairseq.models import register_model, register_model_architecture

from models.sparqle.utils import SparQLeBase, disabled_train
from transformers import AutoModel, AutoTokenizer

# from models.speechtokenizer.model import SpeechTokenizer

import random
random.seed(0)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x)
        return ret.type(orig_type)

@register_model("sparqle_finetune_llm_instruct")
class SparQLeLLMInstruct(SparQLeBase):
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        # if args.speech_encoder_model == 'speechtokenizer':
        #     speech_encoder, config = cls.build_speech_model_speechtoknenizer({
        #         'speech_model_file_path': args.speechtokenizer_ckptpath,
        #         'speech_model_config_path': args.speechtokenizer_configpath,
        #         'freeze_enc': True
        #     }) 
        #     codebook_size = config['codebook_size']
        if args.speech_encoder_model == 'hubertbase':
            speech_encoder, config = cls.build_speech_model_hubertbase({
                'speech_model_file_path': args.speechtokenizer_ckptpath,
                'speech_model_config_path': args.speechtokenizer_configpath,
                'freeze_enc': True
            })
            codebook_size = config.hidden_size
        
        elif args.speech_encoder_model == 'hubertlarge':
            speech_encoder, config = cls.build_speech_model_hubertlarge({
                'speech_model_file_path': args.speechtokenizer_ckptpath,
                'speech_model_config_path': args.speechtokenizer_configpath,
                'freeze_enc': True
            }) 
            codebook_size = config.hidden_size
        else:
            raise NotImplemented   
        Qformer, query_tokens, tokenizer = cls.build_qformer({
            'codebook_size': codebook_size,
            'cross_attention_freq': args.cross_attention_freq,
            'num_query_token': args.num_query_token
        })

        return cls(
            args,
            speech_encoder,
            Qformer, 
            query_tokens,
            tokenizer,
            config,
            codebook_size
        )

    # @classmethod
    # def build_speech_model_speechtoknenizer(cls, args, dictionary=None, embed_tokens=None):
    #     speech_encoder = SpeechTokenizer.load_from_checkpoint(
    #         args['speech_model_config_path'], args['speech_model_file_path']
    #     )
    #     if hasattr(speech_encoder, "decoder"):   # Pruning
    #         del speech_encoder.decoder

    #     with open(args['speech_model_config_path']) as config_file:
    #         config = json.load(config_file)

    #     if args['freeze_enc']:
    #         for name, param in speech_encoder.named_parameters():
    #             param.requires_grad = False
    #         speech_encoder.eval()

    #     return speech_encoder, config

    @classmethod
    def build_speech_model_hubertbase(cls, args, dictionary=None, embed_tokens=None):
        speech_encoder = AutoModel.from_pretrained("facebook/hubert-base-ls960")

        if args['freeze_enc']:
            for name, param in speech_encoder.named_parameters():
                param.requires_grad = False
            speech_encoder.eval()

        return speech_encoder, speech_encoder.config
    
    @classmethod
    def build_speech_model_hubertlarge(cls, args, dictionary=None, embed_tokens=None):
        speech_encoder = AutoModel.from_pretrained('facebook/hubert-large-ll60k', cache_dir=os.environ['CACHE'])

        if args['freeze_enc']:
            for name, param in speech_encoder.named_parameters():
                param.requires_grad = False
            speech_encoder.eval()

        return speech_encoder, speech_encoder.config
    
    @classmethod
    def build_qformer(cls, args):
        tokenizer = cls.init_tokenizer_bert()
        Qformer, query_tokens = cls.init_Qformer_bert(
            args['num_query_token'], args['codebook_size'], args['cross_attention_freq']
        )
        Qformer.resize_token_embeddings(len(tokenizer))

        return Qformer, query_tokens, tokenizer

    @staticmethod
    def add_args(parser):
        parser.add_argument("--cross-attention-freq", type=int)
        parser.add_argument("--num-query-token", type=int)
        parser.add_argument("--speech-model-config-path", type=str)
        parser.add_argument("--speech-model-file-path", type=str)
        parser.add_argument("--qformer-dim", type=str)
        parser.add_argument("--llama-model", type=str)
        parser.add_argument("--pretrained", type=str)
        parser.add_argument("--speech-encoder-model", type=str)
       

    def __init__(
        self,
        args,
        speech_encoder,
        Qformer, 
        query_tokens,
        codebook_size,
        freeze_encoder=True,
        prompt=""
    ):
        super().__init__()
        self.speech_encoder_model = args.speech_encoder_model
        self.tokenizer = self.init_tokenizer_bert()

        self.speech_encoder = speech_encoder
        self.ln_speech = LayerNorm(codebook_size)
        if freeze_encoder:
            for name, param in self.speech_encoder.named_parameters():
                param.requires_grad = False
            self.speech_encoder = self.speech_encoder.eval()
            self.speech_encoder.train = disabled_train
            logging.info("freeze speech encoder")

        self.Qformer, self.query_tokens = Qformer, query_tokens
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model, use_fast=False, token=os.environ['HF_TOKEN'])
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.llm_model = AutoModel.from_pretrained(args.llm_model, cache_dir=os.environ['CACHE'], token=os.environ['HF_TOKEN'])
        
        
        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False

        self.eos_token_id = self.llm_tokenizer("\n", add_special_tokens=False).input_ids[0]
        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        # self.speechtokenizer_padding_idx = 931

        checkpoint = torch.load(args.pretrained, map_location="cpu")
        state_dict = checkpoint["model"]

        self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % args.pretrained)



        self.prompts = {
            "transcription": [
                "<Speech><SpeechHere></Speech> Can you transcribe the speech into a written format?",
                "<Speech><SpeechHere></Speech> Listen to the speech and write down its content.",
                "<Speech><SpeechHere></Speech> What is the content of the speech you heard?",
                "<Speech><SpeechHere></Speech> Please write down the transcription of the speech.",
                "<Speech><SpeechHere></Speech> Please transcribe the speech into a written format.",
                "<Speech><SpeechHere></Speech> Write down the content of the speech you heard.",
                "<Speech><SpeechHere></Speech> Can you write down the transcription of the speech?",
                # "<Speech><SpeechHere></Speech> Put the speech into a written format.",
                # "<Speech><SpeechHere></Speech> Please help me to transcribe the speech into a written format.",
                # "<Speech><SpeechHere></Speech> Recognize the content of the speech you heard.",
                # "<Speech><SpeechHere></Speech> Can you recognize what you heard in the speech?",
                # "<Speech><SpeechHere></Speech> Recognize the speech and write it down in a written format.",
                # "<Speech><SpeechHere></Speech> Listen to the speech and recognize its content.",
                # "<Speech><SpeechHere></Speech> Give me the transcription of the speech you heard.",
                # "<Speech><SpeechHere></Speech> Recognize the speech and give me the transcription.",
            ],
            "translation": [
                "<Speech><SpeechHere></Speech> Can you translate the speech into French?",
                "<Speech><SpeechHere></Speech> Please translate the speech you heard into French.",
                "<Speech><SpeechHere></Speech> Listen to the speech and translate it into French.",
                "<Speech><SpeechHere></Speech> Give me the French translation of this speech.",
                # "<Speech><SpeechHere></Speech> Could you please provide a French translation for the speech?",
                # "<Speech><SpeechHere></Speech> Would you be willing to translate the speech into French for me?",
                # "<Speech><SpeechHere></Speech> Would you be able to render the speech in French?",
                # "<Speech><SpeechHere></Speech> Could you assist me in translating the speech into French?",
                # "<Speech><SpeechHere></Speech> Can you help me convert the speech into French text?",
                # "<Speech><SpeechHere></Speech> Please convert the speech into French text.",
            ]
        }
            

    def prompt_wrap(self, embeds, atts, prompt, multi_prompt=False):
        if prompt:
            if multi_prompt:
                p_before = []
                p_after = []
                for i, p in enumerate(prompt):
                    b, a = p.split("<SpeechHere>")
                    p_before.append(b)
                    p_after.append(a)
                
                p_before_tokens = self.llm_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_before_embeds = self.llm_model.model.embed_tokens(p_before_tokens.input_ids)

                # speech_embeds wrapped with prompts_embeds are padded to the same length here
                p_after_tokens = self.llm_tokenizer(
                    p_after, return_tensors="pt", padding="longest", add_special_tokens=False
                ).to(embeds.device)
                p_after_embeds = self.llm_model.model.embed_tokens(p_after_tokens.input_ids)

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            else:
                batch_size = embeds.shape[0]
                p_before, p_after = prompt.split("<SpeechHere>")

                p_before_tokens = self.llm_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_after_tokens = self.llm_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_before_embeds = self.llm_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
                p_after_embeds = self.llm_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
                
                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            return wrapped_embeds, wrapped_atts
        else:
            return embeds, atts

    def forward(self, samples):
        speech = samples["source"]
        if self.speech_encoder_model == 'speechtokenizer':
            speech_embeds, codes = self.speech_encoder.forward_feature(speech.unsqueeze(1), [0])
            speech_embeds = self.ln_speech(speech_embeds[0].transpose(1, 2))

            speech_atts = codes[0] != self.speechtokenizer_padding_idx
        elif self.speech_encoder_model == 'hubertbase' or self.speech_encoder_model == 'hubertlarge':
            output = self.speech_encoder(speech, attention_mask=samples['padding_mask'])
            speech_embeds = output.last_hidden_state
            speech_embeds = self.ln_speech(speech_embeds)
            
            speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(
                speech.device
            )
        else:
            raise NotImplemented
        
        query_tokens = self.query_tokens.expand(speech_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )
        
        inputs_llama_query = self.llm_proj(query_output.last_hidden_state)
        atts_llama_query = torch.ones(inputs_llama_query.size()[:-1], dtype=torch.long).to(speech.device)

        self.llm_tokenizer.padding_side = "right"
        text = samples["target"][0]
        if self.prompts:
            if True:
                prompt = [random.choice(self.prompts[task]) for task in samples['task']]
            else:
                prompt = random.sample(self.prompts[task], inputs_llama_query.shape[0])
            inputs_llama, atts_llama = self.prompt_wrap(inputs_llama_query, atts_llama_query, prompt, multi_prompt=True)

        llama_tokens = self.llm_tokenizer(
            text, return_tensors="pt",padding="longest", add_special_tokens=False,truncation=True,
            max_length=128,
        ).to(speech.device)

        targets = llama_tokens.input_ids.masked_fill(
            llama_tokens.input_ids == self.llm_tokenizer.pad_token_id, -100
        )
        # empty_targets = (
        #     torch.ones(atts_llama.size(), dtype=torch.long).to(speech.device).fill_(-100)
        # )
        empty_targets = torch.ones([atts_llama.shape[0], atts_llama.shape[1] + 1], dtype=torch.long).to(speech.device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)
        batch_size = inputs_llama_query.shape[0]

        inputs_embeds = self.llm_model.model.embed_tokens(llama_tokens.input_ids)
        
        bos = torch.ones([batch_size, 1],dtype=llama_tokens.input_ids.dtype,device=llama_tokens.input_ids.device,) * self.llm_tokenizer.bos_token_id
        bos_embeds = self.llm_model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]
        
        inputs_embeds = torch.cat([bos_embeds, inputs_llama, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_llama, llama_tokens.attention_mask], dim=1)

        outputs = self.llm_model(inputs_embeds=inputs_embeds,attention_mask=attention_mask,return_dict=True,labels=targets,)
        loss = outputs.loss

        return {"loss": loss}
    
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=30,
        max_length=100,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        
        speech = samples["source"]
        if self.speech_encoder_model == 'speechtokenizer':
            speech_embeds, codes = self.speech_encoder.forward_feature(speech.unsqueeze(1), [0])
            speech_embeds = self.ln_speech(speech_embeds[0].transpose(1, 2))

            speech_atts = codes[0] != self.speechtokenizer_padding_idx
        elif self.speech_encoder_model == 'hubertbase' or self.speech_encoder_model == 'hubertlarge':
            output = self.speech_encoder(speech, attention_mask=samples['padding_mask'])
            speech_embeds = output.last_hidden_state
            speech_embeds = self.ln_speech(speech_embeds)
            
            speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(
                speech.device
            )
        else:
            raise NotImplemented

        

        query_tokens = self.query_tokens.expand(speech_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(query_embeds=query_tokens,encoder_hidden_states=speech_embeds,encoder_attention_mask=speech_atts,return_dict=True,)

        inputs = self.llm_proj(query_output.last_hidden_state)
        atts_llm = torch.ones(inputs.size()[:-1], dtype=torch.long).to(speech.device)

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        prompt = [prompt] * speech.size(0)
        
        gpt_tokens = self.llm_tokenizer(prompt,return_tensors="pt",).to(speech.device)
        attention_mask = torch.cat([atts_llm, gpt_tokens.attention_mask], dim=1)
        
        inputs_embeds = self.llm_model.get_input_embeddings()(gpt_tokens.input_ids.to(torch.int)).to(speech.device)
        inputs_embeds = torch.cat([inputs, inputs_embeds],dim=1)
        
        outputs = self.llm_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask,do_sample=use_nucleus_sampling,top_p=top_p,temperature=0.3,num_beams=num_beams,max_length=max_length,min_length=min_length,eos_token_id=self.eos_token_id,repetition_penalty=repetition_penalty,length_penalty=length_penalty,num_return_sequences=5,)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        output_text = [text.strip() for text in output_text]
        
        return output_text
        
        
@register_model_architecture(model_name="sparqle_finetune_llm_instruct", arch_name="sparqle_finetune_llm_instruct")
def base_architecture(args):
    pass