import re

from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import numpy as np
import copy
import inspect
import warnings
from dataclasses import dataclass

import torch
from torch import nn
import torch.distributed as dist

from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import logging
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import (
    GreedySearchEncoderDecoderOutput, 
    GreedySearchDecoderOnlyOutput, 
    BeamSearchEncoderDecoderOutput,
    BeamSearchDecoderOnlyOutput,
    )
from transformers.generation.logits_process import LogitsProcessorList
from tqdm import tqdm
import time
from data_utils.basic import read_txt_as_list, read_json
from eval_utils import read_results, read_num_and_li_results

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from llama2_ori_repo.llama.model import ModelArgs, Transformer
from llama2_ori_repo.llama.tokenizer import Tokenizer
from neox import init_neox, text_generation
from data_utils.basic import blockPrinting

Role = Literal["system", "user", "assistant"]

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.generation.streamers import BaseStreamer


logger = logging.get_logger(__name__)
class Evaler:
    def __init__(self, topk, tests, test_ans, 
                 eval_txt_path, args, 
                 model=None, tokenizer=None, patterns=None, 
                 early_stop_chars=None, obligations=[]):
        model_name = args.MODEL_NAME
        if not model or not tokenizer:
            if 'llama' in model_name:
                self.llama = 1
                llama2_directory = model_name.split('/models/')[0]
                tokenizer_path = os.path.join(llama2_directory, 'tokenizer.model')
                self.model, self.tokenizer = self.build(
                    ckpt_dir = model_name,
                    tokenizer_path=tokenizer_path,
                    max_seq_len=args.max_seq_len,
                    max_batch_size=args.max_batch_size,
                )
            else: 
                self.llama = 0
                self.model, self.tokenizer = init_neox(model_name)
        else:
            self.llama = 1
            self.model = model
            self.tokenizer = tokenizer
        self.patterns = patterns
        self.tests = tests
        self.test_ans = test_ans
        self.eval_txt_path = eval_txt_path
        self.topk = topk
        
        self.args = args

        self.obligations = obligations
        self.constraints = []
        self.zone_zero = early_stop_chars #in tensors. 0 is '\n', 29962 is ']'; self.tokenizer.encode(char) gives only a list

        self.first_check = 0
        self.top = 1
    def restrict_list_hard(self, tokens, prev_pos, min_prompt_len, input_text_mask, eos_reached, m=0):
        logits = self.model.forward(tokens[:, prev_pos:min_prompt_len], prev_pos)
        logits_last = logits[:, -1]
        # Get the index of the ten tokens of numbers
        top_10_indices = torch.topk(logits_last, k=logits.shape[-1], dim=-1).indices 
        values_to_extract = [29900, 29896, 29906, 29941, 29946, 29945, 29953, 29955, 29947, 29929] # 0-9 token
        top_10_indices_np = top_10_indices.cpu().numpy()
        mask = np.isin(top_10_indices_np, values_to_extract)
        extracted_elements = top_10_indices_np[mask][:10]
        # Convert back to Tensor type
        top_10_indices = torch.tensor(extracted_elements)

        # print('top 10 idx:', top_10_indices) # [:,0:10]
        # Get the token with the m-th highest probability
        next_token = top_10_indices[m]
        next_token = next_token.reshape(-1)
        # print('next token1: ', next_token)

        next_token = torch.where(
            input_text_mask[:, min_prompt_len], tokens[:, min_prompt_len], next_token
            )

        # In addition to getting the token, everything related to last_layer must be updated for the mode.forward of the next cycle
        tokens[:, min_prompt_len] = next_token
        eos_reached |= (~input_text_mask[:, min_prompt_len]) & (
                next_token == self.tokenizer.eos_id
            )
        
        self.first_check = 1 #skip first_check
        return next_token, eos_reached
  
    def first_checking(self, next_tokens, next_tokens_scores):
        this_peer_finished = False
        if self.first_check == 0: #first check
            if self.obligations and (next_tokens not in self.obligations):
                this_peer_finished = True
                #need to force regenerate/reset next tokens to avoid the constraints
                self.first_check = -1 #not begin with nums

            if self.constraints and (next_tokens in self.constraints):
                self.top += 1
                #force regenerate/reset next tokens to avoid the constraints
                next_tokens = torch.argsort(next_tokens_scores, dim=-1, descending=True)[:, self.top-1]
                self.constraints.append(next_tokens)
                self.first_check = -1 #breach of obligs
            else:
                self.constraints.append(next_tokens)
                self.first_check = 1 #check sign passed
        return this_peer_finished, next_tokens
      
    def gen_set_ans(self, tests='', dir_full_test='', dir_time2id=''):
        '''add non-duplicate answer for duplicate queries (no duplicates in sets); 
            not require order a-z within one timestamp anymore
                (to be used in Gdelt & Yago), but may need more time and space'''
        if tests == '':
            tests = self.tests
        dict_qu_ans = {}
        if dir_full_test == '':
            full_test_ans = self.test_ans #dense and time-well-divided dataset; icews14
            for i in tqdm(range(0, len(tests)-1)):
                query = tests[i].split('\n')[-1]
                if query == '':
                    break
                if dict_qu_ans.get(query) == None:
                    dict_qu_ans[query] = set()
                dict_qu_ans[query].add(full_test_ans[i]) #add answers to the set
                time.sleep(0.001)
        else:
            dict_t2id = {}
            if dir_time2id != '':
                dict_t2id = read_json(dir_time2id)
            else:
                print("Attention: icews18 needs its ts2id file to convert time into time_id")
            fulltest = read_txt_as_list(dir_full_test) #only load essentially 
            li_queries = [test.split('\n')[-1] for test in tests]
            #build sets
            for i in range(0, len(li_queries)-1):
                query = li_queries[i]
                if query == '':
                    break
                if dict_qu_ans.get(query) is None:
                    dict_qu_ans[query] = set()
            end_time = li_queries[-3].split(':')[0]
            for line in fulltest:
                quadruple = line.strip().split('\t')
                time_quadruple = dict_t2id[quadruple[3]] if dir_time2id != '' else quadruple[3]
                if int(time_quadruple) > int(end_time):
                    break
                built_query = f"{time_quadruple}: [{quadruple[0]}, {quadruple[1]},"
                if dict_qu_ans.get(built_query) is not None:
                    dict_qu_ans[built_query].add(quadruple[2]) #add answers to the set
            print("duplicate answers checked")
        return dict_qu_ans
    
    def generate_extra_answers(self, m_inloop, k_inloop):
        if self.args.ft == 1:
            raw_answers, answer_regs = self.model_calling(m_inloop) #call for more generated ans
        elif self.llama == 1: #icl llama2
            answer_regs = self.text_completion(m_inloop,  
                                str(self.args.PROMPT),
                                max_gen_len=self.args.max_gen_len,
                                temperature=self.args.TEMPERATURE,
                                #top_p=top_p,
                            )
            answer_regs = [answer_reg['generation'] for answer_reg in answer_regs]
            raw_answers = answer_regs
        else: #icl gpt neox
            raw_answers = text_generation(m_inloop, k_inloop, self.model, self.tokenizer, 
                                        str(self.args.PROMPT), 
                                        #icews14 28, icews18 34, ecola 18, GDELT 16, YAGO 25. 
                                        max_seq_len=34,
                                        verbose=False)
            pattern = re.compile(r'\s*(\d+)\.(.*?)\]')
            answer_regs = re.match(pattern, raw_answers).group(2).strip() \
                if re.match(pattern, raw_answers) else raw_answers
            answer_regs = [answer_regs] 
        return raw_answers, answer_regs

    def build(
        self, 
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
    ): 
        os.environ["RANK"] = "0" # Set for torch.distributed.init_process_group
        
        #os.environ["WORLD_SIZE"] = "4" #  
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(1)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return model, tokenizer #Llama(model, tokenizer)
    
    def bs_generate(
        self, m,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        torch.no_grad() #replace @torch.inference_mode()

        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )
        if self.top <= 10 and m<10:
            next_token, eos_reached = self.restrict_list_hard(tokens, 
                                                              prev_pos, min_prompt_len, 
                                                              input_text_mask, eos_reached, m)

        prev_pos = min_prompt_len
        torch.set_printoptions(profile="full")
        tokens = torch.where(tokens == -1, torch.tensor(0), tokens) 
        

        for cur_pos in range(min_prompt_len+1, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )

            top_sign = self.top-1 if self.first_check == 0 else 0 #first check, or to generate the rest
            next_token = torch.argsort(logits[:, -1], dim=-1, descending=True)[:, top_sign]

            this_peer_finished, next_token = self.first_checking(next_token, logits[:, -1])

            if next_token in self.zone_zero:
                this_peer_finished = True
            ## modification ends 
            next_token = next_token.reshape(-1)

            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            if all(eos_reached) or this_peer_finished: # added this_peer_finished
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)
    
    def text_completion(
        self, m, 
        prompts: List[str],
        temperature: float = 0,
        top_p: float = 0.1,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ): #-> List[CompletionPrediction]:
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(prompts, bos=True, eos=False)]
        '''
        for x in prompts:
            print(x)
            prompt_tokens.append(self.tokenizer.encode(x, bos=False, eos=False))'''
        generation_tokens, generation_logprobs = self.bs_generate(m, 
                                                                prompt_tokens,
                                                                max_gen_len,
                                                                temperature,
                                                                top_p,
                                                                logprobs,
                                                                echo,
                                                            )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]


    def my_generate_top10(self, model_instance, m, gen_length, **kwargs):
        base_model = model_instance.base_model
        
        # original prepare_inputs_for_generation and generation_config
        original_prepare_inputs_for_generation = base_model.prepare_inputs_for_generation
        original_generation_config = getattr(base_model, "generation_config", None)
        
        # prepare_inputs_for_generation and generation_config
        base_model.prepare_inputs_for_generation = model_instance.prepare_inputs_for_generation
        if hasattr(base_model, "model"):
            base_model.model.generation_config = model_instance.generation_config
        else:
            base_model.generation_config = model_instance.generation_config
        
        try:
            # base_model generate_top10
            outputs = self.my_utils_generate_top10(base_model, m, gen_length, **kwargs)
        except Exception as e:
            # prepare_inputs_for_generation
            base_model.prepare_inputs_for_generation = original_prepare_inputs_for_generation
            # recover generation_config
            if original_generation_config is not None:
                base_model.generation_config = original_generation_config
            raise e
        else:
            # recover prepare_inputs_for_generation
            base_model.prepare_inputs_for_generation = original_prepare_inputs_for_generation
            # recover generation_config
            if original_generation_config is not None:
                base_model.generation_config = original_generation_config
            return outputs

    # adopted from "generate" in /transformers/generation/utils.py
    @torch.no_grad()
    def my_utils_generate_top10(self, 
        model_instance,  m, 
        gen_length,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        # max_length=max_length,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ): #-> Union[GenerateOutput, torch.LongTensor]:
        
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        model_instance._validate_model_class()

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation -- update the generation config
            # model attribute accordingly, if it was created from the model config
            if model_instance.generation_config._from_model_config:
                new_generation_config = GenerationConfig.from_model_config(model_instance.config)
                if new_generation_config != model_instance.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use a generation configuration file (see"
                        " https://huggingface.co/docs/transformers/main_classes/text_generation)"
                    )
                    model_instance.generation_config = new_generation_config
            generation_config = model_instance.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        model_instance._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = model_instance._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        model_kwargs["use_cache"] = generation_config.use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(model_instance.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = model_instance._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        # decoder-only models should use left-padding for generation
        if not model_instance.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config.pad_token_id is not None
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        if model_instance.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = model_instance._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if model_instance.config.is_encoder_decoder:
            input_ids, model_kwargs = model_instance._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            if not has_default_max_length:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length

        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(
                f"Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than"
                f" the maximum length ({generation_config.max_length})"
            )
        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if model_instance.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 7. determine generation mode
        is_constraint_gen_mode = (
            generation_config.constraints is not None or generation_config.force_words_ids is not None
        )

        is_contrastive_search_gen_mode = (
            (generation_config.num_beams == 1)
            and generation_config.top_k is not None
            and generation_config.top_k > 1
            and generation_config.do_sample is False
            and generation_config.penalty_alpha is not None
            and generation_config.penalty_alpha > 0
        )

        is_greedy_gen_mode = (
            (generation_config.num_beams == 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is False
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_sample_gen_mode = (
            (generation_config.num_beams == 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is True
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_beam_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is False
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_beam_sample_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is True
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_group_beam_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups > 1)
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_assisted_gen_mode = False
        if assistant_model is not None:
            if not (is_greedy_gen_mode or is_sample_gen_mode):
                raise ValueError(
                    "You've set `assistant_model`, which triggers assisted generate. Currently, assisted generate "
                    "is only supported with Greedy Search and Sample."
                )
            is_assisted_gen_mode = True

        if generation_config.num_beam_groups > generation_config.num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and generation_config.do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if model_instance.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {model_instance.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{model_instance.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        logits_processor = model_instance._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        # 9. prepare stopping criteria
        stopping_criteria = model_instance._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        # 10. go into different generation modes
        if is_assisted_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing assisted generate, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if batch_size > 1:
                raise ValueError("assisted generate is only supported for batch_size = 1")
            if not model_kwargs["use_cache"]:
                raise ValueError("assisted generate requires `use_cache=True`")

            # 11. If the assistant model is an encoder-decoder, prepare its encoder outputs
            if assistant_model.config.is_encoder_decoder:
                assistant_model_kwargs = copy.deepcopy(model_kwargs)
                inputs_tensor, model_input_name, assistant_model_kwargs = assistant_model._prepare_model_inputs(
                    inputs_tensor, assistant_model.generation_config.bos_token_id, assistant_model_kwargs
                )
                assistant_model_kwargs = assistant_model._prepare_encoder_decoder_kwargs_for_generation(
                    inputs_tensor, assistant_model_kwargs, model_input_name
                )
                model_kwargs["assistant_encoder_outputs"] = assistant_model_kwargs["encoder_outputs"]

            # 12. run assisted generate
            return model_instance.assisted_decoding(
                input_ids,
                assistant_model=assistant_model,
                do_sample=generation_config.do_sample,
                logits_processor=logits_processor,
                logits_warper=model_instance._get_logits_warper(generation_config) if generation_config.do_sample else None,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        if is_greedy_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing greedy search, "
                    f"but is {generation_config.num_return_sequences}."
                )
            # 11. run greedy search
            return self.my_utils_greedy_search_top10(model_instance,  #my_utils_greedy_search_top10
                m,              #check m
                gen_length, 
                input_ids, 
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                # max_length=20, 
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif is_contrastive_search_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing contrastive search, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if not model_kwargs["use_cache"]:
                raise ValueError("Contrastive search requires `use_cache=True`")

            return model_instance.contrastive_search(
                input_ids,
                top_k=generation_config.top_k,
                penalty_alpha=generation_config.penalty_alpha,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # 11. prepare logits warper
            logits_warper = model_instance._get_logits_warper(generation_config)

            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = model_instance._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=model_instance.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run sample
            return model_instance.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = model_instance._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=model_instance.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return model_instance.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_beam_sample_gen_mode:
            # 11. prepare logits warper
            logits_warper = model_instance._get_logits_warper(generation_config)

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")
            # 12. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size * generation_config.num_return_sequences,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                max_length=generation_config.max_length,
            )

            # 13. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = model_instance._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams * generation_config.num_return_sequences,
                is_encoder_decoder=model_instance.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 14. run beam sample
            return model_instance.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_group_beam_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if generation_config.num_beams % generation_config.num_beam_groups != 0:
                raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            has_default_typical_p = kwargs.get("typical_p") is None and generation_config.typical_p == 1.0
            if not has_default_typical_p:
                raise ValueError("Decoder argument `typical_p` is not supported with beam groups.")

            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                num_beam_groups=generation_config.num_beam_groups,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = model_instance._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=model_instance.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return model_instance.group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_constraint_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            if generation_config.num_beams <= 1:
                raise ValueError("`num_beams` needs to be greater than 1 for constrained generation.")

            if generation_config.do_sample:
                raise ValueError("`do_sample` needs to be false for constrained generation.")

            if generation_config.num_beam_groups is not None and generation_config.num_beam_groups > 1:
                raise ValueError("`num_beam_groups` not supported yet for constrained generation.")

            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints

            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]`"
                        f"of positive integers, but is {generation_config.force_words_ids}."
                    )

                if (
                    not isinstance(generation_config.force_words_ids, list)
                    or len(generation_config.force_words_ids) == 0
                ):
                    typeerror()

                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                            for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 11. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = model_instance._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=model_instance.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return model_instance.constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

    # adopted from "greedy_search" in /transformers/generation/utils.py 
    def my_utils_greedy_search_top10(self, 
        model_instance,
        m_inloop, 
        gen_length,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ): #-> Union[GreedySearchOutput, torch.LongTensor]:
        r"""
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```"""
        # init values
        stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=gen_length+input_ids.shape[1])])
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else model_instance.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else model_instance.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else model_instance.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else model_instance.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else model_instance.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else model_instance.generation_config.return_dict_in_generate
        )


        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and model_instance.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only



        # prepare model initial inputs
        model_inputs = model_instance.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = model_instance(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if model_instance.config.is_encoder_decoder else (outputs.attentions,)
                )
                if model_instance.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if model_instance.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        next_tokens, _ = self.require_first_to_be(next_tokens_scores)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = model_instance._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model_instance.config.is_encoder_decoder
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )
            

    
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break
            
            # prepare model inputs
            model_inputs = model_instance.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = model_instance(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )


            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if model_instance.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if model_instance.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if model_instance.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            top_sign = self.top-1 if self.first_check == 0 else 0 #first check, or to generate the rest
            next_tokens = torch.argsort(next_tokens_scores, dim=-1, descending=True)[:, top_sign]

            this_peer_finished, next_tokens = self.first_checking(next_tokens, next_tokens_scores)
            
            if next_tokens in self.zone_zero:
                this_peer_finished = True

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = model_instance._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=model_instance.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if model_instance.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids
        
    def require_first_to_be(self, next_tokens_scores, values_to_extract=[29871]):
        top_k_indices = torch.topk(next_tokens_scores, k=next_tokens_scores.shape[-1], dim=-1).indices
        top_k_indices_np = top_k_indices.cpu().numpy()
        mask = np.isin(top_k_indices_np, values_to_extract)
        top_k_indices = top_k_indices_np[mask][0]
        top_k_indices = torch.tensor(top_k_indices)

        next_tokens = top_k_indices.item()
        next_tokens = torch.tensor(next_tokens).reshape(-1)
        current_device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        return next_tokens.to(current_device), top_k_indices.to(current_device)
    
    def my_utils_greedy_search_top10_recursive(self, 
        model_instance,
        m_inloop, 
        gen_length,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ):
        # init values
        print("another day, another destiny")
        stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=gen_length+input_ids.shape[1])])
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else model_instance.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else model_instance.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else model_instance.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else model_instance.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else model_instance.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else model_instance.generation_config.return_dict_in_generate
        )

        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
        dict_parameter_return = {"scores":scores, 
                                 "decoder_attentions":decoder_attentions, 
                                 "cross_attentions":cross_attentions, 
                                 "decoder_hidden_states":decoder_hidden_states} 

        if return_dict_in_generate and model_instance.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        this_peer_finished = False
        #end initialization 

        model_inputs = model_instance.prepare_inputs_for_generation(input_ids, **model_kwargs)
        #most time consuming part:
        outputs = model_instance(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        next_token_logits = outputs.logits[:, -1, :]

        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if model_instance.config.is_encoder_decoder else (outputs.attentions,)
                )
                if model_instance.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if model_instance.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        next_tokens, _ = self.require_first_to_be(next_tokens_scores)

        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = model_instance._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model_instance.config.is_encoder_decoder
        )

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )
        #Time consume: 6-7s

        def _recursive_greedy_search_top10(input_ids, 
                                           model_kwargs, 
                                           unfinished_sequences, 
                                           this_peer_finished, 
                                           eos_token_id_tensor, 
                                           dict_parameter_return):
            def _get_outputs():
                model_inputs = model_instance.prepare_inputs_for_generation(input_ids, **model_kwargs)
                outputs = model_instance(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                return outputs
            
            #Base
            if synced_gpus:
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                if this_peer_finished_flag.item() == 0.0:
                    return input_ids
            if this_peer_finished and not synced_gpus:
                return input_ids
            
            outputs = _get_outputs()
            next_token_logits = outputs.logits[:, -1, :]
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            if return_dict_in_generate:
                if output_scores:
                    dict_parameter_return["scores"] += (next_tokens_scores,)
                if output_attentions:
                    dict_parameter_return["decoder_attentions"] += (
                        (outputs.decoder_attentions,) if model_instance.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if model_instance.config.is_encoder_decoder:
                        dict_parameter_return["cross_attentions"] += (outputs.cross_attentions,)

                if output_hidden_states:
                    dict_parameter_return["decoder_hidden_states"] += (
                        (outputs.decoder_hidden_states,)
                        if model_instance.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            top_sign = self.top-1 if self.first_check == 0 else 0
            next_tokens = torch.argsort(next_tokens_scores, dim=-1, descending=True)[:, top_sign]

            this_peer_finished, next_tokens = self.first_checking(next_tokens, next_tokens_scores)

            if next_tokens in self.zone_zero:
                this_peer_finished = True
                
            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = model_instance._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=model_instance.config.is_encoder_decoder
            )

            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            if stopping_criteria(input_ids, dict_parameter_return["scores"]):
                this_peer_finished = True
            #recursion
            return _recursive_greedy_search_top10(input_ids, 
                                           model_kwargs, 
                                           unfinished_sequences, 
                                           this_peer_finished, 
                                           eos_token_id_tensor, 
                                           dict_parameter_return)
        
        #main cur begins
        input_ids = _recursive_greedy_search_top10(input_ids, model_kwargs, unfinished_sequences,
                                                   this_peer_finished, eos_token_id_tensor,
                                                   dict_parameter_return)

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if model_instance.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids
    
    def model_calling(self, m_inloop):
        ids = self.tokenizer.encode(self.args.PROMPT)
        input_ids = torch.LongTensor([ids]).to('cuda')
        self.first_check = 0 #set check sign = 0 first
        out = self.my_generate_top10(model_instance=self.model, m=m_inloop, #m_inloop useless here now
                                input_ids=input_ids,
                                max_length=self.args.CONTEXT_LEN,
                                gen_length=36,
                                do_sample=False, #with temperature unset
                                #temperature=self.args.TEMPERATURE
                                )
        out_text = self.tokenizer.decode(out[0])
        answer = out_text.replace(self.args.PROMPT, "").replace("\nEND", "").strip()
        answer = answer.replace("\n", "")

        answer_regs = [re.match(pattern, answer).group(1) if re.match(pattern, answer) else answer \
                     for pattern in self.patterns]
        return answer, answer_regs
    
    #@blockPrinting
    def eval(self, c, cnt=0, path_results=None, filter_yes=True):
        query = ""
        c1 = c["c1"]
        c3 = c["c3"]
        c10 = c["c10"]
            
        if path_results is not None: #for processing result files
            test_results = read_results(path_results)
        #preprocess multiple answers
        dict_qu_ans = self.gen_set_ans(dir_full_test=self.args.fulltest, dir_time2id=self.args.time2id)
        set_checked_qu = set()
        num_infer = len(self.tests) #
        for i in tqdm(range(cnt, num_infer)):  # cnt, len(self.tests[:-1])
            his_query = self.tests[i]
            query = his_query.split('\n')[-1]
            #truncated history
            val_trunc = -1
            if len(his_query)-1 > val_trunc and val_trunc != -1: 
                li_his_trunc = his_query.split('\n')[-val_trunc-1:-1] #backward
                li_his_trunc.append(query)
                his_query = "\n".join(li_his_trunc)

            delete = False
            if delete == True: 
                his_query = re.sub(r'\d+:\s', '', his_query)

            ins = '''<s>[INST] <<SYS>> \
            You must be able to correctly predict the next {object_label} from \
            a given text consisting of multiple quadruplets in the form of "{time}:[{subject}, {relation}, {object_label}.{object}]" \
            and the query in the form of "{time}:[{subject}, {relation}," in the end.\n\
            You must generate {object_label}.{object}\n\n<</SYS>>'''
            self.args.PROMPT = ins + his_query + '[/INST]' if self.args.instruct_yes else his_query

            if query not in set_checked_qu:
                set_checked_qu.add(query)
                hello = "For"
                
            else:
                hello = "Duplicate query:"
            print(hello, query)
            if query == '': #probably the end 
                continue
            print("Given answers", dict_qu_ans[query], "with", self.test_ans[i], "as the gt")

            content_to_write = []
            content_to_write2 = []
            m_inloop = -1
            filter_m_count = -1
            k_inloop = self.topk  # k
            self.constraints = []
            self.top = 1 #reset top
            exist_num = 0   
            if path_results is not None: #for processing result files
                num_Test, li_results = read_num_and_li_results(test_results[i])
                exist_num = len(li_results)
                if int(num_Test) != i:
                    print(num_Test, i)
                    raise ValueError("Test id and i do not match.")
            while m_inloop < k_inloop-1:  # Use while to allow changing "m"
                m_inloop += 1
                filter_m_count += 1
                with torch.no_grad(): #loops for one history_query
                    if path_results is None: #or self.args.ft==1
                        raw_ans, answer_regs = self.model_calling(m_inloop) 
                        print(str(m_inloop) + "-th time, I would say, ", answer_regs)
                    else:
                        
                        if m_inloop >= exist_num:
                            if not filter_yes:
                                break
                            else:
                                print("call of duty")
                                raw_ans, answer_regs = self.generate_extra_answers(m_inloop, k_inloop)
                                print(str(m_inloop) + "-th time, I would say, ", answer_regs)
                        else:
                            #existing results
                            raw_ans = answer_regs = [li_results[m_inloop]] 
                            pattern = re.compile(r'.*?[\d:@][._](.*)\]') #'\s*(\d+)\.(.*?)\]')
                            answer_regs = [re.match(pattern, answer_regs[0]).group(2).strip()] \
                                if re.match(pattern, answer_regs[0]) else answer_regs
                            print(str(m_inloop) + " read ", answer_regs)
                            self.top += 1

                    content_to_write.append('\n' + str(answer_regs))
                    content_to_write2.append('\n' + str(raw_ans))

                    #check multiple regex 
                    bingo = False
                    dict_qu_ans_lower = [ans.lower() for ans in dict_qu_ans[query]]
                    for answer in answer_regs:
                        answerlow = answer.lower()
                        gtlow = self.test_ans[i].lower()
                        if answer == '':
                            content_to_write.append("(none string; removed)")
                            k_inloop += 1
                            filter_m_count -= 1
                            print("increased k: " + str(k_inloop))
                            break
                        if (answerlow != gtlow and answerlow in dict_qu_ans_lower) and filter_yes: #first_check = -1 if to check breach of obligation
                            print("Got another answer: " + answer + ", ignored.")
                            content_to_write.append("(ignored gt)")
                            k_inloop += 1
                            filter_m_count -= 1
                            print("increased k: " + str(k_inloop))
                            break
                        elif answerlow == gtlow:
                            bingo = True
                            if filter_m_count == 0:
                                c1 += 1
                                c3 += 1
                                c10 += 1
                            elif 0 < filter_m_count < 3:
                                c3 += 1
                                c10 += 1
                            elif 3 <= filter_m_count < 10:
                                c10 += 1
                            print("Bingo! Line: ", i, "count after filtering: ", filter_m_count + 1, "all count: ", \
                            m_inloop + 1, "answer: ", answer, "gt: ", self.test_ans[i])
                            break
                    if bingo:
                        break

            hits_1 = c1 / (i + 1)
            hits_3 = c3 / (i + 1)
            hits_10 = c10 / (i + 1)
            '''
            print("hit1=", c1, "/", str(i+1), "=", hits_1)
            print("hit3=", c3, "/", str(i+1), "=", hits_3)
            print("hit10=", c10, "/", str(i+1), "=", hits_10)'''

            with open(self.eval_txt_path, "a", encoding="utf-8") as fout:
                if self.args.ft == 1:
                    fout.write('current model: ' + self.args.LORA_CHECKPOINT_DIR + ', \n')
                else:
                    fout.write('current model: ' + self.args.MODEL_NAME + ', \n')
                fout.write(self.args.output_file + ' currently finished: ' + str(i + 1) + '; results: \n')
                fout.write("Hits@1: " + str(round(hits_1, 3)) + "\n")
                fout.write("Hits@3: " + str(round(hits_3, 3)) + "\n")
                fout.write("Hits@10: " + str(round(hits_10, 3)) + "\n")
                fout.write(str(c1) + "\n")
                fout.write(str(c3) + "\n")
                fout.write(str(c10) + "\n\n")

            with open(self.args.output_file, 'a', encoding='utf-8') as f:
                f.write('{"Test'+str(i)+'": ["' + ', '.join(content_to_write) + '"]}, \n\n')
            with open(self.args.output_file.replace(".txt", "_raw.txt"), 'a', encoding='utf-8') as f:
                f.write('{"Test'+str(i)+'": ["' + ', '.join(content_to_write2) + '"]}, \n\n')

            print('processing: ' + self.args.output_file, i + 1)
            time.sleep(0.001)