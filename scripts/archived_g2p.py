# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Cisco Systems, Inc. and others.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# This model is for Fairseq v1.0, which doesnt have a binary distribution yet

import re
import ast
import time
from argparse import Namespace
from collections import namedtuple

import numpy as np
import torch
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output


Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints")

class G2PPytorch:

    def __init__(self, beam_size=1):
        parser = options.get_interactive_generation_parser()

        if beam_size == 1:
            input_args = ['data/infer_data/', '--path', 'models/checkpoint_best.pt',
                          '--beam', '1', '--remove-bpe', '--sampling', '-s',
                          'word', '-t', 'phon']
        else:
            input_args = ['data/infer_data/', '--path', 'models/checkpoint_best.pt',
                          '--beam', str(beam_size), '--remove-bpe', '-s', 'word', '-t', 'phon']

        args = options.parse_args_and_arch(parser, input_args=input_args)
        self.cfg = args
        if isinstance(self.cfg, Namespace):
            self.cfg = convert_namespace_to_omegaconf(self.cfg)

        utils.import_user_module(self.cfg.common)

        if self.cfg.interactive.buffer_size < 1:
            self.cfg.interactive.buffer_size = 1
        if self.cfg.dataset.max_tokens is None and self.cfg.dataset.batch_size is None:
            self.cfg.dataset.batch_size = 1

        assert (
                not self.cfg.generation.sampling or self.cfg.generation.nbest == self.cfg.generation.beam
        ), "--sampling requires --nbest to be equal to --beam"
        assert (
                not self.cfg.dataset.batch_size
                or self.cfg.dataset.batch_size <= self.cfg.interactive.buffer_size
        ), "--batch-size cannot be larger than --buffer-size"

        # Fix seed for stochastic decoding
        if self.cfg.common.seed is not None and not self.cfg.generation.no_seed_provided:
            np.random.seed(self.cfg.common.seed)
            utils.set_torch_seed(self.cfg.common.seed)

        self.use_cuda = torch.cuda.is_available() and not self.cfg.common.cpu

        # Setup task, e.g., translation
        self.task = tasks.setup_task(self.cfg.task)

        # Set dictionaries
        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary

        # Handle tokenization and BPE
        self.tokenizer = self.task.build_tokenizer(self.cfg.tokenizer)
        self.bpe = self.task.build_bpe(self.cfg.bpe)

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(self.cfg.generation.replace_unk)

        self.models = None
        self.generator = None
        self.max_positions = None

    def encode_fn(self, x):
        if self.tokenizer is not None:
            x = self.tokenizer.encode(x)
        if self.bpe is not None:
            x = self.bpe.encode(x)
        return x

    def decode_fn(self, x):
        if self.bpe is not None:
            x = self.bpe.decode(x)
        if self.tokenizer is not None:
            x = self.tokenizer.decode(x)
        return x

    @staticmethod
    def make_batches(lines, cfg, task, max_positions, encode_fn):
        def encode_fn_target(x):
            return encode_fn(x)

        if cfg.generation.constraints:
            # Strip (tab-delimited) contraints, if present, from input lines,
            # store them in batch_constraints
            batch_constraints = [list() for _ in lines]
            for i, line in enumerate(lines):
                if "\t" in line:
                    lines[i], *batch_constraints[i] = line.split("\t")

            # Convert each List[str] to List[Tensor]
            for i, constraint_list in enumerate(batch_constraints):
                batch_constraints[i] = [
                    task.target_dictionary.encode_line(
                        encode_fn_target(constraint),
                        append_eos=False,
                        add_if_not_exist=False,
                    )
                    for constraint in constraint_list
                ]

        if cfg.generation.constraints:
            constraints_tensor = pack_constraints(batch_constraints)
        else:
            constraints_tensor = None

        tokens, lengths = task.get_interactive_tokens_and_lengths(lines, encode_fn)

        itr = task.get_batch_iterator(
            dataset=task.build_dataset_for_inference(
                tokens, lengths, constraints=constraints_tensor
            ),
            max_tokens=cfg.dataset.max_tokens,
            max_sentences=cfg.dataset.batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        ).next_epoch_itr(shuffle=False)
        for batch in itr:
            ids = batch["id"]
            src_tokens = batch["net_input"]["src_tokens"]
            src_lengths = batch["net_input"]["src_lengths"]
            constraints = batch.get("constraints", None)

            yield Batch(
                ids=ids,
                src_tokens=src_tokens,
                src_lengths=src_lengths,
                constraints=constraints,
            )

    def load_g2p_model(self):
        # Load ensemble
        overrides = ast.literal_eval(self.cfg.common_eval.model_overrides)
        self.models, _model_args = checkpoint_utils.load_model_ensemble(
            utils.split_paths(self.cfg.common_eval.path),
            arg_overrides=overrides,
            task=self.task,
            suffix=self.cfg.checkpoint.checkpoint_suffix,
            strict=(self.cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=self.cfg.checkpoint.checkpoint_shard_count,
        )

        # Optimize ensemble for generation
        for model in self.models:
            if model is None:
                continue
            if self.cfg.common.fp16:
                model.half()
            if self.use_cuda and not self.cfg.distributed_training.pipeline_model_parallel:
                model.cuda()
            model.prepare_for_inference_(self.cfg)

        # Initialize generator
        self.generator = self.task.build_generator(self.models, self.cfg.generation)

        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(), *[model.max_positions() for model in self.models]
        )

    @staticmethod
    def _remove_stress(word):
        letter_pattern = r'[a-zA-Z]+'
        return ' '.join([re.match(letter_pattern, i).group(0) for i in word.split()])

    def decode_word(self, word: str, with_stress=False) -> str:
        if len(word.split()) > 1:
            raise Exception(f"Input {word} contains multiple words. This function expects one word")

        inputs = [' '.join([char for char in word.lower()])]
        results = []
        for batch in self.make_batches(inputs, self.cfg, self.task, self.max_positions, self.encode_fn):
            bsz = batch.src_tokens.size(0)
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            constraints = batch.constraints
            if self.use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
                if constraints is not None:
                    constraints = constraints.cuda()

            sample = {
                "net_input": {
                    "src_tokens": src_tokens,
                    "src_lengths": src_lengths,
                },
            }
            translate_start_time = time.time()
            translations = self.task.inference_step(
                self.generator, self.models, sample, constraints=constraints
            )
            translate_time = time.time() - translate_start_time
            list_constraints = [[] for _ in range(bsz)]
            if self.cfg.generation.constraints:
                list_constraints = [unpack_constraints(c) for c in constraints]
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.tgt_dict.pad())
                constraints = list_constraints[i]
                results.append(
                    (
                        id,
                        src_tokens_i,
                        hypos,
                        {
                            "constraints": constraints,
                            "time": translate_time / len(translations),
                        },
                    )
                )

        # sort output to match input order
        for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
            src_str = ''
            if self.src_dict is not None:
                src_str = self.src_dict.string(src_tokens, self.cfg.common_eval.post_process)

            # Process top predictions
            for hypo in hypos[: min(len(hypos), self.cfg.generation.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=self.align_dict,
                    tgt_dict=self.tgt_dict,
                    remove_bpe=self.cfg.common_eval.post_process,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(self.generator),
                )
                detok_hypo_str = self.decode_fn(hypo_str)
                if not with_stress:
                    return self._remove_stress(detok_hypo_str)
                return detok_hypo_str

