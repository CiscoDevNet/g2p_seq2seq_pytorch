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

from argparse import Namespace
from collections import namedtuple
import os.path
import re
from typing import Generator, List, Optional, Tuple

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import encoders
from fairseq.models.transformer import TransformerModel
from fairseq.tasks.translation import TranslationTask
from fairseq.token_generation_constraints import pack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output
import numpy as np
import torch

INFER_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/infer_data/')
USER_DIR = os.path.join(os.path.dirname(__file__), 'transformer')


class G2PPytorch:

    def __init__(self, beam_size: Optional[int] = 5) -> None:
        parser = options.get_interactive_generation_parser()
        input_args = [INFER_DATA_DIR, '--user-dir', USER_DIR,
                      '--beam', str(beam_size), '--remove-bpe',
                      '-s', 'word', '-t', 'phon']
        self.cfg = options.parse_args_and_arch(parser, input_args=input_args)
        self.batch = namedtuple("Batch", "ids src_tokens src_lengths constraints")  # type: ignore

        utils.import_user_module(self.cfg)
        self.cfg.buffer_size = max(self.cfg.buffer_size, 1)
        if self.cfg.max_tokens is None and self.cfg.batch_size is None:
            self.cfg.batch_size = 1

        # Fix seed for stochastic decoding
        if self.cfg.seed is not None and not self.cfg.no_seed_provided:
            np.random.seed(self.cfg.seed)
            utils.set_torch_seed(self.cfg.seed)

        self.use_cuda = torch.cuda.is_available() and not self.cfg.cpu

        # Setup task, e.g., translation
        self.task = tasks.setup_task(self.cfg)

        # Set dictionaries
        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary

        # Handle tokenization and BPE
        self.tokenizer = encoders.build_tokenizer(self.cfg)
        self.bpe = encoders.build_bpe(self.cfg)

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(self.cfg.replace_unk)

        self.models: List[TransformerModel] = []
        self.generator = None
        self.max_positions = (0, 0)

    def encode_fn(self, x: str) -> str:
        if self.tokenizer is not None:
            x = self.tokenizer.encode(x)
        if self.bpe is not None:
            x = self.bpe.encode(x)
        return x

    def decode_fn(self, x: str) -> str:
        if self.bpe is not None:
            x = self.bpe.decode(x)
        if self.tokenizer is not None:
            x = self.tokenizer.decode(x)
        return x

    def make_batches(self, lines: List[str], args: Namespace,
                     task: TranslationTask, max_positions: Tuple[int, int]) -> Generator:  # type: ignore
        if args.constraints:
            # Strip (tab-delimited) constraints, if present, from input lines,
            # store them in batch_constraints
            batch_constraints: List[List[str]] = [list() for _ in lines]
            for i, line in enumerate(lines):
                if "\t" in line:
                    lines[i], *batch_constraints[i] = line.split("\t")

            # Convert each List[str] to List[Tensor]
            for i, constraint_list in enumerate(batch_constraints):
                batch_constraints[i] = [
                    task.target_dictionary.encode_line(
                        self.encode_fn(constraint),
                        append_eos=False,
                        add_if_not_exist=False,
                    )
                    for constraint in constraint_list
                ]

        tokens = [
            task.source_dictionary.encode_line(
                self.encode_fn(src_str), add_if_not_exist=False
            ).long()
            for src_str in lines
        ]

        if args.constraints:
            constraints_tensor = pack_constraints(batch_constraints)
        else:
            constraints_tensor = None

        lengths = [t.numel() for t in tokens]
        itr = task.get_batch_iterator(
            dataset=task.build_dataset_for_inference(
                tokens, lengths, constraints=constraints_tensor
            ),
            max_tokens=args.max_tokens,
            max_sentences=args.batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        ).next_epoch_itr(shuffle=False)
        for batch in itr:
            ids = batch["id"]
            src_tokens = batch["net_input"]["src_tokens"]
            src_lengths = batch["net_input"]["src_lengths"]
            constraints = batch.get("constraints", None)

            yield self.batch(
                ids=ids,
                src_tokens=src_tokens,
                src_lengths=src_lengths,
                constraints=constraints,
            )

    def load_model(self, library_root: str, g2p_model_path: str) -> None:
        # Load ensemble
        self.models, _model_args = checkpoint_utils.load_model_ensemble(
            [os.path.join(library_root, g2p_model_path)],
            task=self.task,
            strict=(self.cfg.checkpoint_shard_count == 1),
            num_shards=self.cfg.checkpoint_shard_count,
        )

        # Optimize ensemble for generation
        for model in self.models:
            if self.cfg.fp16:
                model.half()
            if self.use_cuda and not self.cfg.pipeline_model_parallel:
                model.cuda()
            model.prepare_for_inference_(self.cfg)

        # Initialize generator
        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(), *[model.max_positions() for model in self.models]
        )
        self.generator = self.task.build_generator(self.models, self.cfg)

    @staticmethod
    def _remove_stress(word: str) -> str:
        letter_pattern = r'[a-zA-Z]+'
        result: List[str] = []
        for elem in word.split():
            match = re.match(letter_pattern, elem)
            elem = match.group(0) if match else elem
            result.append(elem)
        return ' '.join(result)

    def decode_word(self, word: str, with_stress: Optional[bool] = False) -> str:
        if len(word.split()) > 1:
            raise Exception(f"Input {word} contains multiple words. This function expects one word")

        inputs = [' '.join(list(word.lower()))]
        results = []
        for batch in self.make_batches(inputs, self.cfg, self.task, self.max_positions):
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
            translations = self.task.inference_step(
                self.generator, self.models, sample, constraints=constraints
            )
            for i, (_, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.tgt_dict.pad())
                results.append(
                    (
                        src_tokens_i,
                        hypos,
                    )
                )

        # sort output to match input order
        result_string = ''
        for src_tokens, hypos in sorted(results, key=lambda x: x[0]):  # type: ignore
            if self.src_dict is not None:
                src_str = self.src_dict.string(src_tokens, self.cfg.remove_bpe)

            # Process top predictions
            for hypo in hypos[: min(len(hypos), self.cfg.nbest)]:
                _, hypo_str, _ = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=self.align_dict,
                    tgt_dict=self.tgt_dict,
                    remove_bpe=self.cfg.remove_bpe,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(self.generator),
                )
                result_string = self.decode_fn(hypo_str)
                if not with_stress:
                    result_string = self._remove_stress(result_string)
                break
        return result_string


def load_g2p_model(library_root: str, g2p_model_path: str) -> G2PPytorch:
    model = G2PPytorch()
    model.load_model(library_root, g2p_model_path)
    return model
