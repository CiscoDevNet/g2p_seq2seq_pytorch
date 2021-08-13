#!/bin/bash
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


mkdir prep
mkdir orig
mkdir bpe
mkdir final_model_data

read -p "Enter training data folder path: " train_data # g2p-seq2seq-pytorch/data/learning_data
read -p "Enter tokenizer path: " tokenizer # fairseq/examples/translation/mosesdecoder/scripts/tokenizer
read -p "Enter subword_nmt path: " subword_nmt # fairseq/examples/translation/subword-nmt/subword_nmt

src=word
tgt=phon
DECODER=tokenizer
tmp=orig
TRAIN=bpe/train.combined
BPE_CODE=prep/code
BPE_TOKENS=10000
BPEROOT=subword_nmt
prep=prep

for L in $src $tgt; do
  for f in train.$L valid.$L test.$L; do
    cat $train_data/$f | perl $DECODER/tokenizer.perl -threads 1 -l en > $tmp/$f
  done
done

rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
    done
done

fairseq-preprocess --source-lang word --target-lang phon --trainpref $prep/train --validpref $prep/valid --testpref $prep/test --destdir final_model_data --workers 20

rm -r orig
rm -r bpe
rm -r prep
