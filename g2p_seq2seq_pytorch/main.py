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


"""
Translate raw text with a trained model
"""
from .g2p import G2PPytorch

MODEL_ROOT = 'g2p_seq2seq_pytorch/models'
MODEL_NAME = 'checkpoint_v01.pt'


def main() -> None:
    model = G2PPytorch()
    model.load_model(MODEL_ROOT, MODEL_NAME)
    print(model.decode_word("vijay"))
    print(model.decode_word("karthik"))


if __name__ == "__main__":
    main()
