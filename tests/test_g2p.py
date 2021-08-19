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


import pytest


@pytest.mark.skip(reason="Model is not checked into git")
@pytest.mark.parametrize(
    "query, stress, expected_value",
    [
        ("dresden", True, "D R EH1 Z D AH0 N"),
        ("dresden", False, "D R EH Z D AH N"),
        ("KolbEck", True, "K OW1 L B EH2 K"),
        ("Kolbeck", False, "K OW L B EH K"),
        ("hidalgo", True, "HH IH0 D AE1 L G OW2"),
        ("hidalgo", False, "HH IH D AE L G OW"),
        ("tejas", True, "T EY1 JH AH0 Z"),
        ("tejas", False, "T EY JH AH Z"),
        ("dugal", True, "D UW1 G AH0 L"),
        ("dugal", False, "D UW G AH L"),
        ("wyoming", True, "W AY1 AH0 M IH0 NG"),
        ("wyoming", False, "W AY AH M IH NG"),
        ("maoris", True, "M AW1 R IH0 S"),
        ("maoris", False, "M AW R IH S"),
    ],
)
def test_g2p_beam(g2p_beam, query, stress, expected_value):
    g2p_result = g2p_beam.decode_word(query, stress)
    assert g2p_result == expected_value


@pytest.mark.skip(reason="Model is not checked into git")
@pytest.mark.parametrize(
    "query, stress, expected_value",
    [
        ("dresden", True, "D R EH1 Z D AH0 N"),
        ("dresden", False, "D R EH Z D AH N"),
        ("KolbEck", True, "K OW1 L B EH2 K"),
        ("Kolbeck", False, "K OW L B EH K"),
        ("hidalgo", True, "HH IH0 D AE1 L G OW2"),
        ("hidalgo", False, "HH IH D AE L G OW"),
        ("tejas", True, "T EY1 JH AH0 Z"),
        ("tejas", False, "T EY JH AH Z"),
        ("dugal", True, "D UW1 G AH0 L"),
        ("dugal", False, "D UW G AH L"),
        ("wyoming", True, "W AY1 AH0 M IH0 NG"),
        ("wyoming", False, "W AY AH M IH NG"),
        ("maoris", True, "M AW1 R IH0 S"),
        ("maoris", False, "M AW R IH S"),
    ],
)
def test_g2p_no_beam(g2p_no_beam, query, stress, expected_value):
    g2p_result = g2p_no_beam.decode_word(query, stress)
    assert g2p_result == expected_value


@pytest.mark.skip(reason="Model is not checked into git")
def test_invalid_input(g2p_beam):
    with pytest.raises(Exception):
        g2p_beam.decode_word("two words")
