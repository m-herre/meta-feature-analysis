from __future__ import annotations

from mfa.data.split_decoder import decode_split_index


def test_decode_split_index_examples() -> None:
    assert decode_split_index(0) == (0, 0)
    assert decode_split_index(3) == (1, 0)
    assert decode_split_index(29) == (9, 2)

