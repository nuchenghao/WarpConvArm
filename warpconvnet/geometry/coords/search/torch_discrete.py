from typing import Sequence
import hashlib


def _int_sequence_hash(arr: Sequence[int]) -> int:  # noqa: F821
    x = hash(arr[0])
    for i in range(1, len(arr)):
        x = (x * 31 + hash(arr[i])) & 0xFFFFFFFF  # Keep it within 32-bit range
    return x


def string_hash(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) & 0xFFFFFFFF
