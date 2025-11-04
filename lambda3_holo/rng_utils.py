# SPDX-License-Identifier: MIT
from numpy.random import Generator, PCG64, SeedSequence

def make_rng(seed: int, stream_tag: str) -> Generator:
    """
    Create an independent RNG stream from a common integer seed and a tag.
    This guarantees independence across subsystems & presets.
    """
    ss = SeedSequence(seed, spawn_key=[hash(stream_tag) & 0xffffffff])
    return Generator(PCG64(ss))
