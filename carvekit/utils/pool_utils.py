"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool

Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].

License: Apache License 2.0
"""
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Callable, Collection, List


def thread_pool_processing(func: Callable[[Any], Any], data: Iterable, workers=18):
    """
    Passes all iterator data through the given function

    Args:
        workers (int, default=18): Count of workers.
        func (Callable[[Any], Any]): function to pass data through
        data (Iterable): input iterator

    Returns:
        List[Any]: list of results

    """
    with ThreadPoolExecutor(workers) as p:
        return list(p.map(func, data))


def batch_generator(iterable: Collection, n: int = 1) -> Iterable[Collection]:
    """
    Splits any iterable into n-size packets

    Args:
        iterable (Collection): iterator
        n (int, default=1): size of packets

    Returns:
        Iterable[Collection]: new n-size packet
    """
    it = len(iterable)
    for ndx in range(0, it, n):
        yield iterable[ndx : min(ndx + n, it)]
