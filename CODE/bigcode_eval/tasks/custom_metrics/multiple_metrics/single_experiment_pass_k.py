import json

import numpy as np


def estimator(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def for_file(path):
    with open(path, "r") as f:
        data = json.load(f)
    n = len(data["results"])
    c = len(
        [True for r in data["results"] if r["status"] == "OK" and r["exit_code"] == 0]
    )
    test_num = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    return np.array([estimator(n, c, k) for k in test_num])
