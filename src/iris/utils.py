from math import sqrt
from typing import Tuple, List

import numpy as np


def compute_distances(v1, v2):
    return sqrt(np.sum(np.power((v1 - v2), 2)))


def determine_nearest(array: List[Tuple[int, str, float]]):
    lookup = {}
    for el in array:
        class_ = el[1]
        if class_ in lookup.keys():
            lookup[class_] += 1
        else:
            lookup[class_] = 0

    maximum_value = max(lookup.values())
    for k, v in lookup.items():
        if v == maximum_value:
            return k
