import heapq
import sys
from math import sqrt
from typing import Tuple, List, Generator
from queue import PriorityQueue

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


class ListMerger:
    class Node:
        def __init__(self,
                     val=0.,
                     meta=None,
                     next=None):
            self.val = val
            self.next = next
            self.meta = meta

    def convert_to_nodes(self,
                         l: List[Tuple[int, str, float]]) -> Node:
        head = l[0]
        head_node = self.Node(val=head[2],
                              meta=(head[1], head[0]))
        current = head_node
        length = len(l)
        for i in range(1, length):
            el = l[i]
            new_node = self.Node(val=el[2],
                                 meta=(el[1], el[0]))
            current.next = new_node
            current = new_node
        current.next = None
        return head_node

    def convert_to_list(self,
                        node: Node):
        li = []
        current_node = node
        while current_node:
            data = (node.val, *node.meta)
            li.append(data)
            current_node = current_node.next
        return li

    def merge_k_lists(self,
                      raw_lists: List[List[Tuple[int, str, float]]]):
        lists = [self.convert_to_nodes(raw_list) for raw_list in raw_lists]
        curr = head = self.Node(0)
        queue = []
        count = 0
        for l in lists:
            if l:
                count += 1
                heapq.heappush(queue, (l.val, count, l))
        while len(queue) > 0:
            _, _, curr.next = heapq.heappop(queue)
            curr = curr.next
            if curr.next is not None:
                count += 1
                heapq.heappush(queue, (curr.next.val, count, curr.next))
        return self.convert_to_list(head.next)
