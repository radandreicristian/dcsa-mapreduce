import heapq
from typing import Tuple, List


class Node:
    """A wrapper class for the nodes (for constructing singly linked lists)."""

    def __init__(self,
                 value: float = 0.,
                 metadata: Tuple[int, str] = None,
                 next_node=None):
        self.value = value
        self.next_node = next_node
        self.metadata = metadata


def get_most_frequent(array: List[Tuple[int, float, str]]) -> str:
    """
    Gets the class with the most appearances from a list.

    :param array: A list of tuples containing id, class, and distance
    :return: The class with the most appearances in the tuples.
    """
    lookup = {}
    for el in array:
        class_ = el[2]
        if class_ in lookup.keys():
            lookup[class_] += 1
        else:
            lookup[class_] = 0

    max_frequency = max(lookup.values())
    for k, v in lookup.items():
        if v == max_frequency:
            return k


def convert_to_nodes(list_: List[Tuple[int, str, float]]) -> Node:
    """
    Convert a list to a linked-list Node structure.

    :param list_: A list of tuples
    :return:
    """
    head = list_[0]
    head_node = Node(value=head[2],
                     metadata=(head[0], head[1]))
    current = head_node
    length = len(list_)
    for index in range(1, length):
        el = list_[index]
        new_node = Node(value=el[2],
                        metadata=(el[0], el[1]))
        current.next_node = new_node
        current = new_node
    current.next_node = None
    return head_node


def convert_to_list(node: Node) -> List:
    """
    Converts a linked list stored in a Node to a regular list.

    :param node: The head of the linked list.
    :return: A flat list containing the values in the nodes of the linked list.
    """
    li = []
    current_node = node
    while current_node:
        data = (current_node.value, *current_node.metadata)
        li.append(data)
        current_node = current_node.next_node
    return li


def merge_k_lists(sorted_lists: List[List[Tuple[int, str, float]]]):
    """
    Merge K sorted lists using a min-heap.
    Source: https://leetcode.com/problems/merge-k-sorted-lists/discuss/10511/10-line-python-solution-with-priority-queue/281748

    :param sorted_lists: A list of K sorted lists of neighbour tuples.
    :return: The merged sorted list.
    """
    sorted_linked_lists = [convert_to_nodes(raw_list) for raw_list in sorted_lists]
    curr = head = Node(0)
    queue = []
    count = 0
    for sorted_linked_list in sorted_linked_lists:
        if sorted_linked_list:
            count += 1
            heapq.heappush(queue, (sorted_linked_list.value, count, sorted_linked_list))
    while len(queue) > 0:
        _, _, curr.next_node = heapq.heappop(queue)
        curr = curr.next_node
        if curr.next_node is not None:
            count += 1
            heapq.heappush(queue, (curr.next_node.value, count, curr.next_node))
    return convert_to_list(head.next_node)
