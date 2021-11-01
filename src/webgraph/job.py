from collections import Generator
from typing import Tuple, List

from mrjob.job import MRJob, MRStep


class RevertGraphJob(MRJob):
    """A mapreduce job that wraps the inversion of the edges in the Google web graph."""

    def mapper_nodes(self,
                     input_path: str,
                     _: str) -> Generator[Tuple[int, int], None, None]:
        """
        Reads the raw text file containing source destination node pairs.

        :param input_path: The location of the input file.
        :param _: The URI of the input (unused).
        :return: Key-value tuples with key=source_node and value=destination_node.
        """
        with open(input_path, 'r') as input_file:
            for line in input_file:
                # Just skip the comment lines
                if line.startswith('#'):
                    continue
                source_node, destination_node = map(int, line.split())
                yield source_node, destination_node

    def combiner(self,
                 key: int,
                 values: List[int]) -> Generator[Tuple[int, int], None, None]:
        """
        Combine the pairs with the same key (source_node).
        This step happens locally in the mapper node.

        :param key: Source node.
        :param values: Destination nodes.
        :return: Key-value pairs where the key is the source node and the value is a list of destination nodes.
        """
        yield key, values

    def reducer(self,
                key: int,
                values: List[List[int]]) -> Generator[Tuple[int, int], None, None]:
        """
        Reduce the pairs with the same key (from across different mapper nodes).

        :param key: The source node.
        :param values: Nested list containing lists of neighbours of a single key emitted by different mappers.
        :return: Key-value pairs where they key is the source node and the value is a (flat) list of all neighbours.
        """
        # Flatten the neighbours list, which is nested at this point.
        flat_values = [item for sublist in values for item in sublist]
        for value in flat_values:
            yield value, key

    def steps(self):
        """
        Define the job steps. As there can only be 1 reducer step, another MRStep has to be defined for the sorting.

        :return: The list of the steps of the job.
        """
        return [
            MRStep(mapper_raw=self.mapper_nodes,
                   combiner=self.combiner,
                   reducer=self.reducer)]


if __name__ == '__main__':
    job = RevertGraphJob().run()
