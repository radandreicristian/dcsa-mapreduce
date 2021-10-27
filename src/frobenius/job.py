import math
from typing import List, Tuple

from mrjob.job import MRJob
from mrjob.step import MRStep


class FrobeniusNormJob(MRJob):
    """A mapreduce job that wraps the computation of the Frobenius norm of a matrix."""

    def mapper_matrix(self, input_path, _) -> List[Tuple[int, float]]:
        """
        Reads a 2D matrix from a file. Lines are separated by newlines and columns by spaces.
        :param input_path: The location of the input file.
        :param _: The URI to the input file (unused).
        :return: Key-value tuples with the key being the column index and the value being the square of an element.
        """
        with open(input_path, 'r') as input_file:
            for line in input_file:
                values = line.split()
                for j, value in enumerate(values):
                    value_f = float(value)
                    yield j, value_f * value_f

    def combiner_column(self, key, values) -> List[Tuple[int, float]]:
        """
        Combine the pairs with the same key (column) by adding their values.
        This step happens locally in the same mapper node.

        :param key: The line index.
        :param values: Mapper-node local square values from the key line.
        :return: Key-value pairs, where the key is the column and the value is the local sum of squares for that line.
        """
        yield key, sum(values)

    def reducer_column(self, _, values) -> List[Tuple[str, float]]:
        """
        Reduce the pairs with the same key (column) by adding their values.

        :param _: The column index (unused further).
        :param values: The mapper-node local sums of the key column.
        :return: The column sum for
        """
        yield "column_norm", sum(values)

    def reducer_line(self, _, values) -> List[Tuple[str, float]]:
        """
        Sum the per-column sums into a single value, which is effectively the Frobenius norm.

        :param _: The column index (unused)
        :param values: The final sums corresponding to the key column.
        :return: The Frobenius norm.
        """
        yield "frobenius_norm", math.sqrt(sum(values))

    def steps(self) -> List:
        """
       Define the job steps. As there can only be 1 reducer step, another MRStep has to be defined for the sorting.
       :return: The list of the steps of the job.
       """

        return [
            MRStep(mapper_raw=self.mapper_matrix,
                   combiner=self.combiner_column,
                   reducer=self.reducer_column),
            MRStep(reducer=self.reducer_line)
        ]


if __name__ == '__main__':
    FrobeniusNormJob().run()
