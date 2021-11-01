from typing import Tuple, Generator

import pandas as pd
from mrjob.job import MRJob
from mrjob.step import MRStep

from utils import determine_nearest, compute_distances


class IrisClassificationJob(MRJob):
    """An MRJob for iris classification using KNN,"""

    def configure_args(self) -> None:
        """
        Configure the command line arguments for running the job.
        :return: None.
        """
        super(IrisClassificationJob, self).configure_args()
        self.add_passthru_arg("-k",
                              "--kNearest",
                              type=int,
                              help="How many closest neighbours to consider.")

    def mapper_csv(self,
                   input_path: str,
                   _: str) -> Generator[Tuple[int, Tuple[int, str, float]], None, None]:
        """
        Reads a CSV-format file and outputs key-value pairs that indicate the distance between the test samples and
        the train samples.
        :param input_path: The path to the CSV file.
        :param _: The URI to the CSV file (unused).
        :return: A generator of key-value pairs of type [int, Tuple[int, str, float]], where the key is the id of a test
        sample, and the value is a tuple containing the id of a train sample, its class, and the distance between the
        features of the test and train samples.
        """
        df = pd.read_csv(input_path)
        predictors = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

        test_df = df[df['Species'].isnull()]
        train_df = df[df['Species'].notnull()]

        # Inplace modification of the dataframes results in an encoding error (bytes not 'str' required).
        test_df_deep_copy = test_df.copy(deep=True)
        train_df_deep_copy = train_df.copy(deep=True)

        # Min-max normalize the predictors
        for column in predictors:
            column_min = train_df[column].min()
            column_max = train_df[column].max()
            test_df_deep_copy[column] = (test_df[column] - column_min) / (column_max - column_min)
            train_df_deep_copy[column] = (train_df[column] - column_min) / (column_max - column_min)

        # Iterate through test_sample, train_sample pairs:
        for _, test_sample in test_df_deep_copy.iterrows():
            for _, train_sample in train_df_deep_copy.iterrows():
                distance = compute_distances(train_sample[predictors].to_numpy(),
                                             test_sample[predictors].to_numpy())
                yield test_sample['Id'], (train_sample['Id'], train_sample['Species'], distance)

    def reducer(self,
                key: str,
                values: Tuple[int, str, float]) -> Generator[Tuple[int, str], None, None]:
        """
        For a given test sample id, sort the distances to the train samples and return the class with most values.

        In this case it does not make sense to use a mapper-local combiner because we need all the distances to be
        sported. Using a sorting+merging strategy wouldn't improve the performance
        :param key: The ID of a sample in the test set.
        :param values: List[Tuple(int, str, float)]
        :return:
        """

        k_nearest = sorted(values, key=lambda x: x[2])[:self.options.kNearest]
        label = determine_nearest(k_nearest)
        yield key, label

    def steps(self):

        return [
            MRStep(mapper_raw=self.mapper_csv,
                   reducer=self.reducer)
        ]


if __name__ == '__main__':
    job = IrisClassificationJob().run()
