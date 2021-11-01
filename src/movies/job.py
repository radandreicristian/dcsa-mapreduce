from typing import Tuple, Generator, List

import pandas as pd
from mrjob.job import MRJob
from mrjob.step import MRStep
import utils


class TopKeywordsJob(MRJob):
    """A mapreduce job that wraps the top keyword in the movie titles task."""

    def configure_args(self) -> None:
        """
        Configure the command line arguments for running the job.

        :return: None
        """
        super(TopKeywordsJob, self).configure_args()
        self.add_passthru_arg("-m",
                              "--maxWords",
                              type=int,
                              default=10,
                              help="How many words to display from the top.")

    def mapper_csv(self,
                   input_path: str,
                   _: str) -> Generator[Tuple[Tuple[str, str], int], None, None]:
        """
        Reads a CSV-format file and outputs key-value pairs that contain the appearance of words in the title of genres.

        :param input_path: The path to the CSV file.
        :param _: The URI of the CSV (unused).
        :return: A generator of key-value pairs, where the key is a tuple consisting of the word and the genre, and the
        value is 1 (1 appearance).
        """
        df = pd.read_csv(input_path)

        # This is sort of a null value, so drop those lines.
        df = df[df['genres'] != "(no genres listed)"]
        for _, data in df.iterrows():
            title = data["title"]
            clean_title = utils.preprocess_text(title)
            title_words = clean_title.split()
            genres = data["genres"].split('|')
            for genre in genres:
                for word in title_words:
                    yield (word, genre), 1

    def combiner_sum(self,
                     key: Tuple[str, str],
                     values: List[int]) -> Generator[Tuple[Tuple[str, str], int], None, None]:
        """
        Combine the pairs with the same by adding their values. This step happens locally in the mapper node.

        :param key: (word, genre) tuples.
        :param values: The count of the keys from the mapper (always "1").
        :return: Key-value pairs where the key is the word and the value is the node-local sum of the counts.
        """
        yield key, sum(values),

    def reducer_sum(self,
                    key: Tuple[str, str],
                    values: List[int]) -> Generator[Tuple[str, Tuple[str, int]], None, None]:
        """
        Combine the pairs with the same key by adding their values (from all the mapper nodes).
        Here the key becomes the genre, and values becomes a list of tuples containing the word and its appearance.

        :param key: A word.
        :param values: The count of the word per genre from the combiner.
        :return: Key-value pairs where the key is the genre and the value is the tuple consisting of the word and its
        number of total appearances.
        """
        word, genre = key
        yield genre, (word, sum(values))

    def reducer_sort(self,
                     key: str,
                     values: List[Tuple[str, int]]) -> Generator[Tuple[str, Tuple[str, int]], None, None]:
        """
        Sort the tuples from the previous reducer by their total occurrences element. Take the first maxWords pairs for
        each of the genres.

        :param key: The genre.
        :param values: Tuples consisting of a word and its total appearances.
        :return: The first maxWords words, by the number of appearances.
        """
        top_size = self.options.maxWords
        sorted_values = sorted(values,
                               key=lambda x: x[1],
                               reverse=True)[:top_size]
        yield key, sorted_values

    def steps(self) -> List:
        """
        Define the job steps. As there can only be 1 reducer step, another MRStep has to be defined for the sorting.

        :return: The list of the steps of the job.
        """
        return [
            MRStep(mapper_raw=self.mapper_csv,
                   combiner=self.combiner_sum,
                   reducer=self.reducer_sum),
            MRStep(reducer=self.reducer_sort)
        ]


if __name__ == '__main__':
    job = TopKeywordsJob.run()
