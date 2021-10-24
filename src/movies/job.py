import pandas as pd
from mrjob.job import MRJob
from mrjob.step import MRStep
from util.preprocessing import clear_text


class TopKeywordsJob(MRJob):

    def configure_args(self):
        super(TopKeywordsJob, self).configure_args()
        self.add_passthru_arg("-m",
                              "--maxWords",
                              type=int,
                              help="How many words to display from the top.")

    def mapper_csv(self, input_path, _):
        """
        Reads a csv-format file and outputs key-value pairs from its title.
        The keys are the words in the title (after preprocessing), and the values are 1's (no. of appearances).
        :param input_path: The location of the CSV
        :param _: The URI of the CSV (unused)
        :return: The key-value pairs of words in the title and 1.
        """
        df = pd.read_csv(input_path)
        for _, data in df.iterrows():
            title = data['title']
            clean_title = clear_text(title)
            tokens = clean_title.split()
            for token in tokens:
                yield token, 1

    def combiner_sum(self, key, values):
        """
        Combine the pairs with the same key (word) by adding their values. This step happens locally in the mapper node.

        :param key: A word.
        :param values: The count of the word from the mapper (always "1").
        :return: Key-value pairs where the key is the word and the value is the node-local sum of the counts.
        """
        yield key, sum(values),

    def reducer(self, key, values):
        """
        Combine the pairs with the same key (word) by adding their values (from all the mapper nodes).
        There is no need to have an effective key, so the value can become a tuple of (count, word).


        :param key: A word.
        :param values: The count of the word from the combiner.
        :return: Key-value pairs where the key is disregarded and the value is the tuple consisting of the word and
        """
        yield None, (key, sum(values))

    def reducer_sort(self, _, values):
        """
        Sort the tuples from the previous step (this is still a reducer) by their second element.
        :param _: Keys (ignored, as the previous keys are None).
        :param values: The tuples consisting of the final count of the words' appearances.
        :return: The first maxWords words, by the number of appearances.
        """
        top_size = self.options.maxWords
        for word, count in sorted(values,
                                  key=lambda x: x[1],
                                  reverse=True)[:top_size]:
            yield word, count

    def steps(self):
        """
        Define the job steps. As there can only be 1 reducer step, another MRStep has to be defined for the sorting.
        :return: The list of the steps of the job.
        """
        return [
            MRStep(mapper_raw=self.mapper_csv,
                   combiner=self.combiner_sum,
                   reducer=self.reducer),
            MRStep(reducer=self.reducer_sort)
        ]


if __name__ == '__main__':
    job = TopKeywordsJob.run()
