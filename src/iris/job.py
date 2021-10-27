from mrjob.job import MRJob
import pandas as pd


class IrisClassificationJob(MRJob):

    def configure_args(self):
        super(IrisClassificationJob, self).configure_args()
        self.add_passthru_arg("-k",
                              "--kNearest",
                              type=int,
                              help="How many closest neighbours to consider.")


    def normalize_features(self):

    def mapper_csv(self, input_path, input_uri):
        df = pd.read_csv(input_path)
        for _, data, in df.iterrows():

