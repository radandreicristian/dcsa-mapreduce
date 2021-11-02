import time

from job import SortMergeIrisClassificationJob

if __name__ == '__main__':
    start_timestamp = time.time()
    job = SortMergeIrisClassificationJob()
    for i in range(100):
        job.run()
    end_timestamp = time.time()
    print(f"Delta T {end_timestamp - start_timestamp}")
