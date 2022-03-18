import os
import random
import sys

from io_utils.io_utils import load_data, save_data
from data_processing.train_test_split import train_test_split


def main():
    random.seed(42)
    if len(sys.argv) < 2:
        err_msg = 'The source file is not specified!'
        raise ValueError(err_msg)
    src_fname = os.path.normpath(sys.argv[1])
    if len(sys.argv) < 3:
        err_msg = 'The training file is not specified!'
        raise ValueError(err_msg)
    training_fname = os.path.normpath(sys.argv[2])
    if len(sys.argv) < 4:
        err_msg = 'The test file is not specified!'
        raise ValueError(err_msg)
    test_fname = os.path.normpath(sys.argv[3])

    if not os.path.isfile(src_fname):
        raise IOError(f'The file {src_fname} does not exist!')
    dname = os.path.dirname(training_fname)
    if len(dname) > 0:
        if not os.path.isdir(dname):
            raise IOError(f'The directory {dname} does not exist!')
    dname = os.path.dirname(test_fname)
    if len(dname) > 0:
        if not os.path.isdir(dname):
            raise IOError(f'The directory {dname} does not exist!')

    source_data = load_data(src_fname)
    data_for_training, data_for_testing = train_test_split(source_data)
    save_data(training_fname, True, data_for_training)
    save_data(test_fname, True, data_for_testing)


if __name__ == '__main__':
    main()
