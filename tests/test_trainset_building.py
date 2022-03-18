import os
import sys
import unittest

import numpy as np

try:
    from trainset_building.trainset_building import \
        transform_indicator_to_classmatrix
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from trainset_building.trainset_building import \
        transform_indicator_to_classmatrix


class TestTrainsetBuidling(unittest.TestCase):
    def test_transform_indicator_to_classmatrix_1(self):
        indicator = [0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0]
        true_matrix = np.array(
            [
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0]
            ],
            dtype=np.float32
        )
        res = transform_indicator_to_classmatrix(indicator)
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(len(res.shape), 3)
        self.assertEqual(res.shape[0], 1)
        self.assertEqual(res.shape[1:], true_matrix.shape)
        for row_idx in range(true_matrix.shape[0]):
            for col_idx in range(true_matrix.shape[1]):
                msg = f'[{row_idx}, {col_idx}]'
                self.assertAlmostEqual(res[0, row_idx, col_idx],
                                       true_matrix[row_idx, col_idx], msg=msg)

    def test_transform_indicator_to_classmatrix_2(self):
        indicator = [0, 2, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0]
        true_matrix = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0]
            ],
            dtype=np.float32
        )
        res = transform_indicator_to_classmatrix(indicator)
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(len(res.shape), 3)
        self.assertEqual(res.shape[0], 1)
        self.assertEqual(res.shape[1:], true_matrix.shape)
        for row_idx in range(true_matrix.shape[0]):
            for col_idx in range(true_matrix.shape[1]):
                msg = f'[{row_idx}, {col_idx}]'
                self.assertAlmostEqual(res[0, row_idx, col_idx],
                                       true_matrix[row_idx, col_idx], msg=msg)

    def test_transform_indicator_to_classmatrix_3(self):
        indicator = [0, 2, 2, 1, 0, 0, 2, 0, 0, 0, 0, 0]
        true_matrix = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0]
            ],
            dtype=np.float32
        )
        res = transform_indicator_to_classmatrix(indicator)
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(len(res.shape), 3)
        self.assertEqual(res.shape[0], 1)
        self.assertEqual(res.shape[1:], true_matrix.shape)
        for row_idx in range(true_matrix.shape[0]):
            for col_idx in range(true_matrix.shape[1]):
                msg = f'[{row_idx}, {col_idx}]'
                self.assertAlmostEqual(res[0, row_idx, col_idx],
                                       true_matrix[row_idx, col_idx], msg=msg)


if __name__ == '__main__':
    unittest.main(verbosity=2)
