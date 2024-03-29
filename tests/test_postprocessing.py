import os
import sys
import unittest

import numpy as np

try:
    from data_processing.postprocessing import decode_entity
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from data_processing.postprocessing import decode_entity


class TestPostprocessing(unittest.TestCase):
    def test_decode_entity_1(self):
        words = [
            ('Всемирно', 1, 3),
            ('известный', 3, 4),
            ('российский', 4, 5),
            ('дирижер', 5, 6),
            ('Валерий', 6, 7),
            ('Гергиев', 7, 9),
            ('назначен', 9, 10),
            ('главным', 10, 11),
            ('дирижёром', 11, 12),
            ('Мюнхенского', 12, 13),
            ('филармонического', 13, 14),
            ('оркестра', 14, 15),
            ('сезона', 15, 16),
            ('2015', 16, 17),
            ('-', 17, 18),
            ('2016', 18, 19),
            ('.', 19, 20)
        ]
        probas = np.array(
            [
                [0.785751, 0.056005, 0.051484, 0.070663, 0.036097],
                [0.833238, 0.011210, 0.034858, 0.072335, 0.048359],
                [0.851490, 0.057056, 0.009934, 0.059273, 0.022248],
                [0.790059, 0.073516, 0.065282, 0.000668, 0.070475],
                [0.798070, 0.035531, 0.055900, 0.056896, 0.053603],
                [0.832262, 0.043964, 0.033757, 0.016637, 0.073380],
                [0.001158, 0.898743, 0.038385, 0.048147, 0.013567],
                [0.052801, 0.044575, 0.014728, 0.841990, 0.045907],
                [0.472256, 0.028446, 0.471274, 0.019135, 0.008889],
                [0.255073, 0.015395, 0.136109, 0.303709, 0.289713],
                [0.802003, 0.056668, 0.067668, 0.007814, 0.065847],
                [0.831534, 0.031822, 0.051651, 0.043895, 0.041097],
                [0.839506, 0.043976, 0.001130, 0.069636, 0.045752],
                [0.837648, 0.023461, 0.014155, 0.077266, 0.047470],
                [0.866057, 0.026031, 0.057585, 0.026189, 0.024138],
                [0.802629, 0.020013, 0.048540, 0.061235, 0.067583],
                [0.853223, 0.045516, 0.013242, 0.045833, 0.042185],
                [0.877368, 0.041283, 0.046932, 0.025986, 0.008430],
                [0.803484, 0.016858, 0.070242, 0.036445, 0.072971],
                [0.809946, 0.044375, 0.032782, 0.053970, 0.058927],
                [0.852801, 0.045037, 0.022476, 0.056785, 0.022901]
            ],
            dtype=np.float64
        )
        true_entity_bounds = [
            (6, 10)
        ]
        self.assertEqual(decode_entity(probas, words), true_entity_bounds)

    def test_decode_entity_2(self):
        words = [
            ('Всемирно', 1, 3),
            ('известный', 3, 4),
            ('российский', 4, 5),
            ('дирижер', 5, 6),
            ('Валерий', 6, 7),
            ('Гергиев', 7, 9),
            ('назначен', 9, 10),
            ('главным', 10, 11),
            ('дирижёром', 11, 12),
            ('Мюнхенского', 12, 13),
            ('филармонического', 13, 14),
            ('оркестра', 14, 15),
            ('сезона', 15, 16),
            ('2015', 16, 17),
            ('-', 17, 18),
            ('2016', 18, 19),
            ('.', 19, 20)
        ]
        probas = np.array(
            [
                [0.796838, 0.053021, 0.046638, 0.038369, 0.065134],
                [0.822459, 0.028262, 0.047054, 0.051639, 0.050586],
                [0.800919, 0.026783, 0.072563, 0.024914, 0.074821],
                [0.816011, 0.017526, 0.069264, 0.063983, 0.033216],
                [0.878857, 0.048738, 0.027262, 0.017532, 0.027612],
                [0.786287, 0.053934, 0.057514, 0.069344, 0.032921],
                [0.074225, 0.037583, 0.060416, 0.036406, 0.791369],
                [0.925706, 0.049973, 0.006923, 0.009853, 0.007545],
                [0.872278, 0.035567, 0.013706, 0.006138, 0.072311],
                [0.782540, 0.033368, 0.065080, 0.065682, 0.053329],
                [0.813465, 0.046575, 0.017866, 0.050632, 0.071462],
                [0.812269, 0.026303, 0.054771, 0.028629, 0.078027],
                [0.894976, 0.048893, 0.020449, 0.010267, 0.025414],
                [0.884381, 0.034809, 0.003209, 0.052090, 0.025510],
                [0.861689, 0.011617, 0.021719, 0.037124, 0.067851],
                [0.778729, 0.071904, 0.053251, 0.069198, 0.026918],
                [0.916116, 0.034633, 0.015576, 0.001566, 0.032109],
                [0.854982, 0.054031, 0.054031, 0.035709, 0.001247],
                [0.853110, 0.029395, 0.023982, 0.032975, 0.060538],
                [0.826272, 0.065835, 0.042987, 0.064131, 0.000775],
                [0.821763, 0.058852, 0.046281, 0.006966, 0.066138]
            ],
            dtype=np.float64
        )
        true_entity_bounds = [
            (6, 7)
        ]
        self.assertEqual(decode_entity(probas, words), true_entity_bounds)


if __name__ == '__main__':
    unittest.main(verbosity=2)
