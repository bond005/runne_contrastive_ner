import os
import sys
import unittest

try:
    from data_processing.feature_extraction import find_entity_words
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from data_processing.feature_extraction import find_entity_words


class TestFeatureExtraction(unittest.TestCase):
    def test_find_entity_words_1(self):
        words = [
            ('Их', 0, 2),
            ('имена', 3, 8),
            ('—', 9, 10),
            ('Диас', 11, 15),
            ('Кадырбаев', 16, 25),
            ('и', 26, 27),
            ('Азамат', 28, 34),
            ('Тажаяков', 35, 43),
            ('.', 43, 44)
        ]
        entity_start = 11
        entity_end = 25
        true_bounds = (3, 5)
        calculated_bounds = find_entity_words(words, entity_start, entity_end)
        self.assertIsInstance(calculated_bounds, tuple)
        self.assertEqual(calculated_bounds, true_bounds)

    def test_find_entity_words_2(self):
        words = [
            ('Их', 0, 2),
            ('имена', 3, 8),
            ('—', 9, 10),
            ('Диас', 11, 15),
            ('Кадырбаев', 16, 25),
            ('и', 26, 27),
            ('Азамат', 28, 34),
            ('Тажаяков', 35, 43),
            ('.', 43, 44)
        ]
        entity_start = 13
        entity_end = 25
        true_bounds = (3, 5)
        calculated_bounds = find_entity_words(words, entity_start, entity_end)
        self.assertIsInstance(calculated_bounds, tuple)
        self.assertEqual(calculated_bounds, true_bounds)

    def test_find_entity_words_3(self):
        words = [
            ('Их', 0, 2),
            ('имена', 3, 8),
            ('—', 9, 10),
            ('Диас', 11, 15),
            ('Кадырбаев', 16, 25),
            ('и', 26, 27),
            ('Азамат', 28, 34),
            ('Тажаяков', 35, 43),
            ('.', 43, 44)
        ]
        entity_start = 10
        entity_end = 25
        true_bounds = (3, 5)
        calculated_bounds = find_entity_words(words, entity_start, entity_end)
        self.assertIsInstance(calculated_bounds, tuple)
        self.assertEqual(calculated_bounds, true_bounds)

    def test_find_entity_words_4(self):
        words = [
            ('Их', 0, 2),
            ('имена', 3, 8),
            ('—', 9, 10),
            ('Диас', 11, 15),
            ('Кадырбаев', 16, 25),
            ('и', 26, 27),
            ('Азамат', 28, 34),
            ('Тажаяков', 35, 43),
            ('.', 43, 44)
        ]
        entity_start = 11
        entity_end = 24
        true_bounds = (3, 5)
        calculated_bounds = find_entity_words(words, entity_start, entity_end)
        self.assertIsInstance(calculated_bounds, tuple)
        self.assertEqual(calculated_bounds, true_bounds)

    def test_find_entity_words_5(self):
        words = [
            ('Их', 0, 2),
            ('имена', 3, 8),
            ('—', 9, 10),
            ('Диас', 11, 15),
            ('Кадырбаев', 16, 25),
            ('и', 26, 27),
            ('Азамат', 28, 34),
            ('Тажаяков', 35, 43),
            ('.', 43, 44)
        ]
        entity_start = 11
        entity_end = 26
        true_bounds = (3, 5)
        calculated_bounds = find_entity_words(words, entity_start, entity_end)
        self.assertIsInstance(calculated_bounds, tuple)
        self.assertEqual(calculated_bounds, true_bounds)

    def test_find_entity_words_7(self):
        words = [
            ('Их', 0, 2),
            ('имена', 3, 8),
            ('—', 9, 10),
            ('Диас', 11, 15),
            ('Кадырбаев', 16, 25),
            ('и', 26, 27),
            ('Азамат', 28, 34),
            ('Тажаяков', 35, 43),
            ('.', 43, 44)
        ]
        entity_start = 10
        entity_end = 11
        true_bounds = (-1, -1)
        calculated_bounds = find_entity_words(words, entity_start, entity_end)
        self.assertIsInstance(calculated_bounds, tuple)
        self.assertEqual(calculated_bounds, true_bounds)

    def test_find_entity_words_8(self):
        words = [
            ('Их', 0, 2),
            ('имена', 3, 8),
            ('—', 9, 10),
            ('Диас', 11, 15),
            ('Кадырбаев', 16, 25),
            ('и', 26, 27),
            ('Азамат', 28, 34),
            ('Тажаяков', 35, 43),
            ('.', 43, 44)
        ]
        entity_start = 10
        entity_end = 12
        true_bounds = (3, 4)
        calculated_bounds = find_entity_words(words, entity_start, entity_end)
        self.assertIsInstance(calculated_bounds, tuple)
        self.assertEqual(calculated_bounds, true_bounds)


if __name__ == '__main__':
    unittest.main(verbosity=2)
