import os
import sys
import unittest


try:
    from evaluation.evaluation import split_text_by_words
    from evaluation.evaluation import compare_entities
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from evaluation.evaluation import split_text_by_words
    from evaluation.evaluation import compare_entities


class TestEvaluation(unittest.TestCase):
    def test_split_text_by_words_1(self):
        text = 'По предварительным данным, его отравили в аэропорту, ' \
            'когда он направлялся из Малайзии в Макао.'
        source_ners = [('EVENT', 31, 39), ('LOCATION', 40, 51),
                       ('COUNTRY', 77, 85), ('STATE_OR_PROVINCE', 88, 93),
                       ('LOCATION', 74, 85), ('LOCATION', 86, 93)]
        true_words = ['По', 'предварительным', 'данным', ',', 'его', 'отравили',
                      'в', 'аэропорту', ',', 'когда', 'он', 'направлялся', 'из',
                      'Малайзии', 'в', 'Макао', '.']
        true_ners = [('EVENT', 5, 6), ('LOCATION', 6, 8), ('COUNTRY', 13, 14),
                     ('STATE_OR_PROVINCE', 15, 16), ('LOCATION', 12, 14),
                     ('LOCATION', 14, 16)]
        prepared = split_text_by_words(s=text, entities=source_ners)
        self.assertIsInstance(prepared, tuple)
        self.assertEqual(len(prepared), 2)
        prepared_words, prepared_entities = prepared
        self.assertIsInstance(prepared_words, list)
        self.assertIsInstance(prepared_entities, list)
        self.assertEqual(prepared_words, true_words)
        self.assertEqual(prepared_entities, true_ners)

    def test_split_text_by_words_2(self):
        text = 'По предварительным данным, его отравили в аэропорту, ' \
            'когда он направлялся из Малайзии в Макао.'
        source_ners = [('EVENT', 32, 39), ('LOCATION', 40, 51),
                       ('COUNTRY', 77, 84), ('STATE_OR_PROVINCE', 88, 93),
                       ('LOCATION', 74, 85), ('LOCATION', 86, 93)]
        true_words = ['По', 'предварительным', 'данным', ',', 'его', 'отравили',
                      'в', 'аэропорту', ',', 'когда', 'он', 'направлялся', 'из',
                      'Малайзии', 'в', 'Макао', '.']
        true_ners = [('EVENT', 5, 6), ('LOCATION', 6, 8), ('COUNTRY', 13, 14),
                     ('STATE_OR_PROVINCE', 15, 16), ('LOCATION', 12, 14),
                     ('LOCATION', 14, 16)]
        prepared = split_text_by_words(s=text, entities=source_ners)
        self.assertIsInstance(prepared, tuple)
        self.assertEqual(len(prepared), 2)
        prepared_words, prepared_entities = prepared
        self.assertIsInstance(prepared_words, list)
        self.assertIsInstance(prepared_entities, list)
        self.assertEqual(prepared_words, true_words)
        self.assertEqual(prepared_entities, true_ners)

    def test_compare_entities_1(self):
        true_ners = [('EVENT', 5, 6), ('LOCATION', 6, 8), ('COUNTRY', 13, 14),
                     ('STATE_OR_PROVINCE', 15, 16), ('LOCATION', 12, 14),
                     ('LOCATION', 14, 16)]
        predicted_ners = [('EVENT', 5, 6), ('LOCATION', 8, 9),
                          ('COUNTRY', 13, 14), ('STATE_OR_PROVINCE', 15, 16)]
        true_res = {
            'tp': set(),
            'fp': {(8, 9)},
            'fn': {(6, 8), (12, 14), (14, 16)}
        }
        self.assertEqual(
            true_res,
            compare_entities(true_ners, predicted_ners, 'LOCATION')
        )

    def test_compare_entities_2(self):
        true_ners = [('EVENT', 5, 6), ('COUNTRY', 13, 14),
                     ('STATE_OR_PROVINCE', 15, 16)]
        predicted_ners = [('EVENT', 5, 6), ('COUNTRY', 13, 14),
                          ('STATE_OR_PROVINCE', 15, 16)]
        self.assertIsNone(
            compare_entities(true_ners, predicted_ners, 'LOCATION')
        )

    def test_compare_entities_3(self):
        true_ners = [('EVENT', 5, 6), ('LOCATION', 6, 8), ('COUNTRY', 13, 14),
                     ('STATE_OR_PROVINCE', 15, 16), ('LOCATION', 12, 14),
                     ('LOCATION', 14, 16)]
        predicted_ners = [('EVENT', 5, 6), ('LOCATION', 6, 8),
                          ('COUNTRY', 13, 14), ('STATE_OR_PROVINCE', 15, 16),
                          ('LOCATION', 12, 14), ('LOCATION', 14, 16)]
        true_res = {
            'tp': {(6, 8), (12, 14), (14, 16)},
            'fp': set(),
            'fn': set()
        }
        self.assertEqual(
            true_res,
            compare_entities(true_ners, predicted_ners, 'LOCATION')
        )

    def test_compare_entities_4(self):
        true_ners = [('EVENT', 5, 6), ('LOCATION', 6, 8), ('COUNTRY', 13, 14),
                     ('STATE_OR_PROVINCE', 15, 16), ('LOCATION', 12, 14),
                     ('LOCATION', 14, 16)]
        predicted_ners = [('EVENT', 5, 6), ('COUNTRY', 13, 14),
                          ('STATE_OR_PROVINCE', 15, 16), ('LOCATION', 12, 14),
                          ('LOCATION', 14, 16)]
        true_res = {
            'tp': {(12, 14), (14, 16)},
            'fp': set(),
            'fn': {(6, 8)}
        }
        self.assertEqual(
            true_res,
            compare_entities(true_ners, predicted_ners, 'LOCATION')
        )

    def test_compare_entities_5(self):
        true_ners = [('EVENT', 5, 6), ('LOCATION', 6, 8), ('COUNTRY', 13, 14),
                     ('STATE_OR_PROVINCE', 15, 16), ('LOCATION', 12, 14),
                     ('LOCATION', 14, 16)]
        predicted_ners = [('EVENT', 5, 6), ('LOCATION', 2, 3),
                          ('COUNTRY', 13, 14), ('STATE_OR_PROVINCE', 15, 16),
                          ('LOCATION', 12, 14), ('LOCATION', 14, 16)]
        true_res = {
            'tp': {(12, 14), (14, 16)},
            'fp': {(2, 3)},
            'fn': {(6, 8)}
        }
        self.assertEqual(
            true_res,
            compare_entities(true_ners, predicted_ners, 'LOCATION')
        )

    def test_compare_entities_6(self):
        true_ners = [('EVENT', 5, 6), ('LOCATION', 6, 8), ('COUNTRY', 13, 14),
                     ('STATE_OR_PROVINCE', 15, 16), ('LOCATION', 12, 14),
                     ('LOCATION', 14, 16)]
        predicted_ners = [('EVENT', 5, 6), ('LOCATION', 2, 3),
                          ('LOCATION', 6, 8), ('COUNTRY', 13, 14),
                          ('STATE_OR_PROVINCE', 15, 16), ('LOCATION', 12, 14),
                          ('LOCATION', 14, 16)]
        true_res = {
            'tp': {(6, 8), (12, 14), (14, 16)},
            'fp': {(2, 3)},
            'fn': set()
        }
        self.assertEqual(
            true_res,
            compare_entities(true_ners, predicted_ners, 'LOCATION')
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
