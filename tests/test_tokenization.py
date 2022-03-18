import os
import sys
import unittest

from transformers import BertTokenizer

try:
    from data_processing.tokenization import tokenize_text
    from data_processing.tokenization import tokenize_text_with_ners
    from data_processing.tokenization import sentenize_text
    from data_processing.tokenization import find_quoted_substrings
    from data_processing.tokenization import remove_accents, find_substring
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from data_processing.tokenization import tokenize_text
    from data_processing.tokenization import tokenize_text_with_ners
    from data_processing.tokenization import sentenize_text
    from data_processing.tokenization import find_quoted_substrings
    from data_processing.tokenization import remove_accents, find_substring


class TestTokenization(unittest.TestCase):
    def setUp(self):
        bert_path = os.path.join(os.path.dirname(__file__), 'data', 'bert')
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        uncased_path = os.path.join(os.path.dirname(__file__), 'data',
                                         'bert-uncased')
        self.uncased_tokenizer = BertTokenizer.from_pretrained(uncased_path)

    def tearDown(self):
        del self.tokenizer
        del self.uncased_tokenizer

    def test_remove_accents_1(self):
        source_text = 'António Manuel de Oliveira Guterres'
        true_text = 'Antonio Manuel de Oliveira Guterres'
        self.assertEqual(remove_accents(source_text), true_text)

    def test_remove_accents_2(self):
        source_text = 'Ёжик бежал под ёлочкой.'
        true_text = 'Ежик бежал под елочкои.'
        self.assertEqual(remove_accents(source_text), true_text)

    def test_remove_accents_3(self):
        source_text = 'Их имена — Диа́с Кадырба́ев и Азама́т Тажая́ков.'
        true_text = 'Их имена — Диа`с Кадырба`ев и Азама`т Тажая`ков.'
        self.assertEqual(remove_accents(source_text), true_text)

    def test_tokenize_text_1(self):
        s = 'По предварительным данным, его отравили в аэропорту, ' \
            'когда он направлялся из Малайзии в Макао.'
        true_words = [
            ('По', 1, 2),
            ('предварительным', 2, 3),
            ('данным', 3, 4),
            (',', 4, 5),
            ('его', 5, 6),
            ('отравили', 6, 8),
            ('в', 8, 9),
            ('аэропорту', 9, 10),
            (',', 10, 11),
            ('когда', 11, 12),
            ('он', 12, 13),
            ('направлялся', 13, 14),
            ('из', 14, 15),
            ('Малайзии', 15, 16),
            ('в', 16, 17),
            ('Макао', 17, 18),
            ('.', 18, 19)
        ]
        true_subtokens = [
            '[CLS]',
            'По',
            'предварительным',
            'данным',
            ',',
            'его',
            'отрав',
            '##или',
            'в',
            'аэропорту',
            ',',
            'когда',
            'он',
            'направлялся',
            'из',
            'Малайзии',
            'в',
            'Макао',
            '.',
            '[SEP]'
        ]
        true_bounds = [
            None,
            (0, 2),
            (3, 18),
            (19, 25),
            (25, 26),
            (27, 30),
            (31, 36),
            (36, 39),
            (40, 41),
            (42, 51),
            (51, 52),
            (53, 58),
            (59, 61),
            (62, 73),
            (74, 76),
            (77, 85),
            (86, 87),
            (88, 93),
            (93, 94),
            None
        ]
        res = tokenize_text(s, self.tokenizer)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0], true_words)
        self.assertEqual(res[1], true_subtokens)
        self.assertEqual(res[2], true_bounds)

    def test_tokenize_text_2(self):
        s = ' В тексте знак «№» применяется только с относящимся к нему ' \
            'числом, от которого при наборе отделяется пробельным материалом ' \
            '(например,  № 11).'
        true_words = [
            ('В', 1, 2),
            ('тексте', 2, 3),
            ('знак', 3, 4),
            ('«', 4, 5),
            ('№', 5, 6),
            ('»', 6, 7),
            ('применяется', 7, 8),
            ('только', 8, 9),
            ('с', 9, 10),
            ('относящимся', 10, 11),
            ('к', 11, 12),
            ('нему', 12, 13),
            ('числом', 13, 14),
            (',', 14, 15),
            ('от', 15, 16),
            ('которого', 16, 17),
            ('при', 17, 18),
            ('наборе', 18, 19),
            ('отделяется', 19, 20),
            ('пробельным', 20, 23),
            ('материалом', 23, 24),
            ('(', 24, 25),
            ('например', 25, 26),
            (',', 26, 27),
            ('№', 27, 28),
            ('11', 28, 29),
            (')', 29, 30),
            ('.', 30, 31)
        ]
        true_subtokens = [
            '[CLS]',
            'В',
            'тексте',
            'знак',
            '«',
            '№',
            '»',
            'применяется',
            'только',
            'с',
            'относящимся',
            'к',
            'нему',
            'числом',
            ',',
            'от',
            'которого',
            'при',
            'наборе',
            'отделяется',
            'пробел',
            '##ь',
            '##ным',
            'материалом',
            '(',
            'например',
            ',',
            '№',
            '11',
            ')',
            '.',
            '[SEP]']
        true_bounds = [
            None,
            (0 + 1, 1 + 1),
            (2 + 1, 8 + 1),
            (9 + 1, 13 + 1),
            (14 + 1, 15 + 1),
            (15 + 1, 16 + 1),
            (16 + 1, 17 + 1),
            (18 + 1, 29 + 1),
            (30 + 1, 36 + 1),
            (37 + 1, 38 + 1),
            (39 + 1, 50 + 1),
            (51 + 1, 52 + 1),
            (53 + 1, 57 + 1),
            (58 + 1, 64 + 1),
            (64 + 1, 65 + 1),
            (66 + 1, 68 + 1),
            (69 + 1, 77 + 1),
            (78 + 1, 81 + 1),
            (82 + 1, 88 + 1),
            (89 + 1, 99 + 1),
            (100 + 1, 106 + 1),
            (106 + 1, 107 + 1),
            (107 + 1, 110 + 1),
            (111 + 1, 121 + 1),
            (122 + 1, 123 + 1),
            (123 + 1, 131 + 1),
            (131 + 1, 132 + 1),
            (133 + 2, 134 + 2),
            (135 + 2, 137 + 2),
            (137 + 2, 138 + 2),
            (138 + 2, 139 + 2),
            None
        ]
        res = tokenize_text(s, self.tokenizer)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0], true_words)
        self.assertEqual(res[1], true_subtokens)
        self.assertEqual(res[2], true_bounds)

    def test_tokenize_text_3(self):
        s = 'По предварительным данным, его отравили в аэропорту, ' \
            'когда он направлялся из Малайзии в Макао.'
        true_words = [
            ('По', 1, 2),
            ('предварительным', 2, 3),
            ('данным', 3, 4),
            (',', 4, 5),
            ('его', 5, 6),
            ('отравили', 6, 7),
            ('в', 7, 8),
            ('аэропорту', 8, 9),
            (',', 9, 10),
            ('когда', 10, 11),
            ('он', 11, 12),
            ('направлялся', 12, 13),
            ('из', 13, 14),
            ('Малайзии', 14, 17),
            ('в', 17, 18),
            ('Макао', 18, 20),
            ('.', 20, 21)
        ]
        true_subtokens = [
            '[CLS]',
            'по',
            'предварительным',
            'данным',
            ',',
            'его',
            'отравили',
            'в',
            'аэропорту',
            ',',
            'когда',
            'он',
            'направлялся',
            'из',
            'мала',
            '##из',
            '##ии',
            'в',
            'мака',
            '##о',
            '.',
            '[SEP]'
        ]
        true_bounds = [
            None,
            (0, 2),
            (3, 18),
            (19, 25),
            (25, 26),
            (27, 30),
            (31, 39),
            (40, 41),
            (42, 51),
            (51, 52),
            (53, 58),
            (59, 61),
            (62, 73),
            (74, 76),
            (77, 81),
            (81, 83),
            (83, 85),
            (86, 87),
            (88, 92),
            (92, 93),
            (93, 94),
            None
        ]
        res = tokenize_text(s, self.uncased_tokenizer)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0], true_words)
        self.assertEqual(res[1], true_subtokens)
        self.assertEqual(res[2], true_bounds)

    def test_tokenize_text_4(self):
        s = 'António Manuel de Oliveira Guterres'
        true_words = [
            ('António', 1, 4),
            ('Manuel', 4, 6),
            ('de', 6, 7),
            ('Oliveira', 7, 10),
            ('Guterres', 10, 13)
        ]
        true_subtokens = [
            '[CLS]',
            'ant',
            '##oni',
            '##o',
            'man',
            '##uel',
            'de',
            'ol',
            '##ive',
            '##ira',
            'gu',
            '##ter',
            '##res',
            '[SEP]'
        ]
        true_bounds = [
            None,
            (0, 3),
            (3, 6),
            (6, 7),
            (8, 11),
            (11, 14),
            (15, 17),
            (18, 20),
            (20, 23),
            (23, 26),
            (27, 29),
            (29, 32),
            (32, 35),
            None
        ]
        res = tokenize_text(s, self.uncased_tokenizer)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0], true_words)
        self.assertEqual(res[1], true_subtokens)
        self.assertEqual(res[2], true_bounds)

    def test_tokenize_text_5(self):
        s = 'Их имена — Диа́с Кадырба́ев и Азама́т Тажая́ков.'
        true_words = [
            ('Их', 1, 2),
            ('имена', 2, 3),
            ('—', 3, 4),
            ('Диа́с', 4, 6),
            ('Кадырба́ев', 6, 9),
            ('и', 9, 10),
            ('Азама́т', 10, 12),
            ('Тажая́ков', 12, 15),
            ('.', 15, 16)
        ]
        true_subtokens = [
            '[CLS]',
            'их',
            'имена',
            '—',
            'диа',
            '##с',
            'кады',
            '##рба',
            '##ев',
            'и',
            'аза',
            '##мат',
            'та',
            '##жая',
            '##ков',
            '.',
            '[SEP]'
        ]
        true_bounds = [
            None,
            (0, 2),
            (3, 8),
            (9, 10),
            (11, 15),
            (15, 16),
            (17, 21),
            (21, 25),
            (25, 27),
            (28, 29),
            (30, 33),
            (33, 37),
            (38, 40),
            (40, 44),
            (44, 47),
            (47, 48),
            None
        ]
        res = tokenize_text(s, self.uncased_tokenizer)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0], true_words)
        self.assertEqual(res[1], true_subtokens)
        self.assertEqual(res[2], true_bounds)

    def test_tokenize_text_with_ners_1(self):
        s = 'По предварительным данным, его отравили в аэропорту, ' \
            'когда он направлялся из Малайзии в Макао.'
        ne_voc = ['EVENT', 'COUNTRY', 'PERSON', 'STATE_OR_PROVINCE',
                  'WORK_OF_ART']
        ners = [('EVENT', 31, 39), ('COUNTRY', 77, 85),
                ('STATE_OR_PROVINCE', 88, 93)]
        true_subtokens = [
            '[CLS]',
            'По',
            'предварительным',
            'данным',
            ',',
            'его',
            'отрав',
            '##или',
            'в',
            'аэропорту',
            ',',
            'когда',
            'он',
            'направлялся',
            'из',
            'Малайзии',
            'в',
            'Макао',
            '.',
            '[SEP]'
        ]
        true_indicators = [
            [0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        res = tokenize_text_with_ners(s, self.tokenizer, ners, ne_voc)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0], true_subtokens)
        self.assertEqual(res[1], true_indicators)

    def test_tokenize_text_with_ners_2(self):
        s = 'По предварительным данным, его отравили в аэропорту, ' \
            'когда он направлялся из Малайзии в Макао.'
        ne_voc = ['EVENT', 'COUNTRY', 'LOCATION', 'PERSON', 'STATE_OR_PROVINCE',
                  'WORK_OF_ART']
        ners = [('EVENT', 31, 39), ('LOCATION', 40, 51), ('COUNTRY', 77, 85),
                ('STATE_OR_PROVINCE', 88, 93), ('LOCATION', 74, 85),
                ('LOCATION', 86, 93)]
        true_subtokens = [
            '[CLS]',
            'По',
            'предварительным',
            'данным',
            ',',
            'его',
            'отрав',
            '##или',
            'в',
            'аэропорту',
            ',',
            'когда',
            'он',
            'направлялся',
            'из',
            'Малайзии',
            'в',
            'Макао',
            '.',
            '[SEP]'
        ]
        true_indicators = [
            [0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 2, 1, 2, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        res = tokenize_text_with_ners(s, self.tokenizer, ners, ne_voc)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0], true_subtokens)
        self.assertEqual(res[1], true_indicators)

    def test_tokenize_text_with_ners_3(self):
        s = 'Samsung и Nokia будут платить «налог на болванки».'
        ne_voc = ['EVENT', 'LOCATION', 'ORGANIZATION']
        ners = [('ORGANIZATION', 0, 9), ('ORGANIZATION', 10, 15)]
        true_subtokens = [
            '[CLS]',
            'Samsung',
            'и',
            'Nokia',
            'будут',
            'платить',
            '«',
            'налог',
            'на',
            'бол',
            '##ван',
            '##ки',
            '»',
            '.',
            '[SEP]'
        ]
        true_indicators = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        res = tokenize_text_with_ners(s, self.tokenizer, ners, ne_voc)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0], true_subtokens)
        self.assertEqual(res[1], true_indicators)

    def test_tokenize_text_with_ners_4(self):
        s = 'Отныне Samsung и Nokia будут платить «налог на болванки».'
        ne_voc = ['EVENT', 'LOCATION', 'ORGANIZATION']
        ners = [('ORGANIZATION', 7, 14), ('ORGANIZATION', 15, 22)]
        true_subtokens = [
            '[CLS]',
            'Отныне',
            'Samsung',
            'и',
            'Nokia',
            'будут',
            'платить',
            '«',
            'налог',
            'на',
            'бол',
            '##ван',
            '##ки',
            '»',
            '.',
            '[SEP]'
        ]
        true_indicators = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        res = tokenize_text_with_ners(s, self.tokenizer, ners, ne_voc)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0], true_subtokens)
        self.assertEqual(res[1], true_indicators)

    def test_sentenize_text_1(self):
        s = 'Мама мыла раму. Папа мыл синхрофазотрон.  И.И. Петров пинал балду.'
        true_bounds = [
            (0, 15),
            (16, 40),
            (42, 66)
        ]
        self.assertEqual(sentenize_text(s), true_bounds)

    def test_sentenize_text_2(self):
        s = 'Мама мыла раму. Папа мыл синхрофазотрон\n\r\n' \
            'И.И. Петров пинал балду.'
        true_bounds = [
            (0, 15),
            (16, 39),
            (42, 66)
        ]
        self.assertEqual(sentenize_text(s), true_bounds)

    def test_sentenize_text_3(self):
        s = 'Между тем генеральный менеджер «Блюз» Даг Армстронг заявил, ' \
            'что, когда новый игрок прибывает в НХЛ, никто ему не дает ' \
            'никаких гарантий, поскольку «гарантировать то, что ты не ' \
            'сможешь потом выполнить – нельзя». «Мы дали ему четко понять, ' \
            'что ему придется заработать место в основной команде», – ' \
            'цитирует слова Армстронга главная газета Сент-Луиса St. Louis ' \
            'Post-Dispatch.'
        true_bounds = [
            (0, 209),
            (210, 370)
        ]
        self.assertEqual(sentenize_text(s), true_bounds)

    def test_sentenize_text_4(self):
        s = 'Мама мыла раму. Папа мыл\n\r\nсинхрофазотрон'
        true_bounds = [
            (0, 15),
            (16, 41)
        ]
        self.assertEqual(sentenize_text(s), true_bounds)

    def test_sentenize_text_5(self):
        s = 'Мама мыла раму. Папа мыл.\n\r\nсинхрофазотрон'
        true_bounds = [
            (0, 15),
            (16, 25),
            (28, 42)
        ]
        self.assertEqual(sentenize_text(s), true_bounds)

    def test_sentenize_text_6(self):
        s = 'Конца правительственному кризису в Чехии не видно. Связано это ' \
            'с тем, что над его разрешением каждая в своих интересах ' \
            'работают три силы: правительственная правоконсервативная ' \
            'коалиция с ведущей двойкой партий - Гражданская демократическая ' \
            'партия (ODS), \"Традиция. Ответственность. Процветание 09\" ' \
            '(TOP 09); левая оппозиция - Чешская социал-демократическая ' \
            'партия (CSSD) и Коммунистическая партия Чехии и Моравии (KSCM); ' \
            'новый президент Чехии Милош Земан.(1)\n\nПричина затягивания ' \
            'кризиса - президент.'
        true_bounds = [
            (0, 50),
            (51, 458),
            (460, 500)
        ]
        self.assertEqual(sentenize_text(s), true_bounds)

    def test_find_quoted_substrings_1(self):
        s = 'Мама мыла раму.'
        true_spans = []
        self.assertEqual(find_quoted_substrings(s), true_spans)

    def test_find_quoted_substrings_2(self):
        s = 'Мама "мыла раму".'
        true_spans = [(5, 16)]
        self.assertEqual(find_quoted_substrings(s), true_spans)

    def test_find_quoted_substrings_3(self):
        s = 'Мама «мыла раму».'
        true_spans = [(5, 16)]
        self.assertEqual(find_quoted_substrings(s), true_spans)

    def test_find_quoted_substrings_4(self):
        s = 'Мама »мыла раму«.'
        true_spans = []
        self.assertEqual(find_quoted_substrings(s), true_spans)

    def test_find_quoted_substrings_5(self):
        s = 'Мама "мыла раму.'
        true_spans = []
        self.assertEqual(find_quoted_substrings(s), true_spans)

    def test_find_quoted_substrings_6(self):
        s = 'Мама "мыла" раму".'
        true_spans = [(5, 11)]
        self.assertEqual(find_quoted_substrings(s), true_spans)

    def test_find_quoted_substrings_7(self):
        s = 'Мама «мыла» раму».'
        true_spans = [(5, 11)]
        self.assertEqual(find_quoted_substrings(s), true_spans)

    def test_find_quoted_substrings_8(self):
        s = 'Мама «мыла «раму».'
        true_spans = [(5, 17)]
        self.assertEqual(find_quoted_substrings(s), true_spans)

    def test_find_substring_1(self):
        s = 'Мама мыла раму'
        substring = 'Мама'
        true_bounds = (0, 4)
        self.assertEqual(find_substring(s, substring), true_bounds)

    def test_find_substring_2(self):
        s = 'Мама мыла раму'
        substring = 'Папа'
        true_bounds = (-1, -1)
        self.assertEqual(find_substring(s, substring), true_bounds)

    def test_find_substring_3(self):
        s = 'Ма`ма мыла раму'
        substring = 'Мама'
        true_bounds = (0, 5)
        self.assertEqual(find_substring(s, substring), true_bounds)

    def test_find_substring_4(self):
        s = '`Мама мыла раму'
        substring = 'Мама'
        true_bounds = (0, 5)
        self.assertEqual(find_substring(s, substring), true_bounds)

    def test_find_substring_5(self):
        s = 'Ма`ма` мыла раму'
        substring = 'Мама'
        true_bounds = (0, 6)
        self.assertEqual(find_substring(s, substring), true_bounds)

    def test_find_substring_6(self):
        s = ' Ма`м`а` мыла раму'
        substring = 'Мама'
        true_bounds = (1, 8)
        self.assertEqual(find_substring(s, substring), true_bounds)

    def test_find_substring_7(self):
        s = 'Мама мыла раму'
        substring = ''
        true_bounds = (-1, -1)
        self.assertEqual(find_substring(s, substring), true_bounds)

    def test_find_substring_8(self):
        s = 'Мама мыла раму'
        substring = 'Ма`ма'
        with self.assertRaises(ValueError):
            _ = find_substring(s, substring)

    def test_find_substring_9(self):
        s = 'Мама мыла раму'
        substring = ' Мама'
        with self.assertRaises(ValueError):
            _ = find_substring(s, substring)


if __name__ == '__main__':
    unittest.main(verbosity=2)
