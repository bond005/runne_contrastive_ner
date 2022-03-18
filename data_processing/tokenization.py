from typing import List, Tuple, Union
import unicodedata as ud
import warnings

from razdel import tokenize, sentenize
from transformers import BertTokenizer


SENTENIZE_EXCLUSIONS = [
    'st.',
    'св.',
    'г.',
    'с.',
    'ftf.',
    'e.g.',
    'g.',
    'i.e.',
    'e.',
    'vs.',
    'jr.',
    'sr.',
    'smth.',
    'smb.',
    'vip.',
    'v.i.p.',
    'p.',
    'n.',
    'v.',
    'pp.',
    'par.',
    'ex.',
    'pl.',
    'sing.',
    's.',
    'p.s.',
    'p.p.s.',
    're.',
    'rf.',
    'edu.',
    'appx.',
    'in.',
    'sec.',
    'gm.',
    'cm.',
    'qt.',
    'mph.',
    'kph.',
    'ft.',
    'lb.',
    'oz.',
    'pt.',
    'yr.',
    'div.',
    'род.',
    'рожд.'
]


def remove_accents(s: str) -> str:
    res = []
    for c in s:
        norm = ud.normalize('NFKD', c)
        if len(norm) > 1:
            new_res = list(filter(lambda it: ud.combining(it) == 0, norm))
            if len(new_res) == 0:
                res.append('`')
            else:
                res.append(new_res[0])
        elif len(norm) == 1:
            if ud.combining(norm) == 0:
                res.append(norm)
            else:
                res.append('`')
        else:
            res.append('`')
    return "".join(res)


def find_substring(s: str, substring: str) -> Tuple[int, int]:
    if '`' in substring:
        err_msg = f'"{substring}" is a wrong sub-word, because ' \
                  f'it contains "`". It cannot be found in the string "{s}".'
        raise ValueError(err_msg)
    if substring != substring.strip():
        err_msg = f'"{substring}" is a wrong sub-word, because ' \
                  f'it includes initial and/or final spaces.'
        raise ValueError(err_msg)
    if len(substring) == 0:
        return -1, -1
    if '`' not in s:
        start_pos = s.find(substring)
        if start_pos < 0:
            return -1, -1
        return start_pos, start_pos + len(substring)
    found_idx = s.find(substring[0])
    if found_idx < 0:
        return -1, -1
    idx1 = found_idx + 1
    if found_idx > 0:
        while found_idx > 0:
            if s[found_idx - 1] != '`':
                break
            found_idx -= 1
    for idx2 in range(1, len(substring)):
        while idx1 < len(s):
            if s[idx1] != '`':
                break
            idx1 += 1
        if idx1 >= len(s):
            break
        if s[idx1] != substring[idx2]:
            break
        idx1 += 1
    if s[found_idx:idx1].replace('`', '') != substring:
        return -1, -1
    while idx1 < len(s):
        if s[idx1] != '`':
            break
        idx1 += 1
    return found_idx, idx1


def tokenize_text(s: str, tokenizer: BertTokenizer) \
        -> Tuple[List[Tuple[str, int, int]], List[str],
                 List[Union[None, Tuple[int, int]]]]:
    words: List[Tuple[str, int, int]] = []
    subtokens: List[str] = []
    subtoken_bounds: List[Union[None, Tuple[int, int]]] = []
    subtokens.append(tokenizer.cls_token)
    subtoken_bounds.append(None)
    n_bpe = 1
    tokenization_iterator = filter(
        lambda it2: len(s[it2[0]:it2[1]].strip()) > 0,
        map(
            lambda it1: (tuple(it1)[0], tuple(it1)[1]),
            tokenize(s.replace('​', ' '))
        )
    )
    word_bounds = []
    punctuation = {',', '-', ':', ';', '.', ')', '(', '\]', '[', '<', '>',
                   '=', '+', '?', '!'}
    for start_word_pos, end_word_pos in tokenization_iterator:
        cur_word = s[start_word_pos:end_word_pos]
        if len(cur_word.strip()) > 0:
            wordpart_start = -1
            for char_idx, char_val in enumerate(cur_word):
                if char_val in punctuation:
                    if wordpart_start >= 0:
                        word_bounds.append((
                            start_word_pos + wordpart_start,
                            start_word_pos + char_idx
                        ))
                        wordpart_start = -1
                    word_bounds.append((
                        start_word_pos + char_idx,
                        start_word_pos + char_idx + 1
                    ))
                else:
                    if wordpart_start < 0:
                        wordpart_start = char_idx
            if wordpart_start >= 0:
                word_bounds.append((
                    start_word_pos + wordpart_start,
                    start_word_pos + len(cur_word)
                ))
    for start_word_pos, end_word_pos in word_bounds:
        cur_word = s[start_word_pos:end_word_pos]
        bpe = tokenizer.tokenize(cur_word)
        if len(bpe) == 0:
            err_msg = f'The word "{cur_word}" cannot be tokenized!'
            raise ValueError(err_msg)
        if tokenizer.unk_token in bpe:
            subtokens.append(tokenizer.unk_token)
            subtoken_bounds.append((start_word_pos, end_word_pos))
            words.append((cur_word, n_bpe, n_bpe + 1))
            n_bpe += 1
        elif len(bpe) > 1:
            prep_word = remove_accents(cur_word.lower())
            subword_start_pos = 0
            for src in bpe:
                if src.startswith('##'):
                    prep = src[2:]
                else:
                    prep = src
                prep = remove_accents(prep.lower()).replace('`', '')
                found_start, found_end = find_substring(
                    s=prep_word[subword_start_pos:],
                    substring=prep
                )
                if (found_start < 0) or (found_end < 0):
                    err_msg = f'The text {s} cannot be tokenized! "{prep}" is ' \
                              f'not found in the "{prep_word}" from ' \
                              f'{subword_start_pos}. Subwords are: {bpe}'
                    raise ValueError(err_msg)
                subword_start_pos += found_start
                subword_end_pos = subword_start_pos + (found_end - found_start)
                subtokens.append(src)
                subtoken_bounds.append((
                    start_word_pos + subword_start_pos,
                    start_word_pos + subword_end_pos
                ))
                subword_start_pos = subword_end_pos
            if (subtoken_bounds[-1][1] - start_word_pos) < len(prep_word):
                subtoken_bounds[-1] = (
                    subtoken_bounds[-1][0],
                    len(prep_word) + start_word_pos
                )
            words.append((cur_word, n_bpe, n_bpe + len(bpe)))
            n_bpe += len(bpe)
        else:
            subtokens.append(bpe[0])
            subtoken_bounds.append((start_word_pos, end_word_pos))
            words.append((cur_word, n_bpe, n_bpe + 1))
            n_bpe += 1
    subtokens.append(tokenizer.sep_token)
    subtoken_bounds.append(None)
    return words, subtokens, subtoken_bounds


def is_exclusion(s: str) -> bool:
    prep = s.lower()
    ok = False
    for cur_exclusion in SENTENIZE_EXCLUSIONS:
        found_idx = prep.rfind(cur_exclusion)
        if found_idx >= 0:
            if prep[found_idx:] == cur_exclusion:
                if found_idx > 0:
                    ok = (not prep[found_idx - 1].isalnum())
                else:
                    ok = True
        if ok:
            break
    return ok


def find_span(spans: List[Tuple[int, int]], char_position: int) -> int:
    found_idx = -1
    for span_idx, (span_start, span_end) in enumerate(spans):
        if (char_position >= span_start) and (char_position < span_end):
            found_idx = span_idx
            break
    return found_idx


def sentenize_with_exclusions(s: str) -> List[Tuple[int, int]]:
    sentence_bounds = list(map(
        lambda it: (tuple(it)[0], tuple(it)[1]),
        sentenize(s)
    ))
    if len(sentence_bounds) == 0:
        return sentence_bounds
    prepared_sentence_bounds = [sentence_bounds[0]]
    prev_text = s[sentence_bounds[0][0]:sentence_bounds[0][1]].lower()
    for cur_bounds in sentence_bounds[1:]:
        if is_exclusion(prev_text):
            prepared_sentence_bounds[-1] = (
                prepared_sentence_bounds[-1][0],
                cur_bounds[1]
            )
        else:
            prepared_sentence_bounds.append(cur_bounds)
        prev_text = s[cur_bounds[0]:cur_bounds[1]].lower()
    quote_spans = find_quoted_substrings(s)
    if len(quote_spans) == 0:
        return prepared_sentence_bounds
    for quote_span_start, quote_span_end in quote_spans:
        first_sent_idx = find_span(prepared_sentence_bounds, quote_span_start)
        last_sent_idx = find_span(prepared_sentence_bounds, quote_span_end - 1)
        if (first_sent_idx < 0) or (last_sent_idx < 0):
            raise ValueError(f'The sentence "{s}" has incorrect spans!')
        if first_sent_idx < last_sent_idx:
            prepared_sentence_bounds[first_sent_idx] = (
                prepared_sentence_bounds[first_sent_idx][0],
                prepared_sentence_bounds[last_sent_idx][1]
            )
            prepared_sentence_bounds = \
                prepared_sentence_bounds[:(first_sent_idx + 1)] + \
                prepared_sentence_bounds[(last_sent_idx + 1):]
    return prepared_sentence_bounds


def find_quoted_substrings(s: str) -> List[Tuple[int, int]]:
    span_start = -1
    spans = []
    for char_idx, char_val in enumerate(s):
        if char_val in {'"', '\''}:
            if span_start < 0:
                span_start = char_idx
            else:
                span_end = char_idx + 1
                spans.append((span_start, span_end))
                span_start = -1
        elif char_val == '«':
            if span_start < 0:
                span_start = char_idx
        elif char_val == '»':
            if span_start >= 0:
                span_end = char_idx + 1
                spans.append((span_start, span_end))
                span_start = -1
    return spans


def sentenize_text(s: str) -> List[Tuple[int, int]]:
    sent_start = -1
    sentence_bounds = []
    newline_counter = 0
    last_char = ''
    for char_idx, char_val in enumerate(s):
        if char_val in {'\n', '\r'}:
            newline_counter += 1
        else:
            if not char_val.isspace():
                if sent_start < 0:
                    sent_start = char_idx
                else:
                    if newline_counter > 0:
                        if last_char in {'?', '!'}:
                            sent_end = char_idx
                        elif char_val.istitle() or (last_char == '.'):
                            sent_end = char_idx
                        else:
                            sent_end = -1
                        if sent_end >= 0:
                            while sent_end > sent_start:
                                if not s[sent_end - 1].isspace():
                                    break
                                sent_end -= 1
                            if sent_end > sent_start:
                                text = s[sent_start:sent_end].replace('​', ' ')
                                if len(text.strip()) > 0:
                                    for it in sentenize_with_exclusions(text):
                                        sentence_bounds.append((
                                            sent_start + it[0],
                                            sent_start + it[1]
                                        ))
                            sent_start = char_idx
                        newline_counter = 0
                last_char = char_val
    if sent_start >= 0:
        sent_end = len(s)
        while sent_end > sent_start:
            if not s[sent_end - 1].isspace():
                break
            sent_end -= 1
        if sent_end > sent_start:
            text = s[sent_start:sent_end].replace('​', ' ')
            if len(text.strip()) > 0:
                for it in sentenize_with_exclusions(text):
                    sentence_bounds.append((
                        sent_start + it[0],
                        sent_start + it[1]
                    ))
    return sentence_bounds


def tokenize_text_with_ners(s: str, tokenizer: BertTokenizer,
                            ners: List[Tuple[str, int, int]],
                            ne_vocabulary: List[str]) \
        -> Tuple[List[str], List[List[int]]]:
    words, subtokens, subtoken_bounds = tokenize_text(s, tokenizer)
    word_bounds = []
    for cur_word, word_start, word_end in words:
        word_bounds.append((
            subtoken_bounds[word_start][0],
            subtoken_bounds[word_end - 1][1]
        ))
    ne_indicators = []
    for _ in range(len(ne_vocabulary)):
        ne_indicators.append([0 for _ in range(len(subtokens))])
    ne_set = set(map(lambda it: it[0], ners))
    if len(ne_set) == 0:
        return subtokens, ne_indicators
    diff_ne = ne_set - set(ne_vocabulary)
    if len(diff_ne) > 0:
        err_msg = f'The annotation {ners} is wrong because ' \
                  f'it contains unknown entities! ' \
                  f'They are: {sorted(list(diff_ne))}.'
        raise ValueError(err_msg)
    for ne_class, ne_start, ne_end in ners:
        ne_id = ne_vocabulary.index(ne_class)
        start_word_idx = -1
        for word_idx, (word_start, word_end) in enumerate(word_bounds):
            if (ne_start >= word_start) and (ne_start < word_end):
                if ne_start != word_start:
                    warn_msg = f'The annotation {ners} can have errors. ' \
                               f'The entity {(ne_class, ne_start, ne_end)} ' \
                               f'is inexactly found in the text "{s}". ' \
                               f'{ne_start} != {word_start}'
                    warnings.warn(warn_msg)
                start_word_idx = word_idx
                break
        if start_word_idx < 0:
            err_msg = f'The annotation {ners} is wrong. ' \
                      f'The entity {(ne_class, ne_start, ne_end)} ' \
                      f'is not found in the text {s}.'
            raise ValueError(err_msg)
        end_word_idx = -1
        for word_idx, (word_start, word_end) in enumerate(word_bounds):
            if (ne_end > word_start) and (ne_end <= word_end):
                if ne_end != word_end:
                    warn_msg = f'The annotation {ners} can have errors. ' \
                               f'The entity {(ne_class, ne_start, ne_end)} ' \
                               f'is inexactly found in the text "{s}". ' \
                               f'{ne_end} != {word_end}'
                    warnings.warn(warn_msg)
                end_word_idx = word_idx
                break
        if end_word_idx < 0:
            err_msg = f'The annotation {ners} is wrong. ' \
                      f'The entity {(ne_class, ne_start, ne_end)} ' \
                      f'is not found in the text {s}.'
            raise ValueError(err_msg)
        init_ne_subtoken = words[start_word_idx][1]
        fin_ne_subtoken = words[end_word_idx][2]
        for subtoken_idx in range(init_ne_subtoken, fin_ne_subtoken):
            ne_indicators[ne_id][subtoken_idx] = 1
        ne_indicators[ne_id][init_ne_subtoken] = 2
    return subtokens, ne_indicators


def sentenize_text_with_ners(s: str, tokenizer: BertTokenizer,
                             ners: List[Tuple[str, int, int]],
                             ne_vocabulary: List[str]) \
        -> List[Tuple[List[str], List[List[int]]]]:
    if len(ners) != len(set(ners)):
        raise ValueError('Some entities are duplicated!')
    sentence_bounds = sentenize_text(s)
    res = []
    used_entities = set()
    for sent_start, sent_end in sentence_bounds:
        ners_for_sent = []
        for ne_type, ne_start, ne_end in ners:
            if ne_end <= sent_start:
                continue
            if ne_start >= sent_end:
                continue
            if (ne_start < sent_start) or (ne_end > sent_end):
                err_msg = f'The entity ({ne_type}, {ne_start}, {ne_end}) ' \
                          f'is wrong! It is located in more than ' \
                          f'a single sentence. More probably sentence is ' \
                          f'"{s[sent_start:sent_end]}"'
                raise ValueError(err_msg)
            if (ne_type, ne_start, ne_end) in used_entities:
                err_msg = f'The entity ({ne_type}, {ne_start}, {ne_end}) ' \
                          f'is wrong! It is located in more than ' \
                          f'a single sentence. More probably sentence is ' \
                          f'"{s[sent_start:sent_end]}"'
                raise ValueError(err_msg)
            ners_for_sent.append(
                (
                    ne_type,
                    ne_start - sent_start,
                    ne_end - sent_start
                )
            )
            used_entities.add((ne_type, ne_start, ne_end))
        res.append(tokenize_text_with_ners(s[sent_start:sent_end], tokenizer,
                                           ners_for_sent, ne_vocabulary))
    if len(used_entities) != len(ners):
        err_msg = f'Some entities are not used! They are: ' \
                  f'{sorted(list(set(ners) - used_entities))}'
        raise ValueError(err_msg)
    return res
