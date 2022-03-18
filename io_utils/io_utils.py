import codecs
import json
from typing import Dict, List, Tuple


def parse_string(s: str) -> Tuple[int, str, List[Tuple[str, int, int]]]:
    data = json.loads(s)
    if 'id' not in data:
        err_msg = f'The "id" key is not found in the string "{s}".'
        raise ValueError(err_msg)
    if 'sentences' not in data:
        err_msg = f'The "sentences" key is not found in the string "{s}".'
        raise ValueError(err_msg)
    identifier = data['id']
    text = data['sentences']
    ners = []
    if 'ners' in data:
        if len(text.strip()) == 0:
            err_msg = f'The named entities are specified incorrectly ' \
                      f'in the string "{s}".'
            raise ValueError(err_msg)
        for idx, named_entity_info in enumerate(data['ners']):
            if not isinstance(named_entity_info, list):
                err_msg = f'Named entity {idx} is specified incorrectly ' \
                          f'in the string "{s}".'
                raise ValueError(err_msg)
            if len(named_entity_info) != 3:
                err_msg = f'Named entity {idx} is specified incorrectly ' \
                          f'in the string "{s}".'
                raise ValueError(err_msg)
            if not isinstance(named_entity_info[0], int):
                err_msg = f'Named entity {idx} is specified incorrectly ' \
                          f'in the string "{s}".'
                raise ValueError(err_msg)
            if not isinstance(named_entity_info[1], int):
                err_msg = f'Named entity {idx} is specified incorrectly ' \
                          f'in the string "{s}".'
                raise ValueError(err_msg)
            if not isinstance(named_entity_info[2], str):
                err_msg = f'Named entity {idx} is specified incorrectly ' \
                          f'in the string "{s}".'
                raise ValueError(err_msg)
            if named_entity_info[0] > named_entity_info[1]:
                err_msg = f'Named entity {idx} is specified incorrectly ' \
                          f'in the string "{s}".'
                raise ValueError(err_msg)
            if named_entity_info[0] < 0:
                err_msg = f'Named entity {idx} is specified incorrectly ' \
                          f'in the string "{s}".'
                raise ValueError(err_msg)
            if named_entity_info[1] >= len(text):
                err_msg = f'Named entity {idx} is specified incorrectly ' \
                          f'in the string "{s}".'
                raise ValueError(err_msg)
            start_pos = named_entity_info[0]
            end_pos = named_entity_info[1] + 1
            if text[start_pos].isspace():
                err_msg = f'Named entity {idx} is specified incorrectly ' \
                          f'in the string "{s}".'
                raise ValueError(err_msg)
            if text[end_pos - 1].isspace():
                err_msg = f'Named entity {idx} is specified incorrectly ' \
                          f'in the string "{s}".'
                raise ValueError(err_msg)
            ners.append((named_entity_info[2], start_pos, end_pos))
    return identifier, text, ners


def load_data(fname: str) -> Dict[int, Tuple[str, List[Tuple[str, int, int]]]]:
    texts_and_annotations = dict()
    with codecs.open(fname, mode='r', encoding='utf-8') as fp:
        cur_line = fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                identifier, text, ners = parse_string(prep_line)
                if identifier in texts_and_annotations:
                    err_msg = f'Identifier {identifier} is duplicated!'
                    raise ValueError(err_msg)
                ners = sorted(
                    list(set(ners)),
                    key=lambda it: (it[1], it[2], it[0])
                )
                texts_and_annotations[identifier] = (text, ners)
            cur_line = fp.readline()
    return texts_and_annotations


def save_data(fname: str, with_text: bool,
              data: Dict[int, Tuple[str, List[Tuple[str, int, int]]]]):
    with codecs.open(fname, mode='w', encoding='utf-8') as fp:
        for identifier in sorted(list(data.keys())):
            text, ners = data[identifier]
            sample = {
                'id': identifier
            }
            if with_text:
                sample['sentences'] = text
            sample['ners'] = list(map(
                lambda it: [it[1], it[2] - 1, it[0]],
                ners
            ))
            fp.write(json.dumps(obj=sample, ensure_ascii=False) + '\n')
