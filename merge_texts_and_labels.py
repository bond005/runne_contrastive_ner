import codecs
import copy
import json
import os
import sys


def main():
    if len(sys.argv) < 2:
        err_msg = 'The source text file is not specified!'
        raise ValueError(err_msg)
    src_text_fname = os.path.normpath(sys.argv[1])
    if not os.path.isfile(src_text_fname):
        err_msg = f'The file "{src_text_fname}" does not exist!'
        raise ValueError(err_msg)
    if len(sys.argv) < 3:
        err_msg = 'The source annotation file is not specified!'
        raise ValueError(err_msg)
    src_ann_fname = os.path.normpath(sys.argv[2])
    if not os.path.isfile(src_ann_fname):
        err_msg = f'The file "{src_ann_fname}" does not exist!'
        raise ValueError(err_msg)
    if len(sys.argv) < 4:
        err_msg = 'The destination file is not specified!'
        raise ValueError(err_msg)
    dst_fname = os.path.normpath(sys.argv[3])
    dst_dir = os.path.dirname(dst_fname)
    if not os.path.isdir(dst_dir):
        err_msg = f'The directory "{dst_dir}" does not exist!'
        raise ValueError(err_msg)

    annotated_data = dict()
    line_idx = 1
    with codecs.open(src_text_fname, mode='r', encoding='utf-8') as fp:
        cur_line = fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                err_msg = f'The file "{src_text_fname}": ' \
                          f'line {line_idx} is wrong!'
                cur_data = json.loads(prep_line)
                if not isinstance(cur_data, dict):
                    raise ValueError(err_msg)
                if ('id' not in cur_data) or ('sentences' not in cur_data):
                    raise ValueError(err_msg)
                if cur_data['id'] in annotated_data:
                    err_msg2 = err_msg + f' The identifier {cur_data["id"]} ' \
                                         f'is duplicated!'
                    raise ValueError(err_msg2)
                cur_id = cur_data['id']
                annotated_data[cur_id] = {'sentences': cur_data['sentences']}
            cur_line = fp.readline()
            line_idx += 1

    line_idx = 1
    known_IDs = set()
    with codecs.open(src_ann_fname, mode='r', encoding='utf-8') as fp:
        cur_line = fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                err_msg = f'The file "{src_ann_fname}": ' \
                          f'line {line_idx} is wrong!'
                cur_data = json.loads(prep_line)
                if not isinstance(cur_data, dict):
                    raise ValueError(err_msg)
                if ('id' not in cur_data) or ('ners' not in cur_data):
                    raise ValueError(err_msg)
                if cur_data['id'] in known_IDs:
                    err_msg2 = err_msg + f' The identifier {cur_data["id"]} ' \
                                         f'is duplicated!'
                    raise ValueError(err_msg2)
                if cur_data['id'] not in annotated_data:
                    err_msg2 = err_msg + f' The identifier {cur_data["id"]} ' \
                                         f'is unknown!'
                    raise ValueError(err_msg2)
                cur_id = cur_data['id']
                known_IDs.add(cur_id)
                annotated_data[cur_id]['ners'] = cur_data['ners']
            cur_line = fp.readline()
            line_idx += 1
    unused_IDs = set(annotated_data.keys()) - known_IDs
    if len(unused_IDs) > 0:
        unused_IDs = sorted(list(unused_IDs))
        err_msg = f'Some samples do not have any annotation! ' \
                  f'They are: {unused_IDs}'
        raise ValueError(err_msg)

    with codecs.open(dst_fname, mode='w', encoding='utf-8') as fp:
        for cur_id in sorted(list(annotated_data.keys())):
            data_sample = copy.copy(annotated_data[cur_id])
            data_sample['id'] = cur_id
            data_str = json.dumps(data_sample, ensure_ascii=False)
            del data_sample
            fp.write(data_str + '\n')


if __name__ == '__main__':
    main()
