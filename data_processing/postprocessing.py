from typing import List, Tuple

import numpy as np


CLASSES = [
    'O',
    'START',
    'END',
    'MIDDLE',
    'START-END'
]

DP_MATRIX_FOR_INIT_SUBWORD = np.array(
    [
        [1/2, 1/2,   0,   0,   0],  # O -> O START
        [  0,   0,   0,   1,   0],  # START -> MIDDLE
        [1/2, 1/2,   0,   0,   0],  # END -> O START
        [  0,   0,   0,   1,   0],  # MIDDLE -> MIDDLE
        [1/2, 1/2,   0,   0,   0]  # START-END -> O START
    ],
    np.float64
)

DP_MATRIX_FOR_FIN_SUBWORD = np.array(
    [
        [  1,   0,   0,   0,   0],  # O -> O
        [  0,   0, 1/2, 1/2,   0],  # START -> END MIDDLE
        [  0,   0,   0,   0,   0],  # END -> no possible
        [  0,   0, 1/2, 1/2,   0],  # MIDDLE -> END MIDDLE
        [  0,   0,   0,   0,   0]  # START-END -> no possible
    ],
    np.float64
)

DP_MATRIX_FOR_MIDDLE_SUBWORD = np.array(
    [
        [  1,   0,   0,   0,   0],  # O -> O
        [  0,   0,   0,   1,   0],  # START -> MIDDLE
        [  0,   0,   0,   0,   0],  # END -> no possible
        [  0,   0,   0,   1,   0],  # MIDDLE -> MIDDLE
        [  0,   0,   0,   0,   0]  # START-END -> no possible
    ],
    np.float64
)

DP_MATRIX_FOR_WORD = np.array(
    [
        [1/3, 1/3,   0,   0, 1/3],  # O -> O START START-END
        [  0,   0, 1/2, 1/2,   0],  # START -> END MIDDLE
        [1/3, 1/3,   0,   0, 1/3],  # END -> O START START-END
        [  0,   0, 1/2, 1/2,   0],  # MIDDLE -> END MIDDLE
        [1/3, 1/3,   0,   0, 1/3]  # START-END -> O START START-END
    ],
    np.float64
)


def do_viterbi_algorithm(aposteriori: np.ndarray,
                         apriori: List[np.ndarray],
                         time_idx: int, initial_state: int) -> List[int]:
    if time_idx > 0:
        state_list = do_viterbi_algorithm(aposteriori, apriori,
                                          time_idx - 1, initial_state)
        best_state = 0
        best_score = aposteriori[time_idx, best_state] * \
                     apriori[time_idx][state_list[-1], best_state]
        for cur_state in range(1, aposteriori.shape[1]):
            cur_score = aposteriori[time_idx, cur_state] * \
                        apriori[time_idx][state_list[-1], cur_state]
            if cur_score > best_score:
                best_score = cur_score
                best_state = cur_state
        state_list.append(best_state)
    else:
        best_state = 0
        best_score = aposteriori[0, best_state] * \
                     apriori[time_idx][initial_state, best_state]
        for cur_state in range(1, aposteriori.shape[1]):
            cur_score = aposteriori[0, cur_state] * \
                        apriori[time_idx][initial_state, cur_state]
            if cur_score > best_score:
                best_score = cur_score
                best_state = cur_state
        state_list = [best_state]
    return state_list


def decode_entity(proba_matrix: np.ndarray, words: List[Tuple[str, int, int]]) \
        -> List[Tuple[int, int]]:
    if len(proba_matrix.shape) != 2:
        err_msg = f'The probability matrix is wrong! ' \
                  f'Expected 2-D array, got {len(proba_matrix.shape)}-D one.'
        raise ValueError(err_msg)
    if proba_matrix.shape[1] != 5:
        err_msg = f'The probability matrix is wrong! ' \
                  f'Expected col number is 5, got {proba_matrix.shape[1]}.'
        raise ValueError(err_msg)
    for subtoken_idx in range(proba_matrix.shape[0]):
        min_proba = np.min(proba_matrix[subtoken_idx])
        max_proba = np.max(proba_matrix[subtoken_idx])
        proba_sum = np.sum(proba_matrix[subtoken_idx])
        err_msg = f'Row {subtoken_idx} of the probability matrix is not ' \
                  f'a probability distribution! ' \
                  f'{proba_matrix[subtoken_idx].tolist()}'
        if min_proba <= 0.0:
            raise ValueError(err_msg)
        if max_proba >= 1.0:
            raise ValueError(err_msg)
        if abs(proba_sum - 1.0) > 1e-2:
            raise ValueError(err_msg)
    prev_pos = -1
    for word_text, word_start, word_end in words:
        err_msg = f'The word {(word_text, word_start, word_end)} ' \
                  f'has wrong bounds!'
        if word_start < 0:
            raise ValueError(err_msg)
        if word_end > proba_matrix.shape[0]:
            raise ValueError(err_msg)
        if word_start >= word_end:
            raise ValueError(err_msg)
        if prev_pos >= 0:
            if prev_pos != word_start:
                raise ValueError(err_msg + f' {prev_pos} != {word_start}')
        prev_pos = word_end
    init_subtoken_idx = words[0][1]
    fin_subtoken_idx = words[-1][2]
    dp_matrices = []
    for _, word_start, word_end in words:
        subtoken_indices = list(range(word_start, word_end))
        if len(subtoken_indices) > 1:
            dp_matrices.append(DP_MATRIX_FOR_INIT_SUBWORD)
            if len(subtoken_indices) > 2:
                for _ in range(len(subtoken_indices) - 2):
                    dp_matrices.append(DP_MATRIX_FOR_MIDDLE_SUBWORD)
            dp_matrices.append(DP_MATRIX_FOR_FIN_SUBWORD)
        else:
            dp_matrices.append(DP_MATRIX_FOR_WORD)
    decoded_classes = do_viterbi_algorithm(
        aposteriori=proba_matrix[init_subtoken_idx:fin_subtoken_idx],
        apriori=dp_matrices,
        time_idx=fin_subtoken_idx - init_subtoken_idx - 1,
        initial_state=0
    )
    if init_subtoken_idx > 0:
        n = init_subtoken_idx
        decoded_classes = [0 for _ in range(n)] +  decoded_classes
    if fin_subtoken_idx < proba_matrix.shape[0]:
        n = proba_matrix.shape[0] - fin_subtoken_idx
        decoded_classes += [0 for _ in range(n)]
    start_pos = -1
    entity_bounds = []
    for idx, val in enumerate(decoded_classes):
        if val > 0:
            if start_pos < 0:
                start_pos = idx
        else:
            if start_pos >= 0:
                entity_bounds.append((start_pos, idx))
                start_pos = -1
    if start_pos >= 0:
        entity_bounds.append((start_pos, len(decoded_classes)))
    return entity_bounds
