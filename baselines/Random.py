import numpy as np


def random_selection(candidate_set, select_size):
    all_index = np.arange(len(candidate_set))
    select_index = np.random.choice(all_index, select_size, replace=False)
    return select_index
