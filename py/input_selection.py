import numpy as np
from py.distance_metrics import measure_distance

def shuffle(data):
    s = np.arange(data.shape[0])
    np.random.shuffle(s)
    return data[s]

def shuffle_idx(data):
    s = np.arange(data.shape[0])
    np.random.shuffle(s)
    return s

def select_next_inputs(bug_fix_data,
                       bug_fix_labels,
                       heatmaps,
                       target_label,
                       batch_size,
                       ratio,
                       for_underfitting=True,
                       distance_metric='dot',
                       **kwargs):

    num_target_samples = int(batch_size * ratio)
    num_rand_samples = batch_size - num_target_samples

    # #################### #
    # get targeted samples #
    # #################### #
    target_label_idx = np.ravel(np.argwhere(bug_fix_labels == target_label))
    heatmap_for_label = heatmaps[target_label]

    # get samples from bug fixing pool
    candidate_samples_x = bug_fix_data[target_label_idx]
    candidate_samples_y = bug_fix_labels[target_label_idx]

    # remove candidates from bug fixing pool
    bug_fix_data = np.delete(bug_fix_data, target_label_idx, axis=0)
    bug_fix_labels = np.delete(bug_fix_labels, target_label_idx, axis=0)

    # compare distances
    sorted_scores = measure_distance(candidate_samples_x, heatmap_for_label, distance_metric, **kwargs)
  
    # select targets
    if for_underfitting:  
        # these should look the most similar to the heatmaps
        targets_x = candidate_samples_x[sorted_scores[-num_target_samples:]].reshape(-1, 28, 28) 
        targets_y = candidate_samples_y[sorted_scores[-num_target_samples:]]
        remainder_x = candidate_samples_x[sorted_scores[:-num_target_samples]].reshape(-1, 28, 28) 
        remainder_y = candidate_samples_y[sorted_scores[:-num_target_samples]]
    else: #for_overfitting        
        # these should look the least similar to the heatmaps
        targets_x = candidate_samples_x[sorted_scores[:num_target_samples]].reshape(-1, 28, 28) 
        targets_y = candidate_samples_y[sorted_scores[:num_target_samples]]
        remainder_x = candidate_samples_x[sorted_scores[num_target_samples:]].reshape(-1, 28, 28) 
        remainder_y = candidate_samples_y[sorted_scores[num_target_samples:]]
      
    # ################## #  
    # get random samples #
    # ################## #
    s = shuffle_idx(bug_fix_data)
    bug_fix_data = bug_fix_data[s]
    bug_fix_labels = bug_fix_labels[s]

    # get from bug fixing pool
    randos_x = bug_fix_data[:num_rand_samples]
    randos_y = bug_fix_labels[:num_rand_samples]

    # remove from bug fixing pool
    bug_fix_data = bug_fix_data[num_rand_samples:]
    bug_fix_labels = bug_fix_labels[num_rand_samples:]

    # ####################### #
    # put together next batch #
    # ####################### #
    next_batch_x = np.vstack((targets_x, randos_x))
    next_batch_y = np.concatenate((targets_y, randos_y))

    # add back the unselected candidates to the bug fix pool and shuffle
    bug_fix_data = np.vstack((bug_fix_data, remainder_x))
    bug_fix_labels = np.concatenate((bug_fix_labels, remainder_y))

    s = shuffle_idx(bug_fix_data)
    bug_fix_data = bug_fix_data[s]
    bug_fix_labels = bug_fix_labels[s]

    print('Bug Fixing Pool - # of samples remaining:', bug_fix_data.shape[0])

    return next_batch_x, next_batch_y, bug_fix_data, bug_fix_labels