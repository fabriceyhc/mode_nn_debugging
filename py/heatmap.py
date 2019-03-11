import numpy as np

def get_heatmaps(correct_data,
                 correct_labels,
                 misclassified_data, 
                 misclassified_labels, 
                 misclassified_correct_labels, 
                 num_classes,
                 labels=[], 
                 type='hci'):

    heat_maps = []
      
    if len(labels) == 0:
        labels = np.arange(num_classes)
      
    type = type.lower()

    for label in labels:

        if type in ['hci', 'correct']:
            # HCI
            idx = np.argwhere(np.argmax(correct_labels, axis = 1) == label) 
            hm = np.mean(correct_data[idx], axis = 0) 
        elif type in ['hmi', 'false positive', 'fp']:
            # HMI
            idx = np.argwhere(misclassified_correct_labels == label) 
            hm = np.mean(misclassified_data[idx], axis = 0)   
        elif type in ['hwi', 'false negative', 'fn']:
            # HWI
            idx = np.argwhere(np.argmax(misclassified_labels, axis = 1) == label) 
            hm = np.mean(misclassified_data[idx], axis = 0)
        else:
            raise ValueError('pass in valid type')
      
        heat_maps.append(np.array(hm).reshape(28,28))
      
    return labels, np.array(heat_maps)