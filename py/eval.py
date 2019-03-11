import numpy as np
import keras as k

def get_eval_datasets(data, labels, predictions):

    # get the indicies of the misclassified examples
    misclassified = np.not_equal(np.argmax(predictions, axis=1), labels)

    # get the indicies of the correctly classified examples
    correct = np.equal(np.argmax(predictions, axis=1), labels)

    data_correct = data[correct]                  # correctly predicted samples
    labels_correct = predictions[correct]         # correctly predicted labels

    data_incorrect = data[misclassified]          # misclassified examples 
    labels_incorrect = predictions[misclassified] # wrong predicted labels
    labels_corrected = labels[misclassified]      # correct labels for the ones that were misclassified

    out = (data_correct, labels_correct, data_incorrect, labels_incorrect, labels_corrected)

    return out

def replicate_model(model):      
    new_model = k.models.clone_model(model)
    for new_layer, old_layer in zip(new_model.layers, model.layers):
        weights = old_layer.get_weights()
        new_layer.set_weights(weights)
    return new_model