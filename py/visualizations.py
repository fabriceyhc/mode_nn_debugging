import matplotlib.pyplot as plt
import numpy as np
from py.heatmap import get_heatmaps
from py.distance_metrics import measure_distance

def get_plot(dict, title='', xlabel='', ylabel=''):
    plt.plot(range(len(dict)), list(dict.values()))
    plt.xticks(range(len(dict)), list(dict.keys()))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def visualize_array(arr, interpolation='gaussian'):
    plt.imshow(arr, interpolation=interpolation)
    plt.axis('off')
    plt.show()

def get_samples_for_label(X, y, label=0):  
    return X[np.argwhere(y == label)]

def sample_misclassifications(misclassified_data, 
                              misclassified_labels, 
                              misclassified_correct_labels, 
                              num_samples=1, 
                              shuffle=True):
    if shuffle:
        s = np.arange(misclassified_data.shape[0])
        np.random.shuffle(s)
        misclassified_data = misclassified_data[s]
        misclassified_labels = misclassified_labels[s]
        misclassified_correct_labels = misclassified_correct_labels[s]

    # sample a few misclassified examples i labeled as -i
    for i in np.arange(num_samples):
      pred = np.argmax(misclassified_labels[i])
      print('True Label: ' + str(misclassified_correct_labels[i]) + ' Prediction: ' + str(pred))
      plt.imshow(misclassified_data[i].reshape(28,28), interpolation='gaussian')
      plt.axis('off')
      plt.show()

def viz_heatmaps_for_correct_prediction(correct_data,
                                        correct_labels,
                                        num_classes,
                                        labels=[],
                                        cmap='seismic'):
  
    # HCI_i = NUMBERS THAT ARE i AND LABELED i (CORRECT)
    # It's a 0 and we said it was a 0

    if len(labels) == 0:
        labels = np.arange(num_classes)

    fig = plt.figure()
    fig.set_figheight(7)
    fig.set_figwidth(15)

    for label, i in zip(labels, np.arange(len(labels))):

        # Get true indicies for current number
        idx_of_true_labels = np.argwhere(np.argmax(correct_labels, axis = 1) == label) 

        # Take the average of all the numbers that are not i, but predicted to be i
        average_img = np.mean(correct_data[idx_of_true_labels], axis = 0) 

        # Plots
        ax = fig.add_subplot(2, 5, i+1)
        ax.imshow(average_img.reshape(28,28), cmap=cmap, interpolation='gaussian')
        plt.axis('off')
        plt.tight_layout()
        plt.title(label)

    plt.show()

def viz_heatmaps_for_false_positives(misclassified_data, 
                                     misclassified_correct_labels, 
                                     num_classes,
                                     cmap='seismic'):

    # HMI_i = NUMBERS THAT ARE NOT i BUT LABELED AS i (FALSE POSITIVES)
    # It's not a 6, but we said it was

    fig = plt.figure()
    fig.set_figheight(7)
    fig.set_figwidth(15)

    for i in np.arange(num_classes):

        # Get indicies of the true values for current number
        idx_of_true_labels = np.argwhere(misclassified_correct_labels == i) 

        # Take the average of all the numbers that are not i, but predicted to be i
        average_img = np.mean(misclassified_data[idx_of_true_labels], axis = 0) 

        # Plots
        ax = fig.add_subplot(2, 5, i+1)
        ax.imshow(average_img.reshape(28,28), cmap=cmap, interpolation='gaussian')
        plt.axis('off')
        plt.tight_layout()
        plt.title(i)
      
    plt.show()

def viz_heatmaps_for_false_negatives(misclassified_data, 
                                     misclassified_labels, 
                                     num_classes,
                                     cmap='seismic'):

    # HWI_i = NUMBERS THAT ARE i BUT LABELED AS SOMETHING OTHER THAN i (FALSE NEGATIVES)
    # It's a 6, but we said it wasn't

    fig = plt.figure()
    fig.set_figheight(7)
    fig.set_figwidth(15)

    for i in np.arange(num_classes):

        # Get true indicies for current number
        idx_of_true_labels = np.argwhere(np.argmax(misclassified_labels, axis = 1) == i) 

        # Take the average of all the numbers that are not i, but predicted to be i
        average_img = np.mean(misclassified_data[idx_of_true_labels], axis = 0) 

        # Plots
        ax = fig.add_subplot(2, 5, i+1)
        ax.imshow(average_img.reshape(28,28), cmap='seismic', interpolation='gaussian')
        plt.axis('off')
        plt.tight_layout()
        plt.title(i)
      
    plt.show()


def DHCI_i_k(correct_data,
             correct_labels,
             first_label, 
             second_label,
             cmap='seismic'):

    # DHCI_i_k = DIFFERENTIAL HEAT MAP FOR TWO DIFFERENT LABELS

    if first_label == None or second_label == None:
        raise ValueError('Must provide two valid labels to compare.')
  
    fig = plt.figure()
    fig.set_figheight(7)
    fig.set_figwidth(15)

    # HCI_i
    HCI_i_idx = np.argwhere(np.argmax(correct_labels, axis = 1) == first_label) 
    HCI_i_average_img = np.mean(correct_data[HCI_i_idx], axis = 0) 

    # HCI_k
    HCI_k_idx = np.argwhere(np.argmax(correct_labels, axis = 1) == second_label) 
    HCI_k_average_img = np.mean(correct_data[HCI_k_idx], axis = 0) 

    # Differential State
    DHCI_i_k = HCI_i_average_img - HCI_k_average_img

    plot_data = [HCI_i_average_img, HCI_k_average_img, DHCI_i_k]
    plot_titles = ['HCI for ' + str(first_label), 'HCI for ' + str(second_label), 'DHCI for ' + str(first_label) + ' and ' + str(second_label)]

    # Plots
    for data, title, i in zip(plot_data, plot_titles, range(3)):   
        ax = fig.add_subplot(1, 3, i+1)
        ax.imshow(data.reshape(28,28), cmap=cmap, interpolation='gaussian')
        plt.axis('off')
        plt.tight_layout()
        plt.title(title)

    plt.show()

def DHMI_i(correct_data,
           correct_labels,
           misclassified_data,
           misclassified_correct_labels, 
           num_classes,
           cmap='seismic'):

    # DHMI_i = DIFFERENTIAL HEATMAP FOR FALSE POSITIVES

    fig = plt.figure()
    fig.set_figheight(7)
    fig.set_figwidth(15)

    for i in np.arange(num_classes):
      
        # HCI
        HCI_idx = np.argwhere(np.argmax(correct_labels, axis = 1) == i) 
        HCI_average_img = np.mean(correct_data[HCI_idx], axis = 0) 

        # HMI
        HMI_idx = np.argwhere(misclassified_correct_labels == i) 
        HMI_average_img = np.mean(misclassified_data[HMI_idx], axis = 0) 

        # Differential State
        DHMI = HMI_average_img - HCI_average_img

        # Plots
        ax = fig.add_subplot(2, 5, i+1)
        ax.imshow(DHMI.reshape(28,28), cmap=cmap, interpolation='gaussian')
        plt.axis('off')
        plt.tight_layout()
        plt.title(i)
      
    plt.show()

def DHWI_i(correct_data,
           correct_labels,
           misclassified_data,
           misclassified_labels, 
           num_classes,
           cmap='seismic'):

    # DHWI_i = DIFFERENTIAL HEATMAP FOR FALSE NEGATIVES

    fig = plt.figure()
    fig.set_figheight(7)
    fig.set_figwidth(15)

    for i in np.arange(num_classes):
      
        # HCI
        HCI_idx = np.argwhere(np.argmax(correct_labels, axis = 1) == i) 
        HCI_average_img = np.mean(correct_data[HCI_idx], axis = 0) 

        # HWI
        HWI_idx = np.argwhere(np.argmax(misclassified_labels, axis = 1) == i) 
        HWI_average_img = np.mean(misclassified_data[HWI_idx], axis = 0) 

        # Differential State
        DHWI = HWI_average_img - HCI_average_img

        # Plots
        ax = fig.add_subplot(2, 5, i+1)
        ax.imshow(DHWI.reshape(28,28), cmap=cmap, interpolation='gaussian')
        plt.axis('off')
        plt.tight_layout()
        plt.title(i)
      
    plt.show()

def viz_most_least_similar(bug_fix_data,
                           bug_fix_labels,
                           labels,
                           heatmaps,
                           distance_metric='dot',
                           **kwargs):

    if distance_metric == 'minkowski':
        if 'minkowski_power' in kwargs:
            minkowski_power = kwargs['minkowski_power']
        else:
            minkowski_power = 3
            print('Defaut minkowski_power = 3 set...')
    else: 
        minkowski_power = None

    fig = plt.figure()
    fig.set_figheight(7)
    fig.set_figwidth(15)

    for label in labels:

        samples_for_label = bug_fix_data[np.argwhere(bug_fix_labels == label)]
        heatmap_for_label = heatmaps[label]

        sorted_scores = measure_distance(samples_for_label, heatmap_for_label, distance_metric, minkowski_power = minkowski_power)

        # Plots
        ax = fig.add_subplot(3, 10, label+1)
        plt.imshow(heatmaps[label].reshape(28,28), cmap='seismic', interpolation='gaussian')
        plt.axis('off')
        plt.tight_layout()
        plt.title('Heatmap for ' + str(label))

        ax = fig.add_subplot(3, 10, label+11)
        plt.imshow(samples_for_label[sorted_scores[-1]].reshape(28,28), interpolation='gaussian')
        plt.axis('off')
        plt.tight_layout()
        plt.title('Best ' + str(label))

        ax = fig.add_subplot(3, 10, label+21)
        plt.imshow(samples_for_label[sorted_scores[0]].reshape(28,28), interpolation='gaussian')
        plt.axis('off')
        plt.tight_layout()
        plt.title('Worst ' + str(label))



def viz_most_similar(bug_fix_data,
                     bug_fix_labels,
                     labels,
                     heatmaps,
                     distance_metric='dot',
                     **kwargs):

    if distance_metric == 'minkowski':
        if 'minkowski_power' in kwargs:
            minkowski_power = kwargs['minkowski_power']
        else:
            minkowski_power = 3
            print('Defaut minkowski_power = 3 set...')
    else: 
        minkowski_power = None

    fig = plt.figure()
    fig.set_figheight(7)
    fig.set_figwidth(15)

    for label in labels:

        samples_for_label = bug_fix_data[np.argwhere(bug_fix_labels == label)]
        heatmap_for_label = heatmaps[label]

        sorted_scores = measure_distance(samples_for_label, heatmap_for_label, distance_metric, minkowski_power = minkowski_power)

        # Plots
        ax = fig.add_subplot(3, 10, label+1)
        plt.imshow(samples_for_label[sorted_scores[-1]].reshape(28,28), interpolation='gaussian')
        plt.axis('off')
        plt.tight_layout()
        plt.title('Best ' + str(label))

def viz_least_similar(bug_fix_data,
                      bug_fix_labels,
                      labels,
                      heatmaps,
                      distance_metric='dot',
                      **kwargs):

    if distance_metric == 'minkowski':
        if 'minkowski_power' in kwargs:
            minkowski_power = kwargs['minkowski_power']
        else:
            minkowski_power = 3
            print('Defaut minkowski_power = 3 set...')
    else: 
        minkowski_power = None

    fig = plt.figure()
    fig.set_figheight(7)
    fig.set_figwidth(15)

    for label in labels:

        samples_for_label = bug_fix_data[np.argwhere(bug_fix_labels == label)]
        heatmap_for_label = heatmaps[label]

        sorted_scores = measure_distance(samples_for_label, heatmap_for_label, distance_metric, minkowski_power = minkowski_power)

        # Plots
        ax = fig.add_subplot(3, 10, label+1)
        plt.imshow(samples_for_label[sorted_scores[0]].reshape(28,28), interpolation='gaussian')
        plt.axis('off')
        plt.tight_layout()
        plt.title('Worst ' + str(label))
