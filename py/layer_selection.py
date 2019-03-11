import numpy as np
import tensorflow as tf
import keras as k
from py.distance_metrics import bhattacharyya

def Forward_Layer_Select(model, 
                         x_train, 
                         y_train, 
                         x_test, 
                         y_test, 
                         epochs, 
                         similarity_threshold, 
                         verbose=False):

    feature_models= {}
    accuracies = {}
    bhattacharyyas = {}
    bit = (1 * verbose)

    for layer_num, layer in zip(np.arange(1, len(model.layers)), model.layers): 
        
        if verbose:
            print('Layer Number: {}'.format(layer_num))

        # create model container
        feature_model = k.models.Sequential()  

        # add "frozen" (trainable = False) layers incrementally to create submodels
        for i in np.arange(layer_num):
            next_layer = model.layers[i]
            next_layer.trainable = False
            feature_model.add(next_layer) 

        # add output layer
        feature_model.add(k.layers.Dense(10, activation=tf.nn.softmax))

        # compile
        feature_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
        
        # get some nice summary data
        if verbose:
            feature_model.build()
            feature_model.summary()
            
        # train
        feature_model.fit(x_train, y_train, epochs=epochs, verbose=bit)
        feature_models['model_upto_' + str(layer_num)] = feature_model

        # predict (to get predictions)
        y_pred = feature_model.predict(x_test)

        # test (just to know how we did overall)
        if verbose:
            print('Getting Scores...')
            
        scores = feature_model.evaluate(x_test, y_test, verbose=bit)
        accuracies[layer_num] = scores[1]
        if verbose:
            print('Test Acc: {}'.format(scores[1]))

        # measure similarity
        if layer_num == 1:
            last_y_pred = np.ones_like(y_pred)

        similarity = bhattacharyya(y_pred, last_y_pred)
        bhattacharyyas[layer_num] = similarity
        
        if verbose:
            print('Bhattacharyya Similarity: {}'.format(similarity)) 

        cache = (feature_models, accuracies, bhattacharyyas, y_pred)    
            
        if similarity <= similarity_threshold:
            target_layer = layer
            print('Target Layer # {}: {}'.format(layer_num, target_layer))            
            return target_layer, cache
        else:
            last_y_pred = y_pred
        
    # if every hidden layer continually improves performance, then target layer is the output layer
    target_layer = model.layers[-1]
    print('Target Layer is the Output Layer: {}. \n Consider additional training because performance has not yet plateued.'.format(target_layer))  
    return target_layer, cache

def Backward_Layer_Select(model, 
                          x_train, 
                          y_train, 
                          x_test, 
                          y_test, 
                          epochs, 
                          similarity_threshold, 
                          verbose=False):

    feature_models= {}
    accuracies = {}
    bhattacharyyas = {}
    bit = (1 * verbose)

    for layer_num, layer in zip(np.arange(1, len(model.layers)), model.layers): 
        
        if verbose:
            print('Layer Number: {}'.format(layer_num))

        # create model container
        feature_model = k.models.Sequential()  

        # add input layer
        feature_model.add(k.layers.Flatten(input_shape=(28, 28)))

        # add "frozen" (trainable = False) layers incrementally to create submodels
        for i in np.flip(np.arange(1, layer_num)):
            next_layer = model.layers[i]
            next_layer.trainable = False
            feature_model.add(next_layer) 

        # compile
        feature_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
        
        # get some nice summary data
        if verbose:
            feature_model.build()
            feature_model.summary()
            
        # train
        feature_model.fit(x_train, y_train, epochs=epochs, verbose=bit)
        feature_models['model_upto_' + str(layer_num)] = feature_model

        # predict (to get predictions)
        y_pred = feature_model.predict(x_test)

        # test (just to know how we did overall)
        if verbose:
            print('Getting Scores...')
            
        scores = feature_model.evaluate(x_test, y_test, verbose=bit)
        accuracies[layer_num] = scores[1]
        if verbose:
            print('Test Acc: {}'.format(scores[1]))

        # measure similarity
        if layer_num == 1:
            last_y_pred = np.ones_like(y_pred)

        similarity = bhattacharyya(y_pred, last_y_pred)
        bhattacharyyas[layer_num] = similarity
        
        if verbose:
            print('Bhattacharyya Similarity: {}'.format(similarity)) 

        cache = (feature_models, accuracies, bhattacharyyas, y_pred)    
            
        if similarity <= similarity_threshold:
            target_layer = layer
            print('Target Layer # {}: {}'.format(layer_num, target_layer))            
            return target_layer, cache
        else:
            last_y_pred = y_pred
        
    # if every hidden layer continually improves performance, then target layer is the output layer
    target_layer = model.layers[1]
    print('Target Layer is the first hidden layer: {}. \n This model does not appear to have an overfitting problem.'.format(target_layer))  
    return target_layer, cache

def calculate_acc_for_labels(all_labels,
                             correct_labels,
                             num_classes,
                             labels = []):

    accuracies_per_label = {}

    if len(labels) == 0:
        labels = np.arange(num_classes)
        
    for label in np.arange(len(labels)):  

        num_correct_for_label = np.count_nonzero(np.argwhere(np.argmax(correct_labels, axis = 1) == label))
        total_for_label = np.count_nonzero(np.argwhere(all_labels == label))
        acc = num_correct_for_label / total_for_label
        
        accuracies_per_label[label] = acc

    return accuracies_per_label

def get_overfit_labels(train_acc,
                       test_acc,
                       num_classes, 
                       threshold = 0.10,
                       labels = []):

    overfit_labels = {}

    if len(labels) == 0:
        labels = np.arange(num_classes)

    for label in labels:

        diff = train_acc[label] - test_acc[label]

        if diff > threshold:
            overfit_labels[label] = diff

    if len(overfit_labels) == 0:
        print('No overfit labels...')

    return overfit_labels

def get_underfit_labels(acc,
                        num_classes,
                        threshold = 0.90,
                        labels = []):

    underfit_labels = {}

    if len(labels) == 0:
        labels = np.arange(num_classes)

    for label in labels:

        if acc[label] < threshold:
            underfit_labels[label] = acc[label]

    if len(underfit_labels) == 0:
        print('No underfit labels...')

    return underfit_labels

def get_faultiest_label(dict):
    return min(dict, key=dict.get)