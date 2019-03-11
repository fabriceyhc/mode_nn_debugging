import numpy as np
from scipy import spatial
from scipy.stats import wasserstein_distance

# For Probabilility Distributions

def bhattacharyya(a, b):
  '''Calculates the Byattacharyya distance of two matrices.'''

  def normalize(x):
    return x / np.sum(x)

  return 1 - np.sum(np.sqrt(np.multiply(normalize(a), normalize(b))))

# For Images

def measure_distance(data, heatmap, distance_metric, **kwargs):

    distance_metric = distance_metric.lower()
    scores = np.zeros(data.shape[0])

    # NOTE: np.argsort() orders from smallest to largest, which is good for dot products, but bad for every other kind of norm

    if distance_metric == 'dot':
        scores = np.dot(data.reshape(-1, 784), np.ravel(heatmap))
        scores = np.argsort(scores)

    elif distance_metric == 'cosine':
        for i in np.arange(data.shape[0]):
            scores[i] = spatial.distance.cosine(np.ravel(data[i]), np.ravel(heatmap))
        scores = np.flip(np.argsort(scores))

    elif distance_metric == 'manhattan' or distance_metric == 'l1':
        for i in np.arange(data.shape[0]):
            scores[i] = spatial.distance.cityblock(np.ravel(data[i]), np.ravel(heatmap))
            # scores[i] = np.sum(np.abs(np.ravel(data[i]) - np.ravel(heatmap)), axis = 0)
        scores = np.flip(np.argsort(scores))

    elif distance_metric == 'euclidean' or distance_metric == 'l2':        
        # square root of dot product
        # scores = np.sqrt(np.dot(data.reshape(-1, 784), np.ravel(heatmap)))
        for i in np.arange(data.shape[0]):
            scores[i] = spatial.distance.euclidean(np.ravel(data[i]), np.ravel(heatmap))
        scores = np.flip(np.argsort(scores))

    elif distance_metric == 'minkowski' or distance_metric == 'lp':
        if 'minkowski_power' in kwargs:
            minkowski_power = kwargs['minkowski_power']
        else: 
            minkowski_power = 3
            print('Defaut minkowski_power = 3 set...')
        for i in np.arange(data.shape[0]):
            scores[i] = spatial.distance.minkowski(np.ravel(data[i]), np.ravel(heatmap), minkowski_power)
        scores = np.flip(np.argsort(scores))

    elif distance_metric == 'earthmover' or distance_metric == 'wasserstein':
        for i in np.arange(data.shape[0]):
            scores[i] = wasserstein_distance(np.ravel(data[i]), np.ravel(heatmap))
        scores = np.argsort(scores)

    elif distance_metric == 'chebyshev' or distance_metric == 'linf':
        for i in np.arange(data.shape[0]):
            scores[i] = spatial.distance.chebyshev(np.ravel(data[i]), np.ravel(heatmap))
        scores = np.flip(np.argsort(scores))

    elif distance_metric == 'canberra':
        for i in np.arange(data.shape[0]):
            scores[i] = spatial.distance.chebyshev(np.ravel(data[i]), np.ravel(heatmap))
        scores = np.flip(np.argsort(scores))

    elif distance_metric == 'braycurtis':
        for i in np.arange(data.shape[0]):
            scores[i] = spatial.distance.braycurtis(np.ravel(data[i]), np.ravel(heatmap))
        scores = np.flip(np.argsort(scores))

    return scores