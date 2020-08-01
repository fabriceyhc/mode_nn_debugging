# mode_nn_debugging
MODE: Automated Neural Network Model Debugging via State Differential Analysis and Input Selection - Replication Project

In addition to replicating the orig work in tensorflow / keras, this project also investigates using alternative distance metrics for finding the next set of inputs to train on depending on whether the model is underfitting or overfitting. For the case of underfitting, the algorithm seeks to find "best" inputs with the smallest distance from the heatmap representing the "standard" for a given class. For the case of overfitting, the algorithm seeks to find "worst" inputs with the greatest distance from the same class heatmaps. The impact of distance metric has a significant impact on the order and type of inputs selected for subsequent training batches and examples of them can be seen below. 

<img src="/img/best.png" width="49%" style="display:inline;"> <img src="/img/worst.png" width="49%" style="display:inline;">

## Getting Started
In order to run this project, you'll need to clone `master` to your local machine. You'll now need to set up an environment for running the main jupyter notebook and the supporting `py\` functions. I recommend having an anaconda environment and initializing it as follows (from the project directory):

```
conda create -n env_name tensorflow
pip install -r requirements.txt
```

NOTE: This project expects Python 3.6 because tensorflow is currently incompatible with v3.7. 

## Running the Notebook
The notebook is designed to give users a walkthrough of the different component of MODE debugging. This particular experiment was conducted on the MNIST dataset for hand-written digit classification. 

It starts with setting various parameters required by the debugging / training algorithms. It then imports the data and splits it into several sets. The main difference between the normal data science workflow here is that we add a further split of the training data as a reserved pool of samples for fixing the training bugs we'll identify. 

- Train
- Bug-fixing (new)
- Test

Thereafter, a model is created using Keras / Tensorflow and it is cloned after training to create a control model that will be fed additional data at random to serve as a baseline for the performance of MODE iteratively targetting the correction of specific buggy labels. 

The following sections allow the user to inspect the functionality of the various components of MODE:
- targeting layers
- calculating per-label accuracy
- identifying problematic buggy labels
- generating the heatmaps for differential state analysis
- selecting a batch of inputs from the bug-fixing set for targeted training / focus for a given issue. 

The following section focuses on bringing all of these steps together to evaluate the effectiveness of the MODE approach to debugging with different kinds of similarity metrics for input selection. 
- dot product
- cosine similarity
- manhattan (L1) distance
- euclidean (L2) distance
- minkowski (Lp) distance
- chebyshev (L-inf) distance
- earth mover (wasserstein) distance
- canberra distance
- bray curtis dissimilarity

A comparison where we omit the layer selection is also performed (because it is extremely time intensive). This is equivalent to the final layer always being the point at which we reach saturation. 

In the final section, various visualizations are generated so that users can get a better sense of what's happening under the hood.
What do the heat maps look like for correct (DHCI), false positive (DHMI), and false negatives (DHWI)?
What are the best and worst images for improving performance according to a variety of similarity metrics? 
These questions and more are illustrated in this section. 
