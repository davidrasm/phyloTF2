# phyloTF2 Tutorial

*by David Rasmussen*

This tutorial demonstrates the basics of setting up and running a model in phyloTF2. Here we will fit a birth-death model to a simulated phylogeny to estimate the fitness effects of mutations at 10 evolving sites. The full script can be found in the *TestSiteEffectsWrapper.py* script in the *model-wrappers* folder. Other examples of how to set models can be found in the same folder.

### Setting up phyloTF2

First we will import some standard Python packages as well as TensorFlow and Keras. We will also use ETE3 to import a Newick tree and Biopython to import an alignment in a Fasta file. These packages can be installed via conda or pip if you do not already have them.

```python
from ete3 import Tree
import numpy as np
import pandas as pd
from Bio import SeqIO
import tensorflow as tf
assert tf.__version__ >= "2.0"
from tensorflow import keras
```

Next import the phyloTF2 classes:

```python
from phyloTF2.TensorTree import TensorTree
from phyloTF2.FitModelKeras import FitModelKeras
from phyloTF2.TreeLikeLoss import TreeLikeLoss
import phyloTF2.TreeUtils as TreeUtils
from phyloTF2.L1Regularizer import L1Regularizer
```

TensorTree, FitModelKeras and TreeLikeLoss are the three major classes phyloTF2 is built upon. We will also import the L1Regularizer class to demonstrate how we can impose regularization on estimated parameters.

Next we need to load in our data including a tree file with the phylogeny we will fit our model to. The true ancestral features (states) of each node in tree are provided in a corresponding Fasta file.

```python
path = '../test/testTF_randomSiteEffects/'
tree_file = path + 'tree-000.tre'
fasta_file = path + 'tree-000.fasta'
```

***Note:*** phyloTF2 assumes we know or have already reconstructed the ancestral states of all features used to predict fitness. We therefore need to provide ancestral states for internal nodes in the tree. The external nodes (tips) and internal nodes of the tree therefore need to be labeled in order to associate ancestral features with each node. The Fasta file given here has entries for each tip and internal node with the ancestral features provided as a binary string of 0's and 1's. While the ancestral features always need to be provided, these do not need to be given as a Fasta file. See the *CovidSpaceTime.py* wrapper for an example of how ancestral features can be loaded through a standard csv file.

We can use the ETE package to read in the tree from our Newick file:
```python
tree = Tree(tree_file, format=1)
tree, tree_times = TreeUtils.add_tree_times(tree)
tree = TreeUtils.index_branches(tree)
```

ETE will parse the branch lengths from the Newick file but will not provide the time or height of the tree nodes. We add these using ```TreeUtils.add_tree_times(tree)```. The root of the tree will be at time t=0 and the node times will increase in chronological order from past to present. We also index the branches so that each branch in the original tree has a unique ID associated with it.

Next we will use Biopython's SeqIO parser to grab each node's ancestral features from the Fasta file and store them in a dictionary.

```python
features_dic = {}
for record in SeqIO.parse(fasta_file, "fasta"):
    seq = str(record.seq) # 'sequence' of binary ancestral features
    features_dic[record.id] = np.array(list(map(int,list(seq))))
sites = len(features_dic[next(iter(features_dic))])
feature_names = ["site" + str(i) for i in range(sites)]
```

We also assign each site a feature name.

### Setting up the birth-death-sampling model

Now we need to parameterize the birth-death-sampling model. There are five required parameters that need to be initialized.

1. ***Beta:*** transmission or birth rate.
2. ***d:*** death or removal rate.
3. ***gamma:*** migration or transition rate (typically zero).
4. ***s:*** sampling fraction at death/removal.
5. ***rho:*** sampling fraction at present.

The birth and death rates can be provided in any unit of time, but the units need to correspond to the timescale of the tree. So if branch lengths are given in years, the rates need to be provided per year. The *dt* parameter is more of a technical detail, it determines how often a lineage's probability of not having any sampled descendent's gets updated along a lineage. 

```python
beta = np.array([1.0,0.7,0.6])
d = 0.5 # death rate
gamma = 0.0 # no migration here
s = 0.5 # sampling fraction upon removal
rho = 0.5 # sampling fraction at present
dt = 1.0 # time step interval for update pE's along branch 
```

Here we will consider a model where the transmission rate beta can vary over time across three intervals. If we had other time varying parameters they would also need to be provided as a numpy array of the same length. However, here we will assume the other parameters are constant so we can just supply them as a single scalar value. 

The parameters are then stored in a Python dictionary used to initialize the model. In addition to the ```params``` dictionary we also need a ```est_params``` dictionary with keys for parameters that can be estimated. The value (True/False) corresponding to each key determines whether TensorFlow treats the variable as an estimated parameter (a TensorFlow Variable) or as a fixed constant. Here we will just estimate the site (mutational) fitness effects.  

```python
params = {'beta': beta, 'd': d, 'gamma': gamma, 's': s, 'rho': rho, 'dt': dt, 'time_intervals': 0, 'sites':sites}
est_params = {'site_effects': True, 'beta': False, 'd': False, 'gamma': False, 's': False, 'rho': False}
```

### Setting up time intervals

Because birth-death models will often have time-varying parameters (e.g. transmission rates or sampling fractions will change through time) it is often necessary to set up different *time intervals*. In phyloTF2, birth-death model parameters are assumed to be piecewise constant within each time interval but allowed to change between intervals. 

Setting up time intervals is perhaps the most confusing part of setting up an analysis. But there is a set of rules  that must be followed:

* Intervals are defined in terms of the end (most recent) time of each interval.
* Times are given with respect to the root time at time zero (t=0)
* Time intervals must be given in ascending chronological order, from most distant past to most recent
* The last time interval must end at the time of the final event in the tree (usually the most recent tip time).

As a simple example, say we have a tree where the root occurs at 2013 and the most recent sample (tip) is taken at 2020. We would like to split the period between 2013 and 2020 into two intervals, one before 2015 and one after. In this hypothetical example the time intervals would be given as the array or list ```[2015.0,2020.0]```, or ```[2.0,7.0]``` once we convert to tree time because the root is always at time t = 0.

Since time intervals always need to be provided relative to the root time, it can be easier to first define the time intervals on some absolute time scale and then convert them to tree time. For example, in the simulated tree we are using here we know that the last sampling event occurred at absolute time t = 20.0. We would like to have three time intervals corresponding to time 0 to 5.0, 5.0 to 10.0, and 10.0 to 20.0. We therefore first define the time intervals on this absolute time scale and then subtract the absolute root time to get time intervals relative to the root time.

```python
absolute_time = 20.0 # final absolute time of sampling
final_time = max(tree_times) # final time in tree time
root_time = absolute_time - final_time # time of root in absolute time
time_intervals = np.array([5.0, 10.0, 20.0]) # time intervals in absolute time
time_intervals = time_intervals - root_time # time intervals in tree time w/ root at t = 0
params.update(time_intervals = time_intervals)
```

### Setting up the TensorTree

Next, we need to convert the input phylogeny ```tree``` into a TensorTree object. Basically, the TensorTree is a set of TF Tensor objects (n-dim arrays) we need to compute the likelihood of the tree under a birth-death model. This sounds fancy, but here these Tensors are just 1D arrays (vectors) that TF can operate on efficiently. We also pass our ```features_dic``` so that each lineage in the TensorTree is associated with its corresponding array of ancestral features. 

```python
tt = TensorTree(tree,features_dic,**params)
tt.check_line_time_steps()
```

It's not necessary to call ```tt.check_line_time_steps()```, but this checks that there are no negative branch lengths in the tree that could cause problems.

### Building the fitness model

We can now finish building our birth-death model and then call it on our TensorTree ```tt``` to get the fitness of each lineage in the tree and other birth-death model parameters:

```python
model = FitModelKeras(params,est_params)
fit_vals, bdm_params = model.call(tt)
```

Its worth taking a minute to understand what is going on underneath the surface when we call the model on the TensorTree. The call function computes the fitness of each lineage in the tree using a fitness mapping function. This function maps the reconstructed ancestral features of each lineage to its expected fitness. FitModelKeras assumes the fitness effect of each feature (site) has a multiplicative effect on overall fitness. More complex fitness mapping functions can be implemented by creating a model subclass that extends FitModelKeras. The subclass can then implement its own call function with a different fitness mapping function.

Here we will also build the loss function that we will optimize below using gradient descent. In this case, the loss function is just the likelihood of the tree under a birth-death model.

```python
like_loss = TreeLikeLoss(tt,params)
```

### Adding regularization

For complex models with many features, it may be a good idea to add a regularization routine (shrinkage) to avoid over fitting the model. We will use L1 regularization which penalizes the model according to the sum of the absolute values of the site (mutational) fitness effects.

```python
reg = L1Regularizer(0.01,offset=1.0)
```

The first parameter passed to the regularizer sets a value for the factor we want to penalize by. We'll just assume a small value but in practice this hyperparameter would need to be tuned, such as by cross-validation. Note that we include an offset of 1.0 because fitness effects are multiplicative such that neutral effects have a value of 1.0, so we want to penalize fitness effects by how much they deviate from 1.0.

### Fitting the model

Now for the fun part! We can fit our model to learn the fitness mapping function and estimate the fitness effect of each mutation using gradient descent. The gradient descent algorithm then optimizes the likelihood of our phylogeny given the ancestral features under the birth-death model. We will use the Adam optimizer, a variant of gradient descent which adapts the momentum (learning rate) of the algorithm. We will run the optimizer for 1000 epochs (iterations) with a learning rate of 0.001. These training parameters should be adjusted to the problem at hand to ensure convergence.  

```python
optimizer = keras.optimizers.Adam(lr=0.001)
n_epochs = 1000
for epoch in range(1,n_epochs+1):
    with tf.GradientTape() as tape:
        fit_vals, bdm_params = model(tt) # or model call?
        penalty = reg.call(model.site_effects)
        loss = -like_loss.call(fit_vals,bdm_params,tt) + penalty
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if epoch % 10 == 0:
        print("Epoch", epoch, "loss =", str(-loss.numpy()), "penalty =", str(penalty.numpy()), "params =", str(model.trainable_variables[0].numpy()))
```

Notice that when we compute the loss, we take the negative (log) likelihood. This is because the optimizer will by default try to minimize the loss function as this is the norm in machine learning applications where we typically want to minimize a loss function (e.g. a least-squares loss if we were fitting a regression model). Here though we are trying to maximize the likelihood, so we want to flip the sign of the loss function.

After each iteration, the optimizer computes the gradients of the loss function with respect to each trainable variable. In our case, these gradients are the derivatives of the birth-death likelihood with respect to each estimated parameters. The gradients are then applied to the current parameter values to get updated values for the next iteration.

The trainable_variables (parameters) or loss values (likelihood) are TF Tensor objects. If we want to grab them and work with them in standard Python code, we can call the ```numpy()``` method on the Tensor objects to return them as numpy arrays.

### Saving the estimates

Finally, we will save the estimated fitness effects by putting them in a Pandas data frame and then writing them to a csv file:

```python
site_effects_ests = model.trainable_variables[0].numpy().reshape((1,sites))
site_ests_df = pd.DataFrame(site_effects_ests, columns=feature_names)
file_name = "testTF_siteEffects_estimates.csv"
site_ests_df.to_csv(file_name) 
```

