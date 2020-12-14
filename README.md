# phyloTF2

phyloTF2 is a package for efficient likelihood-based phylodynamic learning using birth-death-sampling models. phyloTensorFlow is built on top of TensorFlow 2 using the Keras API. Statistical inference or model training is performed using gradient descent algorithms such as ADAM to optimize the likelihood of a tree under a birth-death-sampling model.  The fitness of any lineage in a phylogenetic tree can depend on reconstructed ancestral features (states). A fitness mapping function is used to translate a lineage’s ancestral features into expected fitness. 

## Set up
We recommend installing a package manager such as [Anaconda](https://anaconda.org/anaconda/python) to satisfy standard Python library dependencies. You can then install [TensorFlow 2.x](https://www.tensorflow.org) using *conda* or *pip*.  You will also need to install the [ETE Toolkit](http://etetoolkit.org) for working with phylogenetic trees in Python. Once these packages are installed you can simply run one of the model wrapper classes below to set up or replicate an analysis. 

For a basic overview of how to set up and train a model in phyloTF2, please see this [tutorial](https://davidrasm.github.io/phyloTF2/).

## Source code
The source code contains just three major classes:

**FitModelKeras:** a class for birth-death-sampling models where the fitness of a lineage can depend on its ancestral features. The __init__ function sets the initial values of all parameters in the birth-death model. The call function implements the fitness mapping function used to translate a lineage’s ancestral features into its expected fitness. The callfunction returns the fitness values of each tree lineage (fit_vals) and other birth-death model parameters (bdm_params) required to compute the likelihood of a tree using TreeLikeLoss. Subclasses of FitModelKeras (e.g. RandomEffectsFitModel) can be created to implement different fitness mapping functions. 

**TensorTree:** provides a data structure for working with phylogenetic trees in TensorFlow. Trees are converted into a set of TF Tensor objects (i.e. n-dim arrays) to vectorize operations involved in computing the likelihood of trees under a birth-death-sampling model.

**TreeLikeLoss:** Computes the likelihood of a phylogenetic tree under a birth-death-sampling model with lineage-specific fitness values.  This is the loss or objective function that is optimized by gradient descent. The call function takes as its arguments the birth-death model parameters including the fitness of all lineages in the tree.

## Model wrappers
Our phylodynamic analysis of SARS-CoV-2 in the United States can be replicated using a set of wrapper functions that help set up and run different models. 

**CovidSpaceTimeWrapper:** fits model with spatiotemporal effects for each geographic region and time interval. Genetic features are not considered.

**CovidGeneticFeaturesWrapper:** fits model with spatiotemporal effects and fitness effects for all genetic variants assuming a multiplicative fitness mapping function. 

**CovidRandomEffectsWrapper:** fits model with spatiotemporal effects, genetic fitness effects and random branch-specific fitness effects.

## SARS-CoV-2 data
As per the data usage agreement with GISAID, we cannot redistribute the original sequence data used to reconstruct SARS-CoV-2 phylogenies. Nevertheless, we provide our best fit ML phylogeny and a feature file which contains the reconstructed ancestral features (states) for all nodes in the tree. These two files can be used together to replicate our SARS-CoV-2 analysis using the model wrappers above.
