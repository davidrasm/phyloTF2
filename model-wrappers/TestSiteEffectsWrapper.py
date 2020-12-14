"""
Created on Wed Jun 24 10:21:53 2020

Basic wrapper for estimating site-specific fitness effects
Assumes ancestral states are known for each node

@author: david
"""
from ete3 import Tree
import numpy as np
import pandas as pd
from Bio import SeqIO
import tensorflow as tf
assert tf.__version__ >= "2.0"
from tensorflow import keras

from phyloTF2.FitModelKeras import FitModelKeras
from phyloTF2.TreeLikeLoss import TreeLikeLoss
from phyloTF2.TensorTree import TensorTree
import phyloTF2.TreeUtils as TreeUtils
from phyloTF2.L1Regularizer import L1Regularizer

"Set tree and Fasta alignment file"
path = '../test/testTF_randomSiteEffects/'
tree_file = path + 'tree-000.tre'
fasta_file = path + 'tree-000.fasta'

"Get tree from newick file"
tree = Tree(tree_file, format=1)
tree, tree_times = TreeUtils.add_tree_times(tree)
tree = TreeUtils.index_branches(tree)

"Get sequences from fasta"
features_dic = {}
for record in SeqIO.parse(fasta_file, "fasta"):
    seq = str(record.seq) # 'sequence' of binary ancestral features
    features_dic[record.id] = np.array(list(map(int,list(seq))))
sites = len(features_dic[next(iter(features_dic))])
feature_names = ["site" + str(i) for i in range(sites)]

"Initial birth-death model params"
beta = np.array([1.0,0.7,0.6])
d = 0.5 # death rate
gamma = 0.0 # no migration here
s = 0.5 # sampling fraction upon removal
rho = 0.5 # sampling fraction at present
dt = 1.0 # time step interval for update pE's along branch 
params = {'beta': beta, 'd': d, 'gamma': gamma, 's': s, 'rho': rho, 'dt': dt, 'time_intervals': 0, 'sites':sites}

"Provide dict with estimated params"
est_params = {'site_effects': True, 'beta': False, 'd': False, 'gamma': False, 's': False, 'rho': False}

"Set up tree for run"
absolute_time = 20.0 # final absolute time of sampling
final_time = max(tree_times) # final time in tree time
root_time = absolute_time - final_time # time of root in absolute time
time_intervals = np.array([5.0, 10.0, 20.0]) # time intervals in absolute time
time_intervals = time_intervals - root_time # time intervals in tree time w/ root at t = 0
params.update(time_intervals = time_intervals)

"Convert tree to TensorTree object"
tt = TensorTree(tree,features_dic,**params)
tt.check_line_time_steps()

"Build fitness model"
model = FitModelKeras(params,est_params)
fit_vals, bdm_params = model.call(tt)

"Build loss function"
like_loss = TreeLikeLoss(tt,params)

"Add regularizer"
reg = L1Regularizer(0.0,offset=1.0)

"Don't actually need to compile model since we are doing our own training"
#model.compile(loss=TreeLikeLoss(tt,params),optimizer="nadam")

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

"Store estimates in pandas dataframe"
site_effects_ests = model.trainable_variables[0].numpy().reshape((1,sites))
site_ests_df = pd.DataFrame(site_effects_ests, columns=feature_names)
file_name = "testTF_siteEffects_estimates.csv"
site_ests_df.to_csv(file_name)

