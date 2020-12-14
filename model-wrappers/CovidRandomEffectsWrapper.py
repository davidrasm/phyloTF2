"""
Created on Fri Jun 26 09:22:53 2020

Wrapper for fitting full covid model with site effects, space x time effects and random effects
Random branch-specific fitness effects are fit under a Brownian motion model of trait (fitness) evolution

@author: david
"""
from ete3 import Tree
import TreeUtils
import numpy as np
from RandomEffectsFitModel import RandomEffectsFitModel
from TreeLikeLoss import TreeLikeLoss
from TensorTree import TensorTree
from L2Regularizer import L2Regularizer
import pandas as pd
from DateTimeUtils import date2FloatYear

import tensorflow as tf
assert tf.__version__ >= "2.0"
from tensorflow import keras

print_estimates = True

"Initial birth-death model params"
sigma = 0.01 # scales variance in Brownian motion model (a small value epsilon gets added to this)
beta = np.array([59.983]*9) # transmission ratre
d = 52.18 # death rate = 1/7 per day
gamma = 0.0 # no migration here
s = np.append([0.],[0.0004]*8) # sampling fraction upon removal
rho = (d/365.25)*0.0004 # sampling fraction at present
dt = 0.1 # time step interval for update pE's along branch 
params = {'sigma': sigma, 'beta': beta, 'd': d, 'gamma': gamma, 's': s, 'rho': rho, 'dt': dt, 'time_intervals': 0}

"Provide dict with estimated params"
est_params = {'sigma': False, 'random_branch_effects': True, 'site_effects': False, 'beta': False, 'd': False, 'gamma': False, 's': False, 'rho': False}

"Import tree and sequences"    
path = './covid-analysis/'

"For spike features"
features_file = path + 'feature-files/hcov_oct2020_bestTree_byRegion_allFeatures.csv'
pastml_path = path + 'pastml/collected_preSep1_dated_pastml/'
tree_file = pastml_path + 'named.tree_phylogeny_mle_cleaned_collected_preSep1_dated_cleaned.nwk'
absolute_time = 2020.67 # absolute time of last sample

"Set up tree for run"
tree = Tree(tree_file, format=1)
tree, tree_times = TreeUtils.add_tree_times(tree)

"Set up time intervals"
final_time = max(tree_times)
root_time = absolute_time - final_time
date_time_intervals = ['2020-01-01',
                        '2020-02-15',
                        '2020-03-15',
                  '2020-04-15',
                  '2020-05-15',
                  '2020-06-15',
                  '2020-07-15',
                  '2020-08-15']
time_intervals = date2FloatYear(date_time_intervals)
time_intervals = np.array(time_intervals) - root_time

"Check time intervals"
time_intervals = time_intervals[time_intervals >= 0.]
if final_time not in time_intervals:
    time_intervals = np.append(time_intervals,final_time)
params.update(time_intervals = time_intervals)

"Get tip/ancestor features from csv file"
df = pd.read_csv(features_file,index_col='node')

"""
 For Debugging Only!!!: Control number of features to include for initial training
"""
#df = df.iloc[:, list(range(50))] # take first 50 features to start

"Set up time-varying features and add duplicate columns for time-varying features"
time_feature = 'REGION'
time_feature_labels = [col for col in df.columns if time_feature in col]
for label in time_feature_labels:
    loc = df.columns.get_loc(label)
    base_label = label+'_t0'
    df.rename(columns={label: base_label},inplace=True)
    for interval in range(1,len(time_intervals)):
        df.insert(loc=loc+interval, column=label+'_t'+str(interval), value=df[base_label])
feature_names = list(df)

features_dic = {}
for index, row in df.iterrows():
    features_dic[index] = row.to_numpy()
sites = len(features_dic[next(iter(features_dic))])
params['sites'] = sites

"Convert tree to TensorTree object"
tree = TreeUtils.index_branches(tree) # only used for models with random branch effects
tt = TensorTree(tree,features_dic,**params)
tt.check_line_time_steps()

"Mask time-varying features outside of line time intervals"
tt.mask_time_features(df,time_feature_labels,time_intervals)

"Get site fitness effect estimates"
path = './covid-results/'
feature_fit_file = path + 'covid_spaceXTimeByRegion_Oct2020_siteEffects_rhoSampling_allMonthlyIntervals_estimates.csv'
#feature_fit_file = path + 'covid_spaceXTimeByRegion_Oct2020_siteEffects_rhoSampling_extraTimeIntervals_estimates.csv'
feature_effects_df = pd.read_csv(feature_fit_file)
feature_effects_df.drop(['Unnamed: 0'],axis=1,inplace=True)
if feature_effects_df.columns.tolist() != feature_names: print('WARNING: Feature lists do not match!')
params.update(fitness_effects = feature_effects_df.loc[0].values)

"Build fitness model"
branches = np.max(tt.birth_branch_indexes) + 1 # num of branches: plus one b/c indexed from 0 
model = RandomEffectsFitModel(params,est_params,branches)

"Build loss function"
like_loss = TreeLikeLoss(tt,params)

"Add regularizer"
reg = L2Regularizer(0.0,offset=1.0)

optimizer = keras.optimizers.Adam(lr=0.001)
n_epochs = 5000
for epoch in range(1,n_epochs+1):
    #model.store_line_branch_effects()
    with tf.GradientTape() as tape:
        fit_vals, bdm_params = model(tt) # or model call?
        reg_penalty = reg.call(model.site_effects)
        penalty = model.get_penalty(tt) # Brownian motion penalty on fit evolution
        loss = -(like_loss.call(fit_vals,bdm_params,tt) + penalty) + reg_penalty
    gradients = tape.gradient(loss, model.trainable_variables)
    if tf.math.reduce_any(tf.math.is_nan(gradients[0])):
        print("Gradient is NaN")
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    model.check_line_branch_effects()
    model.clip_line_branch_effects(min_thresh=0.1,max_thresh=10.) # Enforce constraints on branch effects
    if epoch % 10 == 0:
        if print_estimates:
            
            #min_site_effect = np.min(model.trainable_variables[0].numpy())
            #max_site_effect = np.max(model.trainable_variables[0].numpy())
            min_branch_effect = np.min(model.trainable_variables[0].numpy())
            max_branch_effect = np.max(model.trainable_variables[0].numpy())
            
            print("Epoch", epoch, "loss =", str(-loss.numpy()), "penalty =", str(penalty.numpy()), "min_branch_effect =", str(min_branch_effect), "max_branch_effect =", str(max_branch_effect))
            #print("Epoch", epoch, "loss =", str(-loss.numpy()), "penalty =", str(penalty.numpy()), "min_branch_effect =", str(min_branch_effect), "max_branch_effect =", str(max_branch_effect), "min_site_effect =", str(min_site_effect), "max_site_effect =", str(max_site_effect))
            #print("Epoch", epoch, "loss =", str(-loss.numpy()), "penalty =", str(penalty.numpy()), "params =", str(model.trainable_variables[0].numpy()))
            #print("Epoch", epoch, "loss =", str(-loss.numpy()), "penalty =", str(penalty.numpy()), "params =", str(model.trainable_variables[0].numpy()), "betas =", str(model.trainable_variables[1].numpy()))
        else:
            print("Epoch", epoch, "loss =", str(-loss.numpy()), "penalty =", str(penalty.numpy())) 

"Store estimates in pandas dataframe"
#site_effects_ests = model.trainable_variables[0].numpy().reshape((1,sites))
#site_ests_df = pd.DataFrame(site_effects_ests, columns=feature_names)
#file_name = "covid_fullModel_fixedSiteEffects_newTimeIntervals_recodedFeatures_estimates.csv"
#folder = "./"
#site_ests_df.to_csv(folder + file_name)

"Store lineage fit vals"
line_branch_indexes = tt.line_branch_indexes
line_fit_vals, birth_fit_vals = fit_vals
line_fit_vals = line_fit_vals.numpy().flatten()
line_start_times = tt.line_start_times
line_end_times = tt.line_end_times

"Store line random branch effects as well"
line_branch_effects = model.get_line_branch_effects(tt)
line_branch_effects = line_branch_effects.numpy().flatten()

data = {'BranchIndexes':line_branch_indexes,'LineStartTimes': line_start_times, 'LineEndTimes': line_end_times, 'LineFitVals':line_fit_vals, 'LineBranchEffects': line_branch_effects}  
line_fit_df = pd.DataFrame(data)
    
"Store line features in df as well so we can decompose fitness components afterwards"
line_feature_data = tt.line_seqs.numpy()
line_feature_df = pd.DataFrame(line_feature_data, columns=feature_names)
line_fit_df = pd.concat([line_fit_df, line_feature_df], axis=1)
line_fit_df.to_csv("covid_fullModel_fixedSiteEffects_Oct2020_allMonthlyIntervals_lineFitEstimates.csv",index=False)



