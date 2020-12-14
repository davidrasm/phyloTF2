"""
Created on Fri Jun 19 14:06:08 2020

Compute variance in total fitness and fitness components through time
Total variance is computed assuming variances between components are additive
But not necessarily independent, i.e. covariances are accounted for

@author: david
"""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import TreeUtils
from ete3 import Tree
from DateTimeUtils import date2FloatYear

def get_var_df(time_intervals,tree_file, df, est_df):
    
    tree = Tree(tree_file, format=1)
    tree, tree_times = TreeUtils.add_tree_times(tree)
    final_time = max(tree_times)
    absolute_time = 2020.67 # absolute time of last sample
    root_time = absolute_time - final_time

    "Set up time intervals (in tree time) if have time-varying params"
    time_intervals = np.array(time_intervals) - root_time
    time_intervals = time_intervals[time_intervals >= 0.]
    if final_time not in time_intervals:
        time_intervals = np.append(time_intervals,final_time)
    
    "Get fitness of each lineage based on different components"
    spatial_feature = 'REGION'
    spatial_features = [col for col in est_df.columns if spatial_feature in col]
    
    "Fitness based on antigenic effects"
    genetic_features = [col for col in est_df.columns if col not in spatial_features]
    
    "Create df to hold variance at each time (rows) attributable to each feature/component (columns)"
    spatial_features_no_intervals = ['REGION_' + str(i) for i in range(1,11)]
    var_df_columns = genetic_features + spatial_features_no_intervals + ['GENETIC','SPATIAL','RANDOM','TOTAL','TOTAL_DIRECT','TOTAL_SUMMED','GENETIC_SPATIAL_COV','GENETIC_RANDOM_COV','SPATIAL_RANDOM_COV','LINEAR_AVG_FIT','LINEAR_VAR_FIT']
    var_df = pd.DataFrame(index=absolute_times,columns=var_df_columns,dtype='float64')
    for idx,t in enumerate(absolute_times):
        
        tree_time = t - root_time
        
        "Find lineages present in tree at time point t"
        time_df = df[(df['LineStartTimes'] <= tree_time) & (df['LineEndTimes'] >= tree_time)]
        lines = len(time_df.index)
        
        "Find time-varying features in this time interval"
        diffs = time_intervals - tree_time
        diffs[diffs < 0.0] = np.Inf
        curr_time_interval = np.argmin(diffs)
        t_index_str = 't'+str(curr_time_interval)
        t_spatial_features = [f for f in spatial_features if t_index_str in f] 
        
        "Get fitnees effects of all features"
        all_features = genetic_features + t_spatial_features
        anc_features = time_df[all_features].values
        all_effects = np.log(est_df[all_features].values) # log transform so multiplicative effects become additive
        
        """
            Create a feature_effects matrix where columns give the marginal effect of each feature on each lineages (rows) fitness
            The (log) fitness effect of a feature is zero if lineage does not have feature
        """
        tiled_effects = np.tile(all_effects, (lines, 1)) # replicate fit effects for each lineage (row)
        feature_effects = np.multiply(anc_features,tiled_effects) # element wise multiplication 
        
        "Add random fitness effects"
        random_fit = np.reshape(np.log(time_df['LineBranchEffects'].values), (-1, 1))
        all_features.append('RANDOM')
        feature_effects = np.hstack((feature_effects,random_fit))
        
        "Compute cov among all features and put in new df"
        feature_cov = np.cov(feature_effects,rowvar=False) # compute cov among features (columns) 
        feature_cov_df = pd.DataFrame(feature_cov, index=all_features, columns=all_features)
        clean_spatial_features = [f.replace('_'+t_index_str,'') for f in t_spatial_features]
        feature_cov_df.rename(columns=dict(zip(t_spatial_features, clean_spatial_features)),inplace=True) # remove time index from spatial features names
        feature_cov_df.rename(index=dict(zip(t_spatial_features, clean_spatial_features)),inplace=True)
        
        "Loop through features grabing feature name and variance"
        for feature in feature_cov_df.columns:
             var_df.at[t,feature] = feature_cov_df[feature][feature] 
        
        "Compute var/cov for genetic feautres"
        genetic_cov_df = feature_cov_df.loc[genetic_features,genetic_features]
        genetic_var = genetic_cov_df.sum(axis=0).sum() # sum over both dims to sum var and cov
        var_df.at[t,'GENETIC'] = genetic_var
        
        "Compute var/cov for spatial components"
        spatial_cov_df = feature_cov_df.loc[clean_spatial_features,clean_spatial_features]
        spatial_var = spatial_cov_df.sum(axis=0).sum() # sum over both dims to sum var and cov
        var_df.at[t,'SPATIAL'] = spatial_var
        
        "Compute var in random effects"
        random_var = np.var(random_fit)
        #var_df.at[t,'RANDOM'] = random_var # already added above from feature_cov_df
    
        "Compute total"    
        total_var = np.sum(feature_cov, axis=(0,1))
        #total_var = feature_cov_df.sum(axis=0).sum() # should be the same as total_var above
        var_df.at[t,'TOTAL'] = total_var
        var_df.at[t,'TOTAL_DIRECT'] = np.var(np.log(time_df['LineFitVals'].values))
        
        "Compute mean and var in fitness on linear scale"
        var_df.at[t,'LINEAR_AVG_FIT'] = np.mean(time_df['LineFitVals'].values)
        var_df.at[t,'LINEAR_VAR_FIT'] = np.var(time_df['LineFitVals'].values)
        
        "Compute total by summing var/cov among components"
        feature_effects_df = pd.DataFrame(feature_effects, columns=all_features)
        genetic_fit = feature_effects_df[genetic_features].sum(axis=1)
        spatial_fit = feature_effects_df[t_spatial_features].sum(axis=1)
        random_fit = feature_effects_df['RANDOM']
        genetic_spatial_cov = 2*np.cov(genetic_fit,spatial_fit)[0][1]
        genetic_random_cov = 2*np.cov(genetic_fit,random_fit)[0][1]
        spatial_random_cov = 2*np.cov(spatial_fit,random_fit)[0][1]
        
        var_df.at[t,'GENETIC_SPATIAL_COV'] = genetic_spatial_cov
        var_df.at[t,'GENETIC_RANDOM_COV'] = genetic_random_cov
        var_df.at[t,'SPATIAL_RANDOM_COV'] = spatial_random_cov
        
        var_df.at[t,'TOTAL_SUMMED'] = genetic_var + spatial_var + random_var + genetic_spatial_cov + genetic_random_cov + spatial_random_cov
        
    var_df['TOTAL_NO_COV'] = var_df['GENETIC'] + var_df['SPATIAL'] + var_df['RANDOM']
    var_df['FRAC_GENETIC'] = var_df['GENETIC'] / var_df['TOTAL_NO_COV']
    var_df['FRAC_SPATIAL'] = var_df['SPATIAL'] / var_df['TOTAL_NO_COV']
    var_df['FRAC_RANDOM'] = var_df['RANDOM'] / var_df['TOTAL_NO_COV']

    return var_df    


"Set up time intervals in absolute times to compare across trees"
absolute_time = 2020.67 # absolute time of last sample
date_time_intervals = ['2020-01-01',
                        '2020-02-15',
                  '2020-03-15',
                  '2020-04-15',
                  '2020-05-15',
                  '2020-06-15',
                  '2020-07-15',
                  '2020-08-15']
time_intervals = date2FloatYear(date_time_intervals)

"Get grid of time points to compute fitness variance"
N = 30 # number of time points to use
absolute_times = np.linspace(2020.0833, absolute_time, N, endpoint=False)

"Get tree file"
path = './covid-analysis/'
pastml_path = path + 'pastml/collected_preSep1_dated_pastml/'
tree_file = pastml_path + 'named.tree_phylogeny_mle_cleaned_collected_preSep1_dated_cleaned.nwk'

"Get df w/ overall line fitness values"
path = './covid-results/'
line_fit_file = path + "covid_fullModel_fixedSiteEffects_Oct2020_allMonthlyIntervals_lineFitEstimates.csv"
df = pd.read_csv(line_fit_file,index_col='BranchIndexes')

"Get site fitness effect estimates"
site_fit_file = path + 'covid_spaceXTimeByRegion_Oct2020_siteEffects_rhoSampling_allMonthlyIntervals_estimates.csv'
est_df = pd.read_csv(site_fit_file)
est_df.drop(['Unnamed: 0'],axis=1,inplace=True)

bestTree_var_df = get_var_df(time_intervals, tree_file, df, est_df)

bootstrap = True
bootstraps = ['01','03','07','11','12','13','14','16','17','19']
if bootstrap:
    boot_df = pd.DataFrame(index=absolute_times) #columns=features)
    for rep in bootstraps:
        
        if int(rep) < 10:
            pastml_path = '/Users/david/Desktop/bootstrapped_ancestral_features/phylogeny_bootstrap_' + rep + '_cleaned_collected_preSep1_dated_pastml/'
            tree_file = pastml_path + 'named.tree_phylogeny_bootstrap_' + rep + '_cleaned_collected_preSep1_dated.nwk'
        else:
            pastml_path = '/Users/david/Desktop/bootstrapped_ancestral_features/phylogeny_bootstrap_' + rep + '_cleaned_collected_preSep1_secondSet_dated_pastml/'
            tree_file = pastml_path + 'named.tree_phylogeny_bootstrap_' + rep + '_cleaned_collected_preSep1_secondSet_dated.nwk'
        
        "Get df w/ overall line fitness values"
        line_fit_file = path + "covid_fullModel_fixedSiteEffects_Oct2020_allMonthlyIntervals_bootstrap_" + rep + "_lineFitEstimates.csv"        
        df = pd.read_csv(line_fit_file,index_col='BranchIndexes')
    
        "Get site fitness effect estimates"
        site_fit_file = path + 'covid_spaceXTimeByRegion_Oct2020_allMonthlyIntervals_bootstrap_' + rep + '_siteEffects_estimates.csv'
        est_df = pd.read_csv(site_fit_file)
        est_df.drop(['Unnamed: 0'],axis=1,inplace=True)
        
        var_df = get_var_df(time_intervals, tree_file, df, est_df)
        
        boot_df['FRAC_GENETIC_' + rep] = var_df['FRAC_GENETIC']
        boot_df['FRAC_SPATIAL_' + rep] = var_df['FRAC_SPATIAL']
        boot_df['FRAC_RANDOM_' + rep] = var_df['FRAC_RANDOM']
        
        boot_df['LINEAR_AVG_FIT_' + rep] = var_df['LINEAR_AVG_FIT']
        boot_df['LINEAR_VAR_FIT_' + rep] = var_df['LINEAR_VAR_FIT']
        
    boot_df.to_csv("fit_decomp_boot_reps.csv",index='times')
else:
    boot_df = pd.read_csv("fit_decomp_boot_reps.csv") #,index_col='times')

"Reindex by true (absolute) times before plotting"
#var_df['AbsoluteTime'] = var_df.index.values + root_time
#var_df.set_index('AbsoluteTime', inplace=True)
#var_df.drop(var_df[var_df.index < 2020.0].index, inplace=True)

"Plot fraction of variance explained by genetic, spatial and random effects through time"
sns.set(style="darkgrid")
#sns.set_palette(sns.color_palette("cubehelix", 3))
fig, axs = plt.subplots(figsize=(6,4))
var_df = bestTree_var_df
sub_df = var_df[['FRAC_RANDOM','FRAC_SPATIAL','FRAC_GENETIC']]
sub_df = sub_df.rename(columns={"FRAC_RANDOM": "Random", "FRAC_SPATIAL": "Spatial", "FRAC_GENETIC": 'Genetic'})
sns.lineplot(data=sub_df, linewidth=2.5, dashes=False) #, markers=['o']*10)
axs.legend(loc='upper center') #, bbox_to_anchor=(0.8, 0.6), ncol=1)
for rep in bootstraps:
    axs.plot(absolute_times, boot_df['FRAC_RANDOM_' + rep], c=sns.color_palette()[0], lw=1.0, alpha=0.2, label = None)
    axs.plot(absolute_times, boot_df['FRAC_SPATIAL_' + rep], c=sns.color_palette()[1], lw=1.0, alpha=0.2, label = None)
    axs.plot(absolute_times, boot_df['FRAC_GENETIC_' + rep], c=sns.color_palette()[2], lw=1.0, alpha=0.2, label = None)
    
step_freq = 1/12
xticks = np.arange(2020+step_freq,absolute_time,step_freq)
axs.set_xticks(xticks)
labels = ['Feb','Mar','Apr','May','June','July','Aug','Sep']
axs.set_xticklabels(labels) #rotation='vertical'
axs.set_xlabel('Month', fontsize=14)
axs.set_ylabel('Fitness variation explained', fontsize=14) 
plt.savefig('covid_fitVarDecomp_fixedSiteEffects_Oct2020_allMonthlyIntervals_bootstraped_timeSeries.png', dpi=300)

"Plot mean and variance on a linear scale"
# fig, axs = plt.subplots(figsize=(6,4))
# var_df = bestTree_var_df
# sub_df = var_df[['LINEAR_AVG_FIT','LINEAR_VAR_FIT']]
# sub_df = sub_df.rename(columns={'LINEAR_AVG_FIT': 'Mean', 'LINEAR_VAR_FIT': 'Variance'})
# sns.lineplot(data=sub_df, linewidth=2.5, dashes=False) #, markers=['o']*10)
# axs.legend(loc='upper center') #, bbox_to_anchor=(0.8, 0.6), ncol=1)
# for rep in bootstraps:
#     axs.plot(absolute_times, boot_df['LINEAR_AVG_FIT_' + rep], c=sns.color_palette()[0], lw=1.0, alpha=0.2, label = None)
#     axs.plot(absolute_times, boot_df['LINEAR_VAR_FIT_' + rep], c=sns.color_palette()[1], lw=1.0, alpha=0.2, label = None)
# step_freq = 1/12
# xticks = np.arange(2020+step_freq,absolute_time,step_freq)
# axs.set_xticks(xticks)
# labels = ['Feb','Mar','Apr','May','June','July','Aug','Sep']
# axs.set_xticklabels(labels) #rotation='vertical'
# axs.set_xlabel('Month', fontsize=14)
# axs.set_ylabel('Fitness', fontsize=14) 
# plt.savefig('covid_fitVarMean_fixedSiteEffects_Oct2020_extraTimeIntervals_bootstraped_timeSeries.png', dpi=300)

