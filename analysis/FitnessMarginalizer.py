"""
Created on Wed Sep 30 13:30:50 2020

Compute the marginal (average) fitness background of all lineages with a specific feature

@author: david
"""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import TreeUtils
from ete3 import Tree
from DateTimeUtils import date2FloatYear

def get_marg_fit_df(time_intervals,tree_file, df, est_df):
    
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
    marg_fit_df_cols = ['TOTAL_FIT_614D','TOTAL_FIT_614G','GENETIC_FIT_614D','SPATIAL_FIT_614D','GENETIC_FIT_614G','SPATIAL_FIT_614G']
    marg_fit_df = pd.DataFrame(index=absolute_times,columns=marg_fit_df_cols,dtype='float64')
    marg_fit_df.fillna(1.,inplace=True) # populate with ones
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
        #random_fit = np.reshape(np.log(time_df['LineBranchEffects'].values), (-1, 1))
        #all_features.append('RANDOM')
        #feature_effects = np.hstack((feature_effects,random_fit))
        
        "Put feature effects in df"
        feature_effects_df = pd.DataFrame(feature_effects, index=time_df.index, columns=all_features)
        clean_spatial_features = [f.replace('_'+t_index_str,'') for f in t_spatial_features]
        feature_effects_df.rename(columns=dict(zip(t_spatial_features, clean_spatial_features)),inplace=True) # remove time index from spatial features names
    
        "Drop feature we are splitting on so we are only considering background effects"
        feature_effects_df.drop(['nsp12_P323L+S_D614G'], axis=1, inplace=True)
        split_genetic_features = genetic_features.copy()
        split_genetic_features.remove('nsp12_P323L+S_D614G')
        all_features = split_genetic_features + clean_spatial_features
    
        "Compute total fitness contribution from genetic feautres"
        feature_effects_df['TOTAL_FIT'] = feature_effects_df[all_features].sum(axis=1) # assuming multiplicative effects
        feature_effects_df['TOTAL_FIT'] = feature_effects_df['TOTAL_FIT'].apply(np.exp) # convert back to linear scale
        feature_effects_df['GENETIC_FIT'] = feature_effects_df[split_genetic_features].sum(axis=1) # assuming multiplicative effects
        feature_effects_df['GENETIC_FIT'] = feature_effects_df['GENETIC_FIT'].apply(np.exp) # convert back to linear scale
        feature_effects_df['SPATIAL_FIT'] = feature_effects_df[clean_spatial_features].sum(axis=1) # assuming multiplicative effects
        feature_effects_df['SPATIAL_FIT'] = feature_effects_df['SPATIAL_FIT'].apply(np.exp) # convert back to linear scale
        
        "Add binary variable we are splitting on"
        feature_effects_df['D614G'] = time_df['nsp12_P323L+S_D614G']
        gdf = feature_effects_df.groupby(['D614G'])
        means = gdf.mean()
        
        "Means df will not necessarily have both variants if one variant is not present at this time"
        if 0.0 in means.index:
            marg_fit_df.at[t,'TOTAL_FIT_614D'] = means.loc[0.0]['TOTAL_FIT']
            marg_fit_df.at[t,'GENETIC_FIT_614D'] = means.loc[0.0]['GENETIC_FIT']
            marg_fit_df.at[t,'SPATIAL_FIT_614D'] = means.loc[0.0]['SPATIAL_FIT']
        if 1.0 in means.index:
            marg_fit_df.at[t,'TOTAL_FIT_614G'] = means.loc[1.0]['TOTAL_FIT']
            marg_fit_df.at[t,'GENETIC_FIT_614G'] = means.loc[1.0]['GENETIC_FIT']
            marg_fit_df.at[t,'SPATIAL_FIT_614G'] = means.loc[1.0]['SPATIAL_FIT']
            
    "Compute fit ratios"
    marg_fit_df['TOTAL_RELATIVE_FIT'] = marg_fit_df['TOTAL_FIT_614G'] / marg_fit_df['TOTAL_FIT_614D']
    marg_fit_df['GENETIC_RELATIVE_FIT'] = marg_fit_df['GENETIC_FIT_614G'] / marg_fit_df['GENETIC_FIT_614D']
    marg_fit_df['SPATIAL_RELATIVE_FIT'] = marg_fit_df['SPATIAL_FIT_614G'] / marg_fit_df['SPATIAL_FIT_614D']
    
    return marg_fit_df


"Set up time intervals if have time-varying params"    
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
N = 12 # number of time points to use
start_time = 2020.0 #2020.0833 #2020.039
end_time = 2020.497
absolute_times = np.linspace(start_time, end_time, N, endpoint=False)

"Get tree file"
path = './covid-analysis/'
pastml_path = path + 'pastml/collected_preSep1_dated_pastml/'
tree_file = pastml_path + 'named.tree_phylogeny_mle_cleaned_collected_preSep1_dated_cleaned.nwk'

"Get df w/ overall line fitness values"
path = './covid-results/'
line_fit_file = path + "covid_spaceXTimeByRegion_Oct2020_rhoSampling_allMonthlyIntervals_lineFitEstimates.csv"
df = pd.read_csv(line_fit_file,index_col='BranchIndexes')

"Get site fitness effect estimates"
site_fit_file = path + 'covid_spaceXTimeByRegion_Oct2020_siteEffects_rhoSampling_allMonthlyIntervals_estimates.csv'
est_df = pd.read_csv(site_fit_file)
est_df.drop(['Unnamed: 0'],axis=1,inplace=True)

bestTree_marg_fit_df = get_marg_fit_df(time_intervals, tree_file, df, est_df)

bootstrap = True
bootstraps = ['01','03','07','11','12','13','14','16','17','19'] # missing bootstrap rep 07?
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
        line_fit_file = path + "covid_spaceXTimeByRegion_Oct2020_allMonthlyIntervals_bootstrap_" + rep + "_lineFitEstimates.csv"
        df = pd.read_csv(line_fit_file,index_col='BranchIndexes')
    
        "Get site fitness effect estimates"
        site_fit_file = path + 'covid_spaceXTimeByRegion_Oct2020_allMonthlyIntervals_bootstrap_' + rep + '_siteEffects_estimates.csv'
        est_df = pd.read_csv(site_fit_file)
        est_df.drop(['Unnamed: 0'],axis=1,inplace=True)
        
        marg_fit_df = get_marg_fit_df(time_intervals, tree_file, df, est_df)
        
        boot_df['TOTAL_RELATIVE_FIT_' + rep] =  marg_fit_df['TOTAL_RELATIVE_FIT']
        boot_df['GENETIC_RELATIVE_FIT_' + rep] =  marg_fit_df['GENETIC_RELATIVE_FIT']
        boot_df['SPATIAL_RELATIVE_FIT_' + rep] =  marg_fit_df['SPATIAL_RELATIVE_FIT']
        
    boot_df.to_csv("fit_marg_boot_reps.csv",index='times')
else:
    boot_df = pd.read_csv("fit_marg_boot_reps.csv") #,index_col='times')


"Compute time averages"
means = bestTree_marg_fit_df.mean()

"Plot relative total, genetic and spatial fitness for G vs D variant"
sns.set(style="darkgrid")
#sns.set_palette(sns.color_palette("cubehelix", 3))
fig, axs = plt.subplots(figsize=(6,4))
marg_fit_df = bestTree_marg_fit_df
sub_df = marg_fit_df[['TOTAL_RELATIVE_FIT','GENETIC_RELATIVE_FIT','SPATIAL_RELATIVE_FIT']]
sub_df.rename(columns={'TOTAL_RELATIVE_FIT':'Total','GENETIC_RELATIVE_FIT':'Genetic','SPATIAL_RELATIVE_FIT':'Spatial'},inplace=True)
sns.lineplot(data=sub_df, linewidth=2.5, dashes=False) #, markers=['o']*10)
axs.legend(loc='upper right') #, bbox_to_anchor=(0.4, 0.8), ncol=1)
for rep in bootstraps:
    axs.plot(absolute_times, boot_df['TOTAL_RELATIVE_FIT_' + rep], c=sns.color_palette()[0], lw=1.0, alpha=0.2, label = None)
    axs.plot(absolute_times, boot_df['GENETIC_RELATIVE_FIT_' + rep], c=sns.color_palette()[1], lw=1.0, alpha=0.2, label = None)
    axs.plot(absolute_times, boot_df['SPATIAL_RELATIVE_FIT_' + rep], c=sns.color_palette()[2], lw=1.0, alpha=0.2, label = None)

step_freq = 1/12
xticks = np.arange(2020+step_freq,end_time,step_freq)
axs.set_xticks(xticks)
labels = ['Feb','Mar','Apr','May','June','July']
axs.set_xticklabels(labels) #rotation='vertical'
axs.set_xlabel('Time', fontsize=14)
axs.set_ylabel('Relative background fitness (G / D)', fontsize=14) 
plt.savefig('covid_D614G_relativeBackFit_allMonthlyIntervals_bootstraps_timeSeries.png', dpi=300)

"Plot background fit as bar plot"
#fig, axs = plt.subplots(figsize=(2,2.5))
#variants = np.array(['614D','614G'])
#back_fit =  np.array([means['TOTAL_FIT_614D'],means['TOTAL_FIT_614G']])
#sns.barplot(x=variants, y=back_fit) #, capsize=.2)
#axs.set_ylim([0.9,1.2])
##y_displace = axs.patches[0].get_width() * 0.5 
##ylocs = [p.get_y()+y_displace for p in axs.patches]
##plt.errorbar(fit_mles, ylocs, xerr=intervals, fmt='o',color='black', ecolor='grey', elinewidth=1.5, capsize=2.5)
#axs.set_ylabel('Background fitness', fontsize=14)
#axs.set_xlabel('Variant', fontsize=14)
#fig.tight_layout()
#plt.savefig('covid_D614G_avgTotalBackFit_barPlot.png', dpi=300)
