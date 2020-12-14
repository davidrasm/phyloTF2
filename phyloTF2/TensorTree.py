"""
Created on Wed Mar 27 13:22:58 2019

TensorTree class provides a data structure for working with phylogenetic trees in TensorFlow. 
Trees are converted into a set of TF Tensor objects (i.e. n-dim arrays) to vectorize operations involved in computing the likelihood of trees under a birth-death-sampling model.

Updated version appending to lists instead of numpy arrays to init TensorTree

@author: david
"""
import tensorflow as tf
import numpy as np
import random

class TensorTree(object):
    
    """
        __init__: creates a new TensorTree object
        features_dic: feature dictionary with anc features as numpy array for each lineage 'LineageName':features
    """
    def __init__(self,tree,features_dic,**params):
        
        if not tree:
            
            "If tree = None, create empty TensorTree object"
            
        else:
            
            "Convert Tree object to TensorFlow tensor arrays"
            dt = params['dt'] # time step interval for splitting lineages into segments
            time_intervals = params['time_intervals']
            time_indexes = np.arange(0,len(time_intervals))
            final_time = max(time_intervals)
            
            "Numpy arrays to hold vectorized tree"
            line_time_steps = [] # length of each lineage segment in time
            line_back_times = [] # distance of each lineage segment from present
            line_start_times = [] # time at which lineage segment starts
            line_end_times = [] # time at which lineage segment ends
            line_seqs = [] # (ancestral) seq of each lineage segment
            line_names = [] # record of lineage names for debugging
            rho_sampling_events = []
            birth_event_seqs = []
            
            "Added for models with time-varying fitness effects"
            self.line_time_intervals = []
            self.birth_time_intervals = []
            self.sample_time_intervals = []
            
            "Added for models with random branch fitness effects"
            self.line_branch_indexes = [] # unique index for each lineage to map back to lineages in original tree
            self.birth_branch_indexes = []
            self.child_branch_indexes = []
            self.parent_branch_indexes = []
            self.time_to_last_observed = [] # time to last observed (bifurcating node)
    
            "Added for cross-validation methods"
            self.birth_event_times = [] # absolute time of birth events
            self.sample_event_times = []
            self.node_event_times = []
            
            for node in tree.traverse("postorder"):
                
                "Add branch index for retriving branch-specific fitness values"
                branch_index = node.branch_idx
                
                "Root is at time zero with time increasing towards tips"
                end_time = node.time # time closer to present
                start_time = end_time - node.dist  # time further in past
                
                "Get feature array"
                feature_array = features_dic[node.name].tolist()
                sites = len(feature_array)
    
                # Split branch into smaller time segments
                if start_time == end_time:
                    branch_time_segments = [start_time] # Added this to deal with special cases in covid trees
                else:
                    branch_time_segments = np.arange(start_time, end_time, dt).tolist()
                branch_time_segments.append(end_time)

                # For debugging
                #if len(branch_time_segments) <= 1:
                #    print('Could be problem')
            
                # Walking along branch from start to end time
                for i in range(0, len(branch_time_segments) - 1):
                    start_seg_time = branch_time_segments[i]  # start time of current branch segment
                    end_seg_time = branch_time_segments[i + 1]  # end time of current branch segment
                    back_time = final_time - end_seg_time  # time elapsed from the present
                    time_step = end_seg_time - start_seg_time  # duration of current branch segment

                    # Appened line segement data to arrays
                    line_names.append(node.name + "_seg_" + str(i + 1))
                    line_time_steps.append(time_step)
                    line_back_times.append(back_time)
                    line_seqs.append(feature_array)
                    line_start_times.append(start_seg_time)
                    line_end_times.append(end_seg_time)

                    interval = time_indexes[np.where(time_intervals >= end_seg_time)][0]
                    self.line_time_intervals.append(interval)

                    self.line_branch_indexes.append(branch_index)
                
                if node.is_leaf():
                    
                    if np.isclose(end_time,final_time):
                        rho_sample = True
                    else:
                        rho_sample = False
                    rho_sampling_events.append(rho_sample)

                    sampling_interval = time_indexes[np.where(time_intervals >= end_time)][0]
                    self.sample_time_intervals.append(sampling_interval)
                    self.sample_event_times.append(end_time)

                    self.parent_branch_indexes.append(node.parent.branch_idx)
                    self.child_branch_indexes.append(branch_index)
                    self.node_event_times.append(end_time)
                    time_to_last = end_time - node.parent.time
                    self.time_to_last_observed.append(time_to_last)
                    
                elif len(node.children) > 1:

                    birth_event_seqs.append(feature_array)

                    interval = time_indexes[np.where(time_intervals >= end_time)][0]  # interval of birth time
                    self.birth_time_intervals.append(interval)
                    self.birth_event_times.append(end_time)
                    self.birth_branch_indexes.append(branch_index)
                    
                    if node.parent:
                        self.parent_branch_indexes.append(node.parent.branch_idx)
                        self.child_branch_indexes.append(branch_index)
                        self.node_event_times.append(end_time)
                        time_to_last = end_time - node.parent.time
                        self.time_to_last_observed.append(time_to_last)

            "Convert to numpy arrays"
            self.line_time_intervals = np.array(self.line_time_intervals, dtype=int)
            self.birth_time_intervals = np.array(self.birth_time_intervals, dtype=int)
            self.sample_time_intervals = np.array(self.sample_time_intervals, dtype=int)
            self.line_branch_indexes = np.array(self.line_branch_indexes, dtype=int)
            self.birth_branch_indexes = np.array(self.birth_branch_indexes, dtype=int)
            self.child_branch_indexes = np.array(self.child_branch_indexes, dtype=int)
            self.parent_branch_indexes = np.array(self.parent_branch_indexes, dtype=int)  # use empty to ensure proper size
            self.time_to_last_observed = np.array(self.time_to_last_observed, dtype=float)  # time to last observed (bifurcating node)
            self.birth_event_times = np.array(self.birth_event_times, dtype=float)
            self.sample_event_times = np.array(self.sample_event_times, dtype=float)
            self.node_event_times = np.array(self.node_event_times, dtype=float)

            "Set up Tensor objects"
            self.num_line_segs = len(line_names)
            self.num_birth_events = len(self.birth_time_intervals) #int(len(birth_event_seqs) / sites)
            self.num_sample_events = len(rho_sampling_events)
        
            self.line_time_steps = tf.constant(line_time_steps,shape=(self.num_line_segs,1),dtype=tf.float32)
            self.line_back_times = tf.constant(line_back_times,shape=(self.num_line_segs,1),dtype=tf.float32)
            self.line_seqs = tf.constant(line_seqs,shape=(self.num_line_segs,sites),dtype=tf.float32)
            self.rho_sampling_events = tf.constant(rho_sampling_events,shape=(self.num_sample_events,1),dtype=tf.bool)
            self.birth_event_seqs = tf.constant(birth_event_seqs,shape=(self.num_birth_events,sites),dtype=tf.float32)
            
            "Variables not used by TensorFlow - stored as np arrays"
            self.line_start_times = line_start_times
            self.line_end_times = line_end_times
    
    """
        Check for negative or zero branch or lineage segment lengths
    """
    def check_line_time_steps(self,threshold=1e-06):
        
        line_time_steps = self.line_time_steps.numpy()
        min_time_step = np.min(line_time_steps)
        if min_time_step <= 0.:
            print('Found negative or zero branch lenghts in TensorTree')
        if min_time_step <= threshold:
            line_time_steps[line_time_steps < threshold] = threshold
        self.line_time_steps = tf.constant(line_time_steps,shape=(self.num_line_segs,1),dtype=tf.float32)
    
        min_back_time = np.min(self.line_back_times)
        if min_back_time < 0.:
            print('Found negative line back times in TensorTree')
    
        min_ttlo = np.min(self.time_to_last_observed)
        if min_ttlo < 0.:
            print('Found negative or zero time to last observed values in TensorTree')
        "We should also set a threshold on these otherwise we might divide by zero in BM penalty term"
        if min_ttlo <= threshold:
            self.time_to_last_observed[self.time_to_last_observed < threshold] = threshold
            
    """
        Mask time varying features (set to zero) for time intervals outside of lineage segments
        Was never happy with this -- can we find a better way?
    """    
    def mask_time_features(self,df,feature_labels,time_intervals):
                
        line_seqs_array = self.line_seqs.numpy()
        birth_seqs_array = self.birth_event_seqs.numpy()
        sites = len(line_seqs_array[0,:])
        
        for label in feature_labels: 
            
            cols = [col for col in df.columns if label in col] # find the columns that correspond to this feature
            locs = [df.columns.get_loc(col) for col in cols]
            
            time_mask_locs = {}
            for interval in range(len(time_intervals)):
                time_mask_locs[interval] = [locs[idx] for idx, col in enumerate(cols) if 't'+str(interval) not in col] # find the column locs for this feature that do not correspond to this time interval

            for i in range(self.num_line_segs):
                interval = self.line_time_intervals[i]
                line_seqs_array[i,time_mask_locs[interval]] = 0
                
            for i in range(self.num_birth_events):
                interval = self.birth_time_intervals[i]
                birth_seqs_array[i,time_mask_locs[interval]] = 0
        
        "Convert arrays back to tensors"
        self.line_seqs = tf.constant(line_seqs_array,shape=(self.num_line_segs,sites),dtype=tf.float32)
        self.birth_event_seqs = tf.constant(birth_seqs_array,shape=(self.num_birth_events,sites),dtype=tf.float32)  
    
    
    def cross_section(self,cut_times,val_period):
        
        """"
            Cross-section tree at cut time t into training and validation periods
            Currently a 'lazy' implementation b/c we don't actually cut branches
            But rather partition branches into two sets based on where their line_end_time falls
            Birth and sampling events are partitioned based on exact times
            val_period sets length of validation period
        """
        
        val_trees = []
        train_trees = []
        
        for ct in cut_times: # for each fold
        
            val_tt = TensorTree(None,None)
            train_tt = TensorTree(None,None)
            
            "Create boolean masks/arrays for lines to include/exclude in validation set"
            lines_in = np.full(self.num_line_segs, True) # lines in training set
            lines_in[self.line_end_times > ct] = False
            lines_out = np.full(self.num_line_segs, True) # lines not in training set
            lines_out[self.line_end_times < ct] = False
            lines_out[self.line_end_times > (ct+val_period)] = False
            
            "Split variables/tensors between validation and training tensor trees"
            train_tt.num_line_segs = np.sum(lines_in)
            train_tt.line_time_intervals = self.line_time_intervals[lines_in]
            train_tt.line_branch_indexes = self.line_branch_indexes[lines_in]
            train_tt.line_seqs = tf.boolean_mask(self.line_seqs, lines_in) 
            train_tt.line_back_times = tf.boolean_mask(self.line_back_times, lines_in) 
            train_tt.line_time_steps = tf.boolean_mask(self.line_time_steps, lines_in)
            
            val_tt.num_line_segs = np.sum(lines_out)
            val_tt.line_time_intervals = self.line_time_intervals[lines_out]
            val_tt.line_branch_indexes = self.line_branch_indexes[lines_out]
            val_tt.line_seqs = tf.boolean_mask(self.line_seqs, lines_out) 
            val_tt.line_back_times = tf.boolean_mask(self.line_back_times, lines_out) 
            val_tt.line_time_steps = tf.boolean_mask(self.line_time_steps, lines_out)
            
            "Create boolean masks/arrays for births"
            births_in = np.full(self.num_birth_events, True) # birth events in training set
            births_in[self.birth_event_times > ct] = False
            births_out = np.full(self.num_birth_events, True) # births not in training set
            births_out[self.birth_event_times < ct] = False
            births_out[self.birth_event_times > (ct+val_period)] = False
            
            train_tt.num_birth_events = np.sum(births_in)
            train_tt.birth_event_seqs = tf.boolean_mask(self.birth_event_seqs, births_in) 
            train_tt.birth_time_intervals = self.birth_time_intervals[births_in]
            train_tt.birth_branch_indexes = self.birth_branch_indexes[births_in]
           
            val_tt.num_birth_events =  np.sum(births_out)
            val_tt.birth_event_seqs = tf.boolean_mask(self.birth_event_seqs, births_out) 
            val_tt.birth_time_intervals = self.birth_time_intervals[births_out]
            val_tt.birth_branch_indexes = self.birth_branch_indexes[births_out]
            
            "Create boolean arrays for samples"
            samples_in = np.full(self.num_sample_events, True)
            samples_in[self.sample_event_times > ct] = False
            samples_out = np.full(self.num_sample_events, True) # births not in training set
            samples_out[self.sample_event_times < ct] = False
            samples_out[self.sample_event_times > (ct+val_period)] = False
            
            train_tt.num_sample_events = np.sum(samples_in)
            train_tt.sample_time_intervals = self.sample_time_intervals[samples_in]
            train_tt.rho_sampling_events = tf.boolean_mask(self.rho_sampling_events, samples_in) 
            
            val_tt.num_sample_events = np.sum(samples_out)
            val_tt.sample_time_intervals = self.sample_time_intervals[samples_out]
            val_tt.rho_sampling_events = tf.boolean_mask(self.rho_sampling_events, samples_out)
            
            "Create boolean arrays for parent/child branch indexes"
            indexes_in = np.full(self.parent_branch_indexes.size, True) # lines in training set
            indexes_in[self.node_event_times > ct] = False
            indexes_out = np.full(self.parent_branch_indexes.size, True)
            indexes_out[self.node_event_times < ct] = False
            indexes_out[self.node_event_times > (ct+val_period)] = False 
            
            train_tt.parent_branch_indexes = self.parent_branch_indexes[indexes_in]
            train_tt.child_branch_indexes = self.child_branch_indexes[indexes_in]
            train_tt.time_to_last_observed = self.time_to_last_observed[indexes_in]
            
            val_tt.parent_branch_indexes = self.parent_branch_indexes[indexes_out]
            val_tt.child_branch_indexes = self.child_branch_indexes[indexes_out]
            val_tt.time_to_last_observed = self.time_to_last_observed[indexes_out]
            
            val_trees.append(val_tt)
            train_trees.append(train_tt)
        
        return train_trees, val_trees
    
    def cross_section_original(self,cut_times):
        
        """"
            Cross-section tree at cut time t into training and validation periods
            Currently a 'lazy' implementation b/c we don't actually cut branches
            But rather partition branches into two sets based on where their line_end_time falls
            Birth and sampling events are partitioned based on exact times
        """
        
        val_trees = []
        train_trees = []
        
        for ct in cut_times: # for each fold
        
            val_tt = TensorTree(None,None)
            train_tt = TensorTree(None,None)
            
            "Create boolean masks/arrays for lines to include/exclude in validation set"
            lines_in = np.full(self.num_line_segs, True) # lines in training set
            lines_in[self.line_end_times > ct] = False
            lines_out = np.invert(lines_in) # lines not in training set
            
            "Split variables/tensors between validation and training tensor trees"
            train_tt.num_line_segs = np.sum(lines_in)
            train_tt.line_time_intervals = self.line_time_intervals[lines_in]
            train_tt.line_branch_indexes = self.line_branch_indexes[lines_in]
            train_tt.line_seqs = tf.boolean_mask(self.line_seqs, lines_in) 
            train_tt.line_back_times = tf.boolean_mask(self.line_back_times, lines_in) 
            train_tt.line_time_steps = tf.boolean_mask(self.line_time_steps, lines_in)
            
            train_tt.line_start_times = self.line_start_times[lines_in]
            train_tt.line_end_times = self.line_end_times[lines_in]
            
            val_tt.num_line_segs = np.sum(lines_out)
            val_tt.line_time_intervals = self.line_time_intervals[lines_out]
            val_tt.line_branch_indexes = self.line_branch_indexes[lines_out]
            val_tt.line_seqs = tf.boolean_mask(self.line_seqs, lines_out) 
            val_tt.line_back_times = tf.boolean_mask(self.line_back_times, lines_out) 
            val_tt.line_time_steps = tf.boolean_mask(self.line_time_steps, lines_out)
            
            val_tt.line_start_times = self.line_start_times[lines_out]
            val_tt.line_end_times = self.line_end_times[lines_out]
            
            "Create boolean masks/arrays for births"
            births_in = np.full(self.num_birth_events, True) # birth events in training set
            births_in[self.birth_event_times > ct] = False
            births_out = np.invert(births_in)
            
            train_tt.num_birth_events = np.sum(births_in)
            train_tt.birth_event_seqs = tf.boolean_mask(self.birth_event_seqs, births_in) 
            train_tt.birth_time_intervals = self.birth_time_intervals[births_in]
            train_tt.birth_branch_indexes = self.birth_branch_indexes[births_in]
           
            val_tt.num_birth_events =  np.sum(births_out)
            val_tt.birth_event_seqs = tf.boolean_mask(self.birth_event_seqs, births_out) 
            val_tt.birth_time_intervals = self.birth_time_intervals[births_out]
            val_tt.birth_branch_indexes = self.birth_branch_indexes[births_out]
            
            "Create boolean arrays for samples"
            samples_in = np.full(self.num_sample_events, True)
            samples_in[self.sample_event_times > ct] = False
            samples_out = np.invert(samples_in)
            
            train_tt.num_sample_events = np.sum(samples_in)
            train_tt.sample_time_intervals = self.sample_time_intervals[samples_in]
            train_tt.rho_sampling_events = tf.boolean_mask(self.rho_sampling_events, samples_in) 
            
            val_tt.num_sample_events = np.sum(samples_out)
            val_tt.sample_time_intervals = self.sample_time_intervals[samples_out]
            val_tt.rho_sampling_events = tf.boolean_mask(self.rho_sampling_events, samples_out)
            
            "Create boolean arrays for parent/child branch indexes"
            indexes_in = np.full(self.parent_branch_indexes.size, True) # lines in training set
            indexes_in[self.node_event_times > ct] = False
            indexes_out = np.invert(indexes_in) # lines not in training set
            
            train_tt.parent_branch_indexes = self.parent_branch_indexes[indexes_in]
            train_tt.child_branch_indexes = self.child_branch_indexes[indexes_in]
            train_tt.time_to_last_observed = self.time_to_last_observed[indexes_in]
            
            val_tt.parent_branch_indexes = self.parent_branch_indexes[indexes_out]
            val_tt.child_branch_indexes = self.child_branch_indexes[indexes_out]
            val_tt.time_to_last_observed = self.time_to_last_observed[indexes_out]
            
            val_trees.append(val_tt)
            train_trees.append(train_tt)
        
        return train_trees, val_trees
        
    def split_folds(self,k):
        
        "Split tree into folds for cross-validation"
        val_trees = []
        train_trees = []
        
        line_folds = self.get_splits(self.num_line_segs,k) # lines split into different folds
        birth_folds = self.get_splits(self.num_birth_events,k) # births split into different folds
        sample_folds = self.get_splits(self.num_sample_events,k) # samples split into different folds
        
        for f in range(k): # for each fold
            
            val_tt = TensorTree(None,None)
            train_tt = TensorTree(None,None)
            
            "Create boolean arrays for lines to include/exclude"
            lines_in = np.full(self.num_line_segs, False)
            lines_array = np.array(line_folds[f])
            lines_in[lines_array] = True # In means in validation set
            lines_out = np.invert(lines_in) # Out means in training set
            
            "Update variables for lines in tt"
            val_tt.num_line_segs = len(lines_array)
            val_tt.line_seqs = tf.boolean_mask(self.line_seqs, lines_in) 
            val_tt.line_time_intervals = self.line_time_intervals[lines_in]
            val_tt.line_back_times = tf.boolean_mask(self.line_back_times, lines_in) 
            val_tt.line_time_steps = tf.boolean_mask(self.line_time_steps, lines_in)
            
            train_tt.num_line_segs = self.num_line_segs - len(lines_array)
            train_tt.line_seqs = tf.boolean_mask(self.line_seqs, lines_out) 
            train_tt.line_time_intervals = self.line_time_intervals[lines_out]
            train_tt.line_back_times = tf.boolean_mask(self.line_back_times, lines_out) 
            train_tt.line_time_steps = tf.boolean_mask(self.line_time_steps, lines_out) 
            
            "Create boolean arrays for births"
            births_in = np.full(self.num_birth_events, False)
            births_array = np.array(birth_folds[f])
            births_in[births_array] = True
            births_out = np.invert(births_in)
            
            "Update variables for births in tt"
            val_tt.num_birth_events = len(births_array)
            val_tt.birth_event_seqs = tf.boolean_mask(self.birth_event_seqs, births_in) 
            val_tt.birth_time_intervals = self.birth_time_intervals[births_in]
            
            train_tt.num_birth_events = self.num_birth_events - len(births_array)
            train_tt.birth_event_seqs = tf.boolean_mask(self.birth_event_seqs, births_out) 
            train_tt.birth_time_intervals = self.birth_time_intervals[births_out]
            
            "Create boolean arrays for samples"
            samples_in = np.full(self.num_sample_events, False)
            samples_array = np.array(sample_folds[f])
            samples_in[samples_array] = True
            samples_out = np.invert(samples_in)
            
            "Update variables for sampling events tt"
            val_tt.num_sample_events = len(samples_array)
            val_tt.sample_time_intervals = self.sample_time_intervals[samples_in]
            val_tt.rho_sampling_events = tf.boolean_mask(self.rho_sampling_events, samples_in)
            
            train_tt.num_sample_events = self.num_sample_events - len(samples_array)
            train_tt.sample_time_intervals = self.sample_time_intervals[samples_out]
            train_tt.rho_sampling_events = tf.boolean_mask(self.rho_sampling_events, samples_out) 
        
            val_trees.append(val_tt)
            train_trees.append(train_tt)
            
        return train_trees, val_trees
    
    def get_splits(self,N,k):

        pop = list(range(N))
        random.shuffle(pop)

        fsize = int(N/k)
        folds = []
        for f in range(k):
            start = f*fsize
            if f < k-1:
                end = (f+1)*fsize
            else:
                end = N
            folds.append(pop[start:end])
            
        return folds
        
        
        
        
            
            
            
        
        