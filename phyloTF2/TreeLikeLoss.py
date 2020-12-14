"""
Created on Thu Jun 11 11:29:07 2020

Computes the likelihood of a phylogenetic tree under a birth-death-sampling model with lineage-specific fitness values. 
This is the loss or objective function that is optimized by gradient descent. 

To do:
    -Remove needless self references for local tensor variables in call method
    -Added ability to turn on and off rho sampling in likelihood function

@author: david
"""
# TensorFlow ≥2.0 is required
import tensorflow as tf
assert tf.__version__ >= "2.0"
from tensorflow import keras
import numpy as np

class TreeLikeLoss(keras.losses.Loss):
    
    def __init__(self,tree,params,**kwargs):
        
        super().__init__(**kwargs)
        
        self.time_intervals = params['time_intervals']
        n = len(self.time_intervals)
        self.time_indexes = np.arange(0,n)
        self.final_time = np.max(self.time_intervals)
        line_time_intervals = tf.constant(self.time_intervals[tree.line_time_intervals], dtype=tf.float32)
        self.line_time_ics = tf.reshape(self.final_time - line_time_intervals, [tree.num_line_segs,1])
        
        "Create a tensor of ones"
        self.ones_tensor = tf.ones(shape=[tree.num_line_segs, 1],dtype=tf.dtypes.float32)
        self.ones_sample_tensor = tf.ones(shape=[tree.num_sample_events,1],dtype=tf.dtypes.float32)
        
        "Gather time indexes by lineage and put them in a list of 2D indexes: we need to add +1 to j to account to account for how the pE values are padded below"
        self.gather_list = [[i,j+1] for i, j in zip(list(range(len(tree.line_time_intervals))),tree.line_time_intervals.tolist())]
    
    """
        Call function computes likelihood
        fit_vals: lineage-specific fitness values returned by FitModelKeras
        bdm_params: other birth-death model params
        tree: TensorTree object
        rho_sampling: if True, a fraction (rho) of lineages are assumed to be sampled at present, pE_init = 1 - rho
        iterative_pE: if True, the pE probs for each lineages is computed recursively over each time interval taking into account changes in birth-death params
    """    
    def call(self,fit_vals,bdm_params,tree,rho_sampling=True,iterative_pE=False):
        
        "Unpack fit_vals and params"
        line_fit_vals, birth_fit_vals = fit_vals
        beta_series, d_series, gamma_series, s_series, rho = bdm_params
        
        "Compute pE at line_back_times for all lineage segments at once"
        if rho_sampling:
            self.pE_init = 1.0 - tf.constant(rho, shape=[tree.num_line_segs, 1])
        else:
            self.pE_init = tf.ones(shape=[tree.num_line_segs, 1],dtype=tf.dtypes.float32)
        self.pE = self.pE_init
        if iterative_pE:
            for tx in np.flip(self.time_indexes):
                
                line_betas = beta_series[tx] * line_fit_vals
                
                "Note: will need to update other params when variable, e.g."
                self.d_tensor = self.ones_tensor * d_series[tx]
                self.gamma_tensor = self.ones_tensor * gamma_series[tx]
                self.s_tensor = self.ones_tensor * s_series[tx]
                
                gbd_sum = self.gamma_tensor + line_betas + self.d_tensor 
                cnst_c = tf.sqrt(tf.square(gbd_sum) - 4*self.d_tensor*(1-self.s_tensor)*line_betas)
                cnst_x = (-gbd_sum - cnst_c) / 2
                cnst_y = (-gbd_sum + cnst_c) / 2
                
                "Compute backwards times i.e. time since present at curr time and time of last init cond time_ic"
                time_ic = self.final_time - self.time_intervals[tx] # time of init conditions (in backwards time)
                if tx > 0:
                    time = self.final_time - self.time_intervals[tx-1] # time of next interval (in backwards time)
                else:
                    time = self.final_time
                
                pE_num = (cnst_y + line_betas * self.pE_init) * cnst_x * tf.exp(-cnst_c * time) - cnst_y*(cnst_x + line_betas * self.pE_init) * tf.exp(-cnst_c * time_ic)
                pE_denom = (cnst_y + line_betas * self.pE_init) * tf.exp(-cnst_c * time) - (cnst_x + line_betas * self.pE_init) * tf.exp(-cnst_c * time_ic)
                self.pE_init = (-1/line_betas) * pE_num / pE_denom
                #print(pE_init.eval())
                
                self.pE = tf.concat([self.pE_init, self.pE], 1)
        
        "Compute line and birth betas for time interval of each lineage - now always assumes multiplicative effects"
        line_betas = line_fit_vals * tf.reshape(tf.gather(beta_series,tree.line_time_intervals), [tree.num_line_segs,1])
        birth_betas = birth_fit_vals * tf.reshape(tf.gather(beta_series,tree.birth_time_intervals), [tree.num_birth_events,1])

        "Update other time-varying bdm params"
        self.d_tensor = tf.reshape(tf.gather(d_series,tree.line_time_intervals), [tree.num_line_segs,1])
        self.gamma_tensor = tf.reshape(tf.gather(gamma_series,tree.line_time_intervals), [tree.num_line_segs,1])
        self.s_tensor = tf.reshape(tf.gather(s_series,tree.line_time_intervals), [tree.num_line_segs,1])
        
        "Compute constants for time interval of each lineage"
        gbd_sum = self.gamma_tensor + line_betas + self.d_tensor
        
        "Compute constants -- in rare instances cnst_c_squared can be < 0 (due to numerical error??) in which case taking the sqrt will result in csnt_c having NaNs"
        cnst_c_squared = tf.square(gbd_sum) - 4*self.d_tensor*(1-self.s_tensor)*line_betas
        cnst_c_squared = tf.where(cnst_c_squared < 0.0, 0.0, cnst_c_squared) # remove values < 0.0 before squaring cnst_c
        cnst_c = tf.sqrt(cnst_c_squared)
        cnst_x = (-gbd_sum - cnst_c) / 2
        cnst_y = (-gbd_sum + cnst_c) / 2
        
        "pE values are computed for nearest time index, not the exact time backwards from present of lineage end point"
        "Point of clarification: pE is essentially acting as an initialization condition at the next time interval"
        "If we are assuming all rates are constant along unobserved lineages then pE = pE(t=0) = 1-rho and line_time_ics --> 0"
        if iterative_pE:
            pE = tf.reshape(tf.gather_nd(self.pE, self.gather_list), [tree.num_line_segs,1])
        else:
            pE = self.pE_init
            self.line_time_ics = tf.zeros(shape=[tree.num_line_segs, 1],dtype=tf.dtypes.float32)
        
        "Correct pE values to reflect difference between line_back_times and the interval times"
        pE_num = (cnst_y + line_betas * pE) * cnst_x * tf.exp(-cnst_c * tree.line_back_times) - cnst_y*(cnst_x + line_betas * pE) * tf.exp(-cnst_c * self.line_time_ics)
        pE_denom = (cnst_y + line_betas * pE) * tf.exp(-cnst_c * tree.line_back_times) - (cnst_x + line_betas * pE) * tf.exp(-cnst_c * self.line_time_ics)
        
        "Compute pE and ensure pE_denom does not contain zeros"
        min_denom = 0.01
        pE_denom = tf.where(pE_denom == 0.0, min_denom, pE_denom)
        pE = (-1/line_betas) * pE_num / pE_denom
        
        "Compute pD for all lineage segments at once and ensure pD_denom does not contain zeros"
        pD_denom = ((cnst_y + line_betas * pE) * tf.exp(-cnst_c * tree.line_time_steps)) - (cnst_x + line_betas * pE)
        pD_denom = tf.where(pD_denom == 0.0, min_denom, pD_denom)
        pD = tf.exp(-cnst_c * tree.line_time_steps) * tf.square((cnst_y - cnst_x) / pD_denom)

        "New stuff added for debugging"
        if tf.math.reduce_any(tf.math.is_nan(pD)):
            print("pD values went NaN")
            
        "Soft threshold to catch NaN's in log likelihood - no longer used"
        #threshold = float(-100)
        #log_line_like = tf.math.log(pD)
        #if tf.math.reduce_any(tf.math.is_nan(log_line_like)):
            #log_line_like = tf.where(tf.math.is_nan(log_line_like), threshold, log_line_like)
        #line_like = tf.reduce_sum(log_line_like)
        
        "Compute full tree likelihood"
        line_like = tf.reduce_sum(tf.math.log(pD)) # sum of log likelihoods of each lineage segment
        
        "Compute sample event probs"        
        sample_event_probs = tf.reshape(tf.gather(s_series,tree.sample_time_intervals), [tree.num_sample_events,1])
        d_sample_tensor =  tf.reshape(tf.gather(d_series,tree.sample_time_intervals), [tree.num_sample_events,1])
        sample_event_probs = sample_event_probs * d_sample_tensor
        if rho_sampling:
            rho_event_probs = rho * self.ones_sample_tensor
            sample_event_probs = tf.where(tree.rho_sampling_events, rho_event_probs, sample_event_probs)
        sample_like = tf.reduce_sum(tf.math.log(sample_event_probs)) # sum of log likelihoods of each sampling event -- now computed here rather than in TensorTree to allow for time-varying sampling rates
        
        "Check birth rates for numerical issues"
        if tf.math.reduce_any(tf.math.is_nan(birth_betas)):
            print("Birth rates went NaN")   
        
        "Soft threshold to catch NaN's in likelihood - no longer used"
        #log_birth_like = tf.math.log(2*birth_betas)
        #if tf.math.reduce_any(tf.math.is_nan(log_birth_like)):
        #    log_birth_like = tf.where(tf.math.is_nan(log_birth_like), threshold, log_birth_like)
        #birth_like = tf.reduce_sum(log_birth_like)
        
        "Compute like of birth events"
        #birth_like = tf.reduce_sum(tf.math.log(2*birth_betas)) # old way multiplying by 2x as we would under a MTBD with unknown states
        birth_like = tf.reduce_sum(tf.math.log(birth_betas)) #new way - does not impact inference because 2x factor only adds a constant value to overall likelihood
        
        total_like = line_like + sample_like + birth_like
        
        return total_like
