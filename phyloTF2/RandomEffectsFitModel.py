"""
Created on Thu Jun 11 09:38:53 2020

Subclass of FitModelKeras for fitness mapping functions with random branch-specific fitness effects

Notes:
    It inherits standard site and bdm model parameters from FitModelKeras
    Random branch fitness effects are penalized under a Gaussian/Brownian motion model of fitness evolution

@author: david
"""
# TensorFlow ≥2.0 is required
import tensorflow as tf
assert tf.__version__ >= "2.0"
from tensorflow import keras
import numpy as np
from FitModelKeras import FitModelKeras

class RandomEffectsFitModel(FitModelKeras):
    
    def __init__(self,params,est_params,branches,**kwargs):
        
        "Initialize all standard bdm params and site effects using super()"
        super().__init__(params,est_params,**kwargs)
        
        "Branch-specific random fitness effects"
        self.branches = branches
        branch_effects = np.ones(self.branches) # init vals for random branch effects
        if est_params['random_branch_effects']:
            self.branch_effects = tf.Variable(branch_effects,dtype=tf.float32)
        else:
            self.branch_effects = tf.constant(branch_effects,dtype=tf.float32)
        
        "Sigma scales time-dependent std dev in Gaussian/Brownian motion model"
        if est_params['sigma']:
            #self.sigma = tf.Variable(params['sigma'],dtype=tf.float32,constraint=tf.keras.constraints.non_neg())
            sigma_constraint = lambda x: tf.clip_by_value(x, 0.0, 10)
            self.sigma = tf.Variable(params['sigma'],dtype=tf.float32,constraint=sigma_constraint)
        else:
            self.sigma = tf.constant(params['sigma'],dtype=tf.float32)
            
    """
        Implements fitness mapping function with random fitness effects
    """      
    def call(self, tree):
        
        "Set fitness effects"
        fit_effects_tensor = tf.reshape(self.site_effects,[self.sites, 1]) # reshape so rank = 2 for matmul below
    
        "Update line fit vals - multiplicative site fitness effects with random branch effects"
        log_fit_effects = tf.math.log(fit_effects_tensor) # log transform fitness effects so we "multiply" when we sum below
        line_site_effects = tf.math.exp(tf.matmul(tree.line_seqs,log_fit_effects)) # convert back to linear scale
        line_branch_effects = tf.gather(self.branch_effects, tree.line_branch_indexes)
        line_branch_effects = tf.reshape(line_branch_effects, [tree.num_line_segs,1])
        line_fit_vals = tf.multiply(line_site_effects, line_branch_effects)
        
        "Update birth event fit vals"
        birth_site_effects = tf.math.exp(tf.matmul(tree.birth_event_seqs,log_fit_effects))
        birth_branch_effects = tf.gather(self.branch_effects, tree.birth_branch_indexes)
        birth_branch_effects = tf.reshape(birth_branch_effects, [tree.num_birth_events,1])
        birth_fit_vals = tf.multiply(birth_site_effects, birth_branch_effects)
        
        "Numerical checks in line and birth fit vals"
        line_fit_vals = tf.where(line_fit_vals < 0., tf.zeros_like(line_fit_vals), line_fit_vals)
        #if tf.math.reduce_any(tf.math.is_nan(line_fit_vals)):
        #    print("Line fit values went NaN")
        birth_fit_vals = tf.where(birth_fit_vals < 0., tf.zeros_like(birth_fit_vals), birth_fit_vals)
        #if tf.math.reduce_any(tf.math.is_nan(birth_fit_vals)):
        #    print("Birth fit values went NaN")
        
        "Pack fit_vals and params as tuples"
        fit_vals = (line_fit_vals, birth_fit_vals)
        bdm_params = (self.beta_series,self.d_series,self.gamma_series,self.s_series,self.rho)
        
        return fit_vals, bdm_params
    
    """
        Compute penalty for random fitness effects under a Gaussian/Brownian motion model of fitness (trait) evolution
    """
    def get_penalty(self,tree):
        
        "Compute cost/penalty of fitness shifts between parent and child nodes under Brownian motion evolution"
        fit_shifts = tf.gather(self.branch_effects, tree.child_branch_indexes) - tf.gather(self.branch_effects, tree.parent_branch_indexes)
        times = tree.time_to_last_observed # time to last parent node which was "observed" i.e. ignore unobserved unifications along branch
        
        "Original way"
        #probs = tf.math.exp(-0.5 * fit_shifts**2 / (self.sigma * times)) # variance is proportional to time * sigma
        #penalty = tf.reduce_sum(tf.math.log(probs)) # Sum log prob values
        
        "Rewrote this way so we don't exponentiate (and get really small numbers) before taking log for greater numerically stability"
        epsilon = 0.005 # small value added to denom below to increase numerical stability when sigma * delta_t is << 1.0
        log_probs = (-0.5 * fit_shifts**2) / (self.sigma * times + epsilon) 
        penalty = tf.reduce_sum(log_probs)
    
        return penalty
    
    def get_line_branch_effects(self, tree):
    
        line_branch_effects = tf.gather(self.branch_effects, tree.line_branch_indexes)
        line_branch_effects = tf.reshape(line_branch_effects, [tree.num_line_segs,1])
        
        return line_branch_effects
    
    def set_line_branch_effects(self, val_tree):
        
        "Set random branch effects for branches in validation period based on fitness inherited from parents in training period"
        val_branch_effects = self.branch_effects.numpy() # Get current values of branch effects as numpy array we can operate on
        for idx, branch in enumerate(val_tree.child_branch_indexes):
            parent_branch_index = val_tree.parent_branch_indexes[idx]
            while parent_branch_index in val_tree.child_branch_indexes:
                next_parent_idx = np.where(val_tree.child_branch_indexes == parent_branch_index)[0][0]
                parent_branch_index = val_tree.parent_branch_indexes[next_parent_idx]
            val_branch_effects[branch] = self.branch_effects[parent_branch_index]    
        
        self.branch_effects.assign(tf.constant(val_branch_effects))
        
    def store_line_branch_effects(self):
        
        "Store a copy of current line branch effects"
        self.stored_branch_effects = self.branch_effects.numpy()
        
    def check_line_branch_effects(self):
        
        if tf.math.reduce_any(tf.math.is_nan(self.branch_effects)):
            print("Line branch effects went NaN")
            self.branch_effects = tf.Variable(tf.where(tf.math.is_nan(self.branch_effects), 1.0, self.branch_effects))
            #self.branch_effects = tf.Variable(tf.where(tf.math.is_nan(self.branch_effects), self.stored_branch_effects, self.branch_effects))
    
    
    def clip_line_branch_effects(self,min_thresh=0.1,max_thresh=10.0):
        
        "Need to clip ourselves since passing constraints on tf.Variables does not work if variable is sparse like branch_effects"
        
        min_val = tf.math.reduce_min(self.branch_effects)
        if (min_val < min_thresh):
            self.branch_effects = tf.Variable(tf.where(self.branch_effects < min_thresh, min_thresh, self.branch_effects))
            
        max_val = tf.math.reduce_max(self.branch_effects)
        if (max_val > max_thresh):
            self.branch_effects = tf.Variable(tf.where(self.branch_effects > max_thresh, max_thresh, self.branch_effects))
        
    

