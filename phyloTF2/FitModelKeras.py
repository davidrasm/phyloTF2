"""
Created on Thu Jun 11 09:38:53 2020

A class for birth-death-sampling models where the fitness of a lineage can depend on its ancestral features.
The __init_()_ function sets the initial values of all parameters in the birth-death model. 
The call() function implements the fitness mapping function used to translate a lineage’s ancestral features into its expected fitness. 
The call() function returns the fitness values of each tree lineage (fit_vals) and other birth-death model parameters (bdm_params) required to compute the likelihood of a tree using TreeLikeLoss. 
Subclasses of FitModelKeras (e.g. RandomEffectsFitModel) can be created to implement different fitness mapping functions. 


Notes:
    -Model implicitly assumes that all birth-death model parameters are time varying, and therefore have dimensions 1 x n
    -Time-varying parameters can be provided as a list. Note that all param lists must have size 1 x n
    -However, if only a single value is provided, the parameter is held constant at this value

@author: david
"""
# TensorFlow ≥2.0 is required
import tensorflow as tf
assert tf.__version__ >= "2.0"
from tensorflow import keras
import numpy as np
import sys

class FitModelKeras(keras.Model):
    
    """
        params: parameter dictionary with entries 'ParamName':InitValue
        est_params: boolean dict matching entries of params. True values indicate param
            will be estimated as a TF variable, otherwise param will be TF constant.
    """
    def __init__(self,params,est_params,**kwargs):
        
        super().__init__(**kwargs)
        
        "Initialize model parameters as TF variables"
        self.sites = params['sites']
        
        "Site effects contains the fitness effect of each feature in model"
        if "fitness_effects" in params:
            fitness_effects = params['fitness_effects']
        else: 
            fitness_effects = np.ones(self.sites) # if multiplicative effects
        if est_params['site_effects']:
            self.site_effects = tf.Variable(fitness_effects,dtype=tf.float32)
        else:
            self.site_effects = tf.constant(fitness_effects,dtype=tf.float32)
        
        "Initialize birth-death model parameters"
        n = len(params['time_intervals'])
        
        "Birth/transmission rate beta"
        try:
            beta_series = params['beta'] * np.ones(n)
        except ValueError:
            sys.exit("Initial params must be a scalar value or array with length = # of time intervals")
        if est_params['beta']:
            self.beta_series = tf.Variable(beta_series,dtype=tf.float32,name="beta_series")
        else:
            self.beta_series = tf.constant(beta_series,dtype=tf.float32)
        
        "Death/removal rate d"
        try:
            d_series = params['d'] * np.ones(n)
        except ValueError:
            sys.exit("Initial params must be a scalar value or array with length = # of time intervals")
        if est_params['d']:
            self.d_series = tf.Variable(d_series,dtype=tf.float32)
        else:
            self.d_series = tf.constant(d_series,dtype=tf.float32)
        
        "Migration/transition rate gamma -- should be zero for models assuming known ancestral features"
        try:
            gamma_series = params['gamma'] * np.ones(n)
        except ValueError:
            sys.exit("Initial params must be a scalar value or array with length = # of time intervals")
        if est_params['gamma']:
            self.gamma_series = tf.Variable(gamma_series,dtype=tf.float32)
        else:
            self.gamma_series = tf.constant(gamma_series,dtype=tf.float32)
            
        "Sampling fraction s"
        try:
            s_series = params['s'] * np.ones(n)
        except ValueError:
            sys.exit("Initial params must be a scalar value or array with length = # of time intervals")
        if est_params['s']:
            self.s_series = tf.Variable(s_series,dtype=tf.float32)
        else:
            self.s_series = tf.constant(s_series,dtype=tf.float32)
            
        "Contemporaneous sampling at present rho (assumed to be a single value)"
        if est_params['rho']:
            self.rho = tf.Variable(params['rho'],dtype=tf.float32)
        else:
            self.rho = tf.constant(params['rho'],dtype=tf.float32)
            
    """
        Call function implements fitness mapping function
        Input tree is TensorTree object for which we are computing fit_vals
    """    
    def call(self, tree):
        
        "Set fitness effects"
        fit_effects_tensor = tf.reshape(self.site_effects,[self.sites, 1]) # reshape so rank = 2 for matmul below
    
        "Fitness mapping function with multiplicative fitness effects"
        log_fit_effects = tf.math.log(fit_effects_tensor) # log transform fitness effects so we "multiply" when we sum below
        line_fit_vals = tf.math.exp(tf.matmul(tree.line_seqs,log_fit_effects)) # convert back to linear scale
        birth_fit_vals = tf.math.exp(tf.matmul(tree.birth_event_seqs,log_fit_effects))
        
        "Pack fit_vals and params as tuples"
        fit_vals = (line_fit_vals, birth_fit_vals)
        bdm_params = (self.beta_series,self.d_series,self.gamma_series,self.s_series,self.rho)
        
        return fit_vals, bdm_params
    
    def setRandomSiteEffects(self):
        
        starting_effects = np.random.uniform(0.75, 1.25, self.sites).reshape((1,self.sites))
        self.site_effects = tf.Variable(starting_effects, dtype=tf.float32)
        
    def check_site_effects(self):
        
        if tf.math.reduce_any(tf.math.is_nan(self.site_effects)):
            print("Site effects went NaN")
            self.site_effects = tf.Variable(tf.where(tf.math.is_nan(self.site_effects), 1.0, self.site_effects))
        