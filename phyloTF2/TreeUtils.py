"""

Utility functions for working with trees in ETE

Created on Fri Mar 29 10:25:28 2019

@author: david
"""
import numpy as np
from ete3 import Tree
import pandas as pd
import copy

"""
    Write multi-type tree to file in newick format
"""
def write_MTT_newick(tree,file,nexus=False):
    
    tree_file = open(file, "w") 
        
    for node in tree.traverse("postorder"):
        
        if node.is_leaf():
            
            "Set state for debugging"
            node.add_features(string=node.name)
            
        else:
            
            children = len(node.children)
            if children > 0:
                node_string = "("
                for child in range(children):
                    child_node = node.children[child]
                    child_string = child_node.string + '[&type="' + str(child_node.state) + '"]:' + str(child_node.dist)
                    node_string = node_string + child_string + ","
        
                if node_string[-1] == ",":
                    node_string = node_string[:-1]
            
                node_string = node_string + ")"
                node.add_features(string=node_string)
    
    root = tree.get_tree_root()
    root_string = root.string + '[&type="' + str(root.state) + '"]:' + str(root.dist) + ";"
    
    if nexus:
        tree_file.write("#nexus\n") 
        tree_file.write("Begin trees;\n") 
        tree_file.write("tree TREE = " + root_string + "\n")
        tree_file.write("End;")
    else:
        tree_file.write(root_string)
    
    tree_file.close()

"""
    Add node times/heights to ETE tree
"""    
def add_tree_times(tree):
        
    tree_times = []
    for i in tree.traverse():
        if i.is_root():
            i.add_features(time=0)
            tree_times.append(i.time)
        else:
            i.add_features(time=i.up.time + i.dist)
            tree_times.append(i.time)
    return tree, tree_times

"""
    Index branches with a unique label (only assigns indexes to bifurcating nodes and leaf nodes)
"""
def index_branches(tree):
        
    branch_index = 0
    for node in tree.traverse("postorder"):
        
        if hasattr(node, 'branch_idx'):
            continue # we've already indexed this branch
        
        node.add_features(branch_idx = branch_index)
        curr_node = node.up
        if not curr_node is None:
            while len(curr_node.children) < 2: # i.e. not a bifurcating node
                curr_node.add_features(branch_idx = branch_index)
                #last_node = curr_node.copy() # to check how we're getting to below
                last_node = copy.copy(curr_node) # Lenora's fix to solve recurison problem
                curr_node = curr_node.up
                if curr_node is None:
                    "Can get here if non-bifurcating node occurs along root"
                    curr_node = last_node
                    break
        node.add_features(parent = curr_node)    
        branch_index += 1
    
    return tree

def write_rand_branch_effects(tree,branch_effects,file):
    "Write node features in tree to csv file"
    names = []
    features = []
    for node in tree.traverse("postorder"):
        names.append(node.name)
        branch_effect = branch_effects[node.branch_idx]
        features.append(branch_effect) # or replace with feature to write out
    df = pd.DataFrame(features, index=names, columns=['Feature'])
    df.to_csv(file)
    
    
if __name__ == '__main__':
    
    tree = Tree("simFitTree_forLenora_bRate1.0_dRate0.5_mu0.1_rhoSampling0.5_fit0.75_sim4.tre", format=1)
    tree, tree_times = add_tree_times(tree)
    nexus = False
    write_MTT_newick(tree,'test_MTT_newick.tre',nexus)