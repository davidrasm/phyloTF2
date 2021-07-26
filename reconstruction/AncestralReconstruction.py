"""
Created on Sat Jun 13 15:31:32 2020

@author: david
"""

from pastml.acr import pastml_pipeline
from Bio import SeqIO
from ete3 import Tree
import pandas as pd
import numpy as np

def fasta2csv(fasta_file,csv_file):
    
    "Get sequences from fasta"
    seq_dic = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq_dic[record.id] = [i for i in record.seq] #str(record.seq)
    sites = len(seq_dic[next(iter(seq_dic))])
    traits = ["site" + str(i) for i in range(sites)]
    seqs_df = pd.DataFrame.from_dict(seq_dic, orient='index', columns=traits)
    seqs_df.to_csv(csv_file)
    
    return traits

def reconstruct(tree_file,align_file,csv_file="temp-align.csv"):
    
    if align_file.endswith('.fasta'):
        "Convert fasta to csv"
        traits = fasta2csv(align_file,csv_file) # convert fasta to csv
        data = csv_file # Path to the table containing tip/node annotations, in csv or tab format
    else:
        df = pd.read_csv(align_file, index_col=0)
        traits = [column for column in df.columns]
        data = align_file
    
    tree = tree_file # Path to the tree in newick format
    
    # Columns present in the annotation table,
    # for which we want to reconstruct ancestral states
    columns = traits #['Country']
    
    # Path to the output compressed map visualisation
    html_compressed = "tree-000_map.html"
    
    # (Optional) path to the output tree visualisation
    html = "tree-000_tree.html"
    
    pastml_pipeline(data=data, data_sep=',', columns=columns, tree=tree, verbose=True)
    
    #pastml_pipeline(data=data, data_sep=',', columns=columns, name_column=traits[0], tree=tree,
    #                html_compressed=html_compressed, html=html, verbose=True)


def label_internal_nodes(tree):
    
    internal_cntr = 0 # counter for internal nodes encountered
    for node in tree.traverse("preorder"):
        if node.is_root():
            node.name = 'root'
        else:
            if not node.is_leaf():
                node.name = 'n' + str(internal_cntr)
                internal_cntr += 1        
    return tree

"""
    Reconstruct ancestral states using Sankoff's max parsimony algorithm (see Felsenstein p15-18)
    Parsimony scores are computed assuming all transitions have a cost of one.
    Tip/ancestral features are input and output as dictionaries
    Internal nodes are labeled as n<X> where X is an int determined by the position of the node in a pre-order traversal
"""
def reconstruct_MP(tree,feature_dic):
    
    state_set = set([feature_dic[node.name] for node in tree.traverse() if node.is_leaf()])
    states = len(state_set)
    
    "Add a state2int dict map so this works with non-integer data types"
    state2int = {state:index for index, state in enumerate(state_set)}
    int2state = {index:state for index, state in enumerate(state_set)}
    
    "Post-order traversal to compute costs in terms of required state transitions"
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            costs = [np.inf]*states
            tip_state = state2int[feature_dic[node.name]] # feature_dic[node.name]
            costs[tip_state] = 0
            node.add_features(costs = costs)
        else:
            costs = [0]*states
            for i in range(states):
                child_costs = []
                for child in node.children:
                    temp_costs = [0]*states
                    for j in range(states):
                        temp_costs[j] = child.costs[j]
                        if i != j:
                            temp_costs[j] = temp_costs[j] + 1 # add cost for transitioning between i and j
                    child_costs.append(temp_costs)
                costs[i] = sum([min(c) for c in child_costs])
            node.add_features(costs = costs)
    
    "Pre-order traversal to select anc states based on least cost parsimony score"
    anc_dic = {}
    internal_cntr = 0 # counter for internal nodes encountered
    for node in tree.traverse("preorder"):
        costs = node.costs
        if node.is_root():
            root_state = costs.index(min(costs)) # or np.argmin(node.costs)
            node.add_features(state = root_state)
            anc_dic['root'] = root_state
        else:
            parent_state = node.up.state
            least_cost = min(costs)
            least_cost_state = costs.index(least_cost)
            if parent_state == least_cost_state:
                anc_state = parent_state
            else:
                parent_st_cost = costs[parent_state]
                if parent_st_cost < (least_cost+1):
                    anc_state = parent_state # if parent state costs less than transitioning to least cost state
                else:
                    anc_state = least_cost_state
            node.add_features(state = anc_state)
            if node.is_leaf():
                anc_dic[node.name] = anc_state
            else:
                name = 'n' + str(internal_cntr)
                anc_dic[name] = anc_state
                internal_cntr += 1
    
    "Covert from integers back to original states"
    for k,v in anc_dic.items():
        anc_dic[k] = int2state[v]

    "Should return tree and anc_dic"
    return tree, anc_dic

if __name__ == '__main__':
    
    import time
    
    "Old ML test data"
    #path = './test-sets/testTF_timeVaryingSiteEffects_june2020/'
    #tree_file = path + 'tree-000.tre'
    #fasta_file = path + 'tree-000.fasta'
    #csv_file = path + 'tree-000.csv'
    #reconstruct(tree_file,fasta_file,csv_file)
    
    "New covid test data for testing MP reconstructions"
    tree_file = 'covid_ancestral_D614G_testMP.tre'
    csv_file = 'covid_features_D614G_testMP.csv'
    
    tree = Tree(tree_file, format=1)
    tree = label_internal_nodes(tree)
    #for node in tree.traverse("preorder"): print(node.name)
    
    df = pd.read_csv(csv_file,index_col='node')
    feature_dic = {}
    for index, row in df.iterrows():
        feature_dic[index] = row['nsp12_P323L+S_D614G']
    tic = time.perf_counter()
    tree, anc_dic = reconstruct_MP(tree,feature_dic)
    toc = time.perf_counter()
    elapsed = toc - tic
    print(f"Elapsed time: {elapsed:0.4f} seconds")


    "Plot"
    import balticmod as bt
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import TreeUtils
    import seaborn as sns
    
    "Write tree with fit vals to multi-type Newick file"
    absolute_time = 2020.67
    mtt_file = 'covid_ancestral_D614G_testMP_mtt.tre'
    fig_file = 'covid_ancestral_D614G_testMP_MP.png'
    TreeUtils.write_MTT_newick(tree,mtt_file)
    
    sns.set(style="darkgrid")
    myTree=bt.loadNewick(mtt_file,absoluteTime=False)
    myTree.traverse_tree() ## required to set heights
    myTree.setAbsoluteTime(absolute_time) ## set absolute time of all branches by specifying date of most recent tip
    myTree.treeStats() ## report stats about tree
    
    cmap = mpl.cm.get_cmap('tab10', 10)
    
    fig,ax = plt.subplots(figsize=(20,20),facecolor='w')

    x_attr=lambda k: k.absoluteTime ## x coordinate of branches will be absoluteTime attribute
    c_func=lambda k: 'darkorange' if k.traits['type']=='1' else 'steelblue' ## colour of branches
    s_func=lambda k: 50-30*k.height/myTree.treeHeight ## size of tips
    z_func=lambda k: 100
    
    cu_func=lambda k: 'k' ## for plotting a black outline of tip circles
    su_func=lambda k: 2*(50-30*k.height/myTree.treeHeight) ## black outline in twice as big as tip circle 
    zu_func=lambda k: 99
    myTree.plotTree(ax,x_attr=x_attr,colour_function=c_func) ## plot branches
    myTree.plotPoints(ax,x_attr=x_attr,size_function=s_func,colour_function=c_func,zorder_function=z_func) ## plot circles at tips
    myTree.plotPoints(ax,x_attr=x_attr,size_function=su_func,colour_function=cu_func,zorder_function=zu_func) ## plot circles under tips (to give an outline)
    
    "Add legend the hard way"
    import matplotlib.patches as mpatches
    blue_patch = mpatches.Patch(color='steelblue', label = 'Spike 614D')
    red_patch = mpatches.Patch(color='darkorange', label = 'Spike 614G')
    handles = [blue_patch,red_patch]
    
    "For Regions"
    ax.legend(handles=handles,prop={'size': 24}) #loc='upper left'
    
    "Add month labels as xticks"
    step_freq = 1/12
    xticks = np.arange(2020,absolute_time,step_freq)
    ax.set_xticks(xticks)
    labels = ['Jan','Feb','Mar','Apr','May','June','July','Aug','Sep']
    ax.set_xticklabels(labels, fontsize=24) #rotation='vertical'
    ax.set_xlabel('Time', fontsize=24)
    
    ax.set_ylim(-5,myTree.ySpan+5)
    plt.savefig(fig_file, dpi=300)
    
    


