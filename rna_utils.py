import RNA
import os
from collections import defaultdict
import numpy as np
import forgi.graph.bulge_graph as fgb
import forgi
import scipy.sparse as sp
def get_RNA_secondary_structure_dot_bracket(seq):
    ss, _ = RNA.fold(seq)
    return ss

def get_RNA_secondary_structure_free_energy(seq):
    _ , mfe = RNA.fold(seq)
    return mfe
    
def save_RNA_secondary_structure_svg(seq, ss, file_name):
    RNA.svg_rna_plot(seq, ss, file_name)
    return 0


def dotbracket_to_elements(dotbracket_struct, max_len):
    bg = fgb.BulgeGraph.from_dotbracket(dotbracket_struct)
    es = fgb.BulgeGraph.to_element_string(bg, with_numbers=False)
    
    paddings = max_len - len(es)
        
    for _ in range(paddings):
        es += 'N'
    es = es.upper()            
    return es

def elements_structure_encoding(elements_struct):
    onehot = defaultdict(lambda: np.array([0.1666,0.1666,0.1666,0.1666,0.1666,0.1666]))
    onehot["F"] = np.array([1,0,0,0,0,0])
    onehot["T"] = np.array([0,1,0,0,0,0])
    onehot["S"] = np.array([0,0,1,0,0,0])
    onehot["I"] = np.array([0,0,0,1,0,0])
    onehot["M"] = np.array([0,0,0,0,1,0])
    onehot["H"] = np.array([0,0,0,0,0,1])
    
    onehot_ess = [onehot[ss] for ss in elements_struct]
    return np.array(onehot_ess)


def RNASecondaryStructure2AdjacencyMatrix(ss, max_len, LOG_PLT=False):
    N = max_len
    stack = []
    rows = []
    cols = []
    data = []
    open_pair = set(["(","{","[","<"])
    close_pair = set([")","}","]",">"])
    for i, s in enumerate(ss):
        if s in open_pair:
            stack.append(i)
        elif s in close_pair:
            rows.append(stack[-1])
            cols.append(i)
            data.append(1)
            stack.pop()
        elif s == ".":
            continue
        else:
            raise ValueError("Error : unknown symbol founded")
            
    adj = sp.coo_matrix((data,(rows,cols)),shape=(N,N))
    adj += adj.T

    if LOG_PLT:
        plt.figure(figsize=(5,5))
        plt.imshow(adj.todense())
        plt.title("Sequence Length: {}".format(N))
        plt.show()

    return adj