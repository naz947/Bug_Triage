import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import os
import networkx as nx
from extract import *
import json
from networkx.readwrite import json_graph

 # 'cora', 'citeseer', 'pubmed'

def start(a,b)
    data_name=a
    path=b
    print(data_name)
    print(path)

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(data_name)

    G = nx.from_scipy_sparse_matrix(adj)

    val_index = np.where(val_mask)[0]
    test_index = np.where(test_mask)[0]
    y = y_train+y_val+y_test
    y = np.argmax(y,axis=1)


    for i in range(len(y)):
        if i in val_index:
            G.node[i]['val']=True
            G.node[i]['test']=False
        elif i in test_index:
            G.node[i]['test']=True
            G.node[i]['val']=False
        else:
            G.node[i]['test'] = False
            G.node[i]['val'] = False


    data = json_graph.node_link_data(G)
    classMap = {}
    idMap = {}
    for i in range(len(y)):
        classMap[i]=int(y[i])
        idMap[i] = i
    return data,classMap,idMap,feature.todense()
'''
    with open(path+"/"+path+"-G.json","w") as f:
        json.dump(data,f)
    with open(path+"/"+path+"-id_map.json","w") as f:
        json.dump(idMap,f)
    with open(path+"/"+path+"-class_map.json","w") as f:
        json.dump(classMap,f)
    np.save(open(path+"/"+path+"-feats.npy","wb"), features.todense())
''''
