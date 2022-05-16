import pydot
import matplotlib.pyplot as plt
import networkx as nx

graph = pydot.graph_from_dot_file('data/oil/graph_19_oil_1_2.dot')[0]
graph.write_png('oil.png')
# graph = pydot.graph_from_dot_file('data/graph_37_II-M-N_1_0.dot')[0]
# graph.write_png('data/II-M-N.png')
# import torch
# g = torch.load(open('data/graph_samples_19_II-M-N_1.pkl','r'))
# g = torch.load(open('data/graph_19_II-M-N_1_3.dot','r'))
# print(g)
"""
import pickle
data = pickle.load(open('data/graph_samples_19_II-M-N_1.pkl', 'rb'))
nx.draw(data[2])
plt.savefig('test.png')
# print(data)
"""