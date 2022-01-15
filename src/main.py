'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd
import os
def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb/karate.emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()
node_class = dict()
def read_graph():
	global node_class
	'''
	Reads the input network in networkx.
	'''

	node_class = dict()
	edgelist = list()
	class_num = 1
	class_name_to_num = dict()
	with open(args.input + os.sep + 'cora.content', 'r') as f, open(args.input + os.sep + 'cora.cites','r') as f2:
		for line in f:
			l = line.strip().split()
			class_name = l[-1]
			if class_name not in class_name_to_num:
				class_name_to_num[class_name] = class_num
				class_num += 1
			node_class[l[0]] = class_name #class_name_to_num[class_name]
			for line in f2:
				l = line.strip().split()
				edgelist.append((l[1],l[0]))

	G = nx.DiGraph()
	G.add_edges_from(edgelist)
	for edge in G.edges():
		G[edge[0]][edge[1]]['weight'] = 1
	#G = G.to_undirected()
	return G

def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [list(map(str, walk)) for walk in walks] # convert each vertex id to a string
	model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, epochs=args.iter)
	model.wv.save_word2vec_format(args.output)
	
	return model

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	nx_G = read_graph()
	nx_G = nx.relabel_nodes(nx_G, { n:str(n) for n in nx_G.nodes()})
	G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(args.num_walks, args.walk_length)
	model1 = learn_embeddings(walks)

	node_classification(model1, G.G) #G.G => G를 nx 그래프 형태로 인스턴스화 해놓은 것
	tsne_visualization(model1)
	



def tsne_visualization(model):
  global node_class
  node_ids = model.wv.index_to_key  # list of node IDs
  node_subjects = pd.Series(node_class)
  node_targets = node_subjects.loc[node_ids]

  transform = TSNE  # PCA
  trans = transform(n_components=3)
  node_embeddings_3d = trans.fit_transform(model.wv.vectors)

  alpha = 0.7
  label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
  node_colours = [label_map[target] for target in node_targets]

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  #plt.axes().set(aspect="equal")
  ax.scatter(
      node_embeddings_3d[:, 0],
      node_embeddings_3d[:, 1],
      node_embeddings_3d[:, 2],
      c=node_colours,
      cmap="jet",
      alpha=alpha,
  )
  plt.title("{} visualization of node embeddings".format(transform.__name__))
  plt.show()
  plt.savefig("visualization.png")

def node_classification(model, nx_G):
  K = 7
  kmeans = KMeans(n_clusters=K, random_state=0)
  kmeans.fit(model.wv.vectors)

  for n, label in zip(model.wv.index_to_key, kmeans.labels_):
    nx_G.nodes[n]['label'] = label

  for n in nx_G.nodes(data=True):
    if 'label' not in n[1].keys():
      n[1]['label'] = 7
  plt.figure(figsize=(12, 6), dpi=600)
  nx.draw_networkx(nx_G, pos=nx.layout.spring_layout(nx_G), 
  				node_color=[[n[1]['label'] for n in nx_G.nodes(data=True)]], 
					cmap=plt.cm.rainbow,
          node_shape='.',
          font_size='2'
					)
 
  plt.axis('off')
  plt.savefig('img.png', bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
	args = parse_args()
	main(args)
