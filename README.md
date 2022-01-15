# node2vec

This repository provides a reference implementation of *node2vec* as described in the paper:<br>
> node2vec: Scalable Feature Learning for Networks.<br>
> Aditya Grover and Jure Leskovec.<br>
> Knowledge Discovery and Data Mining, 2016.<br>
> <Insert paper link>

The *node2vec* algorithm learns continuous representations for nodes in any (un)directed, (un)weighted graph. Please check the [project page](https://snap.stanford.edu/node2vec/) for more details. 

### Basic Usage

#### Example
To run *node2vec* on Zachary's karate club network, execute the following command from the project home directory:<br/>
	``python src/main.py --input cora --output cora.emd``

### Citing
If you find *node2vec* useful for your research, please consider citing the following paper:

	@inproceedings{node2vec-kdd2016,
	author = {Grover, Aditya and Leskovec, Jure},
	 title = {node2vec: Scalable Feature Learning for Networks},
	 booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
	 year = {2016}
	}

### Module Version
Core version
gensim==4.1.2
matplotlib==3.3.4
networkx==2.5.1
pandas==1.1.5
scikit-learn==0.24.2

All version
cycler==0.11.0
dataclasses==0.8
decorator==4.4.2
gensim==4.1.2
joblib==1.1.0
kiwisolver==1.3.1
matplotlib==3.3.4
networkx==2.5.1
numpy==1.19.5
pandas==1.1.5
Pillow==8.4.0
pyparsing==3.0.6
python-dateutil==2.8.2
pytz==2021.3
scikit-learn==0.24.2
scipy==1.5.4
six==1.16.0
smart-open==5.2.1
threadpoolctl==3.0.0
