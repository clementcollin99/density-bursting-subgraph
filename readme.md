
# Density Bursting Subgraphs
author : Clément Collin  
institution : Universitat Politècnica de València  
date : 12-01-2024

We implemented an algorithm, that finds Density Bursting Subgraphs (DBS) in the Facebook
Wall Posts dataset. In this dataset, a DBS identifies a fast emerging community of
users who actively interact with each other.

The algorithm has been conceived by Lingyang Chu, Yanyan Zhang, Yu Yang, Lanjun Wang and Jian Pei (c.f. their article _Online density bursting subgraph detection from temporal graphs_).

The algorithm requires to solve two subproblems :  
1. the maximum density subgraph problem, which we addressed using the well-known algorithm proposed by A. V. Goldberg in its article _Finding a Maximum Density Subgraph_ released in 1984.
2. the maximum density subsegment problem, which we solved using the first method described by Kai-min Chung and Hsueh-I Lu in _An Optimal Algorithm for the Maximum-Density Segment Problem_. Altough it's not the most efficient one, it did a great when dealing with small amounts of data.

## Instructions for use
You must download the [Facebook Wall Posts dataset](https://data.mendeley.com/datasets/4dwzvcdsv3/2), unzip it and add the _facebook.txt_ file to this repo, at the location indicated in the tree below.

```
+-- density_bursting_subgraph  
│ +-- data  
│ │ +-- facebook.txt  
│ +-- src  
│ │ +-- __init__.py  
│ │ +-- dbs.py  
│ │ +-- graph.py  
│ │ +-- segment.py
│ +-- notebooks   
│ | +-- dbs.ipynb  
│ │ +-- facebook.html
│ │ +-- facebook.ipynb 
│ | +-- graph.ipynb  
│ | +-- segment.ipynb  
│ +-- README.md  
│ +-- .gitignore  
│ +-- references.bib  
```

Then, i recommend creating an Anaconda virtual environment with python and pip and installing the packages indicated in the requirements.txt file.

## References
cf. references.bib
