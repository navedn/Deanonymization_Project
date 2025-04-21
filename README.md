# Deanonymization_Project

Project Description: In this project, you will address real-world problems by implementing both seed-based and seed-free de-anonymization techniques.

Seed-based De-anonymization:
You are provided with two graphs represented in edgelist files (seed_G1.edgelist and seed_G2.edgelist) and a file containing pairs of seed nodes (seed_mapping.txt). In each edgelist file, there are two columns: the first column represents a node in the graph, and the second column indicates a connected node within the same graph. Each row in the edgelist file signifies an edge between these two nodes.

The seed node pairs file, seed_mapping.txt, consists of two columns. The first column corresponds to node indices in graph G1, while the second column corresponds to node indices in graph G2. Your task is to produce a complete mapping of nodes between G1 and G2, following the same format as the seed_mapping.txt file.

Seed-free De-anonymization:
In this case, you are provided with two graphs, unseed_G1.edgelist and unseed_G2.edgelist, in edgelist file format without any seed nodes. Your objective is to identify and match each node in unseed_G1.edgelist with the most similar node in unseed_G1.edgelist, based on node features. The resulting file should adhere to the same format as the seed node mapping.
Hint: You can utilize the networkx package in Python to read the edgelist files.

Validation Dataset
You are provided with two graphs represented in edgelist files (validation_G1.edgelist and validation_G2.edgelist) and a file containing pairs of seed nodes (seed_mapping.txt). In each edgelist file, there are two columns: the first column represents a node in the graph, and the second column indicates a connected node within the same graph. Each row in the edgelist file signifies an edge between these two nodes. There is an additional file, validation_seed_mapping.txt, which contains the complete mapping of nodes between graphs G1 and G2. In this file, the first column represents node indices from graph G1, while the second column represents the corresponding matched node indices from graph G2. This dataset just for Validation purposes.
Project Submission (One Zip File):

Project Report: A brief report (maximum 3 pages) describing your methods for both de-anonymization techniques.
Source Code and Readme: The source code along with a readme file explaining how to execute it.
Results: Submit an individual result file named with your last name (e.g., Caiproject1output.txt). This file should have two columns, with each row containing a pair of matched nodes where the first value corresponds to a node index in G1 and the second value to the corresponding node index in G2.
