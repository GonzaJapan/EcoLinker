# EcoLinker
A Python-based LCA inventory analysis tool utilizing a bipartite graph structure to represent the relationships between products and processes. This project focuses on tracing material flows and calculating environmental impacts through interconnected nodes, providing a foundational framework for life cycle assessment modeling.

## GNN Example Script

This repository includes a standalone script, [gnn_example_minimal.py](gnn_example_minimal.py), that builds a heterogeneous graph from real Excel source data and trains a simple two-layer GNN for link prediction.

The script creates the following model inputs directly from your tables:

- hp0: process node embeddings
- hq0: product node embeddings
- consumes_edges: product -> process edges (q, p)
- produces_edges: process -> product edges (p, q)

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Main packages are listed in [requirements.txt](requirements.txt):

- pandas
- torch
- sentence-transformers
- openpyxl

## Input Data

For privacy reasons, source paths are not hardcoded.
Pass your private data directory with a command-line argument.

Expected files in your data directory:

- プロセス入出力_20250730.xlsx
- プロセス_IDEA35確認済_20250709.xlsx
- 製品_20250717.xlsx

## Run

```bash
python gnn_example_minimal.py --data-dir /path/to/your/private/data
```

Optional argument:

```bash
python gnn_example_minimal.py \
	--data-dir /path/to/your/private/data \
	--embedding-model-name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

If `--data-dir` is omitted, the script raises an explicit error to avoid accidental path leakage.
