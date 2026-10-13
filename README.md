# Social Network Community Detection and Influence Analysis

A graph analytics project that identifies communities in a social network, analyzes influential users, and visualizes network structure using Python and Streamlit.

## Project Overview

Social networks contain users connected through relationships. Graph analysis helps us understand these relationships, identify groups of closely connected users, and find structurally important users.

This project analyzes the Facebook social network dataset using graph algorithms and presents the results through an interactive Streamlit dashboard.

## Features

- Network statistics: users, connections, network density, and average degree.
- Degree distribution analysis.
- Community detection using the Louvain algorithm.
- Interactive community exploration.
- Community network visualization.
- Influential user analysis using Degree Centrality and PageRank.
- Comparison of Louvain, Girvan–Newman, and Label Propagation.
- Community cohesion analysis using internal and external connections.
- CSV downloads for analysis results.

## Technologies Used

- Python
- NetworkX
- Pandas
- Matplotlib
- Streamlit
- python-louvain
- Jupyter Notebook

## Dataset

This project uses the Facebook combined social network edge list from the Stanford Large Network Dataset Collection (SNAP).

Dataset source: https://snap.stanford.edu/data/facebook_combined.txt.gz

The dataset represents users as nodes and their connections as edges.

Download and extract the dataset, then place the text file here:

`dataset/facebook_combined.txt`

## Project Structure

```text
social-network-community-detection/
├── app.py
├── Untitled.ipynb
├── requirements.txt
├── README.md
├── .gitignore
├── dataset/
│   └── facebook_combined.txt
├── community_analysis_results.csv
├── community_detection_algorithm_comparison.csv
└── community_influential_users.csv
```

## Results

The network analysis produced the following results:

| Metric | Result |
|---|---:|
| Users (nodes) | 4,039 |
| Connections (edges) | 88,234 |
| Network density | 0.0108 |
| Average degree | 43.69 |
| Louvain communities | 15 |
| Louvain modularity | 0.8349 |
| Label Propagation communities | 71 |
| Label Propagation modularity | 0.8175 |
| Girvan–Newman communities | 4 |
| Girvan–Newman modularity | 0.0115 |

**Comparison note:** Louvain and Label Propagation were evaluated on the full network. Girvan–Newman was evaluated on a 500-node subgraph because the algorithm can be computationally expensive on large networks. Therefore, the Girvan–Newman results are not directly comparable with the full-network results.

## Installation

Install Python, then open a terminal in the project directory.

Install the required libraries:

```bash
python -m pip install -r requirements.txt
```

## Run the Dashboard

From the project directory, execute:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

## Run the Notebook

Open Jupyter Notebook and run `Untitled.ipynb` to explore the analysis and graph algorithms.

## Applications

- Social network analysis
- Community structure exploration
- Influential user identification
- Recommendation system research
- Network relationship analysis
- Research into methods that can be adapted to fraud-ring detection

The Facebook dataset is used for social network analysis; this project does not itself detect fraud.

## Limitations

- The dataset contains network connections rather than detailed user interests.
- Detected communities reflect connectivity patterns, not verified real-world interests.
- Girvan–Newman is evaluated on a smaller subgraph.
- The dashboard is an academic prototype and does not use live Facebook data.

## Future Improvements

- Add more interactive graph exploration.
- Compare algorithms on the same graph for a fairer benchmark.
- Add community-level statistics and downloadable reports.
- Deploy the dashboard online.

## Author

Data Science Project — Social Network Community Detection and Influence Analysis
