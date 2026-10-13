
import streamlit as st
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from community import community_louvain
from pathlib import Path
from collections import Counter

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Social Network Community Analysis",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 Social Network Community Detection")
st.write(
    "Explore network statistics, communities, influential users, "
    "and graph algorithm results."
)

# --------------------------------------------------
# Locate the project folder and dataset
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "facebook_combined.txt"

# --------------------------------------------------
# Load the social network
# --------------------------------------------------
@st.cache_resource
def load_graph(file_path):
    return nx.read_edgelist(file_path, nodetype=int)

if not DATASET_PATH.exists():
    st.error(f"Dataset not found: {DATASET_PATH}")
    st.info(
        "Place facebook_combined.txt inside the dataset folder."
    )
    st.stop()

with st.spinner("Loading social network..."):
    G = load_graph(str(DATASET_PATH))

# --------------------------------------------------
# Network overview
# --------------------------------------------------
st.header("1. Network Overview")

number_users = G.number_of_nodes()
number_edges = G.number_of_edges()
density = nx.density(G)
average_degree = (
    sum(dict(G.degree()).values()) / number_users
    if number_users else 0
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Users (Nodes)", f"{number_users:,}")
c2.metric("Connections (Edges)", f"{number_edges:,}")
c3.metric("Network Density", f"{density:.4f}")
c4.metric("Average Degree", f"{average_degree:.2f}")

st.write(f"**Directed graph:** {G.is_directed()}")
st.write(f"**Weighted graph:** {nx.is_weighted(G)}")
st.write(f"**Connected graph:** {nx.is_connected(G)}")

# --------------------------------------------------
# Degree distribution
# --------------------------------------------------
st.header("2. Degree Distribution")

degrees = [degree for _, degree in G.degree()]

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(degrees, bins=40, edgecolor="black")
ax.set_xlabel("Number of Connections")
ax.set_ylabel("Number of Users")
ax.set_title("Distribution of User Connections")
st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# Louvain community detection
# --------------------------------------------------
st.header("3. Louvain Community Detection")

@st.cache_data
def detect_louvain():
    partition = community_louvain.best_partition(
        G,
        random_state=42
    )
    modularity = community_louvain.modularity(partition, G)
    return partition, modularity

with st.spinner("Detecting communities with Louvain..."):
    partition, louvain_modularity = detect_louvain()

community_sizes = Counter(partition.values())

c1, c2 = st.columns(2)
c1.metric("Communities Detected", len(community_sizes))
c2.metric("Modularity", f"{louvain_modularity:.4f}")

sizes_df = pd.DataFrame(
    community_sizes.items(),
    columns=["Community", "Users"]
).sort_values("Users", ascending=False)

st.subheader("Community Sizes")
st.dataframe(sizes_df, use_container_width=True, hide_index=True)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(
    sizes_df["Community"].astype(str),
    sizes_df["Users"]
)
ax.set_xlabel("Community ID")
ax.set_ylabel("Number of Users")
ax.set_title("Number of Users in Each Community")
ax.tick_params(axis="x", rotation=90)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# Interactive community explorer
# --------------------------------------------------
st.header("4. Community Explorer")

community_ids = sorted(community_sizes.keys())
selected_community = st.selectbox(
    "Select a community",
    community_ids,
    format_func=lambda x: f"Community {x} ({community_sizes[x]} users)"
)

community_nodes = [
    node for node in G.nodes()
    if partition[node] == selected_community
]
community_graph = G.subgraph(community_nodes)

internal_edges = community_graph.number_of_edges()
external_edges = sum(
    1 for u, v in G.edges()
    if (partition[u] == selected_community)
    != (partition[v] == selected_community)
)

total_incident_edges = internal_edges + external_edges
internal_ratio = (
    internal_edges / total_incident_edges
    if total_incident_edges else 0
)

c1, c2, c3 = st.columns(3)
c1.metric("Users", len(community_nodes))
c2.metric("Internal Connections", internal_edges)
c3.metric("Internal Connection Ratio", f"{internal_ratio:.2%}")

st.write(f"**Connections to other communities:** {external_edges}")

top_community_users = sorted(
    community_nodes,
    key=lambda node: G.degree(node),
    reverse=True
)[:10]

community_users_df = pd.DataFrame({
    "User ID": top_community_users,
    "Connections": [G.degree(node) for node in top_community_users]
})

st.subheader("Most Connected Users in Selected Community")
st.dataframe(
    community_users_df,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# Community network visualization
# Draw a manageable sample for performance
# --------------------------------------------------
st.header("5. Community Network Visualization")

max_nodes = st.slider(
    "Maximum nodes to display",
    min_value=100,
    max_value=1000,
    value=300,
    step=100
)

if st.button("Generate Network Visualization"):
    # Select the highest-degree users for a readable sample
    sampled_nodes = [
        node for node, degree in
        sorted(G.degree(), key=lambda item: item[1], reverse=True)
        [:max_nodes]
    ]

    sampled_graph = G.subgraph(sampled_nodes).copy()

    with st.spinner("Drawing network graph..."):
        positions = nx.spring_layout(
            sampled_graph,
            seed=42,
            iterations=20
        )

        fig, ax = plt.subplots(figsize=(12, 8))
        node_colors = [
            partition[node] for node in sampled_graph.nodes()
        ]

        nx.draw_networkx(
            sampled_graph,
            positions,
            node_color=node_colors,
            node_size=18,
            width=0.25,
            with_labels=False,
            cmap=plt.cm.tab20,
            ax=ax
        )

        ax.set_title(
            "Louvain Communities in a Sample of the Network"
        )
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

    st.caption(
        "This visualization uses a sample of the network to keep "
        "the dashboard responsive. Colors represent Louvain communities."
    )

# --------------------------------------------------
# Centrality analysis
# --------------------------------------------------
st.header("6. Influential User Analysis")

@st.cache_data
def calculate_centralities():
    degree_c = nx.degree_centrality(G)
    pagerank_c = nx.pagerank(G)
    return degree_c, pagerank_c

with st.spinner("Calculating centrality measures..."):
    degree_centrality, pagerank = calculate_centralities()

measure = st.selectbox(
    "Choose an influence measure",
    ["Number of Connections", "Degree Centrality", "PageRank"]
)

if measure == "Number of Connections":
    scores = dict(G.degree())
elif measure == "Degree Centrality":
    scores = degree_centrality
else:
    scores = pagerank

top_users = sorted(
    scores.items(),
    key=lambda item: item[1],
    reverse=True
)[:10]

top_users_df = pd.DataFrame(
    top_users,
    columns=["User ID", measure]
)
top_users_df["Number of Connections"] = [
    G.degree(user) for user, _ in top_users
]

st.dataframe(top_users_df, use_container_width=True, hide_index=True)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(
    top_users_df["User ID"].astype(str),
    top_users_df[measure]
)
ax.set_xlabel("User ID")
ax.set_ylabel(measure)
ax.set_title(f"Top 10 Users by {measure}")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# Algorithm comparison
# --------------------------------------------------
st.header("7. Community Detection Algorithm Comparison")

st.info(
    "Louvain and Label Propagation were evaluated on the full graph. "
    "Your Girvan–Newman result was evaluated on a 500-node subgraph, "
    "so its modularity and community count are not directly comparable."
)

comparison_df = pd.DataFrame({
    "Algorithm": [
        "Louvain",
        "Girvan-Newman",
        "Label Propagation"
    ],
    "Communities": [15, 4, 71],
    "Modularity": [0.8349, 0.0115, 0.8175],
    "Evaluation Graph": [
        "Full graph (4,039 nodes)",
        "Subgraph (500 nodes)",
        "Full graph (4,039 nodes)"
    ]
})

st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(
    comparison_df["Algorithm"],
    comparison_df["Modularity"]
)
ax.set_ylabel("Modularity")
ax.set_title("Reported Modularity Scores")
ax.tick_params(axis="x", rotation=15)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# Community cohesion analysis
# --------------------------------------------------
st.header("8. Community Cohesion")

@st.cache_data
def calculate_community_analysis():
    rows = []

    for community_id in sorted(set(partition.values())):
        nodes = [
            node for node in G.nodes()
            if partition[node] == community_id
        ]
        subgraph = G.subgraph(nodes)

        internal = subgraph.number_of_edges()

        external = sum(
            1 for u, v in G.edges()
            if (partition[u] == community_id)
            != (partition[v] == community_id)
            and (u in nodes or v in nodes)
        )

        ratio = internal / (internal + external) if internal + external else 0

        rows.append({
            "Community": community_id,
            "Users": len(nodes),
            "Internal Edges": internal,
            "External Edges": external,
            "Internal Ratio": round(ratio, 4)
        })

    return pd.DataFrame(rows).sort_values(
        "Users", ascending=False
    )

cohesion_df = calculate_community_analysis()
st.dataframe(cohesion_df, use_container_width=True, hide_index=True)

st.download_button(
    "Download Community Cohesion CSV",
    data=cohesion_df.to_csv(index=False).encode("utf-8"),
    file_name="community_analysis_results.csv",
    mime="text/csv"
)

st.download_button(
    "Download Influential Users CSV",
    data=top_users_df.to_csv(index=False).encode("utf-8"),
    file_name="top_influential_users.csv",
    mime="text/csv"
)

st.download_button(
    "Download Algorithm Comparison CSV",
    data=comparison_df.to_csv(index=False).encode("utf-8"),
    file_name="community_detection_algorithm_comparison.csv",
    mime="text/csv"
)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.caption(
    "Academic project: Social Network Community Detection "
    "and Influence Analysis using NetworkX and Streamlit."
)
