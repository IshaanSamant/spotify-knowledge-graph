from langchain_community.graphs import Neo4jGraph

# No username or password needed when auth is disabled
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    database="spotify1"
)

# Simple test query
print(graph.query("MATCH (n) RETURN count(n) AS node_count"))
