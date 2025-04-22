import pandas as pd
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load updated CSV
df = pd.read_csv("spotify.csv")

# Drop rows with missing essentials
df = df.dropna(subset=["track_name", "artists"])

# Function to convert each row to a document
def row_to_doc(row):
    text = f"Track: {row['track_name']}. Artist(s): {row['artists']}."
    if 'track_genre' in row and not pd.isna(row['track_genre']):
        text += f" Genre: {row['track_genre']}."
    if 'valence' in row and not pd.isna(row['valence']):
        text += f" Mood score: {round(row['valence'], 2)}."
    if 'tempo' in row and not pd.isna(row['tempo']):
        text += f" Tempo: {round(row['tempo'])} BPM."
    if 'popularity' in row and not pd.isna(row['popularity']):
        text += f" Popularity: {row['popularity']}."

    return Document(
        page_content=text,
        metadata={
            "title": row["track_name"],
            "artist": row["artists"]
        }
    )

docs = [row_to_doc(row) for _, row in df.iterrows()]

# Embed + save FAISS index
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(docs, embedder)
vector_db.save_local("spotify_faiss_index")

print("✅ New FAISS vector store built and saved.")
