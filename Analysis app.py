import os
import types
import torch
from dotenv import load_dotenv
import streamlit as st

# Fix Streamlit x Torch error on file watching
os.environ["STREAMLIT_WATCHDOG_MODE"] = "poll"
if isinstance(torch.classes, types.ModuleType):
    torch.classes.__path__ = []

# Load environment variables
load_dotenv()

from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Environment variables for config
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PW", "12345678")
NEO4J_DATABASE = os.getenv("NEO4J_DB", "spotifykg")
GEMINI_API = os.getenv("GEMINI_API", "AIzaSyB0Ulb3FhRiaVsm1CPB078vF3wPHP9krWQ")

# ----- Graph QA Chain -----
@st.cache_resource
def graph_chain():
    graph = Neo4jGraph(
        url=NEO4J_URL,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=GEMINI_API,
        temperature=0
    )
    chain = GraphCypherQAChain.from_llm(
        allow_dangerous_requests=True,
        graph=graph,
        llm=llm,
        return_intermediate_steps=True,
        verbose=True
    )
    return chain

# ----- Vector Store -----
@st.cache_resource
def load_vector_store():
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("spotify_faiss_index", embedder, allow_dangerous_deserialization=True)

# ----- Routing Decision Logic -----
def decide_routing(llm, user_query: str) -> str:
    instruction = f"""
You are an intelligent router for a music discovery system.

The system has two sources of data:
1. A Knowledge Graph (Neo4j) with structured fields like:
   - Track → valence, tempo, energy, acousticness, danceability, popularity, explicit
   - Relationships between tracks, artists, albums, and genres

2. A Vector Database (FAISS) for unstructured song descriptions or vibe-based matching.

Use the Knowledge Graph if the query involves:
- Mood or sound-based keywords (happy, energetic, sad, acoustic)
- Tempo, energy, danceability, valence, popularity, explicit
- Genre, artist, album, track-specific questions

Use the Vector DB if the query involves:
- Abstract themes (e.g., heartbreak, rebellion, memories)
- Open-ended descriptions about lyrical content or feel

Use hybrid if the query includes both structured and unstructured needs.

Return only one of the following: `kg`, `vector`, or `hybrid`.

Query: "{user_query}"
Response:
"""
    response = llm.invoke(instruction)
    return str(response.content).strip().lower()

# ----- Streamlit UI -----
if __name__ == "__main__":
    st.set_page_config(page_title="Spotify Analysis", layout="centered")
    st.title("🎵 Spotify Analysis Tools")
    st.write("Made using Neo4j, LangChain, Gemini, and FAISS")

    chain = graph_chain()
    vector_db = load_vector_store()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask anything about songs, artists, genres, mood, etc."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=GEMINI_API)
        routing = decide_routing(llm, prompt)

        st.markdown(f"🧭 **Routing decision:** `{routing}`")

        if routing == "kg":
            query_result = chain.invoke(prompt)
            query = query_result["intermediate_steps"][0]["query"]
            context = query_result["intermediate_steps"][1]["context"]
            result = query_result["result"]

            with st.chat_message("assistant"):
                st.code(query, language="cypher")
                st.write(context)
                st.markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})

        elif routing == "vector":
            results = vector_db.similarity_search(prompt, k=5)
            with st.chat_message("assistant"):
                for r in results:
                    st.markdown(f"**🎵 {r.metadata['title']}** by *{r.metadata['artist']}*")
                    st.write(r.page_content[:300] + "...")
            st.session_state.messages.append({"role": "assistant", "content": "[Semantic search results above]"})

        elif routing == "hybrid":
            query_result = chain.invoke(prompt)
            query = query_result["intermediate_steps"][0]["query"]
            context = query_result["intermediate_steps"][1]["context"]
            kg_result = query_result["result"]
            vector_results = vector_db.similarity_search(prompt, k=3)

            with st.chat_message("assistant"):
                st.subheader("🔗 Knowledge Graph")
                st.code(query, language="cypher")
                st.write(context)
                st.markdown(kg_result)

                st.subheader("🔍 Vector Search")
                for r in vector_results:
                    st.markdown(f"**🎵 {r.metadata['title']}** by *{r.metadata['artist']}*")
                    st.write(r.page_content[:300] + "...")

            st.session_state.messages.append({"role": "assistant", "content": "[KG + Vector results above]"})
