import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Retrieve environment variables
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PW", "12345678")
NEO4J_DATABASE = os.getenv("NEO4J_DB", "spotify1")
GEMINI_API = os.getenv("GEMINI_API", "AIzaSyB0Ulb3FhRiaVsm1CPB078vF3wPHP9krWQ")

# ----- Graph QA Chain -----
@st.cache_resource
def graph_chain():
    graph = Neo4jGraph(NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite", google_api_key=GEMINI_API, temperature=0
    )
    chain = GraphCypherQAChain.from_llm(
        allow_dangerous_requests=True,
        graph=graph, llm=llm,
        return_intermediate_steps=True, verbose=True
    )
    return chain

# ----- Vector Store -----
@st.cache_resource
def load_vector_store():
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("spotify_faiss_index", embedder, allow_dangerous_deserialization=True)

# ----- QA Inference -----
def infer(chain, prompt):
    response = chain.invoke(prompt)
    query = response["intermediate_steps"][0]["query"]
    context = response["intermediate_steps"][1]["context"]
    result = response["result"]
    return query, context, result

# ----- Streamlit UI -----
if __name__ == "__main__":
    st.set_page_config(page_title="Spotify Analysis", layout="centered")
    st.title("🎵 Spotify Analysis Tools")
    st.write("Made using Neo4j, LangChain, Gemini, and FAISS")

    tab1, tab2 = st.tabs(["🧠 Graph QA (Gemini + Neo4j)", "🎧 Semantic Song Search (FAISS)"])

    # Tab 1: Graph QA
    with tab1:
        chain = graph_chain()

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask questions about most Streamed Spotify Songs 2023"):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            query, context, result = infer(chain, prompt)
            with st.chat_message("assistant"):
                st.code(query, language="cypher")
                st.write(context)
                st.markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})

    # Tab 2: Semantic Search
    with tab2:
        vector_db = load_vector_store()

        st.subheader("🎧 Semantic Song Search")
        st.write("Find songs by describing a vibe, lyrics, artist style, or genre.")

        sem_query = st.text_input("Search semantically...")

        if sem_query:
            results = vector_db.similarity_search(sem_query, k=5)

            for r in results:
                st.markdown(f"**🎵 {r.metadata['title']}** by *{r.metadata['artist']}*")
                st.write(r.page_content[:250] + "...")
