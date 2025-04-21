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
    return chain, llm

@st.cache_resource
def load_vector_store():
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("spotify_faiss_index", embedder, allow_dangerous_deserialization=True)

def infer_graph(chain, prompt):
    response = chain.invoke(prompt)
    query = response["intermediate_steps"][0]["query"]
    context = response["intermediate_steps"][1]["context"]
    result = response["result"]
    return query, context, result

def infer_vector(vector_db, prompt):
    return vector_db.similarity_search(prompt, k=5)

def decide_routing(llm, prompt):
    instruction = (
        "You are a routing assistant. Based on the user query, decide whether to use:\n"
        "'graph' if it is structured and factual,\n"
        "'vector' if it requires similarity or vibe-based semantic search,\n"
        "or 'both' if both are relevant.\n"
        "Just return one word: graph, vector, or both.\n"
        f"Query: {prompt}"
    )
    response = llm.invoke(instruction).content.strip().lower()
    if response not in ["graph", "vector", "both"]:
        return "vector"  # default fallback
    return response

if __name__ == "__main__":
    st.subheader("Analysis Tools")
    st.write("Made using Neo4j, Langchain, Gemini, and FAISS")

    try:
        chain, llm = graph_chain()
    except Exception as e:
        st.error("❌ Could not connect to Neo4j. Falling back to semantic search only.")
        chain, llm = None, ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=GEMINI_API)

    vector_db = load_vector_store()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about 2023's top streamed Spotify songs"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        routing = decide_routing(llm, prompt)
        st.info(f"🤖 Routing decision: {routing}")

        assistant_reply = ""

        if routing == "graph" and chain:
            try:
                query, context, result = infer_graph(chain, prompt)
                st.chat_message("assistant").code(query, language="cypher")
                st.chat_message("assistant").write(context)
                st.chat_message("assistant").markdown(result)
                assistant_reply = result
            except Exception as e:
                st.chat_message("assistant").markdown("⚠️ Graph search failed. Switching to vector DB...")
                routing = "vector"

        if routing == "vector":
            results = infer_vector(vector_db, prompt)
            combined = "\n".join([f"**🎵 {r.metadata['title']}** by *{r.metadata['artist']}*\n{r.page_content[:250]}..." for r in results])
            st.chat_message("assistant").markdown(combined)
            assistant_reply = combined

        if routing == "both" and chain:
            try:
                query, context, result = infer_graph(chain, prompt)
                st.chat_message("assistant").code(query, language="cypher")
                st.chat_message("assistant").write(context)
                results = infer_vector(vector_db, prompt)
                combined = "\n".join([f"**🎵 {r.metadata['title']}** by *{r.metadata['artist']}*\n{r.page_content[:250]}..." for r in results])
                st.chat_message("assistant").markdown(result + "\n---\n" + combined)
                assistant_reply = result + "\n---\n" + combined
            except Exception as e:
                st.chat_message("assistant").markdown("⚠️ One of the sources failed during combined query.")

        if assistant_reply:
            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
