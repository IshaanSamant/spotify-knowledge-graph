import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

# Load environment variables
load_dotenv()

# Retrieve environment variables
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PW", "12345678")
NEO4J_DATABASE = os.getenv("NEO4J_DB", "spotify1")
GEMINI_API = os.getenv("GEMINI_API","AIzaSyB0Ulb3FhRiaVsm1CPB078vF3wPHP9krWQ")
#GEMINI_API = os.getenv("GEMINI_API", "")

@st.cache_resource
def graph_chain():
    graph = Neo4jGraph(NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite", google_api_key=GEMINI_API, temperature=0
    )
    chain = GraphCypherQAChain.from_llm(allow_dangerous_requests=True,
        graph=graph, llm=llm, return_intermediate_steps=True, verbose=True
    )
    return chain


def infer(chain, prompt):
    response = chain.invoke(prompt)
    query = response["intermediate_steps"][0]["query"]
    context = response["intermediate_steps"][1]["context"]
    result = response["result"]
    return query, context, result


if __name__ == "__main__":
    st.subheader("Analysis Tools")
    st.write("Made using Neo4j, Langchain and Gemini.")
    chain = graph_chain()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("ask questions about most Streamed Spotify Songs 2023"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        query, context, result = infer(chain, prompt)
        with st.chat_message("assistant"):
            st.code(query, language="cypher")
            st.write(context)
            st.markdown(result)
        st.session_state.messages.append({"role": "assistant", "content": result})
