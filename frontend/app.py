import streamlit as st
import requests
import json
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_URL = f"{API_BASE_URL}/query"

st.set_page_config(
    page_title="Semantic Search",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 20 Newsgroups Semantic Search")

st.markdown(
    "Enter a query to search the corpus. The system will check the semantic cache "
    "and retrieve relevant documents."
)

query = st.text_input("Enter your query")

if st.button("Search") and query:

    with st.spinner("Processing query..."):

        response = requests.post(
            API_URL,
            json={"query": query}
        )

    if response.status_code != 200:
        st.error("API Error")
        st.write(response.text)
        st.stop()

    data = response.json()

    st.subheader("Query Result")

    col1, col2 = st.columns(2)

    with col1:
        if data["cache_hit"]:
            st.success("✅ Cache HIT")
        else:
            st.warning("⚠️ Cache MISS")

    with col2:
        st.info(f"Dominant Cluster: {data['dominant_cluster']}")

    if data["matched_query"]:
        st.write("**Matched Cached Query:**")
        st.code(data["matched_query"])

    if data["similarity_score"]:
        st.write(f"Similarity Score: **{data['similarity_score']}**")

    st.divider()

    st.subheader("Retrieved Documents")

    docs = data["result"]["retrieved_documents"]

    for i, doc in enumerate(docs):

        with st.expander(f"Document {i+1} | {doc['newsgroup']} | similarity={doc['similarity']:.3f}"):

            st.write(doc["text"])

    st.divider()

    st.subheader("Raw JSON Response")

    st.json(data)
