import requests
import streamlit as st

API_URL = "http://localhost:3000/search"

st.title("Document Search")

query = st.text_input("Enter your search query")

if st.button("Search"):
    if query:
        response = requests.post(API_URL, json={"query": query})
        data = response.json()

        st.subheader("Answer")
        st.write(data["answer"])

        st.subheader("Citations")
        for citation in data["citations"]:
            with st.expander(f"{citation['document_name']} - Page {citation['page_number']}"):
                st.write(citation["text"])