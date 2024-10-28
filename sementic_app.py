import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Initialize Pinecone and model
pc = Pinecone(api_key="83a4ae9d-7e3b-4fba-b809-a5f02dc8c191")
index_name = "semantic-search"

index = pc.Index(index_name)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Side panel for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Search Page", "Dataset Page", "About Me"])

if page == "Search Page":
    st.title("Semantic Search for Passages")
    
    # User input for the query
    new_query = st.text_input("Enter your search query:")
    
    # Add a button to trigger the search
    if st.button("Search"):
        if new_query:
            with st.spinner("Searching..."):
                try:
                    # Embed the new query
                    new_query_embedding = model.encode(new_query).tolist()

                    # Search in Pinecone
                    results = index.query(vector=new_query_embedding, top_k=3, include_metadata=True)

                    # Check if results are returned
                    if not results["matches"]:
                        st.write("No matching passages found.")
                    else:
                        # Display results
                        st.subheader("Top Matching Passages:")
                        for match in results["matches"]:
                            passage_text = match.get("metadata", {}).get("text", "No passage text available.")
                            similarity = match.get("score", 0.0)
                            st.write(f"**Passage:** {passage_text}")
                            st.write(f"**Similarity Score:** {similarity:.4f}")
                            st.write("---")

                except Exception as e:
                    st.error(f"An error occurred: {e}")

        else:
            st.warning("Please enter a search query.")

elif page == "Dataset Page":
    st.title("Dataset Information")
    # Add content or functionality related to the dataset here
    st.write("Details about the dataset will be displayed here.")

elif page == "About Me":
    st.title("About Me")
    st.write("This is a simple semantic search application built using Streamlit and Pinecone.")
    st.write("You can enter a query to search for related passages. Enjoy exploring!")
