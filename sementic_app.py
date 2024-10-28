import json
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Initialize Pinecone and model
pc = Pinecone(api_key="83a4ae9d-7e3b-4fba-b809-a5f02dc8c191")
index_name = "semantic-search"

index = pc.Index(index_name)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load data from GitHub
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/shahinshabab/sementic_search/main/train_dataset.json"
    response = requests.get(url)
    data = json.loads(response.text)
    return data
data = load_data()

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
    # Dataset details
    st.write("### Name: Google Natural Questions Dataset 2020")
    st.write(
        """
        Natural Questions contains 307K training examples, 8K examples for development, and a further 8K examples for testing.
        
        In the paper, we demonstrate a human upper bound of 87% F1 on the long answer selection task, and 76% on the short answer selection task.
        
        We believe that matching human performance on this task will require significant progress in natural language understanding; we encourage you to help make this happen.
        """
    )
    st.write("**Used Rows for Training:** 10,000 rows")
    st.write("**Displayed Sample Size:** 2,000 rows")
    # Convert the loaded data to a DataFrame for visualization
    df = pd.json_normalize(data)
    # Display a sample of the dataset
    st.write("### Sample Data")
    st.dataframe(df.head())
    # Calculate relevant and irrelevant passage counts
    relevant_count = df['passages'].apply(lambda x: sum(p['is_selected'] for p in x))
    irrelevant_count = df['passages'].apply(lambda x: len(x) - sum(p['is_selected'] for p in x))
    total_relevant = relevant_count.sum()
    total_irrelevant = irrelevant_count.sum()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4>Total Relevant Passages</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center; color: green;'>{total_relevant}</h4>", unsafe_allow_html=True)
    with col2:
        st.markdown("<h4>Total Irrelevant Passages</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center; color: red;'>{total_irrelevant}</h4>", unsafe_allow_html=True)
    # Calculate the passage counts per row
    passage_counts = df['passages'].apply(len)
    # Create a line graph for passage counts
    fig_line = px.line(
        x=range(1, len(passage_counts) + 1),  # Row numbers starting from 1
        y=passage_counts.sort_values(),
        title='Count of Passages per Row',
        labels={'x': 'Row Number', 'y': 'Count of Passages'}
    )
    fig_line.update_layout(xaxis_title="", yaxis_title="Count of Passages")
    st.plotly_chart(fig_line)

elif page == "About Me":
    st.title("About Me")
    st.write("This is a simple semantic search application built using Streamlit and Pinecone.")
    st.write("You can enter a query to search for related passages. Enjoy exploring!")
