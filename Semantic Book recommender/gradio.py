#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from huggingface_hub import login


# In[2]:


load_dotenv()

books = pd.read_csv("books_with_emotions.csv")


# #### Gradio is an opensource python package that allows user to build dashboards spcifically to design machine learning models

# In[3]:


books["large_thumbnail"] = books["thumbnail"] + "&fife=w800" # book cover
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",# when book cover is not present
    books["large_thumbnail"],
)


# In[ ]:


# Log in to Hugging Face (if needed)
login(token=Huggingface_key)

# Define the model for embeddings
model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"

# Create the Hugging Face embeddings
embedding = HuggingFaceEmbeddings(model_name=model_name)

# Create the Chroma vector store using the Hugging Face embeddings
db_books = Chroma.from_documents(documents,
                                 embedding=embedding)


# In[ ]:


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k:int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query,k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs =  books[books["isbn13"].isin(books_list)].head(final_top_k)
    
    if category != "All":
        book_recs = book_recs[book_recs["simpler_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    if tone == "Suprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    if tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    if tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    if tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)
    return book_recs

def recommend_books(
        query: str,
        category: str,
        tone: str

):
    recommendations=retrieve_semantic_recommendations(query,category,tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) >2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]
        
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


categories = ["All"] + sorted(books["simpler_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Susepenseful", "Sad"]


#starting gradio dashboard
#multiple themes are available we chose glass one
with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder= "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices=tones,label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendation")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books, inputs=[user_query,category_dropdown,tone_dropdown], outputs = output)

if __name__ == "__main__":
    dashboard.launch()

