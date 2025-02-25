# Book Recommender System

### Semantic Book Recommender 📚
A personalized book recommendation system that combines semantic search and filters to suggest books based on natural language queries.


### Features:
- Natural Language Search: Users can describe the kind of book they're looking for in natural Language
- Semantic Understanding: Utilizes HuggingFace embeddings for intelligent book matching
- Emotional Analysis: Books are classified by emotional tone (Happy, Surprising, Angry, Suspenseful, Sad)
- : Filter recommendations by fiction/non-fiction categories
- Interactive UI: Built with Gradio
- Large Dataset: Powered by a curated dataset of over 5,000 books with Features

### Tech Stack:
- Backend: Python with LangChain for vector search
- Embeddings: HuggingFace Embeddings
- Vector Store/DB: Chroma DB
- Emotion Analysis: DistilRoBERTa for text classification
- UI Framework: Gradio
- Data Processing: Pandas & NumPy

### Data Pipeline:
1. Initial data cleaning and preprocessing
2. Fiction/Non-fiction classification using zero-shot learning using bart-large-mnli model from HuggingFace
3. Emotional tone analysis using DistilRoBERTa from HuggingFace
4. Vector embeddings(storing in Chroma DB) generation using paraphrase-MiniLM-L3-v2 model from HuggingFace for semantic search


### Key Components
- Vector Database: Allows us to find the most similar books to a query
- Text Classification & Sentiment Analysis: Sorting books into fiction and non-fiction(Users can also filter based on this category) and emotional tone related to the book
- Dashboard: Integrating everythong together
 
### References
- Data source used: https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata/data
- Reference material: https://www.youtube.com/watch?v=Q7mS1VHm3Yw
- Emotion analysis model by J. Hartmann
- Built with HuggingFace embeddings
