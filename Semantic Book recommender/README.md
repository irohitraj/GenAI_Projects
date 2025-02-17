# Book Recommender System

### Semantic Book Recommender 📚
A personalized book recommendation system that combines semantic search and filters to suggest books based on natural language queries.

### Features:
- Natural Language Search: Users can describe the kind of book they're looking for in plain English
- Semantic Understanding: Utilizes HuggingFace embeddings for intelligent book matching
- Emotional Analysis: Books are classified by emotional tone (Happy, Surprising, Angry, Suspenseful, Sad)
- Category Filtering: Filter recommendations by fiction/non-fiction categories
- Interactive UI: Clean, modern interface built with Gradio
- Large Dataset: Powered by a curated dataset of over 5,000 books with rich metadata

### Tech Stack:
- Backend: Python with LangChain for vector search
- Embeddings: HuggingFace Embeddings
- Vector Store: Chroma DB
- Emotion Analysis: DistilRoBERTa for text classification
- UI Framework: Gradio
- Data Processing: Pandas & NumPy

### Data Pipeline:
1. Initial data cleaning and preprocessing
2. Fiction/Non-fiction classification using zero-shot learning
3. Emotional tone analysis using DistilRoBERTa
4. Vector embeddings generation for semantic search
5. Integration with Chroma vector store


### Getting Started:

### Key Components
- Data Processing
- Sentiment Analysis
- Recommendation Engine
 
### References
- Data source used: https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata/data
- Reference material: https://www.youtube.com/watch?v=Q7mS1VHm3Yw
- Emotion analysis model by J. Hartmann
- Built with HuggingFace embeddings
