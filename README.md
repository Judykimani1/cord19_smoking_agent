# 🧠 COVID & Smoking QnA Agent

This project is an end-to-end Natural Language Processing (NLP) pipeline built in Google Colab for analyzing the relationship between smoking and COVID-19. It leverages biomedical research papers to create an intelligent chatbot with Named Entity Recognition (NER) and sentiment analysis capabilities.

---

## 🚀 Project Overview

This notebook performs the following key tasks:

1. **📦 Install dependencies** (e.g., LlamaIndex, HuggingFace, Gradio)
2. **📥 Load & preprocess the dataset**
   - Use the [CORD-19 dataset](https://www.semanticscholar.org/cord19) stored in Google Drive
   - Clean abstracts and filter by domain-specific keywords (e.g., "smoking", "COVID", "lung cancer")
3. **🔗 Extend metadata**
   - Add global health records (e.g., WHO publications) with relevant content
4. **📊 Validate the dataset**
   - Generate a summary report with statistics (row counts, nulls, average text length)
   - Save reports and visualizations (term frequency, wordclouds)
5. **🧠 Keyword extraction**
   - Use `KeyBERT` to extract top terms across abstracts
6. **🔍 Create vector embeddings**
   - Chunk cleaned abstracts
   - Use **BioBERT** via HuggingFace for embedding
   - Store embeddings in a **FAISS vector store** for efficient retrieval
7. **💬 Build a chatbot**
   - Powered by **Llama 3.2 1B Instruct Mango**
   - Uses LlamaIndex’s `as_query_engine()` and `as_chat_engine()` APIs
   - Includes **contextual memory** and **medical prompt tuning**
8. **🧬 Biomedical NER + Sentiment Analysis**
   - Annotate chatbot responses using `d4data/biomedical-ner-all`
   - Run sentiment classification on the response text
9. **🌐 Gradio Interface**
   - Interactive UI with:
     - Chat window
     - JSON view of NER entities
     - Real-time chatbot answers

---

## 📂 Dataset

The dataset used in this project is stored **privately in Google Drive** and includes:

- `filtered_cord19_text_sha.csv` — CORD-19 filtered abstracts
- `gpe_or_loc_in_body.csv` — WHO or geographical health data (optional)

📌 **Note**: The dataset is not included in this repo due to size and licensing.  
👉 To run the notebook, **upload these CSVs to your Google Drive** and ensure paths are correctly referenced.

```python
from google.colab import drive
drive.mount('/content/drive')
