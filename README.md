# Cricket Rules Assistant

A retrieval-augmented assistant that answers questions about Scarborough cricket playing conditions from the league rulebook. Answers are generated only from retrieved PDF passages and include page citations.

Each question is searched on its own. The chat shows the answer and the passages it came from.

## Stack

- Python
- Streamlit
- Google Gemini for embeddings and answers
- Pinecone for vector search
- pypdf for PDF text extraction

## How it works

1. Extract text from the playing-conditions PDF and keep the page number.
2. Split each page into 200-word chunks with a 40-word overlap.
3. Embed the chunks with Gemini and store them in a Pinecone index, along with the page and source file.
4. Embed the user's question, retrieve the top matching passages, and ask Gemini to answer only from that text.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install streamlit python-dotenv google-genai pypdf

**# Copy .env.example to .env and add your keys:**

GEMINI_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=cricket-rules

**## Run**

python check_setup.py
python create_index.py
python ingest.py
streamlit run app.py
