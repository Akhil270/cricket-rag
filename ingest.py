from pathlib import Path
import os
import time

from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai
from google.genai import types

import json
from urllib.request import Request, urlopen
from urllib.parse import quote


# 1. Load the API key.
project_folder = Path(__file__).resolve().parent
load_dotenv(project_folder / ".env")

gemini_key = os.getenv("GEMINI_API_KEY")

if not gemini_key:
    raise ValueError("GEMINI_API_KEY is missing from your .env file.")


# 2. Locate and open the PDF.
pad_path = project_folder / "data" / "Scarborough_Playing_Conditions.pdf"

if not pad_path.is_file():
    raise FileNotFoundError(f"PDF not found: {pad_path}")

reader = PdfReader(str(pad_path))


# 3. Extract text and preserve page numbers.
pages = []

for page_number, page in enumerate(reader.pages, start=1):
    text = (page.extract_text() or "").strip()

    if text:
        pages.append({
            "page_number": page_number,
            "text": text,
        })
    else:
        print(f"Warning: No text found on page {page_number}.")

print(f"Extracted text from {len(pages)} pages of the PDF.")


# 4. Split each page into overlapping chunks.
chunk_size = 200
overlap = 40

if not 0 <= overlap < chunk_size:
    raise ValueError("Overlap must be between 0 and chunk_size - 1.")

chunks = []

for page in pages:
    words = page["text"].split()
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])

        chunks.append({
            "text": chunk_text,
            "page_number": page["page_number"],
            "source": pad_path.name,
        })

        if end >= len(words):
            break

        start = end - overlap

print(f"Created {len(chunks)} chunks.")

if not chunks:
    raise ValueError("No chunks available to embed.")


# 5. Preview the first chunk.
first_chunk = chunks[0]

print("\nFirst chunk:")
print("Page:", first_chunk["page_number"])
print("Source:", first_chunk["source"])
print("Text:", first_chunk["text"])


# 6. Connect to Gemini.
client = genai.Client(api_key=gemini_key)

# Read Pinecone settings.
pinecone_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

if not pinecone_key or not index_name:
    raise ValueError(
        "Set PINECONE_API_KEY and PINECONE_INDEX_NAME in .env."
    )

# 7. Get the Pinecone index's host and configuration.
describe_request = Request(
    url=f"https://api.pinecone.io/indexes/{quote(index_name, safe='')}",
    headers={"Api-Key": pinecone_key},
)

with urlopen(describe_request, timeout=60) as response:
    index_details = json.load(response)

if not index_details["status"]["ready"]:
    raise ValueError("The Pinecone index is not ready yet.")

if index_details["dimension"] != 3072:
    raise ValueError("The Pinecone index must have 3072 dimensions.")

if index_details["metric"] != "cosine":
    raise ValueError("The Pinecone index must use cosine similarity.")

host = index_details["host"]


# 8. Embed and upload each chunk.
uploaded_count = 0

for chunk_number, chunk in enumerate(chunks):
    # Generate an embedding for this chunk.
    embedding_response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk["text"],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=3072,
        ),
    )

    if (
        not embedding_response.embeddings
        or not embedding_response.embeddings[0].values
    ):
        raise ValueError(f"No embedding returned for chunk {chunk_number}.")

    embedding = embedding_response.embeddings[0].values

    # Stable IDs update existing records when the script runs again.
    record = {
        "id": f"scarborough-2026-chunk-{chunk_number}",
        "values": embedding,
        "metadata": {
            "text": chunk["text"],
            "page_number": chunk["page_number"],
            "source": chunk["source"],
        },
    }

    payload = {
        "namespace": "scarborough-2026",
        "vectors": [record],
    }

    upload_request = Request(
        url=f"https://{host}/vectors/upsert",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Api-Key": pinecone_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urlopen(upload_request, timeout=60) as response:
        result = json.load(response)

    uploaded_count += result["upsertedCount"]

    print(
        f"Uploaded chunk {chunk_number + 1}/{len(chunks)} "
        f"(page {chunk['page_number']})"
    )

    # Space out embedding requests.
    # This reduces request frequency but doesn't guarantee quota availability.
    time.sleep(2)

print(f"\nFinished! Uploaded {uploaded_count} chunks.")