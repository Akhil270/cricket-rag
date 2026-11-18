import json
import os
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from dotenv import load_dotenv


# Load your existing settings.
project_folder = Path(__file__).resolve().parent
load_dotenv(project_folder / ".env")

api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

if not api_key or not index_name:
    raise ValueError(
        "Set PINECONE_API_KEY and PINECONE_INDEX_NAME in .env."
    )

# Describe the index we want Pinecone to create.
settings = {
    "name": index_name,
    "vector_type": "dense",
    "dimension": 3072,
    "metric": "cosine",
    "spec": {
        "serverless": {
            "cloud": "aws",
            "region": "us-east-1",
        }
    },
}

# Send the creation request to Pinecone.
request = Request(
    url="https://api.pinecone.io/indexes",
    data=json.dumps(settings).encode("utf-8"),
    headers={
        "Api-Key": api_key,
        "Content-Type": "application/json",
    },
    method="POST",
)

try:
    with urlopen(request, timeout=60) as response:
        result = json.load(response)

    print("Index creation requested:", result["name"])
    print("Refresh the Pinecone Indexes page to check its status.")

except HTTPError as error:
    print("Pinecone returned an error:", error.code)
    print(error.read().decode("utf-8"))