from dotenv import load_dotenv
import os

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

print("Gemini key loaded:", bool(gemini_key))
print("Pinecone key loaded:", bool(pinecone_key))