import os
import json
from dotenv import load_dotenv
import pinecone
import openai

# Load environment variables
load_dotenv()

# Initialize Pinecone and OpenAI
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create or connect to Pinecone index
index_name = "rag"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        pod_type="p1"
    )
index = pinecone.Index(index_name)

# Load review data
with open("reviews.json") as f:
    data = json.load(f)

# Process data and create embeddings
processed_data = []
for review in data["reviews"]:
    response = openai.Embedding.create(
        input=review['review'],
        model="text-embedding-ada-002"
    )
    embedding = response['data'][0]['embedding']
    processed_data.append(
        {
            "id": review["professor"],
            "values": embedding,
            "metadata": {
                "review": review["review"],
                "subject": review["subject"],
                "stars": review["stars"],
            }
        }
    )

# Upsert the embeddings into Pinecone
upsert_response = index.upsert(
    vectors=processed_data,
    namespace="ns1",
)
print(f"Upserted count: {len(processed_data)}")

# Print index statistics
print(index.describe_index_stats())
