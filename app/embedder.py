import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embeddings(text_chunks):
    embeddings = []
    for chunk in text_chunks:
        response = openai.Embedding.create(
            input=chunk,
            model="text-embedding-3-small"
        )
        embeddings.append(response['data'][0]['embedding'])
    return embeddings
