import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from google import genai



load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")


# Create Gemini client (new SDK requires Client, not configure)
client = genai.Client(api_key=API_KEY)

# Model name used by Gemini during generation
model = "models/gemini-2.5-flash"




def create_embedding(text_list):
    try:
        # Generate embeddings for input text using Ollama
        r = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "bge-m3", "input": text_list},
            timeout=50
        )
        r.raise_for_status()
        return r.json()['embeddings']
    except requests.RequestException as e:
        print(f"Error creating embeddings: {e}")
        return []


# Load precomputed embeddings of video subtitle chunks
df = joblib.load('embeddings.joblib')

# Stack embeddings into matrix for cosine similarity
embeddings_matrix = np.vstack(df['embedding'].values)

incoming_query = input('Ask a question: ')

# Convert user question into embedding
query_embedding = create_embedding([incoming_query])[0]

top_results = 10

# Compare query embedding with all subtitle embeddings
similarities = cosine_similarity(embeddings_matrix, [query_embedding]).flatten()

# Pick indices of most relevant subtitle chunks
max_idx = similarities.argsort()[::-1][0:top_results]

new_df = df.iloc[max_idx]


# Prompt contains only the most relevant chunks + user question
prompt = f'''I am teaching machine learning in my course. 
Here are video subtitle chunks containing video title, video number, start time in seconds, 
end time in seconds, the text at that time:

---------------------------------------
{new_df[['title','number','start','end','text']].to_json(orient='records',force_ascii=False)}
---------------------------------------
Question: "{incoming_query}"

User asked this question related to the video chunks, you have to answer in a human way 
(don't mention the above format, its just for you) where and how much content is taught in which video 
(in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, 
tell him that you can only answer questions related to the course
'''

with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)


def inference(prompt: str) -> str:
    try:
        # Local LLM inference via Ollama (not used in final answer)
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "timeout": 100
            }
        )
        r.raise_for_status()
        return r.json().get('response', '')
    except requests.RequestException as e:
        print(f"Error during inference: {e}")
        return "Error generating response"


def inference_gemini(prompt):
    # Final answer generation using Gemini
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    return response.text.strip()


response = inference_gemini(prompt)
print(response)

with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)
