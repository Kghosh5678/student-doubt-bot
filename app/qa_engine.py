import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def answer_question(question, context_chunks):
    prompt = "Answer the question based only on the context below.\n\n"
    for i, chunk in enumerate(context_chunks):
        prompt += f"Context {i+1}:\n{chunk}\n\n"
    prompt += f"Question: {question}\nAnswer:"

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()
