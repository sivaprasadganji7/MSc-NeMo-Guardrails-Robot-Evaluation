import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # or "gpt-4" if you have access
    messages=[
        {"role": "system", "content": "You are a helpful movie expert."},
        {"role": "user", "content": "Recommend a good sci-fi movie."}
    ],
    temperature=0.7,
    max_tokens=150
)

print(response.choices[0].message.content)