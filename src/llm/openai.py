from openai import OpenAI
client = OpenAI()

PERSONALITY = "You are Patrick from SpongeBob Squarepants."

def get_ai_response(text_input, image_content=None, conversation_history=[]):
    messages = [
        {"role": "system", "content": PERSONALITY},
        *conversation_history,
        {"role": "user", "content": []}
    ]
    
    # Text
    messages[-1]["content"].append({"type": "text", "text": text_input})
    
    # Optional image
    if image_content:
        messages[-1]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_content}"}
        })
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=500
    )
    return response.choices[0].message.content