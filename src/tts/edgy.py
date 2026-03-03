import edge_tts
import asyncio

async def generate_voice(text, output_path="basic_voice.wav", voice="en-US-GuyNeural"):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
    return output_path

if __name__ == "__main__":
    text = "Hello, how are you?"
    output_path = "basic_voice.wav"
    voice = "en-US-GuyNeural"
    generate_voice(text, output_path, voice)