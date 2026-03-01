# Guide: Creating an AI with Custom Voice, Personality, and Multimodal Input

A comprehensive step-by-step guide for building an AI assistant that uses your microphone and camera as input, with a custom voice and personality you define.

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Technology Choices](#technology-choices)
4. [Project Setup](#project-setup)
5. [Step 1: Microphone Input & Speech-to-Text](#step-1-microphone-input--speech-to-text)
6. [Step 2: Camera Input & Visual Processing](#step-2-camera-input--visual-processing)
7. [Step 3: Personality & LLM Integration](#step-3-personality--llm-integration)
8. [Step 4: Text-to-Speech with Custom Voice](#step-4-text-to-speech-with-custom-voice)
9. [Step 5: Orchestrating the Pipeline](#step-5-orchestrating-the-pipeline)
10. [Step 6: Real-Time Interaction Loop](#step-6-real-time-interaction-loop)
11. [Configuration & Customization](#configuration--customization)
12. [Security & Privacy Considerations](#security--privacy-considerations)
13. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Microphone    │────▶│  Speech-to-Text   │────▶│                 │
└─────────────────┘     │  (Whisper, etc.)  │     │                 │
                        └──────────────────┘     │   LLM + Brain    │
┌─────────────────┐     ┌──────────────────┐     │  (OpenAI, etc.)  │
│     Camera      │────▶│ Visual Analysis   │────▶│                 │
└─────────────────┘     │ (YOLO, etc.)      │     │                 │
                        └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Speakers      │◀────│   Text-to-Speech  │◀────│  Response Text   │
└─────────────────┘     │ (Custom Voice)   │     └─────────────────┘
                        └──────────────────┘
```

**Flow:** Audio + Video → Processed Input → LLM (with personality) → Response Text → Custom Voice Output

---

## Prerequisites

Before starting, ensure you have:

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10+ recommended |
| **Hardware** | USB/webcam camera, working microphone |
| **API Keys** | OpenAI (or alternative LLM) |
| **GPU** | NVIDIA GPU with 4GB+ VRAM for inference, 8GB+ recommended for training |
| **GPU (optional for STT)** | Improves Whisper and vision model performance |
| **OS** | Windows, macOS, or Linux |

---

## Technology Choices

### Speech-to-Text (STT)
| Option | Pros | Cons | Cost |
|--------|------|------|------|
| **OpenAI Whisper** | Excellent accuracy, offline possible | Heavier, needs GPU for real-time | Free (self-hosted) |
| **Whisper API** | No setup | Per-minute cost | ~$0.006/min |
| **Google Speech-to-Text** | Very accurate | Requires internet | Free tier, then paid |
| **Vosk** | Fully offline, lightweight | Less accurate than Whisper | Free |

### LLM (Language Model)
| Option | Pros | Cons | Cost |
|--------|------|------|------|
| **OpenAI GPT-4o** | Best quality, multimodal | API cost | Per token |
| **OpenAI GPT-4o-mini** | Fast, cheaper | Slightly less capable | Cheaper |
| **Anthropic Claude** | Strong reasoning | API cost | Per token |
| **Local (Ollama, LM Studio)** | Free, private | Needs GPU, slower | Free |

### Text-to-Speech (TTS) + Voice Conversion

We use a two-stage pipeline: a TTS engine generates speech, then RVC converts it to a custom voice (e.g., an anime character).

**Stage 1 — TTS Engine (generates the base speech):**
| Option | Pros | Cons | Cost |
|--------|------|------|------|
| **Edge TTS** | Free, many voices, fast | Requires internet | Free |
| **Piper TTS** | Fast, fully offline | Less natural | Free |
| **OpenAI TTS** | Good quality, simple | API cost | Per character |
| **Coqui XTTS** | Open source, decent quality | Deprecated, community forks | Free |

**Stage 2 — Voice Conversion (converts to target character voice):**
| Option | Pros | Cons | Cost |
|--------|------|------|------|
| **RVC (via Applio)** | Best quality, active community, easy UI | Needs NVIDIA GPU | Free |
| **RVC (standalone)** | Flexible, scriptable | More manual setup | Free |
| **GPT-SoVITS** | Few-shot cloning, good for anime | Heavier, newer project | Free |

> **Why not ElevenLabs?** Commercial voice cloning services have policies that prohibit cloning voices you don't own the rights to (e.g., anime characters). RVC is open-source and self-hosted, so there are no such restrictions for personal use.

### Vision (Camera)
| Option | Use Case | Notes |
|--------|----------|-------|
| **GPT-4 Vision / Claude Vision** | Send frames to multimodal LLM | Paid API |
| **YOLO (Ultralytics)** | Object detection, pose, segmentation | Free, local, flexible (v8/v9) |
| **OpenCV + YOLO** | People, pets, objects, actions | Full local control, export to ONNX/TensorRT |

---

## Project Setup

### 1. Create Project Directory

```bash
mkdir ai-voice-personality
cd ai-voice-personality
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Create `requirements.txt`

```txt
# Speech-to-Text
openai-whisper>=20231117
# or: pip install openai  # for Whisper API

# Audio
pyaudio>=0.2.13
sounddevice>=0.4.6
numpy>=1.24.0

# Camera / Vision
opencv-python>=4.8.0
Pillow>=10.0.0
ultralytics>=8.0.0   # YOLO (v8/v9) for local detection, pose, segmentation

# LLM
openai>=1.0.0
anthropic>=0.7.0

# TTS (base voice generation — choose one or more)
edge-tts>=6.1.0
# openai>=1.0.0  # already listed above for LLM

# Voice Conversion (RVC — installed separately via Applio, see Step 4)

# Utilities
python-dotenv>=1.0.0
pydub>=0.25.1
```

### 4. Environment Variables (`.env`)

```env
OPENAI_API_KEY=sk-...
# Optional
ANTHROPIC_API_KEY=...
```

> **Note:** RVC/Applio runs entirely locally — no API keys needed for voice conversion.

---

## Step 1: Microphone Input & Speech-to-Text

### 1.1 Record Audio from Microphone

```python
import sounddevice as sd
import numpy as np

def record_audio(duration_sec=5, sample_rate=16000):
    """Record audio from default microphone."""
    print("Recording...")
    audio = sd.rec(
        int(duration_sec * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32
    )
    sd.wait()
    return audio.flatten(), sample_rate
```

### 1.2 Transcribe with Whisper (Local)

```python
import whisper

model = whisper.load_model("base")  # tiny, base, small, medium, large

def transcribe_audio(audio_data, sample_rate=16000):
    result = model.transcribe(
        (audio_data * 32767).astype(np.int16),
        fp16=False,
        language="en"
    )
    return result["text"].strip()
```

### 1.3 Transcribe with Whisper API (Cloud)

```python
import openai
from pydub import AudioSegment
import io

def transcribe_with_api(audio_data, sample_rate=16000):
    # Convert to WAV bytes
    audio_int16 = (audio_data * 32767).astype(np.int16)
    buffer = io.BytesIO()
    # Use scipy or pydub to write WAV
    # ... 
    buffer.seek(0)
    
    with open("temp.wav", "wb") as f:
        f.write(buffer.getvalue())
    
    with open("temp.wav", "rb") as f:
        result = openai.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text"
        )
    return result.strip()
```

### 1.4 Voice Activity Detection (VAD) – Optional

Use `webrtcvad` or `silero-vad` to detect when the user starts/stops speaking, so you don’t process silence.

---

## Step 2: Camera Input & Visual Processing

### 2.1 Capture Frame from Camera

```python
import cv2
import base64

def capture_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Could not read from camera")
    return frame
```

### 2.2 Convert Frame for LLM (Base64)

```python
def frame_to_base64(frame, format="jpeg", quality=85):
    _, buffer = cv2.imencode(f".{format}", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode("utf-8")
```

### 2.3 Send Image to Multimodal LLM (GPT-4 Vision)

```python
def get_visual_context(frame):
    b64 = frame_to_base64(frame)
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        }
    ]
```

### 2.4 Alternative: Local Visual Analysis (YOLO)

Use YOLO (Ultralytics) for fully local object detection, pose, or segmentation. Models run on CPU or GPU and can be exported to ONNX or TensorRT for faster inference.

**Object detection** (people, pets, common objects):

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # n=nano, s=small, m=medium, l=large

def analyze_frame_yolo(frame):
    results = model(frame, verbose=False)[0]
    # results.boxes: xyxy, conf, class_id
    # results.names: class id -> label (e.g. 0 -> "person")
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        detections.append({
            "class": results.names[cls_id],
            "confidence": conf,
            "bbox": box.xyxy[0].tolist()
        })
    return detections
```

**Pose estimation** (skeletons for posture or gesture cues):

```python
pose_model = YOLO("yolov8n-pose.pt")

def analyze_pose(frame):
    results = pose_model(frame, verbose=False)[0]
    # results.keypoints: person keypoints (e.g. nose, shoulders, wrists)
    return results.keypoints
```

Use detection/pose output to build a **scene description** (e.g. “person present”, “dog in frame”) and pass that text to your LLM instead of sending raw images, keeping vision processing entirely on-device.

---

## Step 3: Personality & LLM Integration

### 3.1 Define Personality in System Prompt

```python
PERSONALITY = """
You are a helpful AI assistant named [YOUR NAME]. Your personality is:
- [Trait 1]: e.g., warm and enthusiastic
- [Trait 2]: e.g., curious and asks follow-up questions
- [Trait 3]: e.g., uses humor when appropriate
- Speaking style: Conversational, uses contractions, occasional emoji if it fits.
- Avoid: Being robotic, overly formal, or repeating yourself.
"""
```

### 3.2 Example Personas

**Friendly Coach:**
```
You are Coach Alex, an energetic fitness coach. You're supportive, use 
short punchy sentences, and often say things like "Let's go!" and 
"You've got this!"
```

**Calm Mentor:**
```
You are Sage, a thoughtful mentor. You speak slowly in concept, use 
analogies, and avoid jargon. You often start with "Let me think about 
that..." when considering complex questions.
```

### 3.3 Call LLM with Text + Optional Image

```python
from openai import OpenAI
client = OpenAI()

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
```

---

## Step 4: Text-to-Speech with Custom Voice (RVC Pipeline)

Our TTS pipeline has two stages:
1. **TTS Engine** — converts LLM text response to speech (generic voice)
2. **RVC Voice Conversion** — converts that speech into the target character voice (e.g., Luffy)

### 4.1 Setting Up RVC with Applio

Applio is a user-friendly frontend for RVC. Install it separately from your main project:

```bash
# Clone Applio (do this outside your project directory)
git clone https://github.com/IAHispano/Applio.git
cd Applio

# Run the installer (Windows)
python install.py

# Launch the web UI
python app.py
```

### 4.2 Training a Voice Model

1. **Collect training data**: Gather 10–20 minutes of clean audio clips of the target voice (e.g., Luffy). Remove background music and sound effects using **UVR (Ultimate Vocal Remover)**.
2. **Prepare audio**: WAV format, mono, 44.1kHz or higher.
3. In Applio's **Train** tab:
   - Set a model name (e.g., `luffy_v1`)
   - Upload your cleaned audio files
   - Use **v2 architecture**, **40k or 48k sample rate**
   - Train for 200–500 epochs (monitor loss to avoid overfitting)
4. The trained model will be saved as a `.pth` file with an associated `.index` file.

### 4.3 Stage 1 — Base TTS (Edge TTS)

Edge TTS is free, fast, and produces natural-sounding speech — a good base voice for RVC to convert.

```python
import edge_tts
import asyncio

async def generate_base_tts(text, output_path="tts_output.wav", voice="en-US-GuyNeural"):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
    return output_path

def text_to_base_speech(text, output_path="tts_output.wav"):
    return asyncio.run(generate_base_tts(text, output_path))
```

### 4.4 Stage 2 — RVC Voice Conversion

Use RVC's inference pipeline to convert the base TTS audio to the target voice. You can call Applio's CLI or use the `rvc` library directly:

```python
import subprocess
import os

RVC_MODEL_PATH = "path/to/luffy_v1.pth"
RVC_INDEX_PATH = "path/to/luffy_v1.index"
APPLIO_DIR = "path/to/Applio"

def convert_voice_rvc(input_audio_path, output_path="rvc_output.wav",
                      pitch_shift=0, method="rmvpe"):
    """Convert audio to target voice using Applio CLI."""
    cmd = [
        "python", os.path.join(APPLIO_DIR, "core", "infer.py"),
        "--input", input_audio_path,
        "--output", output_path,
        "--model", RVC_MODEL_PATH,
        "--index", RVC_INDEX_PATH,
        "--pitch", str(pitch_shift),
        "--method", method,
    ]
    subprocess.run(cmd, check=True)
    return output_path
```

### 4.5 Combined TTS + Voice Conversion

```python
def speak_as_character(text, output_path="final_output.wav"):
    """Full pipeline: text → TTS → RVC → character voice."""
    base_audio = text_to_base_speech(text, "tts_temp.wav")
    final_audio = convert_voice_rvc(base_audio, output_path)
    return final_audio
```

### 4.6 Alternative: OpenAI TTS as Base Voice

```python
from openai import OpenAI
client = OpenAI()

def speak_with_openai(text, output_path="tts_output.wav", voice="alloy"):
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text
    )
    with open(output_path, "wb") as f:
        f.write(response.content)
    return output_path
```

### 4.7 Play Audio

```python
import sounddevice as sd
from scipy.io import wavfile

def play_audio(audio_path):
    rate, data = wavfile.read(audio_path)
    sd.play(data, rate)
    sd.wait()
```

---

## Step 5: Orchestrating the Pipeline

### 5.1 Main Pipeline Function

```python
def run_one_turn(use_camera=True):
    # 1. Record
    audio, sr = record_audio(duration_sec=5)
    
    # 2. Transcribe
    text = transcribe_audio(audio, sr)
    if not text:
        return "I didn't hear anything. Try again?"
    
    # 3. Optional: get camera frame
    image_content = None
    if use_camera:
        frame = capture_frame()
        image_content = frame_to_base64(frame)
    
    # 4. Get LLM response
    response_text = get_ai_response(text, image_content)
    
    # 5. Speak it (TTS → RVC → play)
    audio_path = speak_as_character(response_text)
    play_audio(audio_path)
    
    return response_text
```

---

## Step 6: Real-Time Interaction Loop

### 6.1 Continuous Loop with Wake Word (Optional)

```python
def main_loop():
    conversation_history = []
    
    while True:
        print("Speak when ready (or say 'exit' to quit)...")
        audio, sr = record_audio(duration_sec=5)
        text = transcribe_audio(audio, sr)
        
        if "exit" in text.lower() or "goodbye" in text.lower():
            break
            
        if not text:
            continue
            
        # Add to history
        conversation_history.append({"role": "user", "content": text})
        
        # Get response
        response = get_ai_response(text, conversation_history=conversation_history)
        conversation_history.append({"role": "assistant", "content": response})
        
        # Speak (TTS → RVC → play)
        audio_path = speak_as_character(response)
        play_audio(audio_path)
```

### 6.2 Push-to-Talk vs. Continuous Listening

- **Push-to-Talk**: User holds a key/button to record. Simpler and more reliable.
- **Continuous**: Use VAD to detect speech start/end. More natural but needs tuning.

---

## Configuration & Customization

### `config.yaml` Example

```yaml
personality:
  name: "Alex"
  traits:
    - "Friendly and supportive"
    - "Uses simple language"
  system_prompt: "You are Alex, a helpful assistant..."

voice:
  tts_provider: "edge_tts"  # or openai, piper
  tts_voice: "en-US-GuyNeural"
  rvc_model: "path/to/luffy_v1.pth"
  rvc_index: "path/to/luffy_v1.index"
  pitch_shift: 0
  inference_method: "rmvpe"

input:
  microphone_sample_rate: 16000
  camera_index: 0
  use_camera: true
  record_duration_sec: 5

llm:
  provider: "openai"
  model: "gpt-4o"
  max_tokens: 500
```

---

## Security & Privacy Considerations

| Concern | Recommendation |
|---------|----------------|
| **Camera** | Process frames locally when possible; only send to API if needed |
| **Microphone** | Inform users when recording; store no audio by default |
| **API keys** | Use `.env` and never commit keys |
| **Data retention** | Check OpenAI API policies; use opt-out if available |
| **Voice models** | RVC models run locally — no data leaves your machine |
| **Local fallback** | Use local Whisper + local LLM (Ollama) for sensitive use cases |

---

## Troubleshooting

### Microphone not working
- Check default device: `python -c "import sounddevice; print(sounddevice.query_devices())"`
- On Windows, ensure exclusive mode is off for the mic
- Try `sounddevice` instead of `pyaudio` if one fails

### Camera not detected
- Verify camera index: try `0`, `1`, `2` in `cv2.VideoCapture(0)`
- On Windows, ensure no other app is using the camera
- Test with: `python -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened())"`

### Whisper too slow
- Use `tiny` or `base` model
- Use Whisper API instead of local
- Use a GPU and `fp16=True`

### TTS sounds robotic or doesn't match the character
- Choose a base TTS voice with similar pitch/energy to the target character for better RVC conversion
- Train the RVC model with more data or more epochs
- Experiment with `pitch_shift` (positive for higher, negative for lower)
- Try different inference methods (`rmvpe`, `crepe`, `harvest`)
- Use `tts-1-hd` with OpenAI for a higher-quality base voice

### High latency
- Use streaming: stream LLM tokens and TTS as they arrive
- Use smaller models (e.g., `gpt-4o-mini`)
- Consider Edge TTS for faster, free TTS

---

## Next Steps

1. **Streaming**: Stream LLM output and start TTS before the full response is ready.
2. **Wake word**: Add “Hey Assistant” style detection with Porcupine or similar.
3. **Scene and identity**: Use YOLO for detection plus face recognition or ReID to identify people and pets by name; add action detection (e.g. smoking) from objects or pose.
4. **Persistent memory**: Add a vector DB (e.g., ChromaDB) so the AI remembers past conversations.

---

*Last updated: February 2026*
