# Simple Guide: Training a Custom Voice Model with Applio

Applio is an AI voice cloning tool (based on RVC) that lets you train custom voice models from your own audio samples. This guide walks you through the full process.

---

## Prerequisites

- **NVIDIA GPU** (RTX 2000 series or higher recommended; 8GB VRAM minimum for training)
- **Applio** installed ([get it from GitHub](https://github.com/IAHispano/Applio))
- **Audacity** (free) for cleaning audio
- **~10–30 minutes** of clean voice recordings

---

## Part 1: Prepare Your Dataset

Dataset quality is the most important factor for good results.

### Requirements

| Requirement | Details |
|-------------|---------|
| **Duration** | 10–30 minutes total of clean speech |
| **Format** | Lossless only: `.wav` or `.flac` |
| **Quality** | No background noise, reverb, or music |

### Cleaning with Audacity

1. **Noise Reduction**
   - Select a short segment with only background noise
   - *Effect → Noise Reduction and Repair → Noise Reduction*
   - Click "Get Noise Profile"
   - Select the full track → apply the effect

2. **Noise Gate** (removes low-level noise between words)
   - *Effect → Gating → Noise Gate*
   - Adjust threshold as needed

3. **Truncate Silence** (removes long silences)
   - *Effect → Truncate Silence*
   - Use default or light settings

### Vocal Isolation

If your source has music or other sounds, isolate the vocals first. Applio docs cover this in their [UVR (Ultimate Vocal Remover) guide](https://docs.applio.org/guides/uvr/).

### Export Settings

- **WAV**: Microsoft signed 16-bit PCM  
- **FLAC**: Lossless, smaller files

---

## Part 2: Place Your Dataset

1. Open your Applio install folder.
2. Go to: `applio/assets/datasets/`
3. Create a new folder: `your-model-name/`
4. Put all your `.wav` or `.flac` files directly inside it.

**Multi-speaker models:** Add numbered subfolders `0`, `1`, `2`, etc., one per speaker, with their files inside each.

---

## Part 3: Train in Applio

Open Applio and go to the **Train** tab.

### Step 1: Pre-process Dataset

- Enter your model name in the Train tab.
- Choose sample rate: **32k**, **40k**, or **48k** (match your source audio).
- Click **Pre-process Dataset**.

### Step 2: Extract Features

- Click **Extract Features**.
- **Embedder:** Select the right one for your use case.
- **Pitch extraction:** Use **RMVPE**.
- Wait for extraction to finish (progress in the command line).

### Step 3: Train the Model

- Click **Train Model**.
- **Batch size:** 6–8 for ~8GB VRAM; adjust for your GPU.
- **Total epochs:** 200–400 (start around 300).
- **Save every epoch:** 10–50 (e.g. 25).
- Start training and monitor progress with TensorBoard.
- When training is done, click **Train Index**.

### Step 4: Export Your Model

- Go to the **Export Model** section.
- Click **Refresh**.
- Choose your `.pth` and `.index` in the dropdowns.
- Click **Export Model**.

Exported models are saved in the `logs` folder and can be used for inference.

---

## Part 4: Using Your Model

Use the exported `.pth` and `.index` files in Applio’s inference/conversion tab, or with other RVC-compatible tools. In your project, you can point `custom_voice.py` to the model path:

```python
model_path = "path/to/your-exported-model.pth"  # plus .index file
```

---

## Quick Reference

| Step | Action |
|------|--------|
| 1 | Record/collect 10–30 min of clean audio |
| 2 | Clean in Audacity (noise reduction, gate, truncate silence) |
| 3 | Export as WAV or FLAC |
| 4 | Place files in `applio/assets/datasets/your-model-name/` |
| 5 | Pre-process → Extract features → Train → Train index → Export |

---

## Tips

- **Overfitting:** If the model sounds robotic or unstable, stop earlier or reduce epochs.
- **Underfitting:** If it doesn’t sound like the target, train longer or add more data.
- **Resume training:** Use “Start Training” with a higher max epoch and the same model/settings to continue from the last checkpoint.
- **Cloud option:** Applio has Colab notebooks if you don’t have a suitable GPU.

---

## Resources

- [Applio Official Docs](https://docs.applio.org/)
- [Dataset creation guide](https://docs.applio.org/guides/how-create-datasets/)
- [Applio GitHub](https://github.com/IAHispano/Applio)
