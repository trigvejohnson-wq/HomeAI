from rvc_python.infer import RVCInference

def convert_to_custom_voice(input_path, output_path, model_path="path/to/model.pth", device="cuda:0"):
    rvc = RVCInference(device=device)
    rvc.load_model(model_path)
    rvc.infer_file(input_path, output_path)
    return output_path

if __name__ == "__main__":
    # Update these paths to test with your files
    input_path = "basic_voice.wav"  # e.g., output from edge-tts (edgy.py)
    output_path = "custom_voice_output.wav"
    model_path = "path/to/model.pth"  # path to your RVC voice model

    result = convert_to_custom_voice(input_path, output_path, model_path)
    print(f"Converted to custom voice: {result}")