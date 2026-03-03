from rvc_python.infer import RVCInference

def convert_to_custom_voice(input_path, output_path, model_path="path/to/model.pth", device="cuda:0"):
    rvc = RVCInference(device=device)
    rvc.load_model(model_path)
    rvc.infer_file(input_path, output_path)
    return output_path