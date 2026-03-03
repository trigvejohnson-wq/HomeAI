import argparse
import asyncio
from pathlib import Path

import edge_tts

try:
    from .rcv import RCVEngineDefinition, RCVModelDefinition, convert_with_rcv
except ImportError:
    from rcv import RCVEngineDefinition, RCVModelDefinition, convert_with_rcv


async def generate_voice(text, output_path="basic_voice.wav", voice="en-US-GuyNeural"):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
    return output_path


def generate_voice_sync(text, output_path="basic_voice.wav", voice="en-US-GuyNeural"):
    return asyncio.run(generate_voice(text=text, output_path=output_path, voice=voice))


def generate_custom_voice(
    text,
    model_definition: RCVModelDefinition,
    engine_definition: RCVEngineDefinition,
    output_path="custom_voice.wav",
    base_output_path="basic_voice.wav",
    tts_voice="en-US-GuyNeural",
    pitch_shift=None,
    f0_method=None,
):
    """
    Full text -> edge-tts -> RCV conversion helper.
    """
    base_audio_path = generate_voice_sync(
        text=text,
        output_path=base_output_path,
        voice=tts_voice,
    )
    conversion = convert_with_rcv(
        input_audio_path=base_audio_path,
        output_audio_path=output_path,
        model_definition=model_definition,
        engine_definition=engine_definition,
        pitch_shift=pitch_shift,
        f0_method=f0_method,
    )
    return str(conversion.output_audio_path)


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate speech with edge-tts, optionally converting with RCV."
    )
    parser.add_argument("--text", default="Hello, how are you?", help="Text to speak.")
    parser.add_argument(
        "--voice",
        default="en-US-GuyNeural",
        help="Edge TTS voice name for base audio.",
    )
    parser.add_argument(
        "--output-path",
        default="basic_voice.wav",
        help="Output path for base TTS or final custom output.",
    )
    parser.add_argument(
        "--base-output-path",
        default="basic_voice.wav",
        help="Path to save the base TTS audio before RCV conversion.",
    )
    parser.add_argument(
        "--use-rcv",
        action="store_true",
        help="Enable RCV conversion using model/applio arguments.",
    )
    parser.add_argument("--model-name", help="Custom model name label.")
    parser.add_argument("--model-path", help="Path to custom trained .pth file.")
    parser.add_argument("--index-path", help="Path to custom .index file.")
    parser.add_argument("--applio-dir", help="Path to Applio root directory.")
    parser.add_argument(
        "--pitch-shift",
        type=int,
        default=0,
        help="Pitch shift for RCV conversion.",
    )
    parser.add_argument(
        "--f0-method",
        default="rmvpe",
        help="f0 method for RCV conversion.",
    )
    parser.add_argument(
        "--infer-relative-path",
        default="core/infer.py",
        help="Path to inference script relative to applio-dir.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.use_rcv:
        if not args.model_path:
            raise ValueError("--model-path is required when --use-rcv is set.")
        if not args.applio_dir:
            raise ValueError("--applio-dir is required when --use-rcv is set.")
        model_definition = RCVModelDefinition(
            name=args.model_name or Path(args.model_path).stem,
            model_path=Path(args.model_path),
            index_path=Path(args.index_path) if args.index_path else None,
            pitch_shift=args.pitch_shift,
            f0_method=args.f0_method,
        )
        engine_definition = RCVEngineDefinition(
            applio_dir=Path(args.applio_dir),
            infer_relative_path=Path(args.infer_relative_path),
        )
        generated = generate_custom_voice(
            text=args.text,
            model_definition=model_definition,
            engine_definition=engine_definition,
            output_path=args.output_path,
            base_output_path=args.base_output_path,
            tts_voice=args.voice,
        )
        print(generated)
    else:
        generated = generate_voice_sync(
            text=args.text,
            output_path=args.output_path,
            voice=args.voice,
        )
        print(generated)
