import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping


DEFAULT_F0_METHOD = "rmvpe"
DEFAULT_INFER_RELATIVE_PATH = Path("core") / "infer.py"


class RCVDefinitionError(ValueError):
    """Raised when an RCV definition is missing required values."""


class RCVExecutionError(RuntimeError):
    """Raised when the external RCV conversion command fails."""


@dataclass(frozen=True)
class RCVModelDefinition:
    """
    Definition of a custom trained voice model for conversion.

    This object is meant to be passed around the pipeline so model selection
    remains data-driven and easy to swap at runtime.
    """

    name: str
    model_path: Path
    index_path: Path | None = None
    pitch_shift: int = 0
    f0_method: str = DEFAULT_F0_METHOD

    def __post_init__(self):
        clean_name = (self.name or "").strip()
        if not clean_name:
            raise RCVDefinitionError("RCV model definition requires a non-empty 'name'.")
        object.__setattr__(self, "name", clean_name)
        object.__setattr__(self, "model_path", Path(self.model_path).expanduser())
        if self.index_path:
            object.__setattr__(self, "index_path", Path(self.index_path).expanduser())
        if not self.f0_method:
            object.__setattr__(self, "f0_method", DEFAULT_F0_METHOD)

    def validate_paths(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"RCV model file not found for '{self.name}': {self.model_path}"
            )
        if self.index_path and not self.index_path.exists():
            raise FileNotFoundError(
                f"RCV index file not found for '{self.name}': {self.index_path}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        if "name" not in data:
            raise RCVDefinitionError("RCV definition must include 'name'.")
        if "model_path" not in data:
            raise RCVDefinitionError(
                f"RCV definition '{data.get('name', '<unknown>')}' must include 'model_path'."
            )
        return cls(
            name=str(data["name"]),
            model_path=Path(str(data["model_path"])),
            index_path=Path(str(data["index_path"])) if data.get("index_path") else None,
            pitch_shift=int(data.get("pitch_shift", 0)),
            f0_method=str(data.get("f0_method", DEFAULT_F0_METHOD)),
        )

    def to_dict(self):
        return {
            "name": self.name,
            "model_path": str(self.model_path),
            "index_path": str(self.index_path) if self.index_path else None,
            "pitch_shift": self.pitch_shift,
            "f0_method": self.f0_method,
        }


@dataclass(frozen=True)
class RCVEngineDefinition:
    """
    Definition for where/how to execute the RCV inference script.
    """

    applio_dir: Path
    infer_relative_path: Path = DEFAULT_INFER_RELATIVE_PATH
    python_executable: str = sys.executable

    def __post_init__(self):
        object.__setattr__(self, "applio_dir", Path(self.applio_dir).expanduser())
        object.__setattr__(
            self, "infer_relative_path", Path(self.infer_relative_path).expanduser()
        )
        if not self.python_executable:
            object.__setattr__(self, "python_executable", sys.executable)

    @property
    def infer_script_path(self):
        return self.applio_dir / self.infer_relative_path

    def validate_paths(self):
        if not self.applio_dir.exists():
            raise FileNotFoundError(f"RCV applio_dir not found: {self.applio_dir}")
        if not self.infer_script_path.exists():
            raise FileNotFoundError(
                f"RCV inference script not found: {self.infer_script_path}"
            )


@dataclass
class RCVConversionResult:
    output_audio_path: Path
    command: list[str]
    return_code: int
    stdout: str
    stderr: str


class RCVDefinitionRegistry:
    """
    Named store for reusable model definitions.

    This allows pipeline code to select a model by name instead of manually
    wiring every path and inference option each turn.
    """

    def __init__(self, model_definitions: Mapping[str, RCVModelDefinition] | None = None):
        self._definitions: dict[str, RCVModelDefinition] = {}
        if model_definitions:
            for model_name, definition in model_definitions.items():
                if definition.name != model_name:
                    definition = replace(definition, name=model_name)
                self.register(definition)

    def register(self, definition: RCVModelDefinition):
        self._definitions[definition.name] = definition

    def get(self, model_name: str):
        if model_name not in self._definitions:
            available = ", ".join(sorted(self._definitions)) or "<none>"
            raise KeyError(
                f"Unknown RCV model definition '{model_name}'. Available: {available}"
            )
        return self._definitions[model_name]

    def names(self):
        return sorted(self._definitions.keys())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]):
        raw_models = payload.get("models", [])
        registry = cls()

        if isinstance(raw_models, Mapping):
            for model_name, raw_definition in raw_models.items():
                merged = dict(raw_definition or {})
                merged.setdefault("name", model_name)
                registry.register(RCVModelDefinition.from_dict(merged))
            return registry

        if isinstance(raw_models, list):
            for raw_definition in raw_models:
                registry.register(RCVModelDefinition.from_dict(raw_definition))
            return registry

        raise RCVDefinitionError(
            "Definitions payload must contain 'models' as a list or object."
        )

    @classmethod
    def from_json_file(cls, definitions_file: str | Path):
        file_path = Path(definitions_file).expanduser()
        if not file_path.exists():
            raise FileNotFoundError(f"RCV definitions file not found: {file_path}")
        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload)


def build_rcv_command(
    *,
    input_audio_path: str | Path,
    output_audio_path: str | Path,
    model_definition: RCVModelDefinition,
    engine_definition: RCVEngineDefinition,
    pitch_shift: int | None = None,
    f0_method: str | None = None,
):
    input_path = Path(input_audio_path).expanduser().resolve()
    output_path = Path(output_audio_path).expanduser().resolve()
    model_path = model_definition.model_path.expanduser().resolve()

    command = [
        engine_definition.python_executable,
        str(engine_definition.infer_script_path.expanduser().resolve()),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--model",
        str(model_path),
        "--pitch",
        str(model_definition.pitch_shift if pitch_shift is None else pitch_shift),
        "--method",
        f0_method or model_definition.f0_method,
    ]

    if model_definition.index_path:
        command.extend(
            [
                "--index",
                str(model_definition.index_path.expanduser().resolve()),
            ]
        )
    return command


def convert_with_rcv(
    *,
    input_audio_path: str | Path,
    output_audio_path: str | Path,
    model_definition: RCVModelDefinition,
    engine_definition: RCVEngineDefinition,
    pitch_shift: int | None = None,
    f0_method: str | None = None,
    dry_run: bool = False,
):
    input_path = Path(input_audio_path).expanduser().resolve()
    output_path = Path(output_audio_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    model_definition.validate_paths()
    engine_definition.validate_paths()

    command = build_rcv_command(
        input_audio_path=input_path,
        output_audio_path=output_path,
        model_definition=model_definition,
        engine_definition=engine_definition,
        pitch_shift=pitch_shift,
        f0_method=f0_method,
    )

    if dry_run:
        return RCVConversionResult(
            output_audio_path=output_path,
            command=command,
            return_code=0,
            stdout="",
            stderr="",
        )

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RCVExecutionError(
            "RCV conversion failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Exit code: {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    return RCVConversionResult(
        output_audio_path=output_path,
        command=command,
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def convert_with_registry_definition(
    *,
    input_audio_path: str | Path,
    output_audio_path: str | Path,
    model_name: str,
    definition_registry: RCVDefinitionRegistry,
    engine_definition: RCVEngineDefinition,
    pitch_shift: int | None = None,
    f0_method: str | None = None,
    dry_run: bool = False,
):
    model_definition = definition_registry.get(model_name)
    return convert_with_rcv(
        input_audio_path=input_audio_path,
        output_audio_path=output_audio_path,
        model_definition=model_definition,
        engine_definition=engine_definition,
        pitch_shift=pitch_shift,
        f0_method=f0_method,
        dry_run=dry_run,
    )


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run RCV voice conversion using a custom trained model definition."
    )
    parser.add_argument("--input-audio", required=True, help="Input WAV/MP3 audio path.")
    parser.add_argument("--output-audio", required=True, help="Output audio path.")
    parser.add_argument("--applio-dir", required=True, help="Path to the Applio directory.")
    parser.add_argument(
        "--definitions-file",
        help="JSON file containing named RCV model definitions.",
    )
    parser.add_argument(
        "--model-name",
        help="Name of model in definitions file. Required when --definitions-file is used.",
    )
    parser.add_argument(
        "--model-path",
        help="Direct path to model (.pth). Used when --definitions-file is omitted.",
    )
    parser.add_argument(
        "--index-path",
        help="Direct path to model index (.index), optional.",
    )
    parser.add_argument(
        "--pitch-shift",
        type=int,
        default=None,
        help="Override pitch shift (semitones).",
    )
    parser.add_argument(
        "--f0-method",
        default=None,
        help=f"Override f0 method (default: {DEFAULT_F0_METHOD}).",
    )
    parser.add_argument(
        "--infer-relative-path",
        default=str(DEFAULT_INFER_RELATIVE_PATH),
        help="Path from applio-dir to inference script.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used for the external RCV call.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without running conversion.",
    )
    return parser


def _model_definition_from_args(args):
    if args.definitions_file:
        if not args.model_name:
            raise RCVDefinitionError(
                "--model-name is required when --definitions-file is provided."
            )
        registry = RCVDefinitionRegistry.from_json_file(args.definitions_file)
        definition = registry.get(args.model_name)
    else:
        if not args.model_path:
            raise RCVDefinitionError(
                "--model-path is required when --definitions-file is not provided."
            )
        inferred_name = args.model_name or Path(args.model_path).stem
        definition = RCVModelDefinition(
            name=inferred_name,
            model_path=Path(args.model_path),
            index_path=Path(args.index_path) if args.index_path else None,
        )

    if args.pitch_shift is not None:
        definition = replace(definition, pitch_shift=args.pitch_shift)
    if args.f0_method is not None:
        definition = replace(definition, f0_method=args.f0_method)
    return definition


if __name__ == "__main__":
    parsed = _build_arg_parser().parse_args()
    model_def = _model_definition_from_args(parsed)
    engine_def = RCVEngineDefinition(
        applio_dir=Path(parsed.applio_dir),
        infer_relative_path=Path(parsed.infer_relative_path),
        python_executable=parsed.python_executable,
    )
    result = convert_with_rcv(
        input_audio_path=parsed.input_audio,
        output_audio_path=parsed.output_audio,
        model_definition=model_def,
        engine_definition=engine_def,
        dry_run=parsed.dry_run,
    )
    print(f"Model: {model_def.name}")
    print(f"Output audio: {result.output_audio_path}")
    print(f"Command: {' '.join(result.command)}")
    if parsed.dry_run:
        print("Dry run only (no conversion executed).")
