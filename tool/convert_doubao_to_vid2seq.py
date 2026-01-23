import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Minimal parser for strings shaped like "{person: man, action: pour, object: cup, environment: kitchen}".
# The upstream JSON stores each frame's label as a string rather than a JSON object.
def _parse_label(raw: str) -> Dict[str, str]:
    text = raw.strip().strip('"').strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    fields = {}
    for chunk in text.split(','):
        if ':' not in chunk:
            continue
        key, value = chunk.split(':', 1)
        fields[key.strip()] = value.strip()
    return {
        "person": fields.get("person", "unknown"),
        "action": fields.get("action", "unknown"),
        "object": fields.get("object", "unknown"),
        "environment": fields.get("environment", "unknown"),
    }


def _bin_index(frame_idx: int, total_frames: int, bins: int) -> int:
    # Uses the formula in the prompt: floor(bins * frame / N)
    return math.floor(bins * frame_idx / total_frames)


def _collect_frames_count(frames_dir: Path) -> int:
    # Counts files that look like frames (jpg/png/jpeg). Falls back to 0 if none.
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    count = 0
    for pattern in patterns:
        count += sum(1 for _ in frames_dir.glob(pattern))
    return count


def _is_empty_label(label: Dict[str, str]) -> bool:
    """Check if a label contains only whitespace or meaningless content."""
    for key in ["person", "action", "object", "environment"]:
        value = label.get(key, "").strip()
        if value and value not in ["", "unknown", "still"]:
            return False
    return True


def _group_segments(labels: List[Dict[str, str]]) -> List[Tuple[int, int, Dict[str, str]]]:
    if not labels:
        return []
    segments: List[Tuple[int, int, Dict[str, str]]] = []
    start = 0
    prev = labels[0]
    for idx, label in enumerate(labels[1:], start=1):
        if label != prev:
            segments.append((start, idx - 1, prev))
            start = idx
            prev = label
    segments.append((start, len(labels) - 1, prev))
    return segments


def convert(input_json: Path, frames_dir: Path, bins: int = 100) -> List[str]:
    raw_labels = json.loads(input_json.read_text(encoding="utf-8"))
    parsed_labels = [_parse_label(item) for item in raw_labels]

    total_frames = _collect_frames_count(frames_dir)
    if total_frames == 0:
        total_frames = len(parsed_labels)
    if total_frames == 0:
        raise ValueError("No frames or labels found")

    segments = _group_segments(parsed_labels)
    output_lines: List[str] = []
    for f_start, f_end, label in segments:
        if _is_empty_label(label):
            continue
        t_start = _bin_index(f_start, total_frames, bins)
        t_end = _bin_index(f_end, total_frames, bins)
        output_lines.append(
            f"<{t_start}> <{t_end}> {label['person']} {label['action']} {label['object']} {label['environment']}."
        )
    return output_lines


def _default_video_id(path: Path) -> str:
    match = re.search(r"(\d+)", path.stem)
    return match.group(1) if match else path.stem


def convert_file(
    input_json: Path,
    frames_dir: Optional[Path] = None,
    bins: int = 100,
    video_id: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    frames_dir = frames_dir or input_json.parent
    video_id = video_id or _default_video_id(input_json)
    output_path = input_json.parent / f"vid2seq_{video_id}.json"

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite to replace it.")

    lines = convert(input_json, frames_dir, bins=bins)
    output_path.write_text(json.dumps(lines, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def _iter_doubao_files(root: Path) -> Iterable[Path]:
    yield from root.rglob("doubao_results_*.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert doubao results to vid2seq-style tokens")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "input_json",
        type=Path,
        nargs="?",
        help="Path to doubao_results_XXXX.json",
    )
    mode.add_argument(
        "--dataset-root",
        type=Path,
        help="Process all doubao_results_*.json under this root recursively",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        help="Directory containing extracted frames; defaults to the input file's parent",
    )
    parser.add_argument("--bins", type=int, default=100, help="Number of temporal bins (default: 100)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        help="Video identifier used in the output filename; defaults to digits in the input name",
    )
    args = parser.parse_args()

    outputs: List[Path] = []

    if args.dataset_root:
        if not args.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")
        for input_json in _iter_doubao_files(args.dataset_root):
            try:
                out_path = convert_file(
                    input_json=input_json,
                    frames_dir=args.frames_dir or input_json.parent,
                    bins=args.bins,
                    video_id=_default_video_id(input_json),
                    overwrite=args.overwrite,
                )
                outputs.append(out_path)
            except FileExistsError:
                # Skip existing files unless overwrite is set
                continue
    else:
        input_json: Path = args.input_json
        if not input_json.exists():
            raise FileNotFoundError(f"Input file not found: {input_json}")
        out_path = convert_file(
            input_json=input_json,
            frames_dir=args.frames_dir or input_json.parent,
            bins=args.bins,
            video_id=args.video_id,
            overwrite=args.overwrite,
        )
        outputs.append(out_path)

    if not outputs:
        print("No files converted (all outputs already existed?). Use --overwrite to force regeneration.")
    else:
        print(f"Converted {len(outputs)} file(s):")
        for p in outputs:
            print(f"  {p}")


if __name__ == "__main__":
    main()
