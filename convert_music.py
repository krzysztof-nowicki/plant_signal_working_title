"""
Scan a folder and convert MP3/MIDI files to MusicSignal NPZ format.
"""

import json
from pathlib import Path

from converters.mp3_converter import convert_mp3
from converters.midi_converter import convert_midi

SUPPORTED_FORMATS = {
    ".mp3": ("MP3", convert_mp3),
    ".mid": ("MIDI", convert_midi),
    ".midi": ("MIDI", convert_midi),
}


def detect_music_metadata_from_path(file_path: Path):
    """
    Detect music metadata from file path.

    Returns:
        (author, music_type, source)
    """

    path_str = str(file_path).lower()
    filename = file_path.stem

    author = "unknown"
    music_type = "unknown"
    source = str(file_path)

    parent_dir = file_path.parent.name.lower()

    genres = {
        "Electronic": ["electronic", "synth", "edm", "house", "techno"],
        "Rock": ["rock", "metal", "punk", "alternative"],
        "Pop": ["pop", "mainstream"],
        "Classical": ["classical", "symphony", "orchestra"],
        "Jazz": ["jazz", "bebop", "fusion"],
        "Ambient": ["ambient", "experimental", "drone"],
    }

    for genre, keywords in genres.items():
        if any(word in path_str for word in keywords):
            music_type = genre
            break
    else:
        music_type = parent_dir or "unknown"

    if " - " in filename:
        author = filename.split(" - ")[0].strip()
    elif parent_dir:
        author = parent_dir

    return author, music_type, source


def process_music_folder(input_folder: str, output_folder: str):
    """
    Convert every MP3/MIDI file found inside input_folder.
    """

    input_path = Path(input_folder)
    output_path = Path(output_folder)

    if not input_path.exists():
        raise FileNotFoundError(f"Input folder '{input_folder}' does not exist.")

    output_path.mkdir(parents=True, exist_ok=True)

    stats = {
        "found": 0,
        "success": 0,
        "failed": 0,
    }

    processed = []
    errors = []

    for file_path in input_path.rglob("*"):

        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()

        if extension not in SUPPORTED_FORMATS:
            continue

        file_type, converter = SUPPORTED_FORMATS[extension]

        relative_path = file_path.relative_to(input_path)

        output_file = (
                output_path /
                relative_path.with_suffix(".npz")
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)

        author, music_type, source = detect_music_metadata_from_path(file_path)

        stats["found"] += 1

        try:

            print(f"Converting {relative_path}")

            music_signal = converter(
                str(file_path),
                author=author,
                music_type=music_type,
                source=source,
            )

            music_signal.save(str(output_file))

            stats["success"] += 1

            processed.append({
                "file": str(relative_path),
                "output": str(output_file.relative_to(output_path)),
                "type": file_type,
                "author": author,
                "music_type": music_type,
                "duration": music_signal.metadata.length,
                "sampling_rate": music_signal.fs,
            })

            print("   OK")

        except Exception as exc:

            stats["failed"] += 1

            errors.append(f"{relative_path}: {exc}")

            print("   FAILED")

    report = {
        "statistics": stats,
        "processed": processed,
        "errors": errors,
    }

    report_file = output_path / "conversion_report.json"

    with open(report_file, "w", encoding="utf8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("Conversion finished")
    print("=" * 60)
    print(f"Found:     {stats['found']}")
    print(f"Success:   {stats['success']}")
    print(f"Failed:    {stats['failed']}")
    print(f"Output:    {output_path.resolve()}")
    print(f"Report:    {report_file.resolve()}")


if __name__ == "__main__":
    input_folder = './data_music/midi_classical'
    output_folder = './music_data_converted/midi_classical'

    process_music_folder(
        input_folder,
        output_folder,
    )
