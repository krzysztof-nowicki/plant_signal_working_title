"""Scan a data folder and convert MP3 music files to NPZ format.

This script walks through music folders and converts MP3 files into the
standardized MusicSignal NPZ format using the converter from the
'converters' package.
"""

import sys
from pathlib import Path
import json

from converters.mp3_converter import convert_mp3

sys.path.insert(0, str(Path(__file__).parent))


def detect_music_metadata_from_path(file_path):
    """
    Detect music metadata from file path.

    Returns: (author, music_type, source)
    """
    path_str = str(file_path).lower()
    filename = file_path.stem

    # Defaults
    author = "unknown"
    music_type = "unknown"
    source = str(file_path)

    # Try to detect artist and genre from directory structure
    parent_dir = file_path.parent.name.lower()

    # Common music type patterns
    if any(x in path_str for x in ['electronic', 'synth', 'edm', 'house', 'techno']):
        music_type = "Electronic"
    elif any(x in path_str for x in ['rock', 'metal', 'punk', 'alternative']):
        music_type = "Rock"
    elif any(x in path_str for x in ['pop', 'mainstream']):
        music_type = "Pop"
    elif any(x in path_str for x in ['classical', 'symphony', 'orchestra']):
        music_type = "Classical"
    elif any(x in path_str for x in ['jazz', 'bebop', 'fusion']):
        music_type = "Jazz"
    elif any(x in path_str for x in ['ambient', 'experimental', 'drone']):
        music_type = "Ambient"
    else:
        music_type = parent_dir if parent_dir else "unknown"

    # Try to detect author from filename or directory
    if ' - ' in filename:
        author = filename.split(' - ')[0].strip()
    elif parent_dir and parent_dir != 'music':
        author = parent_dir

    return author, music_type, source


def process_music_folder(data_folder='data_music', output_folder='music_data_converted'):
    """
    Scan music folder and convert all MP3 files to standardized NPZ format.
    """
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    data_path = Path(data_folder)

    if not data_path.exists():
        print(f"[ERROR] Music folder '{data_folder}' not found!")
        return

    stats = {
        'mp3': {'found': 0, 'success': 0, 'failed': 0},
    }

    files_processed = []
    errors = []

    for file_path in data_path.rglob('*'):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        if suffix != '.mp3':
            continue

        relative_path = file_path.relative_to(data_path)
        output_file = output_path / f"{file_path.stem}.npz"

        # Detect music metadata from path
        author, music_type, source = detect_music_metadata_from_path(file_path)

        stats['mp3']['found'] += 1

        try:
            print(f"Converting MP3: {relative_path}...", end=' ')
            music_signal = convert_mp3(
                str(file_path),
                author=author,
                music_type=music_type,
                source=source
            )
            music_signal.save(str(output_file))
            stats['mp3']['success'] += 1
            files_processed.append({
                'type': 'MP3',
                'author': author,
                'music_type': music_type,
                'source': str(relative_path),
                'output': output_file.name,
                'duration': music_signal.metadata.length,
                'channels': music_signal.channels,
                'sampling_rate': music_signal.fs
            })
            print("[OK]")

        except (OSError, IOError, ValueError, RuntimeError) as e:
            error_msg = f"Error processing {relative_path}: {str(e)}"
            print("[FAILED]")
            errors.append(error_msg)
            stats['mp3']['failed'] += 1

    # Print summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY - Music Signal Format (NPZ)")
    print("=" * 60)

    total_found = stats['mp3']['found']
    total_success = stats['mp3']['success']
    total_failed = stats['mp3']['failed']

    if total_found > 0:
        print(f"MP3: {total_found} found -> {total_success} OK, {total_failed} FAILED")
    else:
        print("No MP3 files found")

    print("-" * 60)
    print(f"Total: {total_found} files -> {total_success} converted, {total_failed} errors")
    print(f"Output folder: {output_path.absolute()}")

    # Statistics by music type
    music_type_counts = {}
    for f in files_processed:
        mtype = f.get('music_type', 'unknown')
        music_type_counts[mtype] = music_type_counts.get(mtype, 0) + 1

    if music_type_counts:
        print("\nMusic signals by type:")
        for mtype, count in sorted(music_type_counts.items()):
            print(f"  {mtype}: {count}")

    if errors:
        print("\nERRORS:")
        for error in errors[:5]:
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    # Save processing report
    report = {
        'total_found': total_found,
        'total_success': total_success,
        'total_failed': total_failed,
        'stats': stats,
        'files_processed': files_processed,
        'errors': errors,
        'music_type_distribution': music_type_counts
    }

    report_path = output_path / 'music_conversion_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        data_folder_system = sys.argv[1]
        output_folder_system = sys.argv[2] if len(sys.argv) > 2 else 'music_data_converted'
        process_music_folder(data_folder_system, output_folder_system)
    else:
        process_music_folder()
