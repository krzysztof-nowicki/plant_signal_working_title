import os
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from converters.wav_converter import convert_wav
from converters.csv_converter import convert_csv
from converters.edf_converter import convert_edf


def detect_organism_from_path(file_path):
    """
    Detect organism type from file path.
    
    Returns: (organism, species, recording_type, source)
    """
    path_str = str(file_path).lower()
    
    # Defaults
    organism = "unknown"
    species = "unknown"
    recording_type = "unknown"
    source = "unknown"

    if any(x in path_str for x in ['plant', 'drosera', 'mimosa', 'tomato', 'araucaria']):
        organism = "plant"
        source = "Plant_Recording"
        species = os.path.basename(os.path.dirname(path_str))
        recording_type = "plant_type_recording"
            
    elif any(x in path_str for x in ['shroom', 'fungi', 'mycelium', 'pleurotus', 'oyster']):
        organism = "fungi"
        source = "Fungal_Recording"
        
        if 'pleurotus' in path_str or 'oyster' in path_str:
            species = "Pleurotus ostreatus"
        else:
            species = "unknown_species"
        recording_type = "mycelium_recording"
        
    elif any(x in path_str for x in ['human', 'eeg', 'eeg', 'ecg', 'patient', 'baby']):
        organism = "human"
        source = "EEG_Recording"
        
        if 'baby' in path_str:
            species = "Homo sapiens (infant)"
            recording_type = "EEG"
        else:
            species = "Homo sapiens"
            recording_type = "EEG"
    
    return organism, species, recording_type, source


def process_data_folder(data_folder='data', output_folder='data_converted'):
    """
    Scan data folder and convert all biomedical signals to standardized NPZ format.
    """
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    data_path = Path(data_folder)
    
    if not data_path.exists():
        print(f"[ERROR] Data folder '{data_folder}' not found!")
        return
    
    stats = {
        'wav': {'found': 0, 'success': 0, 'failed': 0},
        'csv': {'found': 0, 'success': 0, 'failed': 0},
        'edf': {'found': 0, 'success': 0, 'failed': 0},
    }
    
    files_processed = []
    errors = []
    
    for file_path in data_path.rglob('*'):
        if not file_path.is_file():
            continue
        
        suffix = file_path.suffix.lower()
        relative_path = file_path.relative_to(data_path)
        output_file = output_path / f"{file_path.stem}.npz"
        
        # Detect organism metadata from path
        organism, species, recording_type, source = detect_organism_from_path(file_path)
        
        try:
            if suffix == '.wav':
                stats['wav']['found'] += 1
                print(f"Converting WAV: {relative_path}...", end=' ')
                bio_signal = convert_wav(
                    str(file_path),
                    organism=organism,
                    species=species,
                    recording_type=recording_type,
                    source=source,
                    units="unknown"
                )
                bio_signal.save(str(output_file))
                stats['wav']['success'] += 1
                files_processed.append({
                    'type': 'WAV',
                    'organism': organism,
                    'species': species,
                    'source': str(relative_path),
                    'output': output_file.name
                })
                print("[OK]")
                
            elif suffix == '.csv':
                stats['csv']['found'] += 1
                print(f"Converting CSV: {relative_path}...", end=' ')
                bio_signal = convert_csv(
                    str(file_path),
                    fs=1000,
                    organism=organism,
                    species=species,
                    recording_type=recording_type,
                    source=source,
                    units="unknown"
                )
                bio_signal.save(str(output_file))
                stats['csv']['success'] += 1
                files_processed.append({
                    'type': 'CSV',
                    'organism': organism,
                    'species': species,
                    'source': str(relative_path),
                    'output': output_file.name
                })
                print("[OK]")
                
            elif suffix == '.edf':
                stats['edf']['found'] += 1
                print(f"Converting EDF: {relative_path}...", end=' ')
                bio_signal = convert_edf(
                    str(file_path),
                    organism=organism,
                    species=species,
                    recording_type=recording_type,
                    source=source,
                    units="µV"
                )
                bio_signal.save(str(output_file))
                stats['edf']['success'] += 1
                files_processed.append({
                    'type': 'EDF',
                    'organism': organism,
                    'species': species,
                    'source': str(relative_path),
                    'output': output_file.name
                })
                print("[OK]")
                
        except Exception as e:
            error_msg = f"Error processing {relative_path}: {str(e)}"
            print(f"[FAILED] {error_msg}")
            errors.append(error_msg)
            if suffix == '.wav':
                stats['wav']['failed'] += 1
            elif suffix == '.csv':
                stats['csv']['failed'] += 1
            elif suffix == '.edf':
                stats['edf']['failed'] += 1
    
    # Print summary
    print("\n" + "="*60)
    print("CONVERSION SUMMARY - Standardized BioSignal Format")
    print("="*60)
    
    total_found = sum(s['found'] for s in stats.values())
    total_success = sum(s['success'] for s in stats.values())
    total_failed = sum(s['failed'] for s in stats.values())
    
    for format_type, counts in stats.items():
        if counts['found'] > 0:
            print(f"{format_type.upper()}: {counts['found']} found -> "
                  f"{counts['success']} OK, {counts['failed']} FAILED")
    
    print("-"*60)
    print(f"Total: {total_found} files -> {total_success} converted, {total_failed} errors")
    print(f"Output folder: {output_path.absolute()}")
    
    # Count by organism
    organism_counts = {}
    for f in files_processed:
        org = f.get('organism', 'unknown')
        organism_counts[org] = organism_counts.get(org, 0) + 1
    
    print("\nBioSignals by organism:")
    for org, count in sorted(organism_counts.items()):
        print(f"  {org}: {count}")
    
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
        'organism_distribution': organism_counts
    }
    
    report_path = output_path / 'conversion_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        data_folder = sys.argv[1]
        output_folder = sys.argv[2] if len(sys.argv) > 2 else 'data_converted'
        process_data_folder(data_folder, output_folder)
    else:
        process_data_folder()
