"""Utilities for querying the converted BioSignal NPZ database.

Provides a small helper class to read the conversion report and load
BioSignal objects by simple filters.
"""

import json
from enum import Enum
from pathlib import Path
from typing import List, Dict
from biosignal import BioSignal


class BioSignalOrganismType(Enum):
    """Enumerated organism types used by the converters and database."""

    HUMAN = "human"
    PLANT = "plant"
    MUSHROOM = "fungi"
    UNKNOWN = "unknown"


class BioSignalDatabase:
    """Query and filter BioSignals from standardized NPZ database."""

    def __init__(self, data_folder='data_converted'):
        self.data_path = Path(data_folder)
        self.report_path = self.data_path / 'conversion_report.json'
        self.files = self._load_file_list()

    def _load_file_list(self):
        """Load file manifest from conversion report."""
        if self.report_path.exists():
            with open(self.report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
                return report.get('files_processed', [])
        else:
            # Fallback: scan directory
            return [
                {'output': f.name, 'organism': 'unknown'}
                for f in self.data_path.glob('*.npz')
            ]

    def filter_by_organism(self, organism: str) -> List[Dict]:
        """Find all files for a specific organism."""
        return [f for f in self.files if f.get('organism') == organism]

    def filter_by_type(self, recording_type: str) -> List[Dict]:
        """Find files by recording type."""
        return [f for f in self.files if recording_type.lower() in f.get('source', '').lower()]

    def load_signal(self, filename: str) -> BioSignal:
        """Load a single BioSignal file."""
        return BioSignal.load(str(self.data_path / filename))

    def load_by_organism(self, organism: str, limit: int = None) -> List[BioSignal]:
        """Load all signals for an organism."""
        files = self.filter_by_organism(organism)
        if limit:
            files = files[:limit]

        signals = []
        for f in files:
            try:
                bio = self.load_signal(f['output'])
                signals.append((f['output'], bio))
            except (OSError, IOError, ValueError) as exc:
                # Skip files we cannot read or parse and report the cause.
                print(f"[SKIP] {f['output']}: {exc}")

        return signals

    def get_statistics(self):
        """Summary statistics about the database."""
        stats = {
            'total_files': len(self.files),
            'by_organism': {},
            'by_source': {}
        }

        for f in self.files:
            org = f.get('organism', 'unknown')
            stats['by_organism'][org] = stats['by_organism'].get(org, 0) + 1

            src = f.get('source', 'unknown')
            stats['by_source'][src] = stats['by_source'].get(src, 0) + 1

        return stats

    def list_all(self, organism: str = None):
        """List all files, optionally filtered by organism."""
        files = self.filter_by_organism(organism) if organism else self.files

        for f in files:
            print(f"{f.get('output'):<60} | {f.get('organism'):<8} | {f.get('species'):<30}")
