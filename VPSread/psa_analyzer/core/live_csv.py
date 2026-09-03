"""
Tail a growing CSV produced by eServer, yielding only the newly-appended rows.

eServer logs one row per sample (~1 Hz) into a CSV that keeps growing. Rather
than re-reading the whole file every second, :class:`CsvTailReader` tracks a
byte offset and parses only the bytes added since the last poll. It is robust
to:

* a partial final line (kept in a buffer until the rest arrives),
* the file being truncated / replaced (offset resets automatically),
* CRLF or LF line endings, and ``,`` / ``;`` / tab delimiters (auto-sniffed).

Nothing here imports Qt — the worker thread wraps this; tests drive it directly.
"""

from __future__ import annotations

import csv
from pathlib import Path


def newest_csv(folder: str | Path, pattern: str = "*.csv") -> Path | None:
    """Return the most-recently-modified CSV in ``folder`` (or None)."""
    folder = Path(folder)
    if not folder.is_dir():
        return None
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


class CsvTailReader:
    """
    Incremental reader for a single growing CSV file.

    Usage::

        r = CsvTailReader("D:/PSA_live/log.csv")
        while True:
            for row in r.read_new():   # list of {column: value} dicts
                ...
            time.sleep(1)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.header: list[str] | None = None
        self._offset = 0
        self._remainder = ""
        self._delim = ","

    def reset(self) -> None:
        """Forget all state — re-read the file from the top on next poll."""
        self.header = None
        self._offset = 0
        self._remainder = ""
        self._delim = ","

    @staticmethod
    def _sniff_delimiter(sample: str) -> str:
        try:
            return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
        except csv.Error:
            return ","

    def read_new(self) -> list[dict]:
        """
        Return rows appended since the last call (possibly empty).

        The first non-empty call also parses the header row; subsequent calls
        return data rows only.
        """
        if not self.path.exists():
            return []

        size = self.path.stat().st_size
        if size < self._offset:          # truncated or replaced -> start over
            self.reset()
        if size == self._offset and not self._remainder:
            return []                    # nothing new

        with self.path.open("r", encoding="utf-8-sig",
                            errors="replace", newline="") as f:
            f.seek(self._offset)
            chunk = f.read()
            self._offset = f.tell()

        if not chunk:
            return []

        data = self._remainder + chunk
        lines = data.split("\n")
        self._remainder = lines.pop()    # last element is the partial line

        rows: list[dict] = []
        for raw in lines:
            line = raw.rstrip("\r")
            if not line.strip():
                continue
            if self.header is None:
                self._delim = self._sniff_delimiter(line)
                self.header = next(csv.reader([line], delimiter=self._delim))
                continue
            values = next(csv.reader([line], delimiter=self._delim))
            if len(values) != len(self.header):
                continue                 # skip malformed / mis-aligned line
            rows.append(dict(zip(self.header, values)))
        return rows
