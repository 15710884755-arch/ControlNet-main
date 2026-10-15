"""Download an HTTP file with several parallel range requests.

huggingface_hub's transfer backends stall on some high latency links even though
the CDN keeps serving plain ranged GETs, so the SDXL model downloader falls back
to this helper. Downloads are resumable: finished chunks are recorded in a
``.part.json`` sidecar next to the target file and skipped on the next run.

Usage:
    python scripts/parallel_download.py <url> <destination>
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import urllib.request
from pathlib import Path


DEFAULT_CHUNK_SIZE = 16 * 1024 * 1024
DEFAULT_CONNECTIONS = 8
PASSES = 3
TIMEOUT = 60
USER_AGENT = "controlnet-model-downloader/1.0"


def _open(url, **headers):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **headers})
    return urllib.request.urlopen(request, timeout=TIMEOUT)


def remote_size(url):
    """Return the size the server reports, or None when it reports nothing usable."""
    with _open(url, Range="bytes=0-0") as response:
        content_range = response.headers.get("Content-Range")
        if content_range and content_range.rsplit("/", 1)[-1].strip().isdigit():
            return int(content_range.rsplit("/", 1)[-1])
        length = response.headers.get("Content-Length")
        if length and length.isdigit():
            return int(length)
    return None


def plan_chunks(total, chunk_size=DEFAULT_CHUNK_SIZE):
    """Split ``total`` bytes into inclusive byte ranges."""
    ranges = []
    start = 0
    while start < total:
        end = min(total, start + chunk_size) - 1
        ranges.append((start, end))
        start = end + 1
    return ranges


def journal_path(dest):
    """Sidecar that records which chunks of an interrupted download already landed."""
    dest = Path(dest)
    return dest.with_name(dest.name + ".part.json")


def is_complete(dest):
    """True when the file is on disk and no interrupted download journal is left behind."""
    dest = Path(dest)
    return dest.is_file() and not journal_path(dest).is_file()


class _ChunkJournal:
    """Remembers which chunks already landed on disk."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.done = set()
        if path.is_file():
            try:
                self.done = {int(item) for item in json.loads(path.read_text(encoding="utf-8"))}
            except (OSError, TypeError, ValueError):
                self.done = set()

    def mark(self, index):
        with self.lock:
            self.done.add(index)
            payload = json.dumps(sorted(self.done))
        temp = self.path.with_name(self.path.name + ".tmp")
        temp.write_text(payload, encoding="utf-8")
        os.replace(temp, self.path)

    def discard(self):
        try:
            self.path.unlink()
        except OSError:
            pass


def _fetch_range(url, start, end, attempts=4):
    """Fetch one byte range, returning None when the server ignores ranges."""
    last_error = None
    for _attempt in range(attempts):
        try:
            with _open(url, Range="bytes=%d-%d" % (start, end)) as response:
                if response.status != 206:
                    return None
                data = response.read()
            expected = end - start + 1
            if len(data) != expected:
                raise RuntimeError("short read: %d of %d bytes" % (len(data), expected))
            return data
        except Exception as error:  # noqa: BLE001 - retried above
            last_error = error
    raise RuntimeError("range %d-%d failed: %s" % (start, end, last_error))


def _write_chunk(dest, offset, data):
    with open(dest, "r+b") as handle:
        handle.seek(offset)
        handle.write(data)


def _progress(name, finished, total):
    return "%s: %d/%d chunk(s), %.0f%%" % (name, finished, total, 100.0 * finished / total)


def _download_parallel(url, dest, pending, total_chunks, journal, connections, report):
    """Fetch ``pending`` chunks with a small thread pool, returning the failures."""
    lock = threading.Lock()
    cursor = [0]
    finished = [total_chunks - len(pending)]
    failures = []

    def worker():
        while True:
            with lock:
                if cursor[0] >= len(pending):
                    return
                index, start, end = pending[cursor[0]]
                cursor[0] += 1
            try:
                data = _fetch_range(url, start, end)
                if data is None:
                    raise RuntimeError("server stopped honouring range requests")
                _write_chunk(dest, start, data)
                journal.mark(index)
            except Exception as error:  # noqa: BLE001 - reported by the caller
                with lock:
                    failures.append((index, str(error)))
                report("%s: chunk %d failed, will retry" % (dest.name, index))
                continue
            with lock:
                finished[0] += 1
                message = _progress(dest.name, finished[0], total_chunks)
            report(message)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(min(connections, len(pending)))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return failures


def _download_sequential(url, dest, size, report):
    with _open(url) as response, open(dest, "wb") as handle:
        written = 0
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            handle.write(block)
            written += len(block)
            if written % (64 << 20) < (1 << 20):
                report("%s: %.0f%%" % (dest.name, 100.0 * written / size))
    if dest.stat().st_size != size:
        raise RuntimeError("incomplete download for " + str(dest))
    return dest


def download(url, dest, connections=DEFAULT_CONNECTIONS, chunk_size=DEFAULT_CHUNK_SIZE, log=None):
    """Fetch ``url`` into ``dest``, resuming the chunks an earlier run already wrote."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    report = log or (lambda _message: None)

    size = remote_size(url)
    if not size:
        raise RuntimeError("the server did not report a size for " + url)

    journal = _ChunkJournal(journal_path(dest))
    if is_complete(dest) and dest.stat().st_size == size:
        report(dest.name + ": already downloaded")
        return dest

    chunks = plan_chunks(size, chunk_size)
    pending = [(index, start, end) for index, (start, end) in enumerate(chunks) if index not in journal.done]

    with open(dest, "r+b" if dest.is_file() else "wb") as handle:
        handle.truncate(size)

    if pending:
        first = pending[0]
        data = _fetch_range(url, first[1], first[2])
        if data is None:
            report(dest.name + ": server ignores range requests, downloading in a single stream")
            journal.discard()
            return _download_sequential(url, dest, size, report)
        _write_chunk(dest, first[1], data)
        journal.mark(first[0])

    # A chunk failure is usually a dropped connection, so retry the survivors a few
    # times before giving up on the file.
    last_error = None
    for attempt in range(1, PASSES + 1):
        remaining = [(index, start, end) for index, (start, end) in enumerate(chunks) if index not in journal.done]
        if not remaining:
            break
        report("%s: %.1f MB, pass %d/%d, %d chunk(s) left"
               % (dest.name, size / 1e6, attempt, PASSES, len(remaining)))
        failures = _download_parallel(url, dest, remaining, len(chunks), journal, connections, report)
        if failures:
            last_error = failures[0][1]

    remaining = [index for index in range(len(chunks)) if index not in journal.done]
    if remaining:
        raise RuntimeError("could not finish %s (%d chunk(s) left): %s"
                           % (dest.name, len(remaining), last_error))

    if dest.stat().st_size != size:
        raise RuntimeError("incomplete download for " + str(dest))
    journal.discard()
    report("%s: complete, %.1f MB" % (dest.name, size / 1e6))
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("destination")
    parser.add_argument("--connections", type=int, default=DEFAULT_CONNECTIONS)
    parser.add_argument("--chunk-size-mb", type=int, default=DEFAULT_CHUNK_SIZE // (1024 * 1024))
    args = parser.parse_args()

    download(
        args.url,
        Path(args.destination),
        connections=args.connections,
        chunk_size=args.chunk_size_mb * 1024 * 1024,
        log=lambda message: print(message, flush=True),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
