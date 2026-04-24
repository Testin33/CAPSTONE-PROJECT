"""
organize_tasks.py
-----------------
Scans subject folders (1-s, 2-s, ... 25-s), pairs each CSV with its
matching MP4 by shared timestamp, sorts pairs chronologically, and
copies them into a centralized output directory:

    output/
        task_1_csv/   <- all subjects' Task-1 CSVs (with 'task' column added)
        task_1_mp4/   <- all subjects' Task-1 MP4s (original names, prefixed by subject)
        task_2_csv/
        task_2_mp4/
        ...
        task_5_csv/
        task_5_mp4/

Usage
-----
    python organize_tasks.py                              # source = current dir
    python organize_tasks.py  path/to/subjects           # custom source dir
    python organize_tasks.py  path/to/subjects  --out path/to/output
    python organize_tasks.py  path/to/subjects  --dry-run
"""

import argparse
import re
import shutil
import csv
from pathlib import Path

REBA_STEM = re.compile(r"^(\d{8}-\d{6})_REBA$", re.IGNORECASE)
SUBJECT_DIR = re.compile(r"^\d+-s$", re.IGNORECASE)
N_TASKS = 5


def find_pairs(subject_dir: Path) -> list[tuple[Path, Path]]:
    """
    Return list of (csv_path, mp4_path) sorted by timestamp.
    Pairs are matched by shared timestamp stem (YYYYMMDD-HHMMSS_REBA).
    Raises ValueError if the count doesn't match expectations.
    """
    stems: dict[str, dict[str, Path]] = {}

    for f in subject_dir.iterdir():
        if not f.is_file():
            continue
        m = REBA_STEM.match(f.stem)
        if not m:
            continue
        ts = m.group(1)
        ext = f.suffix.lower()
        if ext not in (".csv", ".mp4"):
            continue
        stems.setdefault(ts, {})
        stems[ts][ext] = f

    complete = {ts: exts for ts, exts in stems.items()
                if ".csv" in exts and ".mp4" in exts}

    if not complete:
        return []

    sorted_pairs = [
        (complete[ts][".csv"], complete[ts][".mp4"])
        for ts in sorted(complete.keys())
    ]
    return sorted_pairs


def add_task_column(src_csv: Path, dest_csv: Path, task_num: int):
    """Copy src_csv to dest_csv, inserting a 'task' column as the first column."""
    with open(src_csv, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        original_fields = reader.fieldnames or []
        new_fields = ["task"] + original_fields

        with open(dest_csv, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=new_fields)
            writer.writeheader()
            for row in reader:
                row["task"] = task_num
                writer.writerow(row)


def organize(source_dir: Path, out_dir: Path, dry_run: bool):
    subject_dirs = sorted(
        [d for d in source_dir.iterdir()
         if d.is_dir() and SUBJECT_DIR.match(d.name)],
        key=lambda d: int(d.name.split("-")[0])
    )

    if not subject_dirs:
        print(f"No subject folders (e.g. 1-s, 2-s) found in '{source_dir}'")
        return

    print(f"Found {len(subject_dirs)} subject folder(s): "
          f"{[d.name for d in subject_dirs]}\n")

    errors = []
    all_pairs: dict[str, list[tuple[Path, Path]]] = {}

    for subj_dir in subject_dirs:
        pairs = find_pairs(subj_dir)
        if len(pairs) != N_TASKS:
            msg = (f"  WARNING: '{subj_dir.name}' has {len(pairs)} pair(s), "
                   f"expected {N_TASKS} — skipping")
            print(msg)
            errors.append(msg)
            continue
        all_pairs[subj_dir.name] = pairs
        print(f"  {subj_dir.name}: {len(pairs)} pairs OK")

    print()

    if not all_pairs:
        print("No valid subjects to process.")
        return

    # Create output folders
    task_csv_dirs = {n: out_dir / f"task_{n}_csv" for n in range(1, N_TASKS + 1)}
    task_mp4_dirs = {n: out_dir / f"task_{n}_mp4" for n in range(1, N_TASKS + 1)}

    if not dry_run:
        for d in list(task_csv_dirs.values()) + list(task_mp4_dirs.values()):
            d.mkdir(parents=True, exist_ok=True)

    total_csv = total_mp4 = 0

    for subj_name, pairs in sorted(all_pairs.items(),
                                   key=lambda x: int(x[0].split("-")[0])):
        for task_num, (csv_path, mp4_path) in enumerate(pairs, start=1):
            prefix = subj_name  # e.g. "3-s"

            dest_csv = task_csv_dirs[task_num] / f"{prefix}_{csv_path.name}"
            dest_mp4 = task_mp4_dirs[task_num] / f"{prefix}_{mp4_path.name}"

            print(f"  [{subj_name}] Task {task_num}")
            print(f"    CSV : {csv_path.name} → {dest_csv.relative_to(out_dir)}")
            print(f"    MP4 : {mp4_path.name} → {dest_mp4.relative_to(out_dir)}")

            if not dry_run:
                add_task_column(csv_path, dest_csv, task_num)
                shutil.copy2(mp4_path, dest_mp4)
                total_csv += 1
                total_mp4 += 1

    print()
    if dry_run:
        print("(dry-run — no files were copied or modified)")
    else:
        print(f"Done.  {total_csv} CSVs written  |  {total_mp4} MP4s copied")
        print(f"Output → '{out_dir}'")

    if errors:
        print(f"\nWarnings ({len(errors)}):")
        for e in errors:
            print(e)


def main():
    parser = argparse.ArgumentParser(
        description="Organise REBA subject folders into task-based output directories."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=".",
        help="Directory containing subject folders 1-s … 25-s (default: current dir)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: <source>/output)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without copying anything",
    )
    args = parser.parse_args()

    source_dir = Path(args.source).resolve()
    out_dir    = Path(args.out).resolve() if args.out else source_dir / "output"

    if not source_dir.is_dir():
        print(f"Error: '{source_dir}' is not a directory.")
        return

    organize(source_dir, out_dir, args.dry_run)


if __name__ == "__main__":
    main()
