
import os
import re
import shutil
import argparse

# ── Pattern: YYYYMMDD-HHMMSS_REBA
FILE_RE = re.compile(r'^(\d{8}-\d{6})_REBA\.(mp4|csv)$', re.IGNORECASE)

# ── Already-prefixed pattern: {n}_{YYYYMMDD-HHMMSS}_REBA
PREFIXED_RE = re.compile(r'^\d+_(\d{8}-\d{6})_REBA\.(mp4|csv)$', re.IGNORECASE)

# ── Sample folder pattern: digits _ YYYYMMDD _ HHMMSS
FOLDER_RE = re.compile(r'^(\d+)_\d{8}_\d{6}$')


# Helpers

def get_base_dir(override=None):
    if override:
        return os.path.abspath(override)
    return os.path.dirname(os.path.abspath(__file__))


def find_pairs_in(folder):
    """Return list of (timestamp_str, mp4_path, csv_path) sorted oldest→newest
    for every complete mp4+csv pair directly inside `folder`."""
    mp4s, csvs = {}, {}
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue
        # Accept both original names and already-prefixed names
        m = FILE_RE.match(fname) or PREFIXED_RE.match(fname)
        if not m:
            continue
        ts, ext = m.group(1), m.group(2).lower()
        if ext == 'mp4':
            mp4s[ts] = fpath
        elif ext == 'csv':
            csvs[ts] = fpath
    common = set(mp4s) & set(csvs)
    pairs = [(ts, mp4s[ts], csvs[ts]) for ts in common]
    pairs.sort(key=lambda x: x[0])   # oldest first
    return pairs


def next_sample_index(base_dir):
    existing = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and FOLDER_RE.match(d)
    ]
    return len(existing) + 1


def ts_to_folder_suffix(ts_str):
    return ts_str.replace('-', '_')


# MODE 1 — organize

def organize(base_dir, expected=5):
    pairs = find_pairs_in(base_dir)

    if not pairs:
        print("No loose REBA file pairs found. Nothing to organize.")
        return

    print(f"Found {len(pairs)} pair(s):")
    for idx, (ts, mp4, csv_) in enumerate(pairs, start=1):
        print(f"  Task {idx}: {ts}")

    if len(pairs) != expected:
        print(f"\nWarning: expected {expected} pairs but found {len(pairs)}.")
        answer = input("Continue anyway? [y/N] ").strip().lower()
        if answer != 'y':
            print("Aborted.")
            return

    i = next_sample_index(base_dir)
    folder_name = f"{i}_{ts_to_folder_suffix(pairs[0][0])}"
    folder_path = os.path.join(base_dir, folder_name)

    os.makedirs(folder_path, exist_ok=True)
    print(f"\nCreated folder: {folder_name}")

    for task_num, (ts, mp4, csv_) in enumerate(pairs, start=1):
        for src in (mp4, csv_):
            dst = os.path.join(folder_path, os.path.basename(src))
            shutil.move(src, dst)
            print(f"  [Task {task_num}] {os.path.basename(src)} → {folder_name}/")

    print(f"\nDone. Sample {i} organized into '{folder_name}'.")


# MODE 2 — distribute

def distribute(base_dir, task_folder_paths):
    """
    For each sample folder, sort its pairs newest→oldest.
    pair[0] (most recent)  → task_folder_paths[0]   (e.g. 1_task)
    pair[1]                → task_folder_paths[1]   (e.g. 2_task)
    pair[2]                → task_folder_paths[2]   (e.g. 3_task)

    Files are copied (originals kept) and renamed:
        {sample_num}_{original_filename}
    """
    # Resolve task folder paths (allow relative to base_dir)
    resolved_task_folders = []
    for tf in task_folder_paths:
        p = tf if os.path.isabs(tf) else os.path.join(base_dir, tf)
        if not os.path.isdir(p):
            print(f"Error: task folder not found: {p}")
            print("Please create the task folders before running --distribute.")
            return
        resolved_task_folders.append(p)

    # Find sample folders sorted by sample number
    sample_dirs = sorted(
        [d for d in os.listdir(base_dir)
         if os.path.isdir(os.path.join(base_dir, d)) and FOLDER_RE.match(d)],
        key=lambda d: int(FOLDER_RE.match(d).group(1))
    )

    if not sample_dirs:
        print("No sample folders found (pattern: {i}_YYYYMMDD_HHMMSS).")
        return

    print(f"Found {len(sample_dirs)} sample folder(s).\n")

    for sample_dir in sample_dirs:
        sample_path = os.path.join(base_dir, sample_dir)
        sample_num  = FOLDER_RE.match(sample_dir).group(1)
        pairs       = find_pairs_in(sample_path)

        if not pairs:
            print(f"  [{sample_dir}] No pairs found, skipping.")
            continue

        # Newest first
        pairs_desc = list(reversed(pairs))

        print(f"  [{sample_dir}]  {len(pairs)} pair(s) — newest → oldest:")
        for i, (ts, _, _) in enumerate(pairs_desc):
            slot = resolved_task_folders[i] if i < len(resolved_task_folders) else "(no slot)"
            slot_name = os.path.basename(slot) if i < len(resolved_task_folders) else slot
            print(f"    {ts}  →  {slot_name}/")

        for i, (ts, mp4, csv_) in enumerate(pairs_desc):
            if i >= len(resolved_task_folders):
                print(f"    Warning: no task folder for position {i+1}, skipping {ts}.")
                continue
            dest_folder = resolved_task_folders[i]
            for src in (mp4, csv_):
                new_name = f"{sample_num}_{os.path.basename(src)}"
                dst = os.path.join(dest_folder, new_name)
                shutil.copy2(src, dst)
        print()

    print("Distribution complete.")
    for i, tf in enumerate(resolved_task_folders, start=1):
        print(f"  {os.path.basename(tf)}/  ({len(os.listdir(tf))} files)")


# Entry point

def main():
    parser = argparse.ArgumentParser(description="Organize Dynamic REBA recordings.")
    parser.add_argument('--base', default=None,
                        help="Base folder (default: folder where this script lives)")
    parser.add_argument('--tasks', type=int, default=5,
                        help="Expected pairs per sample for MODE 1 (default: 5)")
    parser.add_argument('--distribute', action='store_true',
                        help="Run MODE 2: distribute sample files into task folders")
    parser.add_argument('--task-folders', nargs='+', default=None,
                        metavar='FOLDER',
                        help='Task folder names/paths for --distribute, ordered by recency '
                             '(e.g. "1_task" "2_task" "3_task")')
    args = parser.parse_args()

    base_dir = get_base_dir(args.base)
    print(f"Base folder: {base_dir}\n")

    if args.distribute:
        if not args.task_folders:
            parser.error("--distribute requires --task-folders FOLDER1 FOLDER2 ...")
        distribute(base_dir, args.task_folders)
    else:
        organize(base_dir, expected=args.tasks)


if __name__ == '__main__':
    main()
