"""
Module: prepare_data.py

One-time preprocessing for UCF50. Run it once before training:

    python src/prepare_data.py --video_dir data/UCF50 --frame_dir data/UCF50_frames

Steps:
    1. Find every ``.avi`` in ``<video_dir>/<Class>/``.
    2. Remove byte-identical duplicate videos (MD5), keeping one copy of each.
    3. Extract ``--n_frames`` uniformly spaced frames per video to
       ``<frame_dir>/<Class>/<video_stem>/frame_XXXX.jpg``. Videos that are already
       extracted are skipped, so the script can be resumed.
    4. Split the videos into train / val / test **by UCF50 group** (see
       ``video_datasets.group_split``) and write ``<frame_dir>/splits.json``.

Changing ``--seed``, ``--n_val_groups`` or ``--n_test_groups`` only rewrites ``splits.json``.
Frames that already exist are not extracted again.
"""

import argparse
import glob
import os
from collections import Counter
from multiprocessing import Pool

from tqdm import tqdm

from utils import get_frames, store_frames
from video_datasets import (SPLITS_FILENAME, find_duplicate_videos, group_split,
                            parse_ucf50_name, save_splits)


def args_parser(argv=None):
    """
    Parse the preprocessing command-line arguments.

    Args:
        argv (list[str], optional): Argument list (defaults to ``sys.argv``).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Extract UCF50 frames and build group splits')
    parser.add_argument('-vd', '--video_dir', default='data/UCF50',
                        help='Directory with one sub-directory of .avi files per class')
    parser.add_argument('-fd', '--frame_dir', default='data/UCF50_frames',
                        help='Output directory for the extracted frames and splits.json')
    parser.add_argument('-nf', '--n_frames', type=int, default=32,
                        help='Frames stored per video (training samples a subset of these)')
    parser.add_argument('--n_val_groups', type=int, default=3,
                        help='Number of UCF50 groups (of 25) held out for validation')
    parser.add_argument('--n_test_groups', type=int, default=4,
                        help='Number of UCF50 groups (of 25) held out for testing')
    parser.add_argument('--seed', type=int, default=0, help='Seed for the group permutation')
    parser.add_argument('-w', '--workers', type=int, default=6,
                        help='Parallel extraction processes (keep low on memory-limited WSL)')
    return parser.parse_args(argv)


def _extract_one(job):
    """
    Extract frames for one video unless they are already on disk (Pool worker).

    Args:
        job (tuple): (video_path, out_dir, n_frames).

    Returns:
        tuple: (video_path, ok) where ``ok`` is False if the video could not be decoded.
    """
    video_path, out_dir, n_frames = job
    if len(glob.glob(os.path.join(out_dir, '*.jpg'))) == n_frames:
        return video_path, True
    frames, _ = get_frames(video_path, n_frames)
    if not frames:
        return video_path, False
    store_frames(frames, out_dir)
    return video_path, True


def dedupe_videos(videos, video_dir):
    """
    Drop byte-identical duplicate videos, printing each removed pair.

    Args:
        videos (list[str]): Video paths.
        video_dir (str): Dataset root, used to print short relative paths.

    Returns:
        tuple: (unique_videos, duplicates) where ``duplicates`` maps each dropped path to the
        copy that was kept.
    """
    print('Hashing videos to find exact duplicates...')
    duplicates = find_duplicate_videos(videos)
    for dup, kept in sorted(duplicates.items()):
        print(f'  duplicate: {os.path.relpath(dup, video_dir)} == '
              f'{os.path.relpath(kept, video_dir)} (dropped)')
    return [v for v in videos if v not in duplicates], duplicates


def extract_all(videos, frame_dir, n_frames, workers):
    """
    Extract frames for every video in parallel.

    Args:
        videos (list[str]): Video paths laid out as ``<root>/<Class>/<name>.avi``.
        frame_dir (str): Output root; frames go to ``<frame_dir>/<Class>/<name>/``.
        n_frames (int): Frames stored per video.
        workers (int): Number of worker processes.

    Returns:
        tuple: (extracted, failed)
            - extracted (list[tuple[str, str]]): ``(video_path, frame_subdir)`` for each
              successfully extracted video, with ``frame_subdir`` relative to ``frame_dir``.
            - failed (list[str]): Videos that could not be decoded.
    """
    rel_dirs = {v: os.path.join(os.path.basename(os.path.dirname(v)),
                                os.path.splitext(os.path.basename(v))[0]) for v in videos}
    jobs = [(v, os.path.join(frame_dir, rel_dirs[v]), n_frames) for v in videos]
    failed = []
    with Pool(workers) as pool:
        for video, ok in tqdm(pool.imap_unordered(_extract_one, jobs, chunksize=4),
                              total=len(jobs), desc='Extracting frames'):
            if not ok:
                failed.append(video)
    if failed:
        print(f'WARNING: {len(failed)} videos could not be decoded and are skipped: {failed}')
    return [(v, rel_dirs[v]) for v in videos if v not in failed], failed


def build_splits(extracted, classes, n_val_groups, n_test_groups, seed):
    """
    Assign every extracted video to train / val / test by UCF50 group.

    Args:
        extracted (list[tuple[str, str]]): ``(video_path, frame_subdir)`` pairs.
        classes (list[str]): Sorted class names (the label is the index).
        n_val_groups (int): Groups held out for validation.
        n_test_groups (int): Groups held out for testing.
        seed (int): Group-permutation seed.

    Returns:
        tuple: (splits, val_groups, test_groups) where ``splits`` maps each split name to a
        list of ``(frame_subdir, label, group)`` tuples.
    """
    label_dict = {cls: idx for idx, cls in enumerate(classes)}
    groups = [parse_ucf50_name(video)[0] for video, _ in extracted]
    split_names, val_groups, test_groups = group_split(groups, n_val_groups, n_test_groups,
                                                       seed=seed)
    splits = {'train': [], 'val': [], 'test': []}
    for (video, rel_dir), group, split in zip(extracted, groups, split_names):
        label = label_dict[os.path.basename(os.path.dirname(video))]
        splits[split].append((rel_dir, label, group))
    return splits, val_groups, test_groups


def main(args):
    """
    Run the full preprocessing pipeline (dedupe -> extract -> split -> splits.json).

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    classes = sorted(d for d in os.listdir(args.video_dir)
                     if os.path.isdir(os.path.join(args.video_dir, d)))
    videos = sorted(glob.glob(os.path.join(args.video_dir, '*', '*.avi')))
    print(f'Found {len(videos)} videos in {len(classes)} classes.')

    videos, duplicates = dedupe_videos(videos, args.video_dir)
    extracted, failed = extract_all(videos, args.frame_dir, args.n_frames, args.workers)
    splits, val_groups, test_groups = build_splits(extracted, classes, args.n_val_groups,
                                                   args.n_test_groups, args.seed)

    def rel(path):
        return os.path.relpath(path, args.video_dir)
    meta = {'seed': args.seed, 'n_frames': args.n_frames, 'val_groups': val_groups,
            'test_groups': test_groups,
            'removed_duplicates': {rel(d): rel(k) for d, k in duplicates.items()},
            'undecodable': [rel(v) for v in failed]}
    out_path = os.path.join(args.frame_dir, SPLITS_FILENAME)
    save_splits(out_path, classes, splits, meta)

    per_class_test = Counter(label for _, label, _ in splits['test'])
    print(f'Val groups {val_groups}, test groups {test_groups}')
    print(f'Split sizes: { {name: len(s) for name, s in splits.items()} }; test clips per class: '
          f'min {min(per_class_test.values())}, max {max(per_class_test.values())}')
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main(args_parser())
