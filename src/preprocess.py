
import subprocess
import os
import re
import argparse
import json
from pathlib import Path

def detect_scenes(video_path, threshold=0.3):
    """
    Detect scenes using ffmpeg scene detection filter.
    Returns list of timestamps (seconds) where scene changes occur.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null",
        "-"
    ]
    
    print(f"Running scene detection on {video_path}...")
    # Run ffmpeg and capture stderr (where info is printed)
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    
    timestamps = [0.0]
    # Parse output for timestamps
    # [Parsed_showinfo_1 @ ...] n:   0 pts:  12345 pts_time:12.345 ...
    for line in result.stderr.splitlines():
        if "pts_time:" in line:
            match = re.search(r"pts_time:([0-9.]+)", line)
            if match:
                ts = float(match.group(1))
                # Avoid scene changes too close to start or each other
                if ts - timestamps[-1] > 2.0:  # Minimum 2 seconds per clip
                    timestamps.append(ts)
    
    return timestamps

def split_video(video_path, scenes, output_dir, dry_run=False):
    """
    Split video into clips based on scene timestamps.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get total duration
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ], capture_output=True, text=True)
    total_duration = float(probe.stdout.strip())
    
    scenes.append(total_duration)
    
    print(f"DEBUG: Found {len(scenes)-1} scenes to split")

    metadata = {}
    
    for i in range(len(scenes) - 1):
        start = scenes[i]
        end = scenes[i+1]
        duration = end - start
        
        if duration < 3.0:  # Skip very short clips
            continue
            
        clip_filename = f"clip_{i:03d}.mp4"
        clip_path = output_dir / clip_filename
        
        metadata[clip_filename] = {
            "start": start,
            "end": end,
            "duration": duration
        }

        if not dry_run:
            print(f"Creating {clip_path} ({start:.2f}-{end:.2f}, {duration:.2f}s)...")
            # Use simple copy if keyframes align, but re-encoding is safer for precise cuts
            # For speed we try copy fast, but precise seeking is better
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(start),
                "-to", str(end),
                "-i", str(video_path),
                "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "22",
                "-c:a", "copy",
                str(clip_path)
            ], stderr=subprocess.DEVNULL)
            
    # Save metadata
    with open(output_dir / "clips_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to {output_dir / 'clips_metadata.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", default="clips", help="Output directory for clips")
    parser.add_argument("--threshold", type=float, default=0.3, help="Scene detection threshold (0.0-1.0)")
    parser.add_argument("--dry-run", action="store_true", help="Generate metadata only, do not write video files")
    args = parser.parse_args()
    
    scenes = detect_scenes(args.video, args.threshold)
    print(f"Detected {len(scenes)} scenes.")
    split_video(args.video, scenes, args.output, dry_run=args.dry_run)
