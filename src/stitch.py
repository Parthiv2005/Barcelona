
import pandas as pd
import numpy as np
import json
import argparse
import sys
from pathlib import Path

def stitch_clips(clip_indices, output_dir, eagle_root="output", metadata_file="clips/clips_metadata.json"):
    clips = []
    
    # Load clip metadata
    with open(metadata_file, "r") as f:
        all_metadata = json.load(f)
    
    # Collect data for requested clips
    for idx in clip_indices:
        clip_name = f"clip_{int(idx):03d}.mp4"
        if clip_name not in all_metadata:
            print(f"Warning: Metadata for {clip_name} not found.")
            continue
            
        clip_meta = all_metadata[clip_name]
        clip_output_dir = Path(eagle_root) / f"clip_{int(idx):03d}"
        
        raw_data_path = clip_output_dir / "raw_data.json"
        meta_path = clip_output_dir / "metadata.json"
        
        if not raw_data_path.exists():
            print(f"Warning: Data for {clip_name} not processed yet.")
            continue
            
        df = pd.read_json(raw_data_path).fillna(value=np.nan)
        with open(meta_path) as f:
            eagle_meta = json.load(f)
            
        clips.append({
            "name": clip_name,
            "start": clip_meta["start"],
            "end": clip_meta["end"],
            "duration": clip_meta["duration"],
            "df": df,
            "team_mapping": eagle_meta.get("team_mapping", {}),
            "fps": eagle_meta.get("fps", 25)
        })
    
    if not clips:
        print("No valid clips found to stitch.")
        return

    # Sort by start time
    clips.sort(key=lambda x: x["start"])
    
    print(f"Stitching {len(clips)} clips...")
    
    merged_data = []
    global_team_mapping = {}
    last_max_id = 0
    current_time = clips[0]["start"]
    
    # Reference for team alignment (0 = team A, 1 = team B)
    # We use spatial average of team 0 from the first clip as anchor?
    # Or just assume consistency for now as a baseline.
    # Advanced: Check if centroid of Team 0 in Clip N matches Team 0 or 1 in Clip N+1.
    
    ref_team_0_centroid = None
    
    output_fps = clips[0]["fps"]

    for i, clip in enumerate(clips):
        df = clip["df"]
        fps = clip["fps"]
        team_map = clip["team_mapping"]
        
        # Determine if we need to flip teams
        # Calculate centroid of Team 0 in first 10 frames of current clip
        # Compare with last known centroid of Team 0 from previous clip
        
        current_team_0_centroid = None
        # Extract Team 0 positions
        team_0_positions = []
        for _, row in df.head(10).iterrows():
            for col in df.columns:
                if "Ball" in col or "video" in col: continue
                try:
                    pid = str(int(col.split("_")[1]))
                    if team_map.get(pid) == 0:
                        pos = row[col]
                        if isinstance(pos, (list, tuple, np.ndarray)) and len(pos) == 2:
                            team_0_positions.append(pos)
                except: pass
        
        if team_0_positions:
            current_team_0_centroid = np.mean(team_0_positions, axis=0)
        
        # Decide scaling/flipping
        flip_teams = False
        if i > 0 and ref_team_0_centroid is not None and current_team_0_centroid is not None:
            # Distance from Ref 0 to Curr 0
            dist_00 = np.linalg.norm(ref_team_0_centroid - current_team_0_centroid)
            # Hypothsize flip: Distance from Ref 0 to Curr 1 (assume symmetric pitch or large shift?)
            # Actually, better heuristic: Team 0 is usually Left, Team 1 Right?
            # Or just minimize distance logic.
            # If dist_00 is very large (> 50m?), maybe swapped?
            # Let's assume stability for now unless drift is huge.
            pass # TODO: Implement strict checking if needed.
            
        # Update reference centroid for next clip (use last 10 frames)
        # ... logic omitted for brevity in V1, assuming consistent labeling for short sequence
        
        # Rename columns to avoid ID collision
        # New ID = Old ID + last_max_id + 1
        # Also limit column renaming to Player_X -> Player_Y
        
        rename_map = {}
        for col in df.columns:
            if "Player" in col or "Goalkeeper" in col or "Referee" in col:
                parts = col.split("_")
                try:
                    pid = int(parts[1])
                    new_id = pid + last_max_id + 100 # Add buffer
                    
                    # Store mapping (Old ID -> New ID) to update team map
                    old_id_str = str(pid)
                    if old_id_str in team_map:
                        team = team_map[old_id_str]
                        if flip_teams: team = 1 - team
                        global_team_mapping[str(new_id)] = team
                    
                    # Reconstruct column name
                    new_col = f"{parts[0]}_{new_id}"
                    if len(parts) > 2: 
                        suffix = "_".join(parts[2:])
                        new_col = f"{new_col}_{suffix}"
                    
                    rename_map[col] = new_col
                except:
                    pass
        
        df_renamed = df.rename(columns=rename_map)
        last_max_id += 1000 # Buffer for next clip
        
        # Timeline alignment
        # Insert gap frames if needed
        # Gap = (Clip Start) - (Last Time)
        # Frame Count = Gap * fps
        
        # Simpler approach: Just append rows?
        # But we need to maintain 'fps' pacing for video.
        # If there is a gap, we should insert empty rows.
        
        start_time = clip["start"]
        if i > 0:
            prev_end = clips[i-1]["end"]
            gap = start_time - prev_end
            if gap > 0.1: # 100ms gap
                empty_frames = int(gap * output_fps)
                print(f"Gap of {gap:.2f}s between clips. Inserting {empty_frames} empty frames.")
                # Create empty df with same columns? No, just empty rows.
                # Actually visualize.py iterates rows. Columns can be NaN.
                if empty_frames > 0:
                    empty_df = pd.DataFrame(index=range(empty_frames), columns=df_renamed.columns)
                    merged_data.append(empty_df)

        merged_data.append(df_renamed)
        
        # Update ref centroid for next iteration (using updated IDs? No comparison uses positions)
        
    final_df = pd.concat(merged_data, ignore_index=True, sort=False)
    
    # Save output
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    final_df.to_json(out_path / "raw_data.json", orient="records")
    
    with open(out_path / "metadata.json", "w") as f:
        json.dump({
            "fps": output_fps,
            "team_mapping": global_team_mapping,
            "stitched": True,
            "clips": [c["name"] for c in clips]
        }, f, indent=4)
        
    print(f"Stitched data saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", nargs="+", required=True, help="List of clip indices to stitch (e.g. 146 147 148)")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()
    
    stitch_clips(args.clips, args.output)
