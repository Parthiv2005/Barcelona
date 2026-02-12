
import pandas as pd
import cv2
import numpy as np
from mplsoccer import Pitch
import matplotlib.pyplot as plt
from io import BytesIO
import sys
import argparse
import json
import os
from pathlib import Path

# Add project root to path to import eagle modules
sys.path.append(os.getcwd())
try:
    from eagle.utils.io import write_video
except ImportError:
    # Fallback if eagle not found or running from wrong dir
    print("Warning: Could not import eagle.utils.io. Using local write_video if needed.")
    def write_video(frames, path, fps):
        if not frames:
            return
        h, w, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for frame in frames:
            out.write(frame)
        out.release()

def visualize(input_dir, output_path=None):
    input_dir = Path(input_dir)
    raw_data_path = input_dir / "raw_data.json"
    metadata_path = input_dir / "metadata.json"
    
    if not raw_data_path.exists() or not metadata_path.exists():
        print(f"Error: Missing data files in {input_dir}. Data processing might be incomplete.")
        return

    print(f"Loading data from {input_dir}...")
    df = pd.read_json(raw_data_path).fillna(value=np.nan)
    
    with open(metadata_path) as f:
        metadata = json.load(f)
        
    fps = metadata.get("fps", 25)
    team_mapping = metadata.get("team_mapping", {})
    
    # Identify columns to draw
    to_draw = [x for x in df.columns if "video" not in x and x not in ["Bottom_Left", "Top_Left", "Top_Right", "Bottom_Right"]]
    
    # Setup Pitch
    pitch = Pitch(pitch_type="uefa", pitch_color="#1a1a1a", line_color="white", goal_type="box")
    
    out_frames = []
    print(f"Rendering {len(df)} frames...")
    
    for i, row in df.iterrows():
        if i % 50 == 0:
            print(f"Processing frame {i}/{len(df)}")
            
        buffer = BytesIO()
        # Create figure (Vertical pitch)
        fig, ax = plt.subplots(figsize=(8, 12))
        pitch.draw(ax)
        fig.set_facecolor("#1a1a1a") # Dark background

        # Draw Visible Area (Field of View)
        boundaries_cols = ["Bottom_Left", "Top_Left", "Top_Right", "Bottom_Right", "Bottom_Left"]
        if all(col in df.columns for col in boundaries_cols):
             # check if any boundary point is NaN
             if not row[boundaries_cols].isnull().any():
                boundaries = row[boundaries_cols].values.tolist()
                polygon = plt.Polygon(boundaries, facecolor="white", zorder=1, closed=True, alpha=0.2)
                ax.add_patch(polygon)

        # Draw Players and Ball
        for col in to_draw:
            val = row[col]
            if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
                continue
            
            # Ensure coordinates are list/tuple
            pos = row[col]
            if not isinstance(pos, (list, tuple, np.ndarray)):
               continue
            x, y = pos

            if "Ball" in col:
                ax.scatter(x, y, color="yellow", zorder=10, facecolors="yellow", edgecolors="black", s=80, marker='o')
            else:
                # Parse ID
                try:
                    parts = col.split("_")
                    if len(parts) > 1:
                        obj_id = int(parts[1])
                    else:
                        continue
                except ValueError:
                    continue

                if "Goalkeeper" in col:
                    color = "#00ff00" # Green
                else:
                    # Check team mapping
                    # keys in json are strings
                    team = team_mapping.get(str(obj_id))
                    if team == 0:
                        color = "#ff3333" # Red team
                    elif team == 1:
                        color = "#3399ff" # Blue team
                    else:
                        color = "#aaaaaa" # Unknown/Referee

                ax.scatter(x, y, color=color, zorder=5, s=120, edgecolors="black", linewidth=1)
                # Ensure text is readable
                ax.text(x, y, str(obj_id), color="white", fontsize=8, ha='center', va='center', fontweight='bold', zorder=6)

        # Save frame
        plt.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0, facecolor="#1a1a1a")
        plt.close(fig)
        
        buffer.seek(0)
        img_arr = np.frombuffer(buffer.read(), np.uint8)
        img = cv2.imdecode(img_arr, 1)
        
        # Resize if needed to ensure even dimensions for video encoding
        h, w, _ = img.shape
        if h % 2 != 0: h -= 1
        if w % 2 != 0: w -= 1
        img = img[:h, :w, :]
        
        out_frames.append(img)
        buffer.close()

    if output_path is None:
        output_path = input_dir / "birdseye.mp4"
    
    print(f"Saving video to {output_path}")
    write_video(out_frames, str(output_path), fps=fps)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to Eagle output directory (containing raw_data.json)")
    parser.add_argument("--output", type=str, help="Path for output video")
    args = parser.parse_args()
    
    visualize(args.input_dir, args.output)
