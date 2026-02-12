import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # only for macs
from eagle.models import CoordinateModel
from eagle.processor import Processor
from eagle.utils.io import read_video, write_video
import json
from argparse import ArgumentParser
import pandas as pd
import cv2
import numpy as np
from collections import defaultdict


def main():
    parser = ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--disable_summary_stats", action="store_true", help="Skip writing summary_stats.json")
    parser.add_argument(
        "--max_step_meters",
        type=float,
        default=3.0,
        help="Ignore per-frame player movement jumps larger than this (meters) when computing stats.",
    )
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)
    video_name = args.video_path.split("/")[-1].split(".")[0]
    os.makedirs(f"output/{video_name}", exist_ok=True)
    root = f"output/{video_name}"

    frames, fps = read_video(args.video_path, args.fps)
    model = CoordinateModel()
    coordinates = model.get_coordinates(frames, fps, num_homography=1, num_keypoint_detection=3)

    with open(f"{root}/raw_coordinates.json", "w") as f:
        json.dump(coordinates, f, default=float)

    print("Processing Data")

    processor = Processor(coordinates, frames, fps, filter_ball_detections=False)
    df, team_mapping = processor.process_data(smooth=False)
    df.to_json(f"{root}/raw_data.json", orient="records")
    with open(f"{root}/metadata.json", "w") as f:
        json.dump({"fps": fps, "team_mapping": team_mapping}, f, default=str)

    processed_df = processor.format_data(df)
    processed_df.to_json(f"{root}/processed_data.json", orient="records")
    if not args.disable_summary_stats:
        summary_stats = build_summary_stats(df, fps, team_mapping, args.max_step_meters)
        with open(f"{root}/summary_stats.json", "w") as f:
            json.dump(summary_stats, f, default=float, indent=2)

    out = []
    cols = [x for x in df.columns if "video" in x and x not in ["Bottom_Left", "Top_Left", "Top_Right", "Bottom_Right"]]
    for i, row in df.iterrows():
        curr_frame = frames[int(i)].copy()
        for col in cols:
            if pd.isna(row[col]):
                continue
            x, y = row[col]

            if "Ball" in col:
                color = (0, 255, 0)
                bottom_point = (int(x), int(y) - 20)
                top_left = (int(x) - 5, int(y) - 30)
                top_right = (int(x) + 5, int(y) - 30)
                pts = np.array([bottom_point, top_left, top_right]).reshape(-1, 1, 2)
                cv2.drawContours(curr_frame, [pts], 0, color, -1)
            else:
                id = int(col.split("_")[1])
                if "Goalkeeper" in col:
                    color = (0, 255, 0)
                else:
                    if id not in team_mapping:
                        continue
                    team = team_mapping[id]
                    if team == 0:
                        color = (0, 0, 255)
                    else:
                        color = (255, 0, 0)

                cv2.ellipse(curr_frame, (int(x), int(y)), (35, 18), 0, -45, 235, color, 1)
                cv2.putText(curr_frame, str(id), (int(x) - 3, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        kp = coordinates[i]["Keypoints"]
        for x in kp.values():
            cv2.circle(curr_frame, (int(x[0]), int(x[1])), 6, (0, 0, 0), -1)

        out.append(curr_frame)

    write_video(out, f"{root}/annotated.mp4", fps)
    print("Data saved to", root)


def _is_valid_point(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return pd.notna(value[0]) and pd.notna(value[1])
    return False


def _calculate_track_stats(points, fps, max_step_meters):
    total_distance = 0.0
    valid_samples = 0
    segments_used = 0

    prev = None
    for point in points:
        if not _is_valid_point(point):
            prev = None
            continue
        curr = (float(point[0]), float(point[1]))
        valid_samples += 1
        if prev is not None:
            step_distance = float(np.hypot(curr[0] - prev[0], curr[1] - prev[1]))
            if step_distance <= max_step_meters:
                total_distance += step_distance
                segments_used += 1
        prev = curr

    duration_sec = segments_used / fps if fps > 0 else 0.0
    avg_speed_mps = total_distance / duration_sec if duration_sec > 0 else 0.0

    return {
        "distance_m": round(total_distance, 3),
        "duration_s": round(duration_sec, 3),
        "avg_speed_mps": round(avg_speed_mps, 3),
        "samples": valid_samples,
    }


def build_summary_stats(df, fps, team_mapping, max_step_meters):
    summary = {
        "fps": fps,
        "frames_with_people": int(len(df)),
        "max_step_meters": float(max_step_meters),
        "players": {},
        "totals_by_team": {},
        "ball": {},
    }

    totals_by_team = defaultdict(float)
    player_cols = [col for col in df.columns if col.startswith(("Player_", "Goalkeeper_")) and not col.endswith("_video")]

    for col in sorted(player_cols):
        role, id_text = col.split("_", 1)
        player_id = int(id_text)
        stats = _calculate_track_stats(df[col].tolist(), fps, max_step_meters)
        team = team_mapping.get(player_id, "unknown")
        stats["role"] = role
        stats["team"] = team
        summary["players"][str(player_id)] = stats
        if team != "unknown":
            totals_by_team[str(team)] += stats["distance_m"]

    summary["totals_by_team"] = {team: round(distance, 3) for team, distance in sorted(totals_by_team.items())}
    summary["ball"] = _calculate_track_stats(df["Ball"].tolist(), fps, max_step_meters=1e9)

    return summary


if __name__ == "__main__":
    main()
