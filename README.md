# Barcelona

Football broadcast-to-tracking pipeline for extracting player and ball coordinates from match clips.

## What this repo does
- Detects players and ball from broadcast video.
- Tracks objects across frames.
- Projects coordinates onto a standard 105x68 pitch.
- Produces annotated video and structured outputs for analysis.
- Exports `summary_stats.json` with per-player distance, average speed, team totals, and ball travel distance.

## Quick start
1. Install `uv`.
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Run inference:
   ```bash
   uv run main.py --video_path input_video.mp4
   ```
4. Check output under `output/<video_name>/`.

## Extra options
- Tune movement filtering for stats:
  ```bash
  uv run main.py --video_path input_video.mp4 --max_step_meters 2.5
  ```
- Skip stats export:
  ```bash
  uv run main.py --video_path input_video.mp4 --disable_summary_stats
  ```

## Useful paths
- Code: `eagle/`
- Examples: `examples/`
- Documentation: `docs/`
- Outputs: `output/`

## Notes
- GPU is strongly recommended for faster inference.
- Input clips with stable camera angles generally produce better results.

Reference: https://github.com/nreHieW/Eagle
