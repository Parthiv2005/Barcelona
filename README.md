# Barcelona

Football broadcast-to-tracking pipeline for extracting player and ball coordinates from match clips.

## What this repo does
- Detects players and ball from broadcast video.
- Tracks objects across frames.
- Projects coordinates onto a standard 105x68 pitch.
- Produces annotated video and structured outputs for analysis.

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

## Useful paths
- Code: `eagle/`
- Examples: `examples/`
- Documentation: `docs/`
- Outputs: `output/`

## Notes
- GPU is strongly recommended for faster inference.
- Input clips with stable camera angles generally produce better results.

Reference: https://github.com/nreHieW/Eagle
