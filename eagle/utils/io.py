import cv2
import os


def read_video(path: str, fps: int = 24) -> list:
    """
    Read a video file and return a list of frames.
    :param path: Path to the video file.
    :param fps: Frames per second to sample.

    :return: List of frames and the fps of the video.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames = []
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps is None or native_fps <= 0:
        native_fps = float(fps)
    if fps <= 0:
        raise ValueError("fps must be > 0")
    skip = max(1, int(round(native_fps / fps)))
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % skip == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    return frames, fps


def write_video(frames: list, path: str, fps: int = 24, is_rgb: bool = False) -> str:
    """
    Write a list of frames to a video file.
    :param frames: List of images.
    :param path: Path to save the video file.
    :param fps: Frames per second.

    :return: Path to the saved video file.
    """
    if not frames:
        raise ValueError("frames must contain at least one frame")
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for frame in frames:
        if is_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # cv2 expects BGR
        out.write(frame)
    out.release()
    return path
