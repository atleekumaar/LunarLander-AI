"""Video and GIF recording module for trained LunarLander-v3 agents.

Captures agent rollout frames in rgb_array mode and compiles them into MP4 or
animated GIF demonstrations for visual analysis and presentations.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from src.environment import make_lunar_lander_env


def check_video_dependencies() -> dict:
    """Check availability of optional video encoding libraries.

    Returns:
        dict: Status flags for imageio, imageio_ffmpeg, and moviepy.
    """
    status = {"imageio": False, "imageio_ffmpeg": False, "moviepy": False}
    try:
        import imageio
        status["imageio"] = True
    except ImportError:
        pass

    try:
        import imageio_ffmpeg
        status["imageio_ffmpeg"] = True
    except ImportError:
        pass

    try:
        import moviepy
        status["moviepy"] = True
    except ImportError:
        pass

    return status


def record_agent_rollouts(
    model_path: str = "models/ppo_lunarlander_baseline.zip",
    output_dir: str = "results",
    env_id: str = "LunarLander-v3",
    num_episodes: int = 2,
    max_steps_per_episode: int = 1000,
    fps: int = 50,
    seed: int = 42,
    save_gif: bool = False,
) -> List[str]:
    """Record video or GIF files of agent behavior during test episodes.

    Args:
        model_path: Path to the saved PPO model (.zip).
        output_dir: Destination folder for output video/gif files.
        env_id: Gymnasium environment ID.
        num_episodes: Number of episodes to record.
        max_steps_per_episode: Maximum steps per episode.
        fps: Target frames per second for video/gif playback.
        seed: Random seed for episode execution.
        save_gif: Whether to save an animated GIF in addition to or instead of MP4.

    Returns:
        List[str]: List of paths to the successfully generated media files.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    deps = check_video_dependencies()
    if not deps["imageio"]:
        raise RuntimeError(
            "Video recording requires the 'imageio' package. "
            "Please install it using: pip install imageio imageio-ffmpeg"
        )

    import imageio

    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("           LUNAR LANDER v3 - VIDEO RECORDING")
    print("=" * 70)
    print(f"Loading Model:      {model_path}")
    print(f"Output Directory:   {output_dir}")
    print(f"Recording Episodes: {num_episodes}")
    print(f"Playback FPS:       {fps}")
    print(f"Video Dependencies: imageio={deps['imageio']}, ffmpeg={deps['imageio_ffmpeg']}")

    model = PPO.load(model_path)
    env = make_lunar_lander_env(env_id=env_id, render_mode="rgb_array", seed=seed)

    saved_files: List[str] = []

    try:
        for ep in range(1, num_episodes + 1):
            ep_seed = seed + ep * 500
            obs, info = env.reset(seed=ep_seed)
            frames: List[np.ndarray] = []
            ep_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            # Capture initial frame
            initial_frame = env.render()
            if initial_frame is not None:
                frames.append(initial_frame)

            while not (terminated or truncated) and steps < max_steps_per_episode:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                steps += 1

                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            print(f"Episode {ep}: {steps} steps, return={ep_reward:.2f}, captured {len(frames)} frames.")

            if not frames:
                print(f"Warning: No frames captured for episode {ep}. Skipping file creation.")
                continue

            base_name = f"lunarlander_ep{ep}_return_{int(ep_reward)}"

            # Save MP4 video if ffmpeg is available
            if deps["imageio_ffmpeg"]:
                mp4_path = os.path.join(output_dir, f"{base_name}.mp4")
                try:
                    with imageio.get_writer(mp4_path, fps=fps, codec="libx264") as writer:
                        for frame in frames:
                            writer.append_data(frame)
                    print(f"  [MP4 Saved] -> {mp4_path}")
                    saved_files.append(mp4_path)
                except Exception as e:
                    print(f"  Could not save MP4 via imageio ({e}). Attempting GIF export...")
                    save_gif = True
            else:
                print("  imageio_ffmpeg not detected. Defaulting to GIF export.")
                save_gif = True

            # Save GIF if explicitly requested or if ffmpeg was unavailable
            if save_gif:
                gif_path = os.path.join(output_dir, f"{base_name}.gif")
                # Downsample frames slightly if episode is long to prevent giant gif files
                stride = max(1, len(frames) // 300)
                subsampled_frames = frames[::stride]
                gif_fps = max(10, fps // stride)
                imageio.mimsave(gif_path, subsampled_frames, fps=gif_fps, loop=0)
                print(f"  [GIF Saved] -> {gif_path} (subsampled stride={stride})")
                saved_files.append(gif_path)

    finally:
        env.close()

    print("\nRecording finished successfully.")
    print("=" * 70)
    return saved_files


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for video recording."""
    parser = argparse.ArgumentParser(
        description="Record MP4/GIF rollouts of a trained PPO agent on LunarLander-v3."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/ppo_lunarlander_baseline.zip",
        help="Path to the trained PPO model (.zip).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save generated video files (default: results).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="Number of episodes to record (default: 2).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Frames per second for video playback (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable evaluation rollouts (default: 42).",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Generate animated GIF format alongside or in place of MP4.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    record_agent_rollouts(
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_episodes=args.episodes,
        fps=args.fps,
        seed=args.seed,
        save_gif=args.gif,
    )
