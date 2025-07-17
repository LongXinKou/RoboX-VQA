import numpy as np
import decord
from decord import VideoReader
from PIL import Image
import imageio
from typing import List, Tuple, Dict, Any

def read_video_decord(video_path: str) -> List[np.ndarray]:
    vr = VideoReader(video_path)
    frames = [frame.asnumpy() for frame in vr]
    return frames  # RGB格式

def save_video(frames: List[np.ndarray], output_path: str, fps: int = 30) -> None:
    try:
        writer = imageio.get_writer(output_path, fps=fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"Video saved: {output_path}")
    except Exception as e:
        print(f"Error saving video: {e}")

def save_image(image: np.ndarray, output_path: str) -> None:
    img = Image.fromarray(image)
    img.save(output_path)

def generate_meta_information(
    id: str,
    view: str,
    instructions: Any,
    meta_information: Dict[str, Any]
) -> Dict[str, Any]:
    if isinstance(instructions, np.ndarray):
        instructions_list = instructions.reshape(-1).tolist()
    elif isinstance(instructions, list):
        instructions_list = instructions
    else:
        instructions_list = [instructions]
    total_frames = len(instructions_list)
    step_instructions, frame_segment, temporal_segment = get_unique_instruction(instructions_list)
    horizon = len(step_instructions)
    if horizon > 1:
        meta_information["long_episode_index"].append(id)
        meta_information["long_episodes"] += 1
    else:
        meta_information["short_episode_index"].append(id)
        meta_information["short_episodes"] += 1
    return {
        "id": id,
        "view": view,
        "total_frames": total_frames,
        "horizon": horizon,
        "step_instructions": step_instructions,
        "temporal_segment": temporal_segment,
        "frame_segment": frame_segment,
    }

def get_unique_instruction(
    instructions_list: List[str]
) -> Tuple[List[str], List[List[int]], List[List[float]]]:
    total_length = len(instructions_list)
    unique_instruction = []
    step_counter_list = []
    for i, instruction in enumerate(instructions_list):
        if instruction not in unique_instruction:
            unique_instruction.append(instruction)
            step_counter_list.append(i + 1)

    # Action Localization
    step_counter_list.append(total_length)
    frame_segment = [
        [step_counter_list[i], step_counter_list[i + 1] - 1]
        for i in range(len(step_counter_list) - 1)
    ]
    frame_segment[-1][1] = step_counter_list[-1]

    frame_ratio_list = [step_counter / total_length for step_counter in step_counter_list]
    frame_ratio_segment = [
        [frame_ratio_list[i], frame_ratio_list[i + 1]]
        for i in range(len(frame_ratio_list) - 1)
    ]
    frame_ratio_segment[-1][1] = frame_ratio_list[-1]
    frame_ratio_segment[0][0] = 0.0

    return unique_instruction, frame_segment, frame_ratio_segment