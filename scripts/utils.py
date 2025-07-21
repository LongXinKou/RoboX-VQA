import numpy as np
import decord
from decord import VideoReader
from PIL import Image
import imageio
from typing import List, Tuple, Dict, Any

# QA Generation Utilities
def dataset_mapping(base_dir):
    Dataset_Path_Mapping = {
        "bc_z": f"{base_dir}/bc_z/0.1.0",
        "berkeley_autolab_ur5": f"{base_dir}/berkeley_autolab_ur5/0.1.0/",
        "bridge": f"{base_dir}/bridge/0.1.0/",
        "bridge_data_v2": f"{base_dir}/bridge_data_v2/0.0.1/",
        "droid": f"{base_dir}/droid/1.0.0",
        "fractal20220817_data": f"{base_dir}/fractal20220817_data/0.1.0",
        "jaco_play": f"{base_dir}/jaco_play/0.1.0/",
        "robo_set": f"{base_dir}/robo_set/0.0.1/",
        "ucsd_kitchen_dataset_converted_externally_to_rlds": f"{base_dir}/ucsd_kitchen_dataset_converted_externally_to_rlds/0.1.0",
        "utokyo_xarm_bimanual_converted_externally_to_rlds": f"{base_dir}/utokyo_xarm_bimanual_converted_externally_to_rlds/0.1.0/",
        "utokyo_xarm_pick_and_place_converted_externally_to_rlds": f"{base_dir}/utokyo_xarm_pick_and_place_converted_externally_to_rlds/0.1.0/",

        "viola": f"{base_dir}/viola/0.1.0/",
        "stanford_hydra_dataset_converted_externally_to_rlds": f"{base_dir}/stanford_hydra_dataset_converted_externally_to_rlds/0.1.0/",

        "libero_spatial_no_noops": f"{base_dir}/libero_spatial_no_noops/1.0.0",
        "libero_goal_no_noops": f"{base_dir}/libero_goal_no_noops/1.0.0",
        "libero_object_no_noops": f"{base_dir}/libero_object_no_noops/1.0.0",
        "libero_10_no_noops": f"{base_dir}/libero_10_no_noops/1.0.0",

        # long horizon
        "calvin": f"{base_dir}/calvin/1.0.0",
        "franka_kitchen": f"{base_dir}/franka_kitchen/1.0.0",
        "bridge_data_v2_combine": f"{base_dir}/bridge_v2_release/raw/bridge_data_v2/",
        "bridge_data_v2_combine_rss": f"{base_dir}/bridge_v2_release/raw/rss/",

        "bridge_task": f"{base_dir}/bridge/0.1.0/",
        "bridge_data_v2_task": f"{base_dir}/bridge_data_v2/0.0.1/",
        "bridge_data_v2_combine_task": f"{base_dir}/bridge_v2_release/raw/bridge_data_v2/",
        "bridge_data_v2_combine_rss_task": f"{base_dir}/bridge_v2_release/raw/rss/",

        # "taco_play": f"{base_dir}/taco_play/0.1.0/",
        # "taco_play_task": f"{base_dir}/taco_play/0.1.0/",
        "robot_vqa": f"{base_dir}/robot_vqa/0.1.0/",
    }

    Dataset_Task_Mapping = {
        "bc_z": ["Video Caption", "Action Identification", "Object Identification", "Spatial Relationship"],
        "berkeley_autolab_ur5": ["Video Caption", "Action Identification", "Object Identification",
                                 "Spatial Relationship"],
        "bridge": ["Video Caption", "Action Identification", "Object Identification", "Spatial Relationship"],
        "bridge_data_v2": ["Video Caption", "Action Identification", "Object Identification", "Spatial Relationship"],
        "droid": ["Video Caption", "Action Identification", "Object Identification", "Spatial Relationship"],
        "fractal20220817_data": ["Video Caption", "Action Identification", "Object Identification",
                                 "Spatial Relationship"],
        "jaco_play": ["Video Caption", "Action Identification", "Object Identification", "Spatial Relationship"],
        "robo_set": ["Video Caption", "Action Identification", "Object Identification", "Spatial Relationship"],
        "ucsd_kitchen_dataset_converted_externally_to_rlds": ["Video Caption", "Action Identification",
                                                              "Object Identification", "Spatial Relationship"],
        "utokyo_xarm_bimanual_converted_externally_to_rlds": ["Video Caption", "Action Identification",
                                                              "Object Identification", "Spatial Relationship"],
        "utokyo_xarm_pick_and_place_converted_externally_to_rlds": ["Video Caption", "Action Identification",
                                                                    "Object Identification", "Spatial Relationship"],

        "libero_spatial_no_noops": ["Video Caption", "Action Identification", "Object Identification",
                                    "Spatial Relationship"],
        "libero_goal_no_noops": ["Video Caption", "Action Identification", "Object Identification",
                                 "Spatial Relationship"],
        "libero_10_no_noops": ["Video Caption", "Action Identification", "Object Identification",
                               "Spatial Relationship"],

        "calvin": ["Action Identification", "Object Identification", "Spatial Relationship", "Action Ordering",
                   "Action Temporal Localization", "Action Segment Summarization",
                   "Action Segmentation and Summarization"],
        "franka_kitchen": ["Action Identification", "Object Identification", "Spatial Relationship", "Action Ordering",
                           "Action Temporal Localization", "Action Segment Summarization",
                           "Action Segmentation and Summarization"],
        "bridge_data_v2_combine": ["Action Identification", "Object Identification", "Spatial Relationship",
                                   "Action Ordering", "Action Temporal Localization", "Action Segment Summarization",
                                   "Action Segmentation and Summarization"],
        "bridge_data_v2_combine_rss": ["Action Identification", "Object Identification", "Spatial Relationship",
                                       "Action Ordering", "Action Temporal Localization",
                                       "Action Segment Summarization", "Action Segmentation and Summarization"],

        "bridge_task": ["Task Success Detection"],
        "bridge_data_v2_task": ["Task Success Detection"],
        "bridge_data_v2_combine_task": ["Task Success Detection", "Task Planning"],
        "bridge_data_v2_combine_rss_task": ["Task Success Detection", "Task Planning"],

    }
    return Dataset_Path_Mapping, Dataset_Task_Mapping


# Read&Write Video/Image Utilities
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