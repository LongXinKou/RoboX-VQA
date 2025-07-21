"""
RLDS Dataset Reader and Video Extractor

This module provides functionality to read RLDS (Robotic Learning Dataset Specification)
format datasets and extract video sequences with annotations for robotic learning tasks.

"""

import os
import json
import re
import argparse
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm

try:
    from .utils import save_video, generate_meta_information, dataset_mapping
except ImportError:
    from scripts.utils import save_video, generate_meta_information, dataset_mapping


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RLDSDatasetExtractor:
    IMAGE_KEYS = {
        'taco_play': 'rgb_static',
        'viola': 'agentview_rgb',
        'droid': 'exterior_image_1_left',
        'bridge_data_v2': 'image_0',
        'robo_set': 'image_left',
        'default': 'image'
    }
    LANGUAGE_INSTRUCTION_DATASETS = {
        'ucsd_kitchen_dataset_converted_externally_to_rlds',
        'stanford_hydra_dataset_converted_externally_to_rlds',
        'droid',
        'bridge_data_v2',
        'libero_spatial_no_noops',
        'libero_10_no_noops',
        'libero_goal_no_noops',
        'libero_object_no_noops',
        'robo_set',
        'utokyo_xarm_bimanual_converted_externally_to_rlds',
        'utokyo_xarm_pick_and_place_converted_externally_to_rlds'
    }

    def __init__(self, base_dataset_path: str = ''):
        self.base_dataset_path = base_dataset_path
        self.dataset_path_mapping = dataset_mapping(base_dataset_path)

    def get_camera_image(self, step: Dict[str, Any], dataset_name: str) -> np.ndarray:
        observation = step["observation"]
        image_key = self.IMAGE_KEYS.get(dataset_name, self.IMAGE_KEYS['default'])
        return observation[image_key].numpy()

    def get_natural_language_instruction(self, step: Dict[str, Any], dataset_name: str) -> str:
        if dataset_name in self.LANGUAGE_INSTRUCTION_DATASETS:
            instruction = step["language_instruction"].numpy().decode('utf-8')
        else:
            instruction = step["observation"]["natural_language_instruction"].numpy().decode('utf-8')
        if dataset_name == "columbia_cairlab_pusht_real":
            instruction = instruction.split('.')[0]
        return instruction

    def is_episode_valid(self, instruction: str, dataset_name: str) -> bool:
        if dataset_name in ["columbia_cairlab_pusht_real", "utokyo_xarm_pick_and_place_converted_externally_to_rlds"]:
            return True
        return bool(re.search(r'^[a-zA-Z]+( [a-zA-Z]+)*\.?$', instruction))

    def process_dataset(self, dataset_name: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        if dataset_name not in self.dataset_path_mapping:
            raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(self.dataset_path_mapping.keys())}")
        logger.info(f"Processing dataset: {dataset_name}")
        base_dir = self.dataset_path_mapping[dataset_name]
        video_dir = output_dir or os.path.join(base_dir, 'video')
        os.makedirs(video_dir, exist_ok=True)
        annotation_path = os.path.join(video_dir, 'annotation.json')
        meta_info_path = os.path.join(video_dir, 'meta_information.json')
        stats = self._initialize_stats()
        annotations = []
        video_count = 0
        try:
            episodes = tfds.builder_from_directory(base_dir).as_dataset(split='all')
            for episode in tqdm(episodes, desc=f"Processing {dataset_name}"):
                stats['total_episodes'] += 1
                episode_data = self._process_episode(episode, dataset_name)
                if episode_data['is_valid']:
                    video_filename = f"{video_count:06d}.mp4"
                    video_path = os.path.join(video_dir, video_filename)
                    save_video(frames=episode_data['images'], output_path=video_path)
                    annotation = generate_meta_information(
                        id=video_filename,
                        view="third_person",
                        instructions=episode_data['instructions'],
                        meta_information=stats
                    )
                    annotations.append(annotation)
                    video_count += 1
                else:
                    stats['filtered_episodes'] += 1
            stats['useful_episodes'] = stats['total_episodes'] - stats['filtered_episodes']
            self._save_results(annotation_path, meta_info_path, annotations, stats)
            logger.info(f"Processing completed for {dataset_name}")
            logger.info(f"Total episodes: {stats['total_episodes']}")
            logger.info(f"Useful episodes: {stats['useful_episodes']}")
            logger.info(f"Filtered episodes: {stats['filtered_episodes']}")
            return stats
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {str(e)}")
            raise

    def _process_episode(self, episode: Any, dataset_name: str) -> Dict[str, Any]:
        images = []
        instructions = []
        is_valid = True
        reward = None

        for step in episode["steps"]:
            try:
                image = self.get_camera_image(step, dataset_name)
                instruction = self.get_natural_language_instruction(step, dataset_name)

                if step["is_terminal"]:
                    reward = step["reward"].numpy()

                if not self.is_episode_valid(instruction, dataset_name):
                    is_valid = False
                    break

                images.append(image)
                instructions.append(instruction)

            except Exception as e:
                logger.warning(f"Error processing step in episode: {str(e)}")
                is_valid = False
                break

        return {
            'images': images,
            'instructions': instructions,
            'is_valid': is_valid,
            'reward': reward
        }

    def _initialize_stats(self) -> Dict[str, Any]:
        """Initialize statistics tracking dictionary."""
        return {
            "total_episodes": 0,
            "filtered_episodes": 0,
            "useful_episodes": 0,
            "short_episodes": 0,
            "long_episodes": 0,
            "short_episode_index": [],
            "long_episode_index": [],
        }

    def _save_results(self, annotation_path: str, meta_info_path: str,
                     annotations: List[Dict], stats: Dict[str, Any]) -> None:
        """
        Save annotations and metadata to JSON files.

        Args:
            annotation_path: Path to save annotations
            meta_info_path: Path to save metadata
            annotations: List of annotation dictionaries
            stats: Statistics dictionary
        """
        try:
            with open(annotation_path, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, indent=4, ensure_ascii=False)

            with open(meta_info_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=4, ensure_ascii=False)

            logger.info(f"Results saved to {annotation_path} and {meta_info_path}")

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise


def get_available_datasets(base_path: str = '') -> List[str]:
    """
    Get list of available datasets.

    Args:
        base_path: Base directory containing RLDS datasets

    Returns:
        List of available dataset names
    """
    dataset_mapping_dict = dataset_mapping(base_path)
    return list(dataset_mapping_dict.keys())


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Extract videos and annotations from RLDS format datasets"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='bridge_data_v2',
        help='Name of the dataset to process'
    )
    parser.add_argument(
        '--base-path',
        type=str,
        default='',
        help='Base directory containing RLDS datasets'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for videos and annotations (optional)'
    )
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List all available datasets and exit'
    )

    args = parser.parse_args()

    if args.list_datasets:
        datasets = get_available_datasets(args.base_path)
        print("Available datasets:")
        for dataset in sorted(datasets):
            print(f"  - {dataset}")
        return

    extractor = RLDSDatasetExtractor(args.base_path)
    try:
        stats = extractor.process_dataset(args.dataset, args.output_dir)
        print(f"\nProcessing completed successfully!")
        print(f"Dataset: {args.dataset}")
        print(f"Total episodes: {stats['total_episodes']}")
        print(f"Useful episodes: {stats['useful_episodes']}")
        print(f"Filtered episodes: {stats['filtered_episodes']}")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        import sys
        sys.exit(1)
    import sys
    sys.exit(0)


def process_single_dataset(dataset_name: str, base_path: str = ''):
    extractor = RLDSDatasetExtractor(base_path)
    return extractor.process_dataset(dataset_name)

if __name__ == '__main__':
    main()
