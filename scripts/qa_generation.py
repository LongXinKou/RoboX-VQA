import os
import json
import argparse
import shutil

from tqdm import tqdm

try:
    from .utils import dataset_mapping
except ImportError:
    from scripts.utils import dataset_mapping

from qa_generator import QAGenerator


def copy_videos_and_save_json(source_video_dir, source_json_dir, dest_dir, task, stage, QA_Generator):
    '''
    dest_dir/task
    dest_dir/task_instance_numberK.json
    '''
    os.makedirs(dest_dir, exist_ok=True)
    dest_video_dir = os.path.join(dest_dir, task)
    temp_json_path = os.path.join(dest_dir, f'{task}_temp.json')

    with open(source_json_dir, 'r') as file:
        source_annotation = json.load(file)

    if not os.path.exists(dest_video_dir):
        os.makedirs(dest_video_dir)
        print(f"copy video------{task} dataset")
        for i in tqdm(range(len(source_annotation))):
            video_name = source_annotation[i]['id']
            video_path = os.path.join(source_video_dir, video_name)
            new_video_path = os.path.join(dest_video_dir, video_name)
            shutil.copy(video_path, new_video_path)
    
    # Check for existing temporary results
    if os.path.exists(temp_json_path):
        with open(temp_json_path, 'r') as temp_file:
            annotation = json.load(temp_file)
        start_index = len(annotation)
        print(f"Resuming from index {start_index}")
    else:
        annotation = []
        start_index = 0

    print(f'generate qa pairs----{task} dataset')
    for i in tqdm(range(start_index, len(source_annotation))):
        video_name = source_annotation[i]['id']
        instance = QA_Generator.generate_qa_instance(annotation=source_annotation[i], video_name=video_name, stage=stage, task=task)
        if isinstance(instance, list):
            annotation.extend(instance)
        else:
            annotation.append(instance)
        
        if (i + 1) % 500 == 0:
            with open(temp_json_path, 'w') as temp_file:
                json.dump(annotation, temp_file)
   
    # Remove temporary file
    if os.path.exists(temp_json_path):
        os.remove(temp_json_path)
    
    # Save to JSON
    instance_number = len(annotation) // 1000 # K
    dest_json_dir = os.path.join(dest_dir, f'{task}_{instance_number}K.json')

    with open(dest_json_dir, 'w') as json_file:
        json.dump(annotation, json_file, indent=4)

def args_parse():
    parser = argparse.ArgumentParser(description="qa generation")
    parser.add_argument('--base_dir', type=str, default='robot_dataset')
    parser.add_argument('--dest_dir', type=str, default='')
    parser.add_argument('--dataset_name', type=str, default='')
    parser.add_argument('--stage', type=str, default='Pretrain', help="Pretrain, Finetune")
    args = parser.parse_args()
    return args

def main():
    args = args_parse()
    Dataset_Path_Mapping, Dataset_Task_Mapping = dataset_mapping(args.base_dir)
    task_list = Dataset_Task_Mapping[args.dataset_name]
    QA_Generator = QAGenerator(task_list=task_list)
    base_dir = Dataset_Path_Mapping[args.dataset_name]
    dest_dir = os.path.join(args.dest_dir, args.stage)
    if args.dataset_name.endswith('_task'):
        source_video_dir = os.path.join(base_dir, 'task_planning')
        source_json_dir = os.path.join(source_video_dir, 'annotation.json')
    else:
        source_video_dir = os.path.join(base_dir, 'video')
        source_json_dir = os.path.join(source_video_dir, 'annotation.json')

    print(f"{args.stage} Dataset Processing .......")
    print(f"Dataset {args.dataset_name} Processing .......")
    copy_videos_and_save_json(source_video_dir=source_video_dir, source_json_dir=source_json_dir, dest_dir=dest_dir, task=args.dataset_name, stage=args.stage, QA_Generator=QA_Generator)


if __name__ == '__main__':
    main()