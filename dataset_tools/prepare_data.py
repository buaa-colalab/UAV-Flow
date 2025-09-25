import json
import os
import tqdm
from datasets import load_dataset

# ds_sim = load_dataset("wangxiangyu0814/UAV-Flow-Sim")
# ds_real = load_dataset("wangxiangyu0814/UAV-Flow")

def process_parquet_to_folder_dataset(parquet_path, output_folder):
    dataset = load_dataset("parquet", data_files=parquet_path, split="train", streaming=True)
    trajectory_dict = dict()
    for row in tqdm.tqdm(dataset):
        trajectory_id = row['id']
        frame_idx = row['frame_idx'] 
        log = json.loads(row['log'])
        if trajectory_id not in trajectory_dict:
            trajectory_dict[trajectory_id] = {
                'id': trajectory_id,
                'raw_logs': log['raw_logs'],
                'preprocessed_logs': log['preprocessed_logs'],
                'instruction': log['instruction'],
                'instruction_unified': log['instruction_unified'],
                'length': len(log['preprocessed_logs']),
                'images': []
            }
            trajectory_dict[trajectory_id]['images'] = {}
            trajectory_dict[trajectory_id]['images'][frame_idx]= row['image']
        else:
            trajectory_dict[trajectory_id]['images'][frame_idx] = row['image']
            # collect all images in the this trajectory
            if len(trajectory_dict[trajectory_id]['images']) == trajectory_dict[trajectory_id]['length']:
                imgs = [trajectory_dict[trajectory_id]['images'][key] for key in sorted(trajectory_dict[trajectory_id]['images'].keys())]
                trajectory_dict[trajectory_id].pop('images', None)
                os.makedirs(os.path.join(output_folder, trajectory_id), exist_ok=True)
                for i, image in enumerate(imgs):
                    img_path = os.path.join(output_folder, trajectory_id, str(i).zfill(6) + '.jpg')
                    with open(img_path, 'wb') as f:
                        image.save(f, format='JPEG')
                with open(os.path.join(output_folder, trajectory_id, 'log.json'), 'w') as f:
                    json.dump(trajectory_dict[trajectory_id], f)
                tmp = trajectory_dict.pop(trajectory_id, None)
                if tmp is not None:
                    del tmp

train_dir = '/path/to/sim_train_data'
parquet_dir = '/path/to/uav-flow-sim'
for file in os.listdir(parquet_dir):
    if file.endswith('.parquet'):
        process_parquet_to_folder_dataset(os.path.join(parquet_dir, file), train_dir)

