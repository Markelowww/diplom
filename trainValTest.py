import os
import argparse
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_folder', type=str, default='./datasetNO', help='path to OCTDL folder')
parser.add_argument('--output_folder', type=str, default='./datasetNO1', help='path to output folder')
parser.add_argument('--val_ratio', type=float, default=0.1, help='validation size')
parser.add_argument('--test_ratio', type=float, default=0.2, help='test size')

labels = ['NO']
folders = ['train', 'val', 'test']

def main():
    args = parser.parse_args()
    root_folder = Path(args.dataset_folder)
    output_folder = Path(args.output_folder)
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio
    train_ratio = 1 - val_ratio - test_ratio

    for folder in folders:
        for label in labels:
            Path(os.path.join(output_folder, folder, label)).mkdir(parents=True, exist_ok=True)

    for label in tqdm(labels):
        file_names = os.listdir(os.path.join(root_folder, label))
        train_files, test_val_files = train_test_split(file_names, test_size=1 - train_ratio)
        val_files, test_files = train_test_split(test_val_files, test_size=test_ratio / (test_ratio + val_ratio))

        move_files(root_folder, output_folder, train_files, label, 'train')
        move_files(root_folder, output_folder, val_files, label, 'val')
        move_files(root_folder, output_folder, test_files, label, 'test')

def move_files(root_folder, output_folder, files, label, folder):
    for file in files:
        src_path = os.path.join(root_folder, label, file)
        dest_path = os.path.join(output_folder, folder, label, file)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        os.replace(src_path, dest_path)

if __name__ == "__main__":
    main()
