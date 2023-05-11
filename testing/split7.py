import os
import json
import random
import argparse

def clean_file_paths(file_paths):
    cleaned_paths = []
    for path in file_paths:
        cleaned_path = path.replace("\\", "/").replace("../data/Arabidopsis/partial/", "")
        cleaned_paths.append(cleaned_path)
    return cleaned_paths

data_dir = "../data/Arabidopsis/"

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--max-count', type=int, default=5,
                    help='maximum number of plants to include')
parser.add_argument('--limit-dir', type=int, default=20,
                    help='maximum number of directories to include per plant')
parser.add_argument('--limit-plant', type=int, default=1500,
                    help='maximum number of point clouds to include per directory')
parser.add_argument('--seed', type=int, default=42,
                    help='Set a seed for reproducibility')

args = parser.parse_args()

MAX_COUNT = args.max_count
LIMITDIR = args.limit_dir
LIMITPLANT = args.limit_plant
seed = args.seed

plants = ["plant1", "plant2", "plant3", "plant4"]
categories = []

# Shuffle the files
random.seed(seed)  # Set a seed for reproducibility
j = 0
for plant in plants:
    if j == MAX_COUNT:
        break
    j += 1
    partial_dir = os.path.join(data_dir, "partial", plant)
    i = 0
    dirs = os.listdir(partial_dir)
    random.shuffle(dirs)
    for dir in dirs:
        if i == LIMITDIR:
            break
        i += 1
        count = 0
        partial_files = []
        complete_files = []
        k = 0
        for file in os.listdir(os.path.join(partial_dir, dir)):
            #if k == LIMITPLANT:
            #    break
            k += 1
            if file.endswith(".ply") and file != "gt.ply":
                partial_path = os.path.join(partial_dir, dir, file)
                gt_path = os.path.join(partial_dir, dir, "gt.ply")
                if os.path.exists(gt_path):
                    partial_files.append(file)
                    count += 1
        complete_files.append("gt.ply")
        partial_files = clean_file_paths(partial_files)
        complete_files = clean_file_paths(complete_files)
        random.shuffle(partial_files)
        random.shuffle(complete_files)
        if len(partial_files) > LIMITPLANT:
            partial_files = partial_files[:LIMITPLANT]

        # Split the data into train, validation, and test sets
        train_partial = partial_files
        val_partial = []
        test_partial = []

        val_complete = complete_files
        test_complete = complete_files

        # Combine the point clouds for each split into a single dataset
        val_files = val_complete
        test_files = test_complete

        # Combine the occluded point clouds for each plant into the train set
        train_files = train_partial
        # Extract the second folder name as the taxonomy name and combine with the plant name
        taxonomy_name = os.path.join(partial_dir, dir)
        taxonomy_id = taxonomy_name.replace("\\", "/").replace("../data/Arabidopsis/partial/", "")

        categories.append({
            "taxonomy_id": taxonomy_id,
            "taxonomy_name": taxonomy_id,
            "train": train_files,
            "val": val_files,
            "test": test_files
        })

# Save the file names to a JSON file
with open("../completion/category_files/Arabidopsis.json", "w") as f:
    json.dump(categories, f, indent=4)
