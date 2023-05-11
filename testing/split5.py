import os
import json
import random

MAX_COUNT = 300
LIMIT = False
def clean_file_paths(file_paths):
    cleaned_paths = []
    for path in file_paths:
        cleaned_path = path.replace("\\", "/").replace("../data/Arabidopsis/partial/", "")
        #cleaned_path = re.sub(r"plant\d+/", "", cleaned_path)
        cleaned_paths.append(cleaned_path)
    return cleaned_paths

data_dir = "../data/Arabidopsis/"

plants = ["plant1", "plant2", "plant3", "plant4"]
categories = []

# Shuffle the files
random.seed(42)  # Set a seed for reproducibility

for plant in plants:
    partial_dir = os.path.join(data_dir, "partial", plant)
    count = 0
    partial_files = []
    complete_files = []
    for root, dirs, files in os.walk(partial_dir):
        for file in files:
            if file.endswith(".ply") and file != "gt.ply":
                partial_path = os.path.join(root, file)
                gt_path = os.path.join(root, "gt.ply")
                if os.path.exists(gt_path) and (count < MAX_COUNT or not LIMIT):
                    partial_files.append(partial_path)
                    count += 1
            elif file == "gt.ply":
                complete_path = os.path.join(root, file)
                complete_files.append(complete_path)
    partial_files = clean_file_paths(partial_files)
    complete_files = clean_file_paths(complete_files)
    random.shuffle(partial_files)
    random.shuffle(complete_files)

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
    taxonomy_name = os.path.basename(os.path.dirname(train_files[0]))
    taxonomy_id = os.path.join(plant, taxonomy_name)

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
