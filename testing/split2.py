import os
import json
import random

def clean_file_paths(file_paths):
    cleaned_paths = []
    for path in file_paths:
        cleaned_path = path.replace("complete\\", "").replace("partial\\", "").replace("\\", "/")
        cleaned_paths.append(cleaned_path)
    return cleaned_paths

data_dir = "../data/Arabidopsis/"

plants = ["plant1", "plant2", "plant3", "plant4"]
categories = []

# Shuffle the files
random.seed(42)  # Set a seed for reproducibility
for plant in plants:
    complete_dir = os.path.join(data_dir, "complete", plant)
    partial_dir = os.path.join(data_dir, "partial", plant)

    partial_files = [os.path.join("partial", plant, dir, f) for dir in os.listdir(partial_dir) for f in os.listdir(os.path.join(partial_dir, dir)) if f.endswith(".ply")]
    partial_files = clean_file_paths(partial_files)
    random.shuffle(partial_files)

    # Split the data into train, validation, and test sets
    train_partial = partial_files[:int(0.8 * len(partial_files))]
    val_partial = partial_files[int(0.8 * len(partial_files)):int(0.9 * len(partial_files))]
    test_partial = partial_files[int(0.9 * len(partial_files)):]

    complete_files = [os.path.join("complete", plant, dir, f) for dir in os.listdir(complete_dir) for f in os.listdir(os.path.join(complete_dir, dir)) if f.endswith(".ply")]
    complete_files = clean_file_paths(complete_files)

    # Only add complete files to val and test if they exist
    if len(complete_files) >= 1:
        val_complete = complete_files[:int(0.1 * len(complete_files))]
        test_complete = complete_files[int(0.1 * len(complete_files)):]

        # Combine the point clouds for each split into a single dataset
        val_files = val_complete + val_partial
        test_files = test_complete + test_partial

    else:
        # If there are no complete files, use partial files for validation and test
        val_files = val_partial
        test_files = test_partial

    # Combine the occluded point clouds for each plant into the train set
    train_files = train_partial

    categories.append({
        "taxonomy_id": plant,
        "taxonomy_name": plant,
        "train": train_files,
        "val": val_files,
        "test": test_files
    })

# Save the file names to a JSON file
with open("../completion/category_files/Arabidopsis.json", "w") as f:
    json.dump(categories, f, indent=4)
