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

    complete_files = [os.path.join("complete", plant, dir, f) for dir in os.listdir(complete_dir) for f in os.listdir(os.path.join(complete_dir, dir)) if f.endswith(".ply")]
    partial_files = [os.path.join("partial", plant, dir, f) for dir in os.listdir(partial_dir) for f in os.listdir(os.path.join(partial_dir, dir)) if f.endswith(".ply")]
    complete_files = clean_file_paths(complete_files)
    partial_files = clean_file_paths(partial_files)
    random.shuffle(complete_files)
    random.shuffle(partial_files)

    # Split the data into train, validation, and test sets
    train_complete = complete_files[:int(0.8 * len(complete_files))]
    val_complete = complete_files[int(0.8 * len(complete_files)):int(0.9 * len(complete_files))]
    test_complete = complete_files[int(0.9 * len(complete_files)):]

    train_partial = partial_files[:int(0.8 * len(partial_files))]
    val_partial = partial_files[int(0.8 * len(partial_files)):int(0.9 * len(partial_files))]
    test_partial = partial_files[int(0.9 * len(partial_files)):]

    # Combine the occluded point clouds for each plant into a single dataset
    train_files = train_complete + train_partial
    val_files = val_partial[:len(val_partial) // 2] + train_partial[len(train_partial) // 2:] + val_complete
    test_files = test_partial[:len(test_partial) // 2] + train_partial[len(train_partial) // 2:] + test_complete

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
