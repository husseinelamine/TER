import os
import json

data_dir = "../data/Arabidopsis/"


categories = {
    "taxonomy": []
}

plants = ["plant1", "plant2", "plant3", "plant4"]

for plant in plants:
    complete_dir = os.path.join(data_dir, "complete", plant)
    partial_dir = os.path.join(data_dir, "partial", plant)
    
    complete_files = [os.path.join("complete", plant, dir, f) for dir in os.listdir(complete_dir) for f in os.listdir(os.path.join(complete_dir, dir)) if f.endswith(".ply")]
    partial_files = [os.path.join("partial", plant, dir, f) for dir in os.listdir(partial_dir) for f in os.listdir(os.path.join(partial_dir, dir)) if f.endswith(".ply")]
    
    categories["taxonomy"].append({
        "taxonomy_id": plant,
        "taxonomy_name": plant,
        "train": complete_files,
        "val": partial_files[:len(partial_files)//2],
        "test": partial_files[len(partial_files)//2:]
    })

with open("categories.json", "w") as f:
    json.dump(categories, f, indent=4)
