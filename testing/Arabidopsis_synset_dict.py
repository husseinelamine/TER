import os
import json

def clean_file_paths(file_paths):
    cleaned_paths = []
    for path in file_paths:
        cleaned_path = path.replace("\\", "/").replace("../data/Arabidopsis/partial/", "")
        #cleaned_path = re.sub(r"plant\d+/", "", cleaned_path)
        cleaned_paths.append(cleaned_path)
    return cleaned_paths

data_dir = "../data/Arabidopsis/"

plants = ["plant1", "plant2", "plant3", "plant4"]
categories = {}

for plant in plants:
    partial_dir = os.path.join(data_dir, "partial", plant)
    for dir in os.listdir(partial_dir):
        taxonomy_name = os.path.join(partial_dir, dir)
        taxonomy_id = taxonomy_name.replace("\\", "/").replace("../data/Arabidopsis/partial/", "")
        categories[taxonomy_id] = taxonomy_id

# Save the file names to a JSON file
with open("../completion/category_files/Arabidopsis_synset_dict.json", "w") as f:
    json.dump(categories, f)
  