import os
import json
import sys

def create_folder_structure(path):
    folder = {'name': os.path.basename(path), 'type': 'folder', 'children': []}
    if os.path.isdir(path):
        files = os.listdir(path)
        if len(files) > 0:
            files.sort()
            i = 0
            j = 0
            for filename in files:
                file_path = os.path.join(path, filename)
                if os.path.isdir(file_path) and j < 2:
                    j = j + 1
                    folder['children'].append(create_folder_structure(file_path))
                elif not os.path.isdir(file_path) and i < 2:
                    i = i + 1
                    ext = os.path.splitext(filename)[1]
                    if ext in ['.ply']:
                        folder['children'].append({'name': filename, 'type': 'file'})
    return folder

if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print('Usage: python create_json_folder.py <path> [output_file]')
        sys.exit(1)

    path = sys.argv[1]
    output_file = '../completion/category_files/Arabidopsis.json' if len(sys.argv) == 2 else sys.argv[2]

    json_data = json.dumps(create_folder_structure(path), indent=4)
    
    with open(output_file, 'w') as f:
        f.write(json_data)

    print(f"Folder structure written to {output_file}")
