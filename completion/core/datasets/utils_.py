import os
import re
def add_ply_ext_if_needed(path):
    if not path.endswith(".ply"):
        path += ".ply"
    return path
def add_ply_ext_if_needed_list(list):
    for i in range(len(list)):
        list[i] = add_ply_ext_if_needed(list[i])
    return list
def replace_backslash(path):
    # Remplacer les backslashes par des slashes
    path = path.replace("\\", "/")
    # Remplacer plusieurs backslashes consécutifs par un seul slash
    path = re.sub(r"/{2,}", "/", path)
    return path
def get_max_file_number(complete_file_path):
    partial_file_path = complete_file_path.replace("complete", "partial")
    partial_file_path = os.path.split(partial_file_path)[0] + "/"
    max_file_number = -1
    if os.path.exists(partial_file_path):
        for file in os.listdir(partial_file_path):
            if file.endswith(".ply"):
                try:
                    file_number = int(os.path.splitext(file)[0])
                    max_file_number = max(max_file_number, file_number)
                except ValueError:
                    continue
    return max_file_number
def get_this_max_file_number(complete_file_path):
    max_file_number = -1
    # split to take file root path
    complete_file_path = os.path.split(complete_file_path)[0] + "/"
    if os.path.exists(complete_file_path):
        for file in os.listdir(complete_file_path):
            if file.endswith(".ply"):
                try:
                    file_number = int(os.path.splitext(file)[0])
                    max_file_number = max(max_file_number, file_number)
                except ValueError:
                    continue
    return max_file_number
def rename_file(original_path, new_name):
    # Split the path into directory, filename and extension
    directory, filename_ext = os.path.split(original_path)
    filename, ext = os.path.splitext(filename_ext)
    
    # Construct the new path with the desired filename
    new_path = os.path.join(directory, f"{new_name}{ext}")
        
    return new_path

def fix_dataset_path(path, replace_str=None):
    if "Arabidopsis" in path:
        if "complete" in path:
            path = rename_file(path.replace("complete", "partial"), "gt")
        elif "partial" in path and replace_str is not None:
            _, filename_ext = os.path.split(path)
            filename, _ = os.path.splitext(filename_ext)
            path = rename_file(path, f"{replace_str}{filename}")

    return path
