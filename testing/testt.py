import os

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

a = get_max_file_number("D:/M1/S2/TER/Train/SnowflakeNet/data/Arabidopsis/complete/plant1/03-23_PM/03-23_PM_segmented.ply")
print(a)