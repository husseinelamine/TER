import os
import pandas as pd

logs_directory = '../completion/exp/logs'

# Iterate over the folders in the logs directory
for folder_name in os.listdir(logs_directory):
    folder_path = os.path.join(logs_directory, folder_name)
    if os.path.isdir(folder_path):
        train_folder = os.path.join(folder_path, 'train')
        test_folder = os.path.join(folder_path, 'test')
        
        # Load train logs
        train_logs_path = os.path.join(train_folder, os.listdir(train_folder)[0])
        if os.path.isfile(train_logs_path):
            train_logs = pd.read_csv(train_logs_path)
            # Perform analysis or visualization on train logs
            
        # Load test logs
        test_logs_path = os.path.join(test_folder, os.listdir(test_folder)[0])
        if os.path.isfile(test_logs_path):
            test_logs = pd.read_csv(test_logs_path)
            # Perform analysis or visualization on test logs
