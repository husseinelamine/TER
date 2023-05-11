import yaml
import os
import argparse
def increment_path_number(path):

    # Split the path into its components
    dir_path, filename = os.path.split(path)
    name, ext = os.path.splitext(filename)
    
    # Extract the number from the filename
    num_str = ''
    for c in reversed(name):
        if c.isdigit():
            num_str = c + num_str
        else:
            break
    
    # Increment the number and convert it back to a string
    num = int(num_str)
    new_num_str = str(num + 1)
    
    # Replace the old number with the new number in the filename
    new_name = name[:-len(num_str)] + new_num_str
    
    # Reassemble the path and return it
    new_filename = new_name + ext
    new_path = os.path.join(dir_path, new_filename).replace('\\', '/')
    return new_path

def update_yaml_file(file_path = '../completion/configs/train_update.yaml', increment_path = True, increment_epoch = True):
    # check if file exists and write default values and create file if it doesn't exist
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write('model_path: D:/M1/S2/TER/Train/SnowflakeNet/experiments/Arabidopsis0.pth\ninit_epoch: 1')
        exit(0)
    with open(file_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    if increment_path:
        # Update model path
        train_path = data['model_path']
        updated_train_path = increment_path_number(train_path)
        data['model_path'] = updated_train_path

    if increment_epoch:
        # Increment init_epoch by 1
        data['init_epoch'] += 1
    
    # Save updated data to file
    with open(file_path, 'w') as f:
        yaml.dump(
        data,
        f,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    # add arguments for increment_path and increment_epoch boolean flags

    # increment_path
    parser.add_argument('--increment_path', dest='increment_path', action='store_true')
    parser.add_argument('--no-increment_path', dest='increment_path', action='store_false')
    parser.set_defaults(increment_path=True)

    # increment_epoch
    parser.add_argument('--increment_epoch', dest='increment_epoch', action='store_true')
    parser.add_argument('--no-increment_epoch', dest='increment_epoch', action='store_false')
    parser.set_defaults(increment_epoch=True) 

    args = parser.parse_args()

    update_yaml_file(increment_path = args.increment_path, increment_epoch = args.increment_epoch)
