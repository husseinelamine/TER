import open3d
import numpy as np
def read_ply(file_path):
        pc = open3d.io.read_point_cloud(file_path, format='ply')
        ptcloud = np.array(pc.points)
        return ptcloud

def write_ply(file_path, file_content):
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(file_content)
        open3d.io.write_point_cloud(file_path, pc)


def add_ply_ext_if_needed(path):
    if not path.endswith(".ply"):
        path += ".ply"
    return path
def add_ply_ext_if_needed_list(list):
    for i in range(len(list)):
        list[i] = add_ply_ext_if_needed(list[i])
    return list

# test functions

#list = ['../data/Arabidopsis/complete/plant1/03-17_AM/03-17_AM_segmented', '../data/Arabidopsis/complete/plant1/03-17_AM/03-17_AM_segmented.ply']
#list = add_ply_ext_if_needed_list(list)
#print(list)
# test functions

file_path = '../data/Arabidopsis/complete/plant1/03-17_AM/03-17_AM_segmented.ply'
output_path = './test.ply'
ptcloud = read_ply(file_path)
print(ptcloud)
#write_ply(output_path, ptcloud)

