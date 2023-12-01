import glob
import os

root_dir = 'C:/Users/Tomson/BRLAB/tooth/Temporomandibular_movement/movie/2023_06_15'
pattern = os.path.join(root_dir, '*.bag')
bag_files = glob.glob(pattern, recursive=True)
print(bag_files)

num_bags = len(bag_files)
print('num_bags=',num_bags)

for progress, bagfile in enumerate(bag_files):
    path = root_dir + '/' + bag
    os.mkdir(path)
