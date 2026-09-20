import os
import shutil

dir_path = "/remote-home/cs_cs_ywh/CorrNet_Plus-main/CorrNet_Plus_CSLR/dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test"
subfolders = [x for x in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, x))]

for dir in subfolders:
    if "1" in os.listdir(os.path.join(dir_path, dir)):
        continue # 1 dir already exists, assume files are in correct dir
    files = [x for x in os.listdir(os.path.join(dir_path, dir)) if os.path.isfile(os.path.join(dir_path, dir, x))]
    goal_dir = os.path.join(dir_path, dir, "1")
    os.makedirs(goal_dir)
    for f in files:
        shutil.move(os.path.join(dir_path, dir, f), goal_dir)
