import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

work_dir = "work_dirs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d"
folders = [f for f in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, f))]
folders = sorted(folders)  # Sort by timestamp

runs = []
for folder in folders:
    json_file = os.path.join(work_dir, folder, f"{folder}.json")
    if os.path.exists(json_file):
        with open(json_file) as f:
            data = json.load(f)
        mAP = data.get("NuScenes metric/pred_instances_3d_NuScenes/mAP")
        NDS = data.get("NuScenes metric/pred_instances_3d_NuScenes/NDS")
        runs.append((folder, mAP, NDS))

if not runs:
    print("No result files found!")
    exit()

# Plot
x = [r[0] for r in runs]
mAPs = [r[1] for r in runs]
NDSs = [r[2] for r in runs]

plt.figure(figsize=(12, 6))
plt.plot(x, mAPs, marker='o', label='mAP')
plt.plot(x, NDSs, marker='s', label='NDS')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Run (timestamp)')
plt.ylabel('Score')
plt.title('mAP and NDS over different validation/test runs')
plt.legend()
plt.tight_layout()
plt.savefig("mAP_NDS_over_time.png")
print("Saved plot as mAP_NDS_over_time.png") 