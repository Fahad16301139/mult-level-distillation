import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

work_dir = "work_dirs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d"
folders = [f for f in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, f))]
if not folders:
    print("No result folders found!")
    exit()

latest = sorted(folders)[-1]
json_file = os.path.join(work_dir, latest, f"{latest}.json")
if not os.path.exists(json_file):
    print("No result file found in the latest folder!")
    exit()

with open(json_file) as f:
    data = json.load(f)

# Print all main metrics
def print_main_metrics():
    print(f"Results for run: {latest}")
    for metric in [
        'NDS', 'mAP', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']:
        key = f"NuScenes metric/pred_instances_3d_NuScenes/{metric}"
        print(f"{metric:5}: {data.get(key)}")

# Per-class APs at all thresholds
def get_per_class_aps():
    aps = {}
    for k, v in data.items():
        if '_AP_dist_' in k:
            parts = k.split('_')
            cls = parts[0]
            dist = k.split('_AP_dist_')[-1]
            aps.setdefault(cls, {})[dist] = v
    return aps

def print_per_class_aps(aps):
    print("\nPer-class APs:")
    dists = sorted({d for v in aps.values() for d in v})
    header = f"{'Class':20}" + ''.join([f"AP@{d:>5}" for d in dists])
    print(header)
    for cls, v in aps.items():
        row = f"{cls:20}" + ''.join([f"{v.get(d, 0):8.3f}" for d in dists])
        print(row)

# Visualize per-class APs at 0.5m
aps = get_per_class_aps()
if aps:
    ap_05 = {cls: v.get('0.5', 0) for cls, v in aps.items()}
    plt.figure(figsize=(10, 5))
    plt.bar(ap_05.keys(), ap_05.values())
    plt.xticks(rotation=45)
    plt.ylabel("AP @ 0.5m")
    plt.title(f"Per-class APs (0.5m) for run {latest}")
    plt.tight_layout()
    plt.savefig(f"{latest}_per_class_APs.png")
    print(f"\nSaved bar chart as {latest}_per_class_APs.png")

# Visualize all main metrics as a bar chart
main_metrics = ['NDS', 'mAP', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']
main_values = [data.get(f"NuScenes metric/pred_instances_3d_NuScenes/{m}") for m in main_metrics]
plt.figure(figsize=(8, 5))
plt.bar(main_metrics, main_values)
plt.ylabel('Value')
plt.title(f"Main NuScenes Metrics for run {latest}")
plt.tight_layout()
plt.savefig(f"{latest}_main_metrics.png")
print(f"Saved bar chart as {latest}_main_metrics.png")

# Print all metrics as tables
print_main_metrics()
print_per_class_aps(aps) 