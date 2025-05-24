import os
import json
import csv

work_dir = "work_dirs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d"
folders = [f for f in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, f))]
latest = sorted(folders)[-1]
json_file = os.path.join(work_dir, latest, f"{latest}.json")
with open(json_file) as f:
    data = json.load(f)

# Main metrics
main_metrics = ['NDS', 'mAP', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']
main_values = [data.get(f"NuScenes metric/pred_instances_3d_NuScenes/{m}") for m in main_metrics]

print("Main NuScenes Metrics:")
print("| Metric | Value |")
print("|--------|-------|")
for m, v in zip(main_metrics, main_values):
    print(f"| {m:5} | {v:.4f} |")

# Save as CSV
with open(f"{latest}_main_metrics.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Metric", "Value"])
    for m, v in zip(main_metrics, main_values):
        writer.writerow([m, v])

# Per-class APs
aps = {}
for k, v in data.items():
    if '_AP_dist_' in k:
        parts = k.split('_')
        cls = parts[0]
        dist = k.split('_AP_dist_')[-1]
        aps.setdefault(cls, {})[dist] = v

dists = sorted({d for v in aps.values() for d in v})
print("\nPer-class APs:")
header = ["Class"] + [f"AP@{d}" for d in dists]
print("| " + " | ".join(header) + " |")
print("|" + "----|" * len(header))
for cls, v in aps.items():
    row = [cls] + [f"{v.get(d, 0):.4f}" for d in dists]
    print("| " + " | ".join(row) + " |")

# Save as CSV
with open(f"{latest}_per_class_APs.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    for cls, v in aps.items():
        row = [cls] + [v.get(d, 0) for d in dists]
        writer.writerow(row)

print(f"\nSaved tables as {latest}_main_metrics.csv and {latest}_per_class_APs.csv") 