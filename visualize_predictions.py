#!/usr/bin/env python
import argparse
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet3d.apis import init_model
import torch
from PIL import Image, ImageDraw, ImageFont
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize BEVFusion predictions')
    parser.add_argument('--results', default='mini_test_results/all_results.pkl',
                        help='Path to saved results file')
    parser.add_argument('--config', default='projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py',
                        help='Config file for BEVFusion model')
    parser.add_argument('--sample-indices', type=str, default='0,1,2,3,4',
                        help='Comma-separated indices of samples to visualize')
    parser.add_argument('--out-dir', default='visualization_results',
                        help='Output directory for visualizations')
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='Score threshold for filtering predictions')
    return parser.parse_args()

def visualize_bev(points, boxes, scores, labels, class_names, out_file, point_size=1.0, score_thr=0.3):
    """Visualize 3D boxes from bird's eye view."""
    # Filter detections by score
    if len(boxes) > 0:
        mask = scores > score_thr
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
    
    # Create figure and axes
    fig = plt.figure(figsize=(10, 10))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    
    # Plot points
    points_bev = points[:, 0:2]  # Get X and Y coordinates
    ax.scatter(points_bev[:, 0], points_bev[:, 1], s=point_size, c='black', alpha=0.2)
    
    # Plot boxes
    colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names)))
    
    for i, box in enumerate(boxes):
        # Extract BEV box parameters
        x, y, z = box[0], box[1], box[2]  # Center coordinates
        l, w, h = box[3], box[4], box[5]  # Length, width, height
        yaw = box[6]  # Yaw angle
        
        # Create rectangle for BEV
        bev_corners = np.array([
            [-l/2, -w/2],
            [l/2, -w/2],
            [l/2, w/2],
            [-l/2, w/2]
        ])
        
        # Rotate corners by yaw
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        rot_matrix = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])
        
        rotated_corners = np.dot(bev_corners, rot_matrix.T)
        
        # Shift to center position
        rotated_corners[:, 0] += x
        rotated_corners[:, 1] += y
        
        # Create polygon
        polygon = patches.Polygon(rotated_corners, closed=True, 
                                 fill=True, alpha=0.4, 
                                 color=colors[int(labels[i])])
        ax.add_patch(polygon)
        
        # Add class label and score
        plt.text(x, y, f"{class_names[int(labels[i])]} {scores[i]:.2f}", 
                 fontsize=8, color='black', 
                 bbox=dict(facecolor='white', alpha=0.7))
    
    # Add title and legend
    plt.title("Bird's Eye View Object Detection")
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    
    # Add grid
    plt.grid(True)
    
    # Save figure
    canvas.draw()
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    return out_file

def create_summary_figure(results, class_names, config_name, checkpoint_name, out_file):
    """Create a summary figure showing detection statistics"""
    # Collect statistics
    total_objects = 0
    class_counts = {}
    
    for result in results:
        if hasattr(result['result'], 'pred_instances_3d'):
            pred_instances = result['result'].pred_instances_3d
            pred_scores = pred_instances.scores.cpu().numpy()
            pred_labels = pred_instances.labels.cpu().numpy()
            
            # Filter by score threshold
            mask = pred_scores > 0.3
            pred_labels = pred_labels[mask]
            
            # Count by class
            for label in pred_labels:
                class_name = class_names[int(label)]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                total_objects += 1
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot bar chart of class counts
    if class_counts:
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        ax1.bar(classes, counts, color=plt.cm.rainbow(np.linspace(0, 1, len(classes))))
        ax1.set_title('Object Classes Detected')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add count on top of bars
        for i, count in enumerate(counts):
            ax1.text(i, count + 0.1, str(count), ha='center')
    else:
        ax1.text(0.5, 0.5, 'No objects detected', ha='center', va='center', fontsize=14)
        ax1.set_title('Object Classes Detected')
    
    # Add model info
    ax2.axis('off')
    model_info = [
        f"Model: BEVFusion LiDAR",
        f"Config: {os.path.basename(config_name)}",
        f"Samples: {len(results)}",
        f"Total objects: {total_objects}",
    ]
    
    # Add class distribution
    if total_objects > 0:
        model_info.append("\nClass Distribution:")
        for class_name, count in class_counts.items():
            percentage = count / total_objects * 100
            model_info.append(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    ax2.text(0.1, 0.5, '\n'.join(model_info), va='center', fontsize=12)
    ax2.set_title('Detection Summary')
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()
    
    return out_file

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load the results
    if not os.path.exists(args.results):
        print(f"Results file not found: {args.results}")
        return
    
    with open(args.results, 'rb') as f:
        results = pickle.load(f)
    
    print(f"Loaded {len(results)} results")
    
    # Load configuration
    init_default_scope('mmdet3d')
    cfg = Config.fromfile(args.config)
    class_names = cfg.class_names
    
    # Parse sample indices
    try:
        sample_indices = [int(idx) for idx in args.sample_indices.split(',')]
    except:
        sample_indices = [0, 1, 2, 3, 4]
    
    # Filter to selected samples
    if sample_indices:
        sample_indices = [i for i in sample_indices if i < len(results)]
        visualize_results = [results[i] for i in sample_indices]
    else:
        visualize_results = results
    
    print(f"Visualizing {len(visualize_results)} samples")
    
    # Process each selected result
    for i, result_item in enumerate(visualize_results):
        try:
            result = result_item['result']
            sample_id = result_item['sample_id']
            
            # Find the corresponding point cloud file
            # For visualization purposes, we need to load the original point cloud
            result_pkl = f"mini_test_results/result_{sample_id}.pkl"
            if not os.path.exists(result_pkl):
                print(f"Result file not found: {result_pkl}, skipping")
                continue
                
            # Try to locate the point cloud file from data folder
            # First, look in sample files to find matching ID
            sample_files = os.listdir('data/nuscenes/samples/LIDAR_TOP/')
            potential_files = [f for f in sample_files if sample_id in f]
            
            if potential_files:
                points_file = os.path.join('data/nuscenes/samples/LIDAR_TOP/', potential_files[0])
            else:
                # If not found, use any sample file for visualization
                points_file = os.path.join('data/nuscenes/samples/LIDAR_TOP/', sample_files[0])
            
            # Load points
            points = np.fromfile(points_file, dtype=np.float32).reshape(-1, 5)
            
            # Extract prediction results
            if hasattr(result, 'pred_instances_3d'):
                pred_instances = result.pred_instances_3d
                pred_bboxes = pred_instances.bboxes.tensor.cpu().numpy()
                pred_scores = pred_instances.scores.cpu().numpy()
                pred_labels = pred_instances.labels.cpu().numpy()
                
                # Create BEV visualization
                out_file = os.path.join(args.out_dir, f"sample_{i}_{sample_id}_bev.png")
                visualize_bev(points, pred_bboxes, pred_scores, pred_labels, 
                              class_names, out_file, score_thr=args.score_thr)
                print(f"Saved BEV visualization to {out_file}")
        
        except Exception as e:
            print(f"Error processing result {i}: {e}")
    
    # Create summary figure
    summary_file = os.path.join(args.out_dir, "detection_summary.png")
    create_summary_figure(results, class_names, args.config, "N/A", summary_file)
    print(f"Saved summary figure to {summary_file}")
    
    # Create HTML report
    html_file = os.path.join(args.out_dir, "visualization_report.html")
    with open(html_file, 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>BEVFusion Prediction Visualization</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
                h1, h2 { color: #333; }
                .summary { margin-bottom: 30px; }
                .sample { margin-bottom: 30px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; }
                img { max-width: 100%; border: 1px solid #eee; }
            </style>
        </head>
        <body>
            <h1>BEVFusion Prediction Visualization</h1>
            
            <div class="summary">
                <h2>Detection Summary</h2>
                <img src="detection_summary.png" alt="Detection Summary">
            </div>
            
            <h2>Sample Visualizations</h2>
        """)
        
        for i, result_item in enumerate(visualize_results):
            sample_id = result_item['sample_id']
            bev_file = f"sample_{i}_{sample_id}_bev.png"
            
            if os.path.exists(os.path.join(args.out_dir, bev_file)):
                f.write(f"""
                <div class="sample">
                    <h3>Sample {i}: {sample_id}</h3>
                    <div class="bev">
                        <h4>Bird's Eye View</h4>
                        <img src="{bev_file}" alt="BEV Visualization">
                    </div>
                </div>
                """)
        
        f.write("""
        </body>
        </html>
        """)
    
    print(f"Created HTML report at {html_file}")

if __name__ == "__main__":
    main() 