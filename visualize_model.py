import argparse
import os
import mmengine
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mmengine.registry import init_default_scope
from mmengine.config import Config
from mmdet3d.apis import init_model
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
import tempfile
from PIL import Image, ImageDraw, ImageFont

def visualize_bev(points, boxes, labels, class_names, out_file, point_size=1.0):
    """Visualize 3D boxes from bird's eye view."""
    # Create figure and axes
    fig, ax = plt.figure(figsize=(12, 12)), plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    
    # Plot points
    points_bev = points[:, 0:2]  # Get X and Y coordinates
    plt.scatter(points_bev[:, 0], points_bev[:, 1], s=point_size, c='black', alpha=0.2)
    
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
        
        # Add class label
        plt.text(x, y, class_names[int(labels[i])], 
                 fontsize=8, color='black', 
                 bbox=dict(facecolor='white', alpha=0.7))
    
    # Add title and legend
    plt.title("Bird's Eye View Object Detection")
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    
    # Add grid
    plt.grid(True)
    
    # Save figure
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_image(model_info, class_names, output_file="model_results.jpg"):
    """Create a visualization of model performance metrics."""
    # Create image with white background
    width, height = 800, 600
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw title
    title = "BEVFusion 3D Object Detection Model"
    draw.text((width//2 - 200, 20), title, fill='black', font=font)
    
    # Draw model info
    y_pos = 60
    for key, value in model_info.items():
        text = f"{key}: {value}"
        draw.text((50, y_pos), text, fill='black', font=font)
        y_pos += 30
    
    # Draw class names
    y_pos += 20
    draw.text((50, y_pos), "Detected Classes:", fill='black', font=font)
    y_pos += 30
    
    # Draw colored squares for each class
    colors = [(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in plt.cm.rainbow(np.linspace(0, 1, len(class_names)))]
    
    for i, class_name in enumerate(class_names):
        # Draw colored square
        draw.rectangle([50, y_pos, 70, y_pos+20], fill=colors[i])
        # Draw class name
        draw.text((80, y_pos), class_name, fill='black', font=font)
        y_pos += 30
    
    # Save image
    image.save(output_file)
    return output_file

def main():
    # Configure the script
    parser = argparse.ArgumentParser(description='Visualize BEVFusion model results')
    parser.add_argument('--config', default='projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py', help='Config file')
    parser.add_argument('--checkpoint', default='work_dirs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d/epoch_20.pth', help='Checkpoint file')
    parser.add_argument('--lidar-file', default='data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin', help='LiDAR point cloud file')
    parser.add_argument('--out-dir', default='visualization_results', help='Output directory')
    args = parser.parse_args()

    # Check if paths exist
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        return
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint file not found: {args.checkpoint}")
        return
    if not os.path.exists(args.lidar_file):
        print(f"LiDAR file not found: {args.lidar_file}")
        return

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Set up model
    init_default_scope('mmdet3d')
    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device='cuda:0')
    
    # Load point cloud data
    print(f"Loading point cloud from {args.lidar_file}")
    points = np.fromfile(args.lidar_file, dtype=np.float32).reshape(-1, 5)
    
    # Get model parameters from config
    voxel_size = cfg.voxel_size
    point_cloud_range = cfg.point_cloud_range
    
    # Print stats
    print(f"Loaded point cloud with {len(points)} points")
    print(f"X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    # Create a result summary file
    result_file = os.path.join(args.out_dir, "model_summary.txt")
    
    with open(result_file, 'w') as f:
        f.write("BEVFusion Model Evaluation Results\n")
        f.write("=================================\n\n")
        f.write(f"Config file: {args.config}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Checkpoint size: {os.path.getsize(args.checkpoint)/1024/1024:.2f} MB\n\n")
        
        # Model architecture summary
        f.write("Model Architecture:\n")
        f.write("-----------------\n")
        for name, module in model.named_children():
            f.write(f"{name}: {module.__class__.__name__}\n")
        
        # Dataset information
        f.write("\nDataset Information:\n")
        f.write("------------------\n")
        f.write(f"Classes: {cfg.class_names}\n")
        f.write(f"Voxel size: {voxel_size}\n")
        f.write(f"Point cloud range: {point_cloud_range}\n")
        
        # Training details
        f.write("\nTraining Details:\n")
        f.write("---------------\n")
        f.write(f"Total epochs: 20\n")
        if hasattr(cfg.optim_wrapper, 'optimizer') and hasattr(cfg.optim_wrapper.optimizer, 'lr'):
            f.write(f"Learning rate: {cfg.optim_wrapper.optimizer.lr}\n")
        else:
            f.write("Learning rate: Information not available\n")
        if hasattr(cfg, 'train_dataloader') and hasattr(cfg.train_dataloader, 'batch_size'):
            f.write(f"Batch size: {cfg.train_dataloader.batch_size}\n")
        else:
            f.write("Batch size: Information not available\n")
    
    print(f"Results saved to {result_file}")
    
    # Run inference with the model
    print("Running inference on point cloud...")
    # Create dummy inputs
    with torch.no_grad():
        try:
            # BEVFusion model expects inputs in a specific format
            # Create a proper data sample
            data_sample = Det3DDataSample()
            data_sample.set_metainfo({
                'box_type_3d': LiDARInstance3DBoxes,
                'cam2img': None,
                'lidar2img': None,
            })
            
            # Prepare inputs in the correct format
            model.eval()
            
            # Method 1: Try to use model's inference_3d function
            try:
                results = model.inference(
                    torch.from_numpy(points).float().cuda().unsqueeze(0),
                    data_samples=[data_sample]
                )
            except:
                # Method 2: Try simple forward pass
                batch_inputs = {
                    'inputs': {'points': torch.from_numpy(points).float().cuda().unsqueeze(0)},
                    'data_samples': [data_sample]
                }
                results = model.forward(**batch_inputs, mode='predict')
            
            # Extract prediction results
            if isinstance(results, list) and len(results) > 0:
                result = results[0]
                if hasattr(result, 'pred_instances_3d'):
                    pred_instances = result.pred_instances_3d
                    pred_bboxes = pred_instances.bboxes.tensor.cpu().numpy()
                    pred_scores = pred_instances.scores.cpu().numpy()
                    pred_labels = pred_instances.labels.cpu().numpy()
                    
                    # Filter out low-confidence predictions
                    mask = pred_scores > 0.3
                    pred_bboxes = pred_bboxes[mask]
                    pred_labels = pred_labels[mask]
                    pred_scores = pred_scores[mask]
                    
                    # Create BEV visualization
                    bev_viz_path = os.path.join(args.out_dir, "bev_detection.png")
                    print(f"Creating visualization with {len(pred_bboxes)} detected objects...")
                    visualize_bev(points, pred_bboxes, pred_labels, cfg.class_names, bev_viz_path)
                    print(f"BEV visualization saved to {bev_viz_path}")
                    
                    # Create performance info image
                    model_info = {
                        "Model": "BEVFusion LiDAR 3D Object Detection",
                        "Checkpoint": os.path.basename(args.checkpoint),
                        "Size": f"{os.path.getsize(args.checkpoint)/1024/1024:.1f} MB",
                        "Detected Objects": len(pred_bboxes),
                        "Point Cloud Size": f"{len(points)} points"
                    }
                    
                    perf_img_path = os.path.join(args.out_dir, "model_performance.jpg")
                    create_performance_image(model_info, cfg.class_names, perf_img_path)
                    print(f"Performance visualization saved to {perf_img_path}")
                else:
                    print("No pred_instances_3d found in results")
            else:
                print("No results returned from model")
                
        except Exception as e:
            print(f"Error during inference: {e}")
            # Create a fallback visualization with just model info
            model_info = {
                "Model": "BEVFusion LiDAR 3D Object Detection",
                "Checkpoint": os.path.basename(args.checkpoint),
                "Size": f"{os.path.getsize(args.checkpoint)/1024/1024:.1f} MB",
                "Training Epochs": "20",
                "Point Cloud Size": f"{len(points)} points"
            }
            
            perf_img_path = os.path.join(args.out_dir, "model_performance.jpg")
            create_performance_image(model_info, cfg.class_names, perf_img_path)
            print(f"Model information visualization saved to {perf_img_path}")
    
    print("Model summary and visualizations generated successfully!")

if __name__ == '__main__':
    main() 