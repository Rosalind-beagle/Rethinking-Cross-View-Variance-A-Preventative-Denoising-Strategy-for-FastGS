# File: utils/visualize.py
import torch
import torchvision
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import numpy as np

_stats_files_cleared = set()


@torch.no_grad()
def compute_variance_and_save_heatmap(iteration, viewpoint_cam, gaussians, pipe, bg, render_fastgs, model_path, args=None):
    """
    Compute actual variance, generate and save a heatmap image, and return the normalized variance.
    Red means high variance (potential specular/floater), Blue/Dark means low variance.
    """
    # 1. Calculate Variance: M2 / N
    eps = 1e-8
    variance = gaussians.error_running_M2 / (gaussians.view_observe_count + eps)
    
    # 2. Filter out unobserved points (count == 0) to avoid false signals
    observed_mask = gaussians.view_observe_count > 0
    variance[~observed_mask] = 0.0

    if not observed_mask.any():
        return torch.zeros_like(variance) # No data yet
        
    valid_variance = variance[observed_mask]
    
    # 3. Normalize Variance for visualization (0 to 1)
    v_min = valid_variance.min()
    v_max = torch.quantile(valid_variance, 0.99) 
    
    # Prevent degenerate bounds. If max == min (e.g., all variances are 0), matplotlib auto-expands to [-0.1, 0.1]
    if v_max <= v_min + 1e-7:
        v_max = v_min + 1e-5
    
    variance_norm = torch.clamp((variance - v_min) / (v_max - v_min + eps), 0.0, 1.0)
    
    # # log variance statistics
    # save_dir = os.path.join(model_path, "variance_heatmaps")
    # os.makedirs(save_dir, exist_ok=True)
    
    # raw_mean = valid_variance.mean().item()
    # raw_max = valid_variance.max().item()
    # raw_min = valid_variance.min().item()
    
    # valid_norm = variance_norm[observed_mask]
    # norm_mean = valid_norm.mean().item()
    # norm_q99 = torch.quantile(valid_norm, 0.99).item()
    # norm_q95 = torch.quantile(valid_norm, 0.95).item()
    
    # stats_file = os.path.join(save_dir, "variance_stats.txt")

    # global _stats_files_cleared
    # if stats_file not in _stats_files_cleared:
    #     if os.path.exists(stats_file):
    #         os.remove(stats_file)
    #     _stats_files_cleared.add(stats_file)

    # write_header = not os.path.exists(stats_file)
    # with open(stats_file, "a", encoding="utf-8") as f:
    #     if write_header:
    #         f.write("Iteration,Raw_Mean,Raw_Max,Raw_Min,Norm_Mean,Norm_Q99,Norm_Q95\n")
    #     f.write(f"{iteration},{raw_mean:.8f},{raw_max:.8f},{raw_min:.8f},{norm_mean:.8f},{norm_q99:.8f},{norm_q95:.8f}\n")
    
    # # 4. Backup original colors (DC components)
    # original_dc = gaussians._features_dc.clone()
    
    # # 5. Create Heatmap Colors
    # heatmap_dc = torch.zeros_like(original_dc)
    # SH_C0 = 0.28209479177387814 
    
    # # Map to Red (Channel 0) and Blue (Channel 2)
    # heatmap_dc[:, 0, 0] = (variance_norm.flatten() * 1.5 - 0.5) / SH_C0 # Red
    # heatmap_dc[:, 0, 2] = ((1.0 - variance_norm.flatten()) * 0.5) / SH_C0 # Blue
    
    # # Apply the fake colors
    # gaussians._features_dc.copy_(heatmap_dc)
    
    # # 6. Render the image with fake colors
    # render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, args.mult, get_flag=False)
    # rendered_image = render_pkg["render"]
    
    # # 7. Restore the original colors immediately!
    # gaussians._features_dc.copy_(original_dc)
    
    # # 8. Save the image
    # img_path = os.path.join(save_dir, f"heatmap_iter_{iteration}.png")

    # # Save the ground truth image for comparison
    # gt_path = os.path.join(save_dir, f"gt_iter_{iteration}.png")
    # torchvision.utils.save_image(viewpoint_cam.original_image, gt_path)

    # # Convert tensor to numpy format for matplotlib
    # img_np = rendered_image.detach().permute(1, 2, 0).cpu().numpy()
    # img_np = np.clip(img_np, 0, 1)
    
    # # Create matplotlib figure matching the image aspect ratio
    # fig, ax = plt.subplots(figsize=(8, 8 * img_np.shape[0] / img_np.shape[1]))
    # ax.imshow(img_np)
    # ax.axis('off')
    
    # # Create a custom colormap that exactly matches the SH feature modification logic
    # colors = []
    # for v in np.linspace(0, 1, 256):
    #     r = min(1.0, max(0.0, v * 1.5))
    #     g = 0.5
    #     b = min(1.0, max(0.0, 1.0 - v * 0.5))
    #     colors.append((r, g, b))
    # cmap = LinearSegmentedColormap.from_list("variance_cmap", colors)
    
    # # Add a Scale Bar corresponding to the real Variance numerical range
    # sm = ScalarMappable(norm=plt.Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label("Variance")
    
    # plt.savefig(img_path, bbox_inches='tight', dpi=150)
    # plt.close(fig)

    # # 9. Save the histogram of normalized variance
    # hist_path = os.path.join(save_dir, f"histogram_iter_{iteration}.png")
    
    # fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
    # valid_norm_np = valid_norm.detach().cpu().numpy()
    
    # ax_hist.hist(valid_norm_np, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    # ax_hist.set_title(f"Normalized Variance Histogram (Iteration {iteration})")
    # ax_hist.set_xlabel("Normalized Variance")
    # ax_hist.set_ylabel("Count (Gaussians)")
    # ax_hist.set_xlim(0.0, 1.0)
    # ax_hist.grid(axis='y', linestyle='--', alpha=0.7)
    
    # plt.savefig(hist_path, bbox_inches='tight', dpi=150)
    # plt.close(fig_hist)

    return variance_norm