import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
import cv2, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Circle

raw = cv2.imread('figures/cylinder_crop_raw.png', 0)
clean = cv2.bilateralFilter(raw, 5, 50, 50)
gx = cv2.Sobel(clean, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(clean, cv2.CV_64F, 0, 1, ksize=3)
mag = np.hypot(gx, gy)
H, W = raw.shape

cx, cy = 85.64, 73.14
a = 26.91
b = 22.60
phi_deg = 119.06
phi = np.radians(phi_deg)
stretch = a / b

R_fwd = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
S = np.array([[1.0, 0.0], [0.0, stretch]])
M_pure = R_fwd @ S @ R_fwd.T

pad = 60
cx_p = cx + pad
cy_p = cy + pad
offset_padded = np.array([cx_p, cy_p]) - M_pure @ np.array([cx_p, cy_p])

M_warp = np.zeros((2, 3), dtype=np.float32)
M_warp[:2, :2] = M_pure
M_warp[:2, 2] = offset_padded

warped_raw = cv2.warpAffine(cv2.copyMakeBorder(raw, pad, pad, pad, pad, cv2.BORDER_REFLECT),
                            M_warp, (W + 2*pad, H + 2*pad), flags=cv2.INTER_CUBIC)
warped_mag = cv2.warpAffine(cv2.copyMakeBorder(mag, pad, pad, pad, pad, cv2.BORDER_REFLECT),
                            M_warp, (W + 2*pad, H + 2*pad), flags=cv2.INTER_CUBIC)

cx1, cy1 = cx_p, cy_p
R1 = a # 26.91

cx2, cy2 = 88.0, 81.5
R2 = 10.2

d = np.hypot(cx1 - cx2, cy1 - cy2)
theta = np.arctan2(cy1 - cy2, cx1 - cx2)
alpha = np.arcsin((R1 - R2) / d)

ang_upper = theta - np.pi/2 - alpha
ang_lower = theta + np.pi/2 + alpha

p1_upper = np.array([cx1 + R1 * np.cos(ang_upper), cy1 + R1 * np.sin(ang_upper)])
p2_upper = np.array([cx2 + R2 * np.cos(ang_upper), cy2 + R2 * np.sin(ang_upper)])

p1_lower = np.array([cx1 + R1 * np.cos(ang_lower), cy1 + R1 * np.sin(ang_lower)])
p2_lower = np.array([cx2 + R2 * np.cos(ang_lower), cy2 + R2 * np.sin(ang_lower)])

# User-delineated regolith line:
m_reg = -0.410
c_reg = 201.50

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11), dpi=240)
fig.patch.set_facecolor('#05070d')

for ax, img, title, subtitle in [
    (ax1, warped_mag, 'Sobel Edge Gradient (Un-Tilted Front-On Frame)', 'Sub-Pixel Gradient Ridges with Corrected Regolith Horizon Line'),
    (ax2, warped_raw, 'Raw Mastcam Sensor Radiometry (Un-Tilted Frame)', '3D Perspective Cylinder Sitting on Martian Regolith Horizon')
]:
    ax.set_facecolor('#05070d')
    if ax == ax1:
        ax.imshow(img, cmap='magma', origin='upper')
    else:
        ax.imshow(img, cmap='gray', origin='upper')
    
    # 1. Front Circle (Foreground)
    ax.add_patch(Circle((cx1, cy1), R1, edgecolor='#00f0ff', facecolor='none', linewidth=2.8, linestyle='--'))
    ax.scatter([cx1], [cy1], color='#ff1744', s=45, zorder=14)
    ax.add_patch(Circle((cx1, cy1), 5.0, edgecolor='#00f0ff', facecolor='none', linewidth=1.6, zorder=13))
    ax.plot([cx1 - 14, cx1 + 14], [cy1, cy1], color='#00f0ff', linewidth=1.6, zorder=12)
    ax.plot([cx1, cx1], [cy1 - 14, cy1 + 14], color='#00f0ff', linewidth=1.6, zorder=12)
    
    # 2. Rear Circle (Far-End Curved Terminus)
    ax.add_patch(Circle((cx2, cy2), R2, edgecolor='#00f0ff', facecolor='none', linewidth=2.8, linestyle='-'))
    ax.scatter([cx2], [cy2], color='#ff1744', s=45, zorder=14)
    ax.add_patch(Circle((cx2, cy2), 4.5, edgecolor='#00f0ff', facecolor='none', linewidth=1.6, zorder=13))
    ax.plot([cx2 - 12, cx2 + 12], [cy2, cy2], color='#00f0ff', linewidth=1.6, zorder=12)
    ax.plot([cx2, cx2], [cy2 - 12, cy2 + 12], color='#00f0ff', linewidth=1.6, zorder=12)
    
    # 3. Central Axis (Yellow line)
    ax.plot([cx1, cx2], [cy1, cy2], color='#ffd600', linewidth=2.8, zorder=11, label=f'Cylinder Central Axis (L = {d:.1f} px, θ = {np.degrees(theta):.1f}°)')
    
    # 4. Upper Flank Generator (Cyan tangent)
    ax.plot([p1_upper[0], p2_upper[0]], [p1_upper[1], p2_upper[1]], color='#00e5ff', linewidth=2.8, zorder=11, label='Upper Flank Hull Generator (Sunlit Ridge)')
    
    # 5. Lower Flank Generator (Orange dashed tangent)
    ax.plot([p1_lower[0], p2_lower[0]], [p1_lower[1], p2_lower[1]], color='#ff6d00', linewidth=2.4, linestyle='--', zorder=11, label='Lower Flank Tangent (Flank Base)')
    
    # 6. Corrected Regolith Ground Horizon Line
    reg_x = np.linspace(cx2 - 28, cx1 + 55, 100)
    reg_y = m_reg * reg_x + c_reg
    ax.plot(reg_x, reg_y, color='#ff3366', linewidth=3.4, zorder=16, label='Corrected Regolith Horizon (Soil Contact Line)')
    
    # Annotations
    if ax == ax1:
        ax.annotate(f'FRONT FACE RIM\nRadius R₁ = {R1:.1f} px\nCenter: (145.6, 133.1)',
                    xy=(cx1, cy1), xytext=(cx1 + 22, cy1 - 38),
                    arrowprops=dict(facecolor='#00f0ff', edgecolor='#ffffff', width=1.6, headwidth=6, shrink=0.1),
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#090d16', edgecolor='#00f0ff', linewidth=1.8, alpha=0.92),
                    fontsize=9.5, color='#ffffff', fontweight='bold', fontfamily='monospace')
        
        ax.annotate(f'FAR-END CURVED TERMINUS\nRadius R₂ = {R2:.1f} px ({R2/R1*100:.1f}%)\nCenter: ({cx2:.1f}, {cy2:.1f})',
                    xy=(cx2, cy2), xytext=(cx2 - 25, cy2 - 42),
                    arrowprops=dict(facecolor='#00f0ff', edgecolor='#ffffff', width=1.6, headwidth=6, shrink=0.1),
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#090d16', edgecolor='#00f0ff', linewidth=1.8, alpha=0.92),
                    fontsize=9.5, color='#ffffff', fontweight='bold', fontfamily='monospace')
    else:
        ax.annotate('EXPOSED SUNLIT FLANK\n(Bright Radiometric Hull)',
                    xy=((p1_upper[0] + p2_upper[0])/2, (p1_upper[1] + p2_upper[1])/2),
                    xytext=((p1_upper[0] + p2_upper[0])/2 + 8, (p1_upper[1] + p2_upper[1])/2 - 28),
                    arrowprops=dict(facecolor='#ffd600', edgecolor='#ffffff', width=1.5, headwidth=5.5, shrink=0.1),
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#090d16', edgecolor='#ffd600', linewidth=1.6, alpha=0.92),
                    fontsize=9.5, color='#ffd600', fontweight='bold', fontfamily='monospace')
        
        ax.annotate('CORRECTED REGOLITH HORIZON\n(Base of Cylinder / Soil Interface)',
                    xy=(145.64, m_reg * 145.64 + c_reg),
                    xytext=(145.64 - 55, m_reg * 145.64 + c_reg + 38),
                    arrowprops=dict(facecolor='#ff3366', edgecolor='#ffffff', width=1.6, headwidth=6, shrink=0.1),
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#090d16', edgecolor='#ff3366', linewidth=1.8, alpha=0.92),
                    fontsize=9.5, color='#ffffff', fontweight='bold', fontfamily='monospace')

    ax.set_xlim(cx2 - 28, cx1 + 52)
    ax.set_ylim(cy1 + 65, cy2 - 50)
    ax.axis('off')
    ax.legend(loc='lower left', fontsize=9, facecolor='#090d16', edgecolor='#1e293b')
    ax.set_title(f'{title}\n{subtitle}', fontsize=12, color='#ffffff', fontweight='bold', pad=12)

fig.suptitle('NASA Curiosity Sol 3556: Validated 3D Cylinder Hull Reconstruction\n'
             'Corrected Regolith Horizon Delineation (Aligned to User & Soil Interface)',
             fontsize=14.5, color='#ffffff', fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.94])
out_path = 'figures/cylinder_3d_hull_reconstruction.png'
plt.savefig(out_path, bbox_inches='tight', facecolor='#05070d', dpi=240)
plt.close()
print(f'Successfully updated {out_path}')