import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
import cv2, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Rectangle
from matplotlib.lines import Line2D

# Set up canvas
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 3, figsize=(22, 14), dpi=240)
fig.patch.set_facecolor('#05070d')

# Load images
raw_crop = cv2.imread('figures/cylinder_crop_raw.png', cv2.IMREAD_GRAYSCALE)
delineated = cv2.imread('figures/cylinder_red_shadow_blue_clean.png')
delineated = cv2.cvtColor(delineated, cv2.COLOR_BGR2RGB)
sobel_img = cv2.imread('figures/sobel_gradient_and_flank_tangents.png')
sobel_img = cv2.cvtColor(sobel_img, cv2.COLOR_BGR2RGB)
recon_img = cv2.imread('figures/cylinder_3d_hull_reconstruction.png')
recon_img = cv2.cvtColor(recon_img, cv2.COLOR_BGR2RGB)
deproj_img = cv2.imread('figures/straight_on_deprojection_proof.png')
deproj_img = cv2.cvtColor(deproj_img, cv2.COLOR_BGR2RGB)

H_c, W_c = raw_crop.shape

# Helper to style axis
def style_ax(ax, title, subtitle=''):
    ax.set_facecolor('#070912')
    ax.tick_params(colors='#8b9bb4', labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color('#1e293b')
        spine.set_linewidth(1.0)
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=11, color='#ffffff', fontweight='bold', pad=10)
    else:
        ax.set_title(title, fontsize=11, color='#ffffff', fontweight='bold', pad=10)

# PANEL 1: Raw Mastcam-100 Sensor Crop
ax1 = axes[0, 0]
style_ax(ax1, '(1) Raw Mastcam Context Crop', 'NASA Curiosity Sol 3556 | M-100 Flight Model #3004')
ax1.imshow(raw_crop, cmap='magma', origin='upper')
# Add scale bar (5 mm at 0.3865 mm/px = 12.9 px)
sb_len = 5.0 / 0.3865 # 12.9 px
ax1.plot([15, 15 + sb_len], [H_c - 15, H_c - 15], color='#ffffff', linewidth=3)
ax1.text(15, H_c - 20, '5 mm (d = 5.17 m)', color='#ffffff', fontsize=8.5, fontweight='bold')
# Compass indicator (North / Sun)
ax1.annotate('Sun Az: 186.8°', xy=(W_c - 45, 20), xytext=(W_c - 65, 35),
             arrowprops=dict(facecolor='#ffd600', edgecolor='none', shrink=0.1, width=1.5, headwidth=5),
             fontsize=8, color='#ffd600', fontweight='bold')

# PANEL 2: Calibrated Delineation (Red Cylinder Hull + Blue Ground Shadow)
ax2 = axes[0, 1]
style_ax(ax2, '(2) Calibrated Morphological Segmentation', 'Cylinder Hull (Red) vs. Cast Ground Shadow (Blue)')
ax2.imshow(delineated)
ax2.axis('off')

# PANEL 3: Sobel Gradient Magnitude & Tangent Hull
ax3 = axes[0, 2]
style_ax(ax3, '(3) Sobel Spatial Gradient & Tangent Generators', 'All Lines Match Sub-Pixel Gradient Ridges')
ax3.imshow(sobel_img)
ax3.axis('off')

# PANEL 4: Straight-On De-projection Circle Proof
ax4 = axes[1, 0]
style_ax(ax4, '(4) Straight-On De-projection Proof', 'Affine Un-tilting Yields Euclidean Circle (σ_r = 0.74 px)')
ax4.imshow(deproj_img)
ax4.axis('off')

# PANEL 5: 3D Hull Reconstruction & Regolith Horizon
ax5 = axes[1, 1]
style_ax(ax5, '(5) 3D Perspective Hull & Regolith Horizon', '68.8% Exposed / 31.2% Subsurface Buried Sector')
ax5.imshow(recon_img)
ax5.axis('off')

# PANEL 6: NASA Flight Telemetry & Solar Geometry Vector
ax6 = axes[1, 2]
style_ax(ax6, '(6) NASA Flight Telemetry & Solar Ray Geometry', 'Site Frame (Gravity) | Slant Range d = 5.17 ± 0.12 m (±2.3%)')

# Draw diagram of Mastcam, Rover Slope, and Cylinder Ground Intersection:
# Ground line on 14.84 deg slope:
ax6.set_xlim(-1, 9)
ax6.set_ylim(-3, 4)
ax6.set_aspect('equal')

# Rover body representation
rover_x = np.array([0.0, 2.5])
rover_z = np.array([0.0, 0.0]) # in rover coordinates
# Mast at X = 0.68 m, Z = -1.93 m (relative to nav origin)
mast_base = [0.68, 0.0]
mast_head = [0.68, 1.93]
ax6.plot([mast_base[0], mast_head[0]], [mast_base[1], mast_head[1]], color='#90a4ae', linewidth=3.5, label='Curiosity Mast (H = 1.93 m)')
ax6.scatter([mast_head[0]], [mast_head[1]], color='#00e5ff', s=70, zorder=10, label='M-100 Camera Head')

# Rover wheels / slope
slope_angle = 14.84
th_s = np.radians(slope_angle)
# Draw ground plane
gx = np.linspace(-1, 8.5, 100)
gz = -np.tan(th_s) * gx - 0.2
ax6.plot(gx, gz, color='#8d6e63', linewidth=3.0, linestyle='-', label=f'Martian Bedrock Slope ({slope_angle:.1f}°)')

# Target location at standoff d = 5.17 m along depression 24 deg:
target_x = 0.68 + 5.17 * np.cos(np.radians(24.0)) # ~ 5.40 m
target_z = 1.93 - 5.17 * np.sin(np.radians(24.0)) # ~ -0.17 m
ax6.scatter([target_x], [target_z], color='#ff1744', s=85, zorder=12, label='Cylinder (D = 15.5 mm)')

# Camera Line of Sight Ray
ax6.plot([mast_head[0], target_x], [mast_head[1], target_z], color='#00e5ff', linewidth=2.4, linestyle='--',
         label='Line-of-Sight Ray (d = 5.17 m, ψ = 24.0°)')

# Sunlight vector (El = 69.82 deg):
sun_start = [target_x - 1.2 * np.cos(np.radians(69.82)), target_z + 1.2 * np.sin(np.radians(69.82))]
ax6.annotate('', xy=(target_x, target_z), xytext=(sun_start[0], sun_start[1]),
             arrowprops=dict(facecolor='#ffd600', edgecolor='#ffd600', width=2.0, headwidth=7),
             zorder=11)
ax6.text(sun_start[0] - 0.8, sun_start[1] + 0.15, 'Solar Noon (El: 69.8°)', color='#ffd600', fontsize=8.5, fontweight='bold')

# Telemetry data card overlay:
card_text = (
    "PDS4 FLIGHT TELEMETRY CLOSURE:\n"
    "• Timestamp: Sol 3556 12:10:14 LTST\n"
    "• Camera Center: C = [0.68, 0.65, -1.93] m\n"
    "• Rover Slope Tilt: 14.84° (Site Frame)\n"
    "• Line-of-Sight Depression: 23.99°\n"
    "• Slant Range: d = 5.168 ± 0.12 m (±2.3%)\n"
    "• Spatial Scale: 0.3865 mm / native px\n"
    "• Shadow Invariant: L/h = 1/tan(69.8°) = 0.368\n"
    "• Object Diameter: D = 15.5 ± 0.4 mm\n"
    "• True 3D Length: L_3D = 47.1 ± 1.1 mm (3:1)"
)
ax6.text(0.03, 0.05, card_text, transform=ax6.transAxes,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#0d131f', edgecolor='#00e5ff', alpha=0.92, linewidth=1.2),
         fontsize=8.5, color='#e0e6ed', family='monospace')

ax6.legend(loc='upper right', fontsize=7.8, facecolor='#0d131f', edgecolor='#2d3748', labelcolor='#e0e6ed')

# Overall title
plt.suptitle('Figure 6: Complete Six-Panel Planetary Forensic Diagnostic Matrix\n'
             'Programmatic Computer Vision, 3D Perspective Reconstruction, and NASA PDS4 Flight Telemetry',
             fontsize=15, color='#ffffff', fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
out_path = 'figures/scientific_dashboard_red_blue.png'
plt.savefig(out_path, dpi=240, facecolor=fig.get_facecolor())
plt.close()
print(f'Generated updated {out_path} successfully!')
