"""
Synthesize the 3D perspective shift straight down the cylinder's longitudinal axis,
with the camera oriented so the ground contact shadow is kept strictly at the BOTTOM (6 o'clock)
and the lit top hull / flank is at the TOP (12 o'clock).
"""
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec

workspace = r"d:\tmp\CYLINDER ANALYSIS"
raw = cv2.imread(os.path.join(workspace, "figures/cylinder_crop_raw.png"), cv2.IMREAD_GRAYSCALE)
clean = cv2.bilateralFilter(raw, d=5, sigmaColor=50, sigmaSpace=50)
H, W = raw.shape  # 140, 180

# 1. Sobel Gradient
gx = cv2.Sobel(clean, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(clean, cv2.CV_64F, 0, 1, ksize=3)
mag = np.hypot(gx, gy)

# 2. User's exact yellow line top plane generator:
p_tail = np.array([46.0, 20.3])
p_apex = np.array([97.9, 49.2])
d_hull = p_apex - p_tail
alpha_deg = float(np.degrees(np.arctan2(d_hull[1], d_hull[0])))  # 29.06 deg
alpha_rad = np.radians(alpha_deg)

# Cylinder face center and semi-axes:
cx, cy = 85.49, 71.48
a_fit = 26.47  # major semi-axis
b_fit = 21.25  # minor semi-axis
phi_deg = alpha_deg + 90.0  # 119.06 deg (strictly perpendicular to yellow line)

# Aspect ratio and tilt:
aspect_ratio = b_fit / a_fit  # 0.8028
theta_deg = float(np.degrees(np.arccos(aspect_ratio)))  # 36.60 deg
stretch_max = a_fit / b_fit  # 1.2457

# 3. Perspective shift de-projection function with shadow strictly at the bottom:
def orbit_view(img, t, out_dim=280, target_radius=75.0):
    current_stretch = 1.0 + t * (stretch_max - 1.0)
    current_scale = target_radius / a_fit
    
    # Align major axis to X, minor to Y:
    rad_align = np.radians(-phi_deg)
    R_align = np.array([
        [np.cos(rad_align), -np.sin(rad_align)],
        [np.sin(rad_align),  np.cos(rad_align)]
    ])
    
    # Scale and stretch:
    S = np.array([
        [current_scale, 0.0],
        [0.0, current_scale * current_stretch]
    ])
    
    # Display rotation of 180 deg puts lit top hull at 12 o'clock and shadow at 6 o'clock (BOTTOM):
    rad_disp = np.radians(180.0)
    R_disp = np.array([
        [np.cos(rad_disp), -np.sin(rad_disp)],
        [np.sin(rad_disp),  np.cos(rad_disp)]
    ])
    
    M_2x2 = R_disp @ S @ R_align
    offset = np.array([out_dim / 2.0, out_dim / 2.0]) - M_2x2 @ np.array([cx, cy])
    
    M_warp = np.zeros((2, 3), dtype=np.float32)
    M_warp[:2, :2] = M_2x2
    M_warp[:2, 2] = offset
    
    warped = cv2.warpAffine(img, M_warp, (out_dim, out_dim), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return warped, M_warp

# Generate the sequence of views:
view_33 = orbit_view(raw, 0.33)[0]
view_66 = orbit_view(raw, 0.66)[0]
view_100, M_final = orbit_view(raw, 1.0)
mag_100, _ = orbit_view(mag.astype(np.float32), 1.0)
clean_100, _ = orbit_view(clean, 1.0)

# 4. Generate Comprehensive Publication-Quality Presentation
plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 11), dpi=300)
fig.patch.set_facecolor('#07090e')
gs = GridSpec(2, 12, figure=fig, hspace=0.28, wspace=0.35)

# Top Row: 4 subplots spanning 3 columns each
ax1 = fig.add_subplot(gs[0, 0:3])
ax2 = fig.add_subplot(gs[0, 3:6])
ax3 = fig.add_subplot(gs[0, 6:9])
ax4 = fig.add_subplot(gs[0, 9:12])

# Bottom Row: 3 subplots spanning 4 columns each
ax5 = fig.add_subplot(gs[1, 0:4])
ax6 = fig.add_subplot(gs[1, 4:8])
ax7 = fig.add_subplot(gs[1, 8:12])

# Panel 1: Original View with User's Yellow Line and Red Arrow
ax1.set_facecolor('#07090e')
ax1.imshow(raw, cmap='gray', origin='upper', extent=[0, W, H, 0])
ax1.plot([p_tail[0], p_apex[0]], [p_tail[1], p_apex[1]], color='#ffff00', linewidth=2.5,
         label=f'Top Plane Generator ($\\alpha = {alpha_deg:.1f}^\\circ$)')
# Draw red curved arrow:
import matplotlib.patches as patches
style = "Simple, tail_width=2.5, head_width=8, head_length=8"
kw = dict(arrowstyle=style, color="#ff1744", linewidth=1.5)
arrow = patches.FancyArrowPatch((p_tail[0]-5, p_tail[1]), (p_apex[0]-5, p_tail[1]-12),
                               connectionstyle="arc3,rad=-.4", **kw)
ax1.add_patch(arrow)
ax1.text(35, 10, "Camera Swing\n(Red Arrow)", color='#ff1744', fontsize=9.5, fontweight='bold')
ax1.set_title("1. Oblique Perspective\n(User Yellow Line & Red Arrow)", fontsize=11, fontweight='bold', color='#e0e6ed')
ax1.set_xlim(0, W); ax1.set_ylim(H, 0)
ax1.legend(loc='lower left', fontsize=8, facecolor='#121824', edgecolor='#ffff00')

# Panel 2: 33% Shift
ax2.set_facecolor('#07090e')
ax2.imshow(view_33[20:260, 20:260], cmap='gray', origin='upper')
ax2.annotate("Top Hull (Up)", xy=(120, 45), color='#ffff00', fontsize=8, fontweight='bold', ha='center')
ax2.annotate("Shadow (Bottom)", xy=(120, 195), color='#40c4ff', fontsize=8, fontweight='bold', ha='center')
ax2.set_title(f"2. Perspective Shift: 33%\n$\\theta_{{eff}} = {theta_deg*0.33:.1f}^\\circ$", fontsize=11, fontweight='bold', color='#e0e6ed')
ax2.axis('off')

# Panel 3: 66% Shift
ax3.set_facecolor('#07090e')
ax3.imshow(view_66[20:260, 20:260], cmap='gray', origin='upper')
ax3.annotate("Top Hull (Up)", xy=(120, 45), color='#ffff00', fontsize=8, fontweight='bold', ha='center')
ax3.annotate("Shadow (Bottom)", xy=(120, 195), color='#40c4ff', fontsize=8, fontweight='bold', ha='center')
ax3.set_title(f"3. Perspective Shift: 66%\n$\\theta_{{eff}} = {theta_deg*0.66:.1f}^\\circ$", fontsize=11, fontweight='bold', color='#e0e6ed')
ax3.axis('off')

# Panel 4: 100% Head-On View (Looking Straight Down Longitudinal Axis)
ax4.set_facecolor('#07090e')
ax4.imshow(view_100[20:260, 20:260], cmap='gray', origin='upper')
circ4 = Circle((120, 120), radius=75.0, edgecolor='#00e5ff', facecolor='none', linewidth=2.5,
               label='Euclidean Circle ($R=75\\,\\mathrm{px}$)')
ax4.add_patch(circ4)
ax4.plot(120, 120, '+', color='#ff1744', markersize=10, markeredgewidth=2)
ax4.annotate("Top Hull (Up)", xy=(120, 45), color='#ffff00', fontsize=8, fontweight='bold', ha='center')
ax4.annotate("Shadow (Bottom)", xy=(120, 195), color='#40c4ff', fontsize=8, fontweight='bold', ha='center')
ax4.set_title(f"4. 100% Head-On View\n(Straight Down Axis: $\\theta = {theta_deg:.1f}^\\circ$)", fontsize=11, fontweight='bold', color='#00e5ff')
ax4.legend(loc='lower left', fontsize=8, facecolor='#121824', edgecolor='#00e5ff')
ax4.axis('off')

# Bottom Row: Large High-Resolution Head-On Analysis Products (Shadow at Bottom!)
# Panel 5: Large Raw Head-On Face
ax5.set_facecolor('#07090e')
ax5.imshow(view_100, cmap='gray', origin='upper')
circ5 = Circle((140, 140), radius=75.0, edgecolor='#00e5ff', facecolor='none', linewidth=2.5,
               label='True Circle ($R=75\\,\\mathrm{px}$)')
ax5.add_patch(circ5)
ax5.plot(140, 140, '+', color='#ff1744', markersize=10, markeredgewidth=2)
ax5.annotate("Lit Top Hull (Up - 12 o'clock)", xy=(140, 65), xytext=(140, 32),
             ha='center', color='#ffff00', fontsize=10, fontweight='bold',
             arrowprops=dict(facecolor='#ffff00', edgecolor='none', shrink=0.08, width=1.5, headwidth=5))
ax5.annotate("Ground Contact Shadow (Bottom - 6 o'clock)", xy=(140, 215), xytext=(140, 252),
             ha='center', color='#40c4ff', fontsize=10, fontweight='bold',
             arrowprops=dict(facecolor='#40c4ff', edgecolor='none', shrink=0.08, width=1.5, headwidth=5))
ax5.set_title("5. Raw Sensor Head-On View (Shadow Kept at Bottom)", fontsize=12, fontweight='bold', color='#00e5ff')
ax5.legend(loc='lower left', fontsize=8.5, facecolor='#121824', edgecolor='#00e5ff')
ax5.set_xlim(10, 270); ax5.set_ylim(270, 10)

# Panel 6: Sobel Gradient Magnitude Head-On
ax6.set_facecolor('#07090e')
ax6.imshow(mag_100, cmap='magma', origin='upper')
circ6 = Circle((140, 140), radius=75.0, edgecolor='#00e5ff', facecolor='none', linewidth=2.5, linestyle='--',
               label='Conformal Circular Rim')
ax6.add_patch(circ6)
ax6.annotate("Upper Rim Arc (Sunlit)", xy=(140, 65), xytext=(140, 32),
             ha='center', color='#00e5ff', fontsize=9.5, fontweight='bold',
             arrowprops=dict(facecolor='#00e5ff', edgecolor='none', shrink=0.08, width=1.2, headwidth=4))
ax6.annotate("Lower Rim Arc (Shadow Boundary)", xy=(140, 215), xytext=(140, 252),
             ha='center', color='#00e5ff', fontsize=9.5, fontweight='bold',
             arrowprops=dict(facecolor='#00e5ff', edgecolor='none', shrink=0.08, width=1.2, headwidth=4))
ax6.set_title("6. Sobel Gradient Magnitude (Head-On Conformal Rim)", fontsize=12, fontweight='bold', color='#e0e6ed')
ax6.legend(loc='lower left', fontsize=8.5, facecolor='#121824', edgecolor='#00e5ff')
ax6.set_xlim(10, 270); ax6.set_ylim(270, 10)

# Panel 7: Bilateral Denoised Head-On
ax7.set_facecolor('#07090e')
ax7.imshow(clean_100, cmap='gray', origin='upper')
circ7 = Circle((140, 140), radius=75.0, edgecolor='#76ff03', facecolor='none', linewidth=2.5,
               label='Head-On Rim Boundary')
ax7.add_patch(circ7)
ax7.annotate("Top (12 o'clock)", xy=(140, 65), xytext=(140, 32),
             ha='center', color='#76ff03', fontsize=9.5, fontweight='bold',
             arrowprops=dict(facecolor='#76ff03', edgecolor='none', shrink=0.08, width=1.2, headwidth=4))
ax7.annotate("Shadow (6 o'clock)", xy=(140, 215), xytext=(140, 252),
             ha='center', color='#76ff03', fontsize=9.5, fontweight='bold',
             arrowprops=dict(facecolor='#76ff03', edgecolor='none', shrink=0.08, width=1.2, headwidth=4))
ax7.set_title("7. Bilateral Denoised Head-On Face (Shadow at Bottom)", fontsize=12, fontweight='bold', color='#76ff03')
ax7.legend(loc='lower left', fontsize=8.5, facecolor='#121824', edgecolor='#76ff03')
ax7.set_xlim(10, 270); ax7.set_ylim(270, 10)

plt.suptitle("NASA Curiosity Sol 3556: 3D Perspective Shift Straight Down Longitudinal Axis\nTop Plane Angle Basis ($\\alpha = 29.1^\\circ$) | Ground Contact Shadow Kept Strictly at Bottom (6 o'clock)",
             fontsize=14, fontweight='bold', color='#ffffff', y=0.98)

out_fig = os.path.join(workspace, "figures/camera_orbit_straight_down_axis.png")
plt.savefig(out_fig, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print(f"Saved {out_fig}")

# Also save standalone high-res images:
out_headon_raw = os.path.join(workspace, "face_straight_down_axis_raw.png")
cv2.imwrite(out_headon_raw, view_100[20:260, 20:260])
print(f"Saved {out_headon_raw}")

out_headon_clean = os.path.join(workspace, "face_straight_down_axis_clean.png")
cv2.imwrite(out_headon_clean, clean_100[20:260, 20:260])
print(f"Saved {out_headon_clean}")

out_headon_sobel = os.path.join(workspace, "face_straight_down_axis_sobel.png")
mag_norm = np.clip(mag_100[20:260, 20:260] / 400.0 * 255.0, 0, 255).astype(np.uint8)
cv2.imwrite(out_headon_sobel, cv2.applyColorMap(mag_norm, cv2.COLORMAP_MAGMA))
print(f"Saved {out_headon_sobel}")
