"""
Generate rigorous, accurate front-on (straight-on) de-projection views of the
Sol 3556 Martian Cylinder face using the top plane / flank angle as the geometric basis.
Orientation strictly locked so the ground contact shadow is at the BOTTOM (6 o'clock)
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
from matplotlib.patches import Ellipse, Circle
from scipy.optimize import least_squares

workspace = r"d:\tmp\CYLINDER ANALYSIS"
raw = cv2.imread(os.path.join(workspace, "figures/cylinder_crop_raw.png"), cv2.IMREAD_GRAYSCALE)
clean = cv2.bilateralFilter(raw, d=5, sigmaColor=50, sigmaSpace=50)
H, W = raw.shape  # 140, 180

# 1. Compute Sobel Gradient
gx = cv2.Sobel(clean, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(clean, cv2.CV_64F, 0, 1, ksize=3)
mag = np.hypot(gx, gy)

# 2. Geometric Basis: User's confirmed yellow line along top plane / upper hull
# Start point: (46.0, 20.3), End point: (97.9, 49.2)
p_tail = np.array([46.0, 20.3])
p_apex = np.array([97.9, 49.2])
d_hull = p_apex - p_tail
alpha_deg = float(np.degrees(np.arctan2(d_hull[1], d_hull[0])))  # 29.06 deg
alpha_rad = np.radians(alpha_deg)

# The cylinder front face is strictly perpendicular to the cylinder axis:
phi_deg = alpha_deg + 90.0  # 119.06 deg
phi_rad = np.radians(phi_deg)

# Extract Sobel ridge points along the bright rim
pts = []
for x in range(70, 106, 2):
    y = 45 + np.argmax(mag[45:60, x])
    if mag[y, x] > 200:
        pts.append((x, y))
for y in range(55, 95, 2):
    x = 55 + np.argmax(mag[y, 55:75])
    if mag[y, x] > 180:
        pts.append((x, y))
for x in range(70, 95, 2):
    y = 88 + np.argmax(mag[88:102, x])
    if mag[y, x] > 200:
        pts.append((x, y))
for y in range(52, 72, 2):
    x = 105 + np.argmax(mag[y, 105:112])
    if mag[y, x] > 250:
        pts.append((x, y))
pts = np.array(pts)

# Fit ellipse constrained strictly perpendicular to the top plane generator:
def resid(p):
    cx, cy, a, b = p
    cp, sp = np.cos(phi_rad), np.sin(phi_rad)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    u = dx * cp + dy * sp
    v = -dx * sp + dy * cp
    return (u / a)**2 + (v / b)**2 - 1.0

opt = least_squares(resid, [85.67, 71.57, 26.30, 21.54], bounds=([70, 55, 18, 12], [100, 85, 38, 30]))
cx, cy, a_fit, b_fit = opt.x
aspect_ratio = b_fit / a_fit
theta_deg = float(np.degrees(np.arccos(aspect_ratio)))
stretch_primary = a_fit / b_fit  # 1.2209

print(f"Top Plane Locked Ellipse: Center=({cx:.2f}, {cy:.2f}), a={a_fit:.2f}, b={b_fit:.2f}")
print(f"Aspect ratio b/a = {aspect_ratio:.4f}, Tilt theta = {theta_deg:.2f} deg, Stretch = {stretch_primary:.4f}")

# 3. Construct Front-On De-projection Function
# Locked with extra_rot = 180.0 so that:
# - Ground contact shadow is at the BOTTOM (6 o'clock, +Y)
# - Lit top hull / flank is at the TOP (12 o'clock, -Y)
def deproject(img, cx, cy, phi_deg, stretch, out_dim=300, target_radius=75.0, extra_rot=180.0):
    scale_overall = target_radius / a_fit
    
    # Step 1: Align ellipse major axis to X, minor axis to Y
    rad_align = np.radians(-phi_deg)
    R_align = np.array([
        [np.cos(rad_align), -np.sin(rad_align)],
        [np.sin(rad_align),  np.cos(rad_align)]
    ])
    
    # Step 2: Un-tilt foreshortening by expanding minor axis (Y) by stretch factor
    S = np.array([
        [scale_overall, 0.0],
        [0.0, scale_overall * stretch]
    ])
    
    # Step 3: 180 deg display rotation to place lit top hull at 12 o'clock and ground shadow at 6 o'clock
    rad_disp = np.radians(extra_rot)
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

# Generate front-on views
front_raw_primary, M_prim = deproject(raw, cx, cy, phi_deg, stretch_primary, 300, 75.0, extra_rot=180.0)
front_mag_primary, _ = deproject(mag.astype(np.float32), cx, cy, phi_deg, stretch_primary, 300, 75.0, extra_rot=180.0)
front_clean_primary, _ = deproject(clean, cx, cy, phi_deg, stretch_primary, 300, 75.0, extra_rot=180.0)

# Alternative comparisons:
# B. 3D ground plane photogrammetric basis (theta = 42.0 deg, stretch = 1.346)
stretch_3d = 1.0 / np.cos(np.radians(42.04))
front_raw_3d, _ = deproject(raw, cx, cy, phi_deg, stretch_3d, 300, 75.0, extra_rot=180.0)

# C. Direct top plane flank generator tilt basis (theta = 29.06 deg, stretch = 1.144)
stretch_flank = 1.0 / np.cos(np.radians(alpha_deg))
front_raw_flank, _ = deproject(raw, cx, cy, phi_deg, stretch_flank, 300, 75.0, extra_rot=180.0)

# 4. Generate Publication-Quality Multi-Panel Diagnostic Plot
plt.style.use('dark_background')
fig = plt.figure(figsize=(18, 12), dpi=300)
fig.patch.set_facecolor('#07090e')

# Panel 1: Oblique view with Top Plane Basis & Orthogonal Ellipse
ax1 = plt.subplot(2, 3, 1)
ax1.set_facecolor('#07090e')
ax1.imshow(raw, cmap='gray', origin='upper', extent=[0, W, H, 0])
# Draw user's top plane yellow line
ax1.plot([p_tail[0], p_apex[0]], [p_tail[1], p_apex[1]], color='#ffff00', linewidth=2.5,
         label=f'User Top Plane Line ($\\alpha = {alpha_deg:.1f}^\\circ$)')
# Draw locked orthogonal ellipse
ell = Ellipse((cx, cy), width=2*a_fit, height=2*b_fit, angle=phi_deg,
              edgecolor='#00e5ff', facecolor='none', linewidth=2.5,
              label=f'Orthogonal Face Ellipse\n($\\phi = {phi_deg:.1f}^\\circ \\perp \\text{{Top Plane}}$)')
ax1.add_patch(ell)
ax1.plot(cx, cy, '+', color='#ff1744', markersize=10, markeredgewidth=2)
ax1.annotate("Lit Top Hull (Up)", xy=(p_tail[0], p_tail[1]), xytext=(p_tail[0]-25, p_tail[1]-10),
             color='#ffff00', fontsize=8.5, fontweight='bold',
             arrowprops=dict(facecolor='#ffff00', edgecolor='none', shrink=0.05, width=1.2, headwidth=4))
ax1.annotate("Ground Contact Shadow (Bottom)", xy=(100, 88), xytext=(40, 115),
             color='#40c4ff', fontsize=8.5, fontweight='bold',
             arrowprops=dict(facecolor='#40c4ff', edgecolor='none', shrink=0.05, width=1.2, headwidth=4))
ax1.set_title("1. Oblique Perspective (Mastcam-100)\nTop Plane Basis $\\alpha = 29.1^\\circ$", fontsize=12, fontweight='bold', color='#e0e6ed')
ax1.set_xlim(0, W)
ax1.set_ylim(H, 0)
ax1.legend(loc='lower left', fontsize=8.0, facecolor='#121824', edgecolor='#00e5ff')

# Panel 2: Sobel Edge Gradient with Top Plane & Fitted Rim Ridge
ax2 = plt.subplot(2, 3, 2)
ax2.set_facecolor('#07090e')
im2 = ax2.imshow(mag, cmap='magma', origin='upper', extent=[0, W, H, 0])
ax2.plot([p_tail[0], p_apex[0]], [p_tail[1], p_apex[1]], color='#ffff00', linewidth=2.0, linestyle=':')
ax2.add_patch(Ellipse((cx, cy), width=2*a_fit, height=2*b_fit, angle=phi_deg,
                      edgecolor='#00e5ff', facecolor='none', linewidth=2.0, linestyle='--'))
ax2.scatter(pts[:, 0], pts[:, 1], color='#00e5ff', s=12, alpha=0.7, label='Orange/Pink Rim Ridge Pts')
ax2.set_title("2. Sobel Edge Gradient Magnitude\nRim Ridge Tracking ($>200\\,\\mathrm{DN/px}$)", fontsize=12, fontweight='bold', color='#e0e6ed')
ax2.set_xlim(0, W)
ax2.set_ylim(H, 0)
ax2.legend(loc='lower left', fontsize=8.5, facecolor='#121824', edgecolor='#00e5ff')

# Panel 3: Primary Front-On View (Raw Mastcam Grayscale) - Shadow at Bottom
ax3 = plt.subplot(2, 3, 3)
ax3.set_facecolor('#07090e')
ax3.imshow(front_raw_primary, cmap='gray', origin='upper')
circ3 = Circle((150, 150), radius=75.0, edgecolor='#00e5ff', facecolor='none', linewidth=2.5,
              label='True Euclidean Circle ($R=75\\,\\mathrm{px}$)')
ax3.add_patch(circ3)
ax3.plot(150, 150, '+', color='#ff1744', markersize=10, markeredgewidth=2)
# Explicit Up/Down Orientations:
ax3.annotate("Lit Top Hull (Up - 12 o'clock)", xy=(150, 75), xytext=(150, 40),
             ha='center', color='#ffff00', fontsize=9.5, fontweight='bold',
             arrowprops=dict(facecolor='#ffff00', edgecolor='none', shrink=0.08, width=1.5, headwidth=5))
ax3.annotate("Ground Shadow (Bottom - 6 o'clock)", xy=(150, 225), xytext=(150, 260),
             ha='center', color='#40c4ff', fontsize=9.5, fontweight='bold',
             arrowprops=dict(facecolor='#40c4ff', edgecolor='none', shrink=0.08, width=1.5, headwidth=5))
ax3.set_title(f"3. Front-On View: Face Normal\nTilt Shift $\\theta = {theta_deg:.1f}^\\circ$ (Shadow at Bottom)", fontsize=12, fontweight='bold', color='#00e5ff')
ax3.set_xlim(50, 250)
ax3.set_ylim(250, 50)
ax3.legend(loc='lower left', fontsize=8.0, facecolor='#121824', edgecolor='#00e5ff')

# Panel 4: Front-On View: Sobel Gradient Proof - Shadow at Bottom
ax4 = plt.subplot(2, 3, 4)
ax4.set_facecolor('#07090e')
ax4.imshow(front_mag_primary, cmap='magma', origin='upper')
circ4 = Circle((150, 150), radius=75.0, edgecolor='#00e5ff', facecolor='none', linewidth=2.5, linestyle='--',
               label='Euclidean Rim Circle')
ax4.add_patch(circ4)
ax4.annotate("Upper Rim Arc", xy=(150, 75), xytext=(150, 42),
             ha='center', color='#00e5ff', fontsize=9, fontweight='bold',
             arrowprops=dict(facecolor='#00e5ff', edgecolor='none', shrink=0.08, width=1.2, headwidth=4))
ax4.annotate("Lower Rim Arc (Ground Contact)", xy=(150, 225), xytext=(150, 258),
             ha='center', color='#00e5ff', fontsize=9, fontweight='bold',
             arrowprops=dict(facecolor='#00e5ff', edgecolor='none', shrink=0.08, width=1.2, headwidth=4))
ax4.set_title("4. Front-On Sobel Edge Gradient\nConformal Circular Rim Proof", fontsize=12, fontweight='bold', color='#e0e6ed')
ax4.set_xlim(50, 250)
ax4.set_ylim(250, 50)
ax4.legend(loc='lower left', fontsize=8.5, facecolor='#121824', edgecolor='#00e5ff')

# Panel 5: Bilateral Cleaned Front-On Face - Shadow at Bottom
ax5 = plt.subplot(2, 3, 5)
ax5.set_facecolor('#07090e')
ax5.imshow(front_clean_primary, cmap='gray', origin='upper')
circ5 = Circle((150, 150), radius=75.0, edgecolor='#76ff03', facecolor='none', linewidth=2.5,
               label='Head-On Rim Boundary')
ax5.add_patch(circ5)
ax5.plot(150, 150, '+', color='#ff1744', markersize=10, markeredgewidth=2)
ax5.annotate("Top (12 o'clock)", xy=(150, 75), xytext=(150, 42),
             ha='center', color='#76ff03', fontsize=9, fontweight='bold',
             arrowprops=dict(facecolor='#76ff03', edgecolor='none', shrink=0.08, width=1.2, headwidth=4))
ax5.annotate("Shadow (6 o'clock)", xy=(150, 225), xytext=(150, 258),
             ha='center', color='#76ff03', fontsize=9, fontweight='bold',
             arrowprops=dict(facecolor='#76ff03', edgecolor='none', shrink=0.08, width=1.2, headwidth=4))
ax5.set_title("5. Bilateral Filtered Head-On View\n(Shadow at Bottom, Lit Hull Up)", fontsize=12, fontweight='bold', color='#76ff03')
ax5.set_xlim(50, 250)
ax5.set_ylim(250, 50)
ax5.legend(loc='lower left', fontsize=8.5, facecolor='#121824', edgecolor='#76ff03')

# Panel 6: Direct Top Plane Tilt Basis (theta = 29.1 deg) - Shadow at Bottom
ax6 = plt.subplot(2, 3, 6)
ax6.set_facecolor('#07090e')
ax6.imshow(front_raw_flank, cmap='gray', origin='upper')
circ6 = Circle((150, 150), radius=75.0, edgecolor='#ffd54f', facecolor='none', linewidth=2.2,
               label='Circle ($R=75\\,\\mathrm{px}$)')
ax6.add_patch(circ6)
ax6.set_title(f"6. Direct Flank Shift Basis\n$\\theta = {alpha_deg:.1f}^\\circ$ (Shadow Kept at Bottom)", fontsize=12, fontweight='bold', color='#ffd54f')
ax6.set_xlim(50, 250)
ax6.set_ylim(250, 50)
ax6.legend(loc='lower left', fontsize=8.5, facecolor='#121824', edgecolor='#ffd54f')

plt.suptitle("NASA Curiosity Sol 3556 Mastcam-100: Rigorous Front-On Face De-projection\nTop Plane Angle Basis ($\\alpha = 29.1^\\circ$) | Ground Contact Shadow Kept Strictly at Bottom",
             fontsize=15, fontweight='bold', color='#ffffff', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

out_fig = os.path.join(workspace, "front_on_face_deprojection_matrix.png")
plt.savefig(out_fig, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print(f"Saved {out_fig}")

# Also save standalone high-res front-on views of the face (shadow strictly at bottom)
out_face_raw = os.path.join(workspace, "face_front_on_raw.png")
cv2.imwrite(out_face_raw, front_raw_primary[50:250, 50:250])
print(f"Saved {out_face_raw}")

out_face_clean = os.path.join(workspace, "face_front_on_clean.png")
cv2.imwrite(out_face_clean, front_clean_primary[50:250, 50:250])
print(f"Saved {out_face_clean}")

out_face_sobel = os.path.join(workspace, "face_front_on_sobel.png")
mag_norm = np.clip(front_mag_primary[50:250, 50:250] / 400.0 * 255.0, 0, 255).astype(np.uint8)
cv2.imwrite(out_face_sobel, cv2.applyColorMap(mag_norm, cv2.COLORMAP_MAGMA))
print(f"Saved {out_face_sobel}")
