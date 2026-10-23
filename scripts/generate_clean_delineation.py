import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
import cv2, numpy as np, matplotlib.pyplot as plt

raw = cv2.imread('figures/cylinder_crop_raw.png')
H, W, _ = raw.shape

# Mathematical parameters in raw crop space
# Front ellipse: cx=85.64, cy=73.14, a=26.91, b=22.60 (or a=25.96, b=16.45)
# Let's use the precise ellipse contour:
# We know from our transform:
# Center 1: (85.64, 73.14)
# Center 2: (38.57, 27.37)
# Upper tangent: (43.56, 18.48) -> (98.82, 49.68)
# Lower tangent: (30.89, 33.66) -> (65.39, 89.73)

# Create masks for the cylinder and shadow:
mask_cylinder = np.zeros((H, W), dtype=np.uint8)
mask_shadow = np.zeros((H, W), dtype=np.uint8)

# 1. Cylinder polygon in raw crop:
# Hull is enclosed by rear cap, upper tangent, front rim, and lower tangent:
# Let's generate points along rear cap arc, upper tangent, front rim arc, and lower tangent:
th_rear = np.linspace(np.arctan2(18.48 - 27.37, 43.56 - 38.57),
                      np.arctan2(33.66 - 27.37, 30.89 - 38.57) + 2*np.pi, 50)
# Rear cap is an ellipse with semi-axes ~10.2 / stretch
rear_pts = []
for t in np.linspace(1.8, 4.8, 30):
    # rear arc
    rx = 38.57 + 8.5 * np.cos(t) - 4.5 * np.sin(t)
    ry = 27.37 + 5.5 * np.cos(t) + 7.5 * np.sin(t)
    rear_pts.append([rx, ry])

# Front rim ellipse:
front_pts = []
phi = np.radians(53.96)
a_front = 25.96
b_front = 16.45
for t in np.linspace(-np.pi/2, np.pi/2, 50):
    ex = a_front * np.cos(t)
    ey = b_front * np.sin(t)
    # rotate by phi
    fx = 85.64 + ex * np.cos(phi) - ey * np.sin(phi)
    fy = 73.14 + ex * np.sin(phi) + ey * np.cos(phi)
    front_pts.append([fx, fy])

# Combine into precise hull polygon:
# Upper tangent: from ~ (43, 18) to (99, 50)
# Lower tangent: from (65, 90) to (31, 34)
poly_cyl = np.array([
    [38, 27], [40, 21], [44, 18], [60, 27], [80, 38], [99, 50],
    [104, 58], [107, 68], [105, 78], [98, 86], [86, 90], [74, 91], [65, 89],
    [52, 73], [42, 55], [33, 38], [31, 33], [33, 29]
], dtype=np.int32)

cv2.fillPoly(mask_cylinder, [poly_cyl], 255)

# 2. Shadow polygon:
# Dark shadow cast to the right of the front face:
# Pixels in raw crop with low DN to the right of front rim:
gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
shadow_dark = (gray < 85) & (gray > 20)
# Region of interest for shadow: X between 90 and 155, Y between 45 and 95
roi_shadow = np.zeros_like(shadow_dark)
roi_shadow[45:95, 95:155] = True
mask_shadow_raw = shadow_dark & roi_shadow

# Clean shadow mask with morphological closing
kernel = np.ones((3, 3), np.uint8)
mask_shadow = cv2.morphologyEx(mask_shadow_raw.astype(np.uint8)*255, cv2.MORPH_CLOSE, kernel)
# Smooth contours
contours, _ = cv2.findContours(mask_shadow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask_shadow_clean = np.zeros_like(mask_shadow)
if contours:
    c_max = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask_shadow_clean, [c_max], -1, 255, -1)

# Ensure shadow doesn't overlap cylinder:
mask_shadow_clean[mask_cylinder > 0] = 0

# Create high-res professional visualization:
# Canvas with alpha overlay
overlay = raw.copy().astype(float)
# Red tint for cylinder:
color_cyl = np.array([25, 30, 255], dtype=float) # BGR: Bright Red/Coral
# Blue tint for shadow:
color_shad = np.array([255, 140, 0], dtype=float) # BGR: Deep Sky Blue

alpha_cyl = 0.42
alpha_shad = 0.48

for c in range(3):
    overlay[:, :, c] = np.where(mask_cylinder > 0,
                                (1 - alpha_cyl) * overlay[:, :, c] + alpha_cyl * color_cyl[c],
                                overlay[:, :, c])
    overlay[:, :, c] = np.where(mask_shadow_clean > 0,
                                (1 - alpha_shad) * overlay[:, :, c] + alpha_shad * color_shad[c],
                                overlay[:, :, c])

overlay = np.clip(overlay, 0, 255).astype(np.uint8)

# Draw crisp neon boundary lines:
contours_cyl, _ = cv2.findContours(mask_cylinder, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(overlay, contours_cyl, -1, (40, 50, 255), 2, cv2.LINE_AA) # Red border

contours_sh, _ = cv2.findContours(mask_shadow_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(overlay, contours_sh, -1, (255, 190, 20), 2, cv2.LINE_AA) # Cyan/Blue border

# Save output
cv2.imwrite('figures/cylinder_red_shadow_blue_clean.png', overlay)
print('Generated cylinder_red_shadow_blue_clean.png successfully!')
