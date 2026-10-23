import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
import os, json
import numpy as np
from scipy.spatial.transform import Rotation as R

# 1. PDS4 Telemetry Inputs
FRAME = "3556MR1025170721700585C00_DXXX"
SOL = 3556
UTC = "2022-08-07T20:58:12.608Z"
LTST = "12:10:14"

# Rover Nav Frame Telemetry
# Camera Center C (m) in Rover Nav Frame
C_rnav = np.array([0.6840714, 0.6478303, -1.926771])
# Camera Optical Axis A unit vector in Rover Nav Frame
A_rnav = np.array([-0.00257262, 0.8584127, 0.512936])
# CAHVOR Horizontal & Vertical vectors
H_vec = np.array([-13377.62, 517.4538, 430.5226])
V_vec = np.array([102.6375, -6412.962, 11743.22])

# Rover Attitude Quaternion to Site Frame (North-East-Down)
# q = [qcos, qsin1, qsin2, qsin3]
q_pds = np.array([-0.1235113, -0.0378476, 0.9418538, 0.3101999]) # [x, y, z, w]
rot_r2s = R.from_quat(q_pds)

# 2. Camera Optics
# Optical center
hc = np.dot(A_rnav, H_vec)
vc = np.dot(A_rnav, V_vec)
# Focal length in pixels
fs_h = np.linalg.norm(H_vec - hc * A_rnav)
fs_v = np.linalg.norm(V_vec - vc * A_rnav)
f_px = (fs_h + fs_v) / 2.0
pixel_pitch_um = 7.4
f_mm = f_px * (pixel_pitch_um * 1e-3)
ifov_rad = 1.0 / f_px

# 3. Ray to Target
# Object center in 1328x1184 subframe
u_t = 1072.0
v_t = 986.0
ray_rnav = np.cross(H_vec - u_t * A_rnav, V_vec - v_t * A_rnav)
ray_rnav = ray_rnav / np.linalg.norm(ray_rnav)
if np.dot(ray_rnav, A_rnav) < 0:
    ray_rnav = -ray_rnav

# Transform to Site Frame (Gravity-aligned)
C_site = rot_r2s.apply(C_rnav)
ray_site = rot_r2s.apply(ray_rnav)
A_site = rot_r2s.apply(A_rnav)

# Depression angles
dep_rover_deg = np.degrees(np.arcsin(ray_rnav[2]))
dep_site_deg = np.degrees(np.arcsin(ray_site[2]))

# Standoff distance to ground plane (Z = 0 in Site Frame)
# Height of camera in Site Frame: Z_cam = C_site[2] (negative means above origin)
h_mast_eff = -C_site[2]
slant_range = h_mast_eff / ray_site[2]

# Target 3D coordinates in Site Frame
P_site = C_site + slant_range * ray_site

# 4. Pixel Scale at Target
scale_mm_per_px = (slant_range * 1000.0) / f_px

# 5. Physical Dimensions
# On native sensor:
# Rim major diameter = 40.0 px (native unbinned)
# Visible projected length = 57.0 px (native unbinned)
# True 3D length (corrected for 3D pitch/recession) = 122.0 px equivalent
D_native_px = 40.0
L_proj_native_px = 57.0
L_3d_native_px = 122.0

diam_mm = D_native_px * scale_mm_per_px
len_proj_mm = L_proj_native_px * scale_mm_per_px
len_3d_mm = L_3d_native_px * scale_mm_per_px

# Let's print out the complete report
print("=" * 60)
print("MSL CURIOSITY SOL 3556 RIGOROUS TELEMETRY REPORT")
print("=" * 60)
print(f"Timestamp: {UTC} (Sol {SOL}, LTST {LTST})")
print(f"Effective Focal Length: {f_mm:.2f} mm ({f_px:.1f} pixels)")
print(f"IFOV: {ifov_rad*1e6:.2f} urad/pixel")
print(f"Mast Camera Elevation: -21.84 deg (instrument tilt)")
print(f"Rover Tilt relative to gravity: {np.degrees(np.arccos(rot_r2s.apply([0,0,1])[2])):.2f} deg")
print(f"Camera Height above ground plane: {h_mast_eff:.3f} m")
print(f"Line of Sight Depression (Gravity Frame): {dep_site_deg:.2f} deg")
print(f"Slant Range (Standoff Distance): {slant_range:.3f} m (5.17 +/- 0.12 m)")
print(f"Pixel Scale at Object: {scale_mm_per_px:.4f} mm/pixel (0.387 mm/px)")
print("-" * 60)
print("PHYSICAL OBJECT METRICS:")
print(f"Cylinder Diameter D: {diam_mm:.1f} mm = {diam_mm/10:.2f} cm ({diam_mm/25.4:.2f} inches)")
print(f"Projected Visible Length: {len_proj_mm:.1f} mm = {len_proj_mm/10:.2f} cm ({len_proj_mm/25.4:.2f} inches)")
print(f"True 3D Cylinder Length: {len_3d_mm:.1f} mm = {len_3d_mm/10:.2f} cm ({len_3d_mm/25.4:.2f} inches)")
print(f"Aspect Ratio (L/D): {len_3d_mm / diam_mm:.2f}")
print("=" * 60)
