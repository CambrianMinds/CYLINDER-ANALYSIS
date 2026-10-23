import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
import numpy as np

# Camera center in Site frame:
# C_site = [-0.4105, -0.1255, -2.1014] m
# Target ray unit vector in Site frame:
# r_site = [-0.5982, -0.6905, 0.4066]
# Sunlight unit vector (pointing towards ground):
# L = [0.3425, 0.0409, 0.9386]

C_site = np.array([-0.41051675, -0.12545922, -2.10138532])
r_site = np.array([-0.59824404, -0.69049735,  0.40659253])
L = np.array([0.34249279, 0.04094225, 0.93862795])

# In cv_cylinder_analysis:
# shadow_bearing = (sun_az + 180 - cam_az) % 360
# Expected shadow length / object height on horizontal ground:
# tan(el) = tan(69.8224 deg) = 2.7210
# Length/Height = 1 / 2.7210 = 0.3675

# Now let's calculate:
# If the target is at distance s along ray r_site:
# P(s) = C_site + s * r_site
# At s = 5.168 m (Z_site = 0.0 m):
# P = [-3.503, -3.694, 0.000] m
# Distance from rover nav origin in horizontal plane: sqrt(3.503^2 + 3.694^2) = 5.09 m

# Sensor parameters:
# Focal length f_px = 13376 native pixels
# Native pixel pitch = 7.4 um
# Physical focal length = 98.98 mm
# IFOV = 1 / 13376 = 7.476e-5 rad/px

print(f"IFOV = {1/13376*1e6:.2f} urad/px")

# At distance s, the physical width corresponding to 1 native pixel perpendicular to line-of-sight:
# dx = s * IFOV = s / 13376
for s in [4.8, 5.0, 5.17, 5.3, 5.5]:
    px_scale = (s * 1000.0) / 13376.0 # mm/px
    # Cylinder major diameter = 53.8 px
    # Visible length = 122.0 px
    diam_cm = 53.8 * px_scale / 10.0
    len_cm = 122.0 * px_scale / 10.0
    print(f"Distance {s:.2f} m: Scale = {px_scale:.3f} mm/px -> Diam = {diam_cm:.2f} cm, Length = {len_cm:.2f} cm")
