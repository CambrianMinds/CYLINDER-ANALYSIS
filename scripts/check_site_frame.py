import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
import numpy as np
from scipy.spatial.transform import Rotation as R

# Quaternion from ROVER_NAV_FRAME to SITE_FRAME:
# In PDS: qcos, qsin1, qsin2, qsin3 (w, x, y, z)
# Rotation direction: Forward (transforms coordinates from Reference SITE to Present ROVER, or vice versa)
# Let's verify both directions:
q_r_to_s = np.array([-0.1235113, -0.0378476, 0.9418538, 0.3101999]) # x, y, z, w
rot = R.from_quat(q_r_to_s)

# Rover nav frame axes in Site Frame:
# Rover +X (forward), +Y (right), +Z (down relative to rover body)
x_rover_in_site = rot.apply([1, 0, 0])
y_rover_in_site = rot.apply([0, 1, 0])
z_rover_in_site = rot.apply([0, 0, 1])

print("Rover +X in Site frame:", x_rover_in_site)
print("Rover +Y in Site frame:", y_rover_in_site)
print("Rover +Z in Site frame:", z_rover_in_site)

# Rover pitch, roll, yaw relative to local level:
# If Site Frame is North, East, Down (NED):
# z_rover_in_site dot [0, 0, 1] is cos(total rover tilt)
cos_tilt = z_rover_in_site[2]
print(f"Total Rover Tilt relative to gravity: {np.degrees(np.arccos(cos_tilt)):.2f} deg")

# Camera optical axis A and target ray r in ROVER frame:
A_rover = np.array([-0.00257262, 0.8584127, 0.512936])
r_rover = np.array([-0.03012078, 0.83946748, 0.5425745])
C_rover = np.array([0.6840714, 0.6478303, -1.926771])

# Transform ray and camera position to Site Frame:
A_site = rot.apply(A_rover)
r_site = rot.apply(r_rover)
C_site = rot.apply(C_rover)

print("\nCamera axis A in Site frame:", A_site)
print("Target ray r in Site frame:", r_site)
print("Camera position C relative to rover origin in Site frame:", C_site)

# Solar vector in Site Frame:
# Solar azimuth = 186.8169 deg (clockwise from North: North=0, East=90, South=180)
# Solar elevation = 69.8224 deg above horizontal
# In NED frame:
# Sun direction vector from surface towards Sun:
# X_sun = cos(el) * cos(az)
# Y_sun = cos(el) * sin(az)
# Z_sun = -sin(el) (Up is negative Z in NED)
sun_az = np.radians(186.8169)
sun_el = np.radians(69.8224)
sun_vec = np.array([np.cos(sun_el) * np.cos(sun_az), np.cos(sun_el) * np.sin(sun_az), -np.sin(sun_el)])
print("\nSolar vector (pointing toward sun) in Site frame:", sun_vec)

# If the sunlight comes from sun_vec, rays of light travel in direction -sun_vec:
light_dir = -sun_vec
print("Light travel direction in Site frame:", light_dir)
