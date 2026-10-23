import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
import xml.etree.ElementTree as ET
import numpy as np

# Load metadata
tree = ET.parse('data/3556mr1025170721700585c00_dxxx.xml')
root = tree.getroot()
ns = {'geom': 'http://pds.nasa.gov/pds4/geom/v1'}

# Extract CAHVOR
vc = root.find('.//geom:Vector_Center', ns)
va = root.find('.//geom:Vector_Axis', ns)
vh = root.find('.//geom:Vector_Horizontal', ns)
vv = root.find('.//geom:Vector_Vertical', ns)
vo = root.find('.//geom:Vector_Optical', ns)
rad = root.find('.//geom:Radial_Terms', ns)

C = np.array([float(vc.find(f'geom:{k}_position', ns).text) for k in ['x', 'y', 'z']])
A = np.array([float(va.find(f'geom:{k}_unit', ns).text) for k in ['x', 'y', 'z']])
H = np.array([float(vh.find(f'geom:{k}_pixel', ns).text) for k in ['x', 'y', 'z']])
V = np.array([float(vv.find(f'geom:{k}_pixel', ns).text) for k in ['x', 'y', 'z']])
O = np.array([float(vo.find(f'geom:{k}_unit', ns).text) for k in ['x', 'y', 'z']])
R = np.array([float(rad.find(f'geom:{k}', ns).text) for k in ['c0', 'c1', 'c2']])

# Solar angles
sun_az = float(root.find('.//geom:solar_azimuth', ns).text)
sun_el = float(root.find('.//geom:solar_elevation', ns).text)
inst_az = float(root.find('.//geom:instrument_azimuth', ns).text)
inst_el = float(root.find('.//geom:instrument_elevation', ns).text)

print(f"Solar Azimuth: {sun_az:.4f} deg, Solar Elevation: {sun_el:.4f} deg")
print(f"Instrument Azimuth: {inst_az:.4f} deg, Instrument Elevation: {inst_el:.4f} deg")
print(f"Camera Position C: {C}")
print(f"Camera Axis A: {A}")

# Cylinder position in the 1328x1184 image:
# In cv_cylinder_analysis.py:
# x0c, y0c, x1c, y1c = 500, 460, 560, 510 in binned (2x2) coordinates
# Center of the cylinder face / object is at binned (536.0, 493.0)
# Native subframe pixel coordinates:
u_target = 536.0 * 2.0  # 1072.0
v_target = 493.0 * 2.0  # 986.0

print(f"Target pixel in image: ({u_target:.1f}, {v_target:.1f})")

# In CAHV, compute line-of-sight unit ray r:
r = np.cross(H - u_target * A, V - v_target * A)
r = r / np.linalg.norm(r)
if np.dot(r, A) < 0:
    r = -r

print(f"Target line of sight unit ray r: {r}")
print(f"Ray depression angle relative to rover horizontal: {np.degrees(np.arcsin(r[2])):.2f} deg")

# Rover ground contact:
# In Curiosity rover coordinates, the origin is on the centerline at the middle wheel level.
# The ground plane directly under the rover wheels is at approximately Z = +0.0 m (or nominal wheel contact).
# But what is the local terrain slope?
# Curiosity was climbing Mount Sharp (Aeolis Mons) in the sulfate-bearing unit at Paraitepuy Pass.
# The regional slope of Mount Sharp in this area is dipping ~3 deg to 8 deg toward the north/northwest (or Gale floor).
# Let's test the standoff distance as a function of ground elevation Z_ground:
print("\n--- STANDOFF DISTANCE VS TERRAIN ELEVATION ---")
for z_g in [-0.5, -0.25, 0.0, 0.25, 0.5]:
    # Ray equation: P = C + s * r  =>  P_z = C_z + s * r_z = z_g  =>  s = (z_g - C_z) / r_z
    s = (z_g - C[2]) / r[2]
    P = C + s * r
    # Pixel scale at this distance:
    # Mastcam M-100 IFOV = 0.074 mrad/px = 7.4e-5 rad/px
    # Or from CAHV: focal length f_px = norm(H - dot(A,H)*A)
    f_px = np.linalg.norm(H - np.dot(A, H) * A)
    mm_per_px = (s * 1000.0) / f_px
    print(f"Z_ground = {z_g:+.2f} m -> Slant Range s = {s:.3f} m, Scale = {mm_per_px:.3f} mm/native px")
    print(f"   3D Position in Rover Nav Frame: [{P[0]:.2f}, {P[1]:.2f}, {P[2]:.2f}] m")
    # Physical dimensions:
    # Cylinder major diameter: 53.8 px
    # Visible length: 122 px
    D_cm = 53.8 * mm_per_px / 10.0
    L_cm = 122.0 * mm_per_px / 10.0
    print(f"   -> Cylinder Diameter = {D_cm:.2f} cm, Visible Length = {L_cm:.2f} cm")
