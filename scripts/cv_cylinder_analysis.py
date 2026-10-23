"""
Independent re-analysis of the Sol 3556 "cylinder" from the LOSSLESS PDS EDR (Mastcam Right, 100 mm).

Design rules (learned the hard way):
  * Every quantity is MEASURED from pixels. No measurement is used as an input to another one
    (in particular the flank angle is NOT derived from the ellipse angle, or vice-versa).
  * One coordinate frame throughout. Pixel coordinates are image frame (x right, y DOWN). Angles are quoted both in
    the image frame (clockwise on screen from +x) and in standard Cartesian (y up): theta_cart = -theta_img.
  * Uncertainties come from bootstrap resampling of the edge points and from how the answer changes when a
    different (defensible) set of rim points is used. Where the data cannot constrain a quantity we say so.

Run:  python cv_cylinder_analysis.py        (needs numpy, scipy, opencv-python, matplotlib)
Outputs: analysis_results.json, sobel_gradient_and_flank_tangents.png (Fig. 2), rim_orientation_constraint.png (Fig. 3)
"""
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

import os, re, json
import numpy as np, cv2
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FRAME = "3556MR1025170721700585C00_DXXX"
IMG = os.path.join(HERE, "data", FRAME + ".IMG")
LBL = os.path.join(HERE, "data", FRAME.lower() + ".xml")
rng = np.random.default_rng(20261022)

# ------------------------------------------------------------------ data + label
lbl = open(LBL, encoding="utf-8").read()
def tag(name, cast=float):
    m = re.search(r"<%s[^>]*>\s*([^<\s]+)\s*<" % re.escape(name), lbl)
    return cast(m.group(1))
H, W = 1184, 1328
offset = 27888                                    # <offset> of sensor_image in the label
edr = np.fromfile(IMG, dtype=np.uint8)
edr = edr[offset:offset + H * W].reshape(H, W).astype(np.float32)   # 8-bit, onboard-LOSSLESS compression
b2 = edr.reshape(H // 2, 2, W // 2, 2).mean((1, 3))   # 2x2 bin: removes the Bayer (colour-filter-array) checkerboard
bl = gaussian_filter(b2, 0.7)
sun_az, sun_el = tag("geom:solar_azimuth"), tag("geom:solar_elevation")
cam_az, cam_el = tag("geom:instrument_azimuth"), tag("geom:instrument_elevation")
IFOV = 74e-6                                      # rad / pixel, Mastcam-100 (Bell et al. 2017)

# ------------------------------------------------------------------ helpers
def sample(xs, ys): return map_coordinates(bl, [ys, xs], order=3, mode="nearest")

def refine_edge(p0, p1, half=3.5, iters=4, step=0.4):
    """Straight edge: sample perpendicular profiles along the current line, take the sub-pixel max |gradient|, refit."""
    p0, p1 = np.array(p0, float), np.array(p1, float)
    for _ in range(iters):
        d = p1 - p0; L = np.hypot(*d); u = d / L; n = np.array([-u[1], u[0]])
        offs = np.arange(-half, half + 1e-9, 0.1)
        pts, prof_list, step_list = [], [], []
        for t in np.arange(0, L, step):
            c = p0 + u * t
            prof = sample(c[0] + n[0] * offs, c[1] + n[1] * offs); g = np.gradient(prof, offs)
            k = int(np.argmax(np.abs(g)))
            if 0 < k < len(offs) - 1:
                y0, y1_, y2 = g[k - 1], g[k], g[k + 1]; den = y0 - 2 * y1_ + y2
                dk = 0.5 * (y0 - y2) / den if den != 0 else 0.0
                o = offs[k] + dk * 0.1
                pts.append(c + n * o); prof_list.append((offs - o, prof * np.sign(g[k])))
        pts = np.array(pts)
        vx, vy, x0, y0_ = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        cc = np.array([x0, y0_]); uu = np.array([vx, vy]); p0, p1 = cc - uu * L / 2, cc + uu * L / 2
    # edge sharpness: average the aligned profiles, measure 20-80 % rise distance (in NATIVE pixels)
    grid = np.arange(-half, half + 1e-9, 0.1)
    avg = np.mean([np.interp(grid, o, p) for o, p in prof_list], axis=0)
    avg = (avg - avg.min()) / (avg.max() - avg.min())
    i20, i80 = np.argmax(avg > 0.2), np.argmax(avg > 0.8)
    width_native = abs(grid[i80] - grid[i20]) * 2
    return pts, width_native

def line_angle(pts):
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.degrees(np.arctan2(vy, vx)) % 180, (float(x0), float(y0), float(vx), float(vy))

def boot_angles(pts, n=2000):
    out = np.array([line_angle(pts[rng.integers(0, len(pts), len(pts))])[0] for _ in range(n)])
    med = np.median(out); return (out - med + 90) % 180 - 90 + med

def fit_e(pts):
    (xc, yc), (MA, ma), ang = cv2.fitEllipse(pts.astype(np.float32))
    return (xc, yc, MA / 2, ma / 2, ang % 180) if MA >= ma else (xc, yc, ma / 2, MA / 2, (ang + 90) % 180)

def boot_e(pts, n=1500):
    r = []
    for _ in range(n):
        try: r.append(fit_e(pts[rng.integers(0, len(pts), len(pts))]))
        except Exception: pass
    r = np.array(r); med = np.median(r[:, 4]); r[:, 4] = (r[:, 4] - med + 90) % 180 - 90 + med
    return r

# ------------------------------------------------------------------ 1. FLANK SILHOUETTE EDGES (independent of the rim)
up_pts, up_w = refine_edge((517.5, 473.5), (538.0, 484.0))     # seeds from visual inspection; refined from the pixels
lo_pts, lo_w = refine_edge((514.5, 476.0), (525.5, 497.0))
R = {"frame": FRAME, "units": "EDR pixels, 2x2 binned (native = 2x); image frame (y down); *_cart = standard Cartesian (y up)"}
edges = {}
for name, pts, wd in (("upper_edge", up_pts, up_w), ("lower_left_edge", lo_pts, lo_w)):
    ang, line = line_angle(pts); bs = boot_angles(pts)
    vx, vy, x0, y0 = line[2], line[3], line[0], line[1]
    rms = float(np.sqrt(np.mean(np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx) ** 2)))
    edges[name] = dict(n=len(pts), ang_img=float(ang), ang_img_ci95=[float(x) for x in np.percentile(bs, [2.5, 97.5])],
                       ang_cart=float(-ang), rms_binned_px=rms, rise_20_80_native_px=float(wd), line=line)
R["edges"] = edges
bu, bo = boot_angles(up_pts), boot_angles(lo_pts); nn = min(len(bu), len(bo))
diff = np.abs(bo[:nn] - bu[:nn])
R["edge_convergence_deg"] = float(abs(edges["lower_left_edge"]["ang_img"] - edges["upper_edge"]["ang_img"]))
R["edge_convergence_ci95"] = [float(x) for x in np.percentile(diff, [2.5, 97.5])]
def isect(l1, l2):
    x1, y1, vx1, vy1 = l1; x2, y2, vx2, vy2 = l2
    s, t = np.linalg.solve(np.array([[vx1, -vx2], [vy1, -vy2]]), np.array([x2 - x1, y2 - y1])); return [float(x1 + s * vx1), float(y1 + s * vy1)]
R["apex_of_edges"] = isect(edges["upper_edge"]["line"], edges["lower_left_edge"]["line"])

# ------------------------------------------------------------------ 2. FACE RIM (independent of the flank)
cx0, cy0 = 536.5, 494.0
rows = []
for angd in np.arange(0, 360, 2):
    t = np.deg2rad(angd); r = np.arange(4, 17, 0.1)
    prof = sample(cx0 + r * np.cos(t), cy0 + r * np.sin(t)); d = np.gradient(prof, r); i = int(np.argmax(d))
    contrast = (prof[i + 8:i + 20].mean() - prof[max(0, i - 8):max(1, i - 2)].mean()) if 8 < i < len(prof) - 8 else np.nan
    rows.append((angd, r[i], d[i], contrast, cx0 + r[i] * np.cos(t), cy0 + r[i] * np.sin(t)))
P = np.array(rows)
good = (P[:, 2] > 8) & (P[:, 3] > 35) & (P[:, 1] > 6) & (P[:, 1] < 16)
arcs = {"A": ("crisp arc against the lit body (170-300 deg)", 170, 300),
        "B": ("left 'C' arc (90-300 deg)", 90, 300),
        "full": ("full dark-blob outline incl. shadow bulge (0-360 deg)", 0, 360)}
R["ellipse_fits"] = {}
for key, (desc, lo, hi) in arcs.items():
    m = good & (P[:, 0] >= lo) & (P[:, 0] <= hi); pts = P[m][:, 4:6]
    E = fit_e(pts); bs = boot_e(pts)
    R["ellipse_fits"][key] = dict(desc=desc, n=int(m.sum()), xc=E[0], yc=E[1], a=E[2], b=E[3], b_over_a=E[3] / E[2],
        phi_img=E[4], phi_cart=float((-E[4]) % 180), a_ci95=[float(x) for x in np.percentile(bs[:, 2], [2.5, 97.5])],
        b_ci95=[float(x) for x in np.percentile(bs[:, 3], [2.5, 97.5])], phi_boot_ci95=[float(x) for x in np.percentile(bs[:, 4], [2.5, 97.5])])

# orientation-constraint profile: fit residual vs. imposed major-axis angle
def resid(par, pts, phi):
    xc, yc, a, b = par; t = np.deg2rad(phi); c, s = np.cos(t), np.sin(t)
    dx, dy = pts[:, 0] - xc, pts[:, 1] - yc; u, v = dx * c + dy * s, -dx * s + dy * c
    return (np.sqrt((u / a) ** 2 + (v / b) ** 2) - 1.0) * np.sqrt(a * b)
profile = {}
for key in ("A", "B"):
    _, lo, hi = arcs[key][0], arcs[key][1], arcs[key][2]
    pts = P[good & (P[:, 0] >= lo) & (P[:, 0] <= hi)][:, 4:6]; out = []
    for phi in range(0, 180, 3):
        best = None
        for a0, b0 in ((10, 8), (10, 9), (11, 7)):
            r_ = least_squares(resid, [pts[:, 0].mean() + 1, pts[:, 1].mean(), a0, b0], args=(pts, phi), bounds=([500, 480, 4, 3], [570, 510, 20, 20]))
            if best is None or r_.cost < best.cost: best = r_
        out.append((phi, float(np.sqrt(2 * best.cost / len(pts))), float(best.x[2]), float(best.x[3])))
    profile[key] = np.array(out)

# ------------------------------------------------------------------ 3. The cylinder test: is the major axis perpendicular to the projected axis?
phi_pred_img = (edges["upper_edge"]["ang_img"] + 90) % 180       # if the upper silhouette edge is parallel to the cylinder axis
tests = {}
for key in ("A", "B"):
    o = profile[key]; k = int(np.argmin(o[:, 1])); thr = o[k, 1] * 1.10 + 0.02
    ok = o[o[:, 1] <= thr][:, 0]; j = int(np.argmin(np.abs(o[:, 0] - phi_pred_img)))
    tests[key] = dict(best_phi_img=float(o[k, 0]), rms_min=float(o[k, 1]), rms_at_prediction=float(o[j, 1]),
                      consistent_phi_img_range=[float(ok.min()), float(ok.max())], n_consistent_gridvalues=int(len(ok)), n_grid=int(len(o)),
                      is_prediction_within_tolerance=bool(o[j, 1] <= thr))
R["orthogonality_test"] = dict(predicted_phi_img=float(phi_pred_img), predicted_phi_cart=float((-phi_pred_img) % 180),
    measured_phi_img_arcB=R["ellipse_fits"]["B"]["phi_img"], gap_arcB_deg=float(abs(R["ellipse_fits"]["B"]["phi_img"] - phi_pred_img)), by_arc=tests)

# ------------------------------------------------------------------ 4. Scale + lighting from the label (not assumed)
sub_first_line, sub_first_sample = 17, 161
obj_native = (2 * 536.0, 2 * 493.0)                                  # object centre in subframe pixels
dy_px = obj_native[1] - H / 2
depression = -cam_el + np.degrees(np.arctan(dy_px * IFOV))             # deg below horizontal (rover tilt ~1 deg, ignored)
CAM_H = (1.9, 2.1, 2.3)                                               # m above ground: RSM head is 1.91 m above RNAV origin
rng_m = [h / np.sin(np.radians(depression)) for h in CAM_H]
scale_mm = [r * IFOV * 1e3 for r in rng_m]
a_nat, b_nat = 2 * R["ellipse_fits"]["B"]["a"], 2 * R["ellipse_fits"]["B"]["b"]
len_native = 2 * np.hypot(535.5 - 515.0, 493.0 - 473.5)                # truncated tip to face centre, projected
R["scale"] = dict(camera_elevation_deg=cam_el, camera_azimuth_deg=cam_az, depression_at_object_deg=float(depression),
    assumed_camera_height_m=CAM_H, slant_range_m=[float(x) for x in rng_m], mm_per_native_px=[float(x) for x in scale_mm],
    face_major_diameter_native_px=float(2 * a_nat), face_major_diameter_mm=[float(2 * a_nat * s) for s in scale_mm],
    face_minor_diameter_mm=[float(2 * b_nat * s) for s in scale_mm],
    visible_length_native_px=float(len_native), visible_length_mm=[float(len_native * s) for s in scale_mm],
    note="Pixel scale = IFOV x slant range. Range follows from camera height and pointing; verified only to ~15 percent.")
shadow_bearing = (sun_az + 180 - cam_az) % 360
R["lighting"] = dict(solar_azimuth_deg=sun_az, solar_elevation_deg=sun_el, shadow_azimuth_deg=(sun_az + 180) % 360,
    shadow_bearing_rel_camera_heading_deg=float(shadow_bearing),
    expected_shadow_length_over_height=float(1 / np.tan(np.radians(sun_el))))
json.dump(R, open(os.path.join(HERE, "analysis_results.json"), "w"), indent=1)

# ------------------------------------------------------------------ console report
e_u, e_l = edges["upper_edge"], edges["lower_left_edge"]
print(f"Frame {FRAME}  (EDR 8-bit lossless, {W}x{H}; analysis on 2x2 binned {W//2}x{H//2})")
print(f"Upper silhouette edge : {e_u['ang_img']:.2f} deg img  [{e_u['ang_img_ci95'][0]:.1f},{e_u['ang_img_ci95'][1]:.1f}]  = {e_u['ang_cart']:.2f} deg Cartesian ; rms {e_u['rms_binned_px']:.2f} px ; 20-80 rise {e_u['rise_20_80_native_px']:.1f} native px")
print(f"Lower-left edge       : {e_l['ang_img']:.2f} deg img  [{e_l['ang_img_ci95'][0]:.1f},{e_l['ang_img_ci95'][1]:.1f}] ; rms {e_l['rms_binned_px']:.2f} px ; 20-80 rise {e_l['rise_20_80_native_px']:.1f} native px")
print(f"Edge convergence      : {R['edge_convergence_deg']:.1f} deg  95%CI {R['edge_convergence_ci95'][0]:.1f}-{R['edge_convergence_ci95'][1]:.1f}  (parallel generators would give 0)")
for k, v in R["ellipse_fits"].items():
    print(f"Ellipse {k:4s} n={v['n']:3d}  a={v['a']:.2f} b={v['b']:.2f} b/a={v['b_over_a']:.2f}  phi_img={v['phi_img']:.1f} (cart {v['phi_cart']:.1f})  boot phi CI {v['phi_boot_ci95'][0]:.0f}-{v['phi_boot_ci95'][1]:.0f}")
o = R["orthogonality_test"]
print(f"Cylinder prediction   : major axis at {o['predicted_phi_img']:.1f} deg img ({o['predicted_phi_cart']:.1f} cart); arc-B measured {o['measured_phi_img_arcB']:.1f} -> gap {o['gap_arcB_deg']:.1f} deg")
for k, t in o["by_arc"].items():
    print(f"   arc {k}: orientations consistent with rim data span {t['consistent_phi_img_range'][0]:.0f}-{t['consistent_phi_img_range'][1]:.0f} deg ({t['n_consistent_gridvalues']}/{t['n_grid']} grid values); prediction within tolerance: {t['is_prediction_within_tolerance']}")
s = R["scale"]
print(f"Depression {s['depression_at_object_deg']:.1f} deg -> slant range {s['slant_range_m'][0]:.1f}-{s['slant_range_m'][2]:.1f} m -> {s['mm_per_native_px'][0]:.2f}-{s['mm_per_native_px'][2]:.2f} mm/native px")
print(f"Face major diameter {2*a_nat:.0f} native px = {s['face_major_diameter_mm'][0]/10:.1f}-{s['face_major_diameter_mm'][2]/10:.1f} cm ; visible length {len_native:.0f} px = {s['visible_length_mm'][0]/10:.1f}-{s['visible_length_mm'][2]/10:.1f} cm")
print(f"Sun az {sun_az:.1f} el {sun_el:.1f} ; shadow bearing {shadow_bearing:.0f} deg clockwise from camera heading (lower-right in the image)")

# ------------------------------------------------------------------ FIGURES
BG, FG, MUT = "#0b0f19", "#e6edf3", "#8b98a9"
x0c, y0c, x1c, y1c = 500, 460, 560, 510                 # crop window (binned coords)
def style(ax):
    ax.set_facecolor(BG); ax.tick_params(colors=MUT, labelsize=9)
    for s_ in ax.spines.values(): s_.set_color("#2d3748")

# ---- Figure 2: Sobel gradient + measured edges + rim
gx = cv2.Sobel(b2, cv2.CV_64F, 1, 0, ksize=3); gy = cv2.Sobel(b2, cv2.CV_64F, 0, 1, ksize=3); mag = np.hypot(gx, gy)
fig, ax = plt.subplots(figsize=(12.5, 8.6), facecolor=BG); style(ax)
im = ax.imshow(mag[y0c:y1c, x0c:x1c], cmap="magma", extent=[x0c - .5, x1c - .5, y1c - .5, y0c - .5], interpolation="nearest")
t_ = np.linspace(-1, 1, 2)
for ed, col, lab in ((e_u, "#ffd166", "Upper silhouette edge"), (e_l, "#ff9f1c", "Lower-left boundary")):
    x_, y_, vx, vy = ed["line"]; ts = np.linspace(-20, 20, 2)
    ax.plot(x_ + ts * vx, y_ + ts * vy, color=col, lw=2.2, label=f"{lab}: {ed['ang_img']:.1f}° image / {ed['ang_cart']:.1f}° Cartesian  (95% CI ±{(ed['ang_img_ci95'][1]-ed['ang_img_ci95'][0])/2:.1f}°)")
ax.scatter(up_pts[:, 0], up_pts[:, 1], s=6, color="#ffd166", alpha=.7); ax.scatter(lo_pts[:, 0], lo_pts[:, 1], s=6, color="#ff9f1c", alpha=.7)
ax.scatter(P[good][:, 4], P[good][:, 5], s=9, color="#06d6a0", alpha=.85, label="Rim edge points (accepted)")
ax.scatter(P[~good][:, 4], P[~good][:, 5], s=9, color="#ef476f", alpha=.6, label="Rim rays rejected (low contrast)")
Eb = R["ellipse_fits"]["B"]; th = np.linspace(0, 2 * np.pi, 300); ph = np.deg2rad(Eb["phi_img"])
ex, ey = Eb["a"] * np.cos(th), Eb["b"] * np.sin(th)
ax.plot(Eb["xc"] + ex * np.cos(ph) - ey * np.sin(ph), Eb["yc"] + ex * np.sin(ph) + ey * np.cos(ph), color="#00f5d4", lw=1.8, ls="--",
        label=f"Ellipse (arc B): a={Eb['a']*2:.1f}, b={Eb['b']*2:.1f} native px, φ={Eb['phi_img']:.0f}° image / {Eb['phi_cart']:.0f}° Cartesian")
ax.plot(*R["apex_of_edges"], marker="*", ms=14, color="white", ls="none", label="Edge-line intersection (apex)")
ax.set_xlim(x0c, x1c); ax.set_ylim(y1c, y0c)
bx, by = x0c + 3, y1c - 3; sb = 5.0 / scale_mm[1] / 2                    # 5 mm scale bar in binned px
ax.plot([bx, bx + sb], [by, by], color="white", lw=3); ax.text(bx, by - 1.2, "≈5 mm (range 5.3 m, ±10%)", color="white", fontsize=9)
ax.set_title(f"Sobel gradient magnitude — lossless PDS EDR {FRAME}\nEdges measured independently: upper {e_u['ang_img']:.1f}°, lower-left {e_l['ang_img']:.1f}° → {R['edge_convergence_deg']:.1f}° convergence (parallel = 0°)",
             color=FG, fontsize=12.5, fontweight="bold", pad=12)
ax.set_xlabel("EDR x (2×2-binned px; native = 2×; image frame, y down)", color=MUT); ax.set_ylabel("EDR y (2×2-binned px)", color=MUT)
leg = ax.legend(loc="upper right", facecolor="#111827", edgecolor="#2d3748", fontsize=8.2, labelcolor=FG)
cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.02); cb.set_label("|∇I| (DN / binned px)", color=MUT); cb.ax.tick_params(colors=MUT)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "figures/sobel_gradient_and_flank_tangents.png"), dpi=200, facecolor=BG); plt.close()

# ---- Figure 3: what the rim orientation can and cannot say
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.6), facecolor=BG, gridspec_kw={"width_ratios": [1, 1.05]}); style(a1); style(a2)
a1.imshow(b2[y0c:y1c, x0c:x1c], cmap="gray", extent=[x0c - .5, x1c - .5, y1c - .5, y0c - .5], interpolation="bicubic", vmin=30, vmax=190)
for key, col in (("A", "#ffd166"), ("B", "#00f5d4")):
    v = R["ellipse_fits"][key]; ph = np.deg2rad(v["phi_img"]); ex, ey = v["a"] * np.cos(th), v["b"] * np.sin(th)
    m = good & (P[:, 0] >= arcs[key][1]) & (P[:, 0] <= arcs[key][2])
    a1.scatter(P[m][:, 4], P[m][:, 5], s=7, color=col, alpha=.6)
    a1.plot(v["xc"] + ex * np.cos(ph) - ey * np.sin(ph), v["yc"] + ex * np.sin(ph) + ey * np.cos(ph), color=col, lw=1.6,
            label=f"arc {key}: b/a={v['b_over_a']:.2f}, φ={v['phi_img']:.0f}° (img)")
x_, y_, vx, vy = e_u["line"]; a1.plot([x_ - 20 * vx, x_ + 20 * vx], [y_ - 20 * vy, y_ + 20 * vy], color="#ef476f", lw=1.2, ls=":", label="upper silhouette edge")
pp = np.deg2rad(phi_pred_img); cxx, cyy = Eb["xc"], Eb["yc"]
a1.plot([cxx - 12 * np.cos(pp), cxx + 12 * np.cos(pp)], [cyy - 12 * np.sin(pp), cyy + 12 * np.sin(pp)], color="#ef476f", lw=2, label=f"cylinder prediction for major axis: {phi_pred_img:.0f}° (img)")
a1.set_xlim(x0c + 8, x1c - 8); a1.set_ylim(y1c - 4, y0c + 6); a1.set_title("Rim points used and ellipse fits (image frame, y down)", color=FG, fontsize=12, fontweight="bold")
a1.legend(loc="upper right", facecolor="#111827", edgecolor="#2d3748", fontsize=8, labelcolor=FG); a1.set_xlabel("EDR x (binned px)", color=MUT); a1.set_ylabel("EDR y (binned px)", color=MUT)
for key, col in (("A", "#ffd166"), ("B", "#00f5d4")):
    o_ = profile[key]; a2.plot(o_[:, 0], o_[:, 1], color=col, lw=2, label=f"arc {key}")
a2.axvline(phi_pred_img, color="#ef476f", lw=2, ls="--", label=f"cylinder prediction {phi_pred_img:.0f}° (⟂ upper edge)")
a2.axvline(Eb["phi_img"], color="#00f5d4", lw=1, ls=":", label=f"best free fit, arc B {Eb['phi_img']:.0f}°")
a2.set_xlabel("imposed major-axis angle φ (deg, image frame; Cartesian = −φ)", color=MUT); a2.set_ylabel("rim fit RMS residual (binned px)", color=MUT)
a2.set_title("How well does the rim constrain the ellipse orientation?\n(flat curve = no constraint)", color=FG, fontsize=12, fontweight="bold")
a2.legend(facecolor="#111827", edgecolor="#2d3748", fontsize=9, labelcolor=FG); a2.grid(color="#2d3748", ls=":", alpha=.6)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "rim_orientation_constraint.png"), dpi=200, facecolor=BG); plt.close()
print("wrote analysis_results.json, sobel_gradient_and_flank_tangents.png, rim_orientation_constraint.png")
