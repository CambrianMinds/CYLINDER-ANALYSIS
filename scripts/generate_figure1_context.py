import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
import cv2, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Load full PDS image
full = cv2.imread('figures/pds_raw_full.png', cv2.IMREAD_GRAYSCALE)
H_f, W_f = full.shape # 1184, 1328

# Cylinder location in full frame:
# Native coordinates: center ~ (1072, 986)
# Bounding box ~ [1072 - 60, 986 - 50] to [1072 + 60, 986 + 50]
x_box = 1072 - 70
y_box = 986 - 55
w_box = 140
h_box = 110

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
fig.patch.set_facecolor('#05070d')
ax.set_facecolor('#05070d')

im = ax.imshow(full, cmap='magma', origin='upper')

# Draw neon callout rectangle around the cylinder
rect = Rectangle((x_box, y_box), w_box, h_box, edgecolor='#00e5ff', facecolor='none', linewidth=2.6, linestyle='-')
ax.add_patch(rect)

# Annotate callout
ax.annotate('Sol 3556 Cylinder Feature\n(See Inset Panels B & C)',
            xy=(x_box + w_box/2, y_box),
            xytext=(x_box - 280, y_box - 80),
            arrowprops=dict(facecolor='#00e5ff', edgecolor='none', shrink=0.08, width=2.0, headwidth=7),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#0d131f', edgecolor='#00e5ff', alpha=0.92, linewidth=1.2),
            fontsize=11, color='#00e5ff', fontweight='bold')

# Scale bar on full frame:
# At distance ~ 5 m, IFOV = 0.074 mrad/px -> 0.386 mm/px
# 50 cm = 500 mm / 0.386 = 1295 px (spans entire width)
# Let's show 10 cm scale bar: 100 mm / 0.386 = 259 px
sb_len = 259
ax.plot([40, 40 + sb_len], [H_f - 40, H_f - 40], color='#ffffff', linewidth=3.5)
ax.text(40, H_f - 55, '10 cm Foreground Scale (d ≈ 5.2 m)', color='#ffffff', fontsize=10, fontweight='bold')

# Metadata overlay card
meta_text = (
    "NASA Curiosity Rover | Mastcam Right (M-100)\n"
    "Frame: 3556MR1025170721700585C00_DXXX\n"
    "Sol 3556 (August 7, 2022) | Paraitepuy Pass\n"
    "Full Sensor Raster: 1328 × 1184 Pixels"
)
ax.text(0.02, 0.96, meta_text, transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#0d131f', edgecolor='#2d3748', alpha=0.9, linewidth=1.0),
        fontsize=9.5, color='#e0e6ed', family='sans-serif')

ax.set_xlim(0, W_f)
ax.set_ylim(H_f, 0)
ax.set_xlabel('Sensor Sample (pixels)', color='#8b9bb4', fontsize=11, labelpad=8)
ax.set_ylabel('Sensor Line (pixels)', color='#8b9bb4', fontsize=11, labelpad=8)
ax.tick_params(colors='#8b9bb4', labelsize=9.5)

plt.title('Figure 1(a): Full Context Planetary Data System (PDS) Raw Sensor Frame\n'
          'NASA Mars Science Laboratory | Gale Crater Traverse | Sulfate-Bearing Unit',
          fontsize=13, color='#ffffff', fontweight='bold', pad=12)

out_path = 'figures/pds_full_with_callout.png'
plt.savefig(out_path, bbox_inches='tight', dpi=240, facecolor=fig.get_facecolor())
plt.close()
print(f'Generated {out_path} successfully!')
