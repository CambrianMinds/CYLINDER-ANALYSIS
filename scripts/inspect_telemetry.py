import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
import xml.etree.ElementTree as ET

tree = ET.parse('data/3556mr1025170721700585c00_dxxx.xml')
ns = {'geom': 'http://pds.nasa.gov/pds4/geom/v1'}

for csd in tree.getroot().findall('.//geom:Coordinate_Space_Definition', ns):
    print("=" * 50)
    for c in csd:
        tag = c.tag.split('}')[-1]
        lines = []
        for x in c.iter():
            xt = x.tag.split('}')[-1]
            txt = (x.text or '').strip()
            if txt and '\n' not in txt:
                lines.append(f"{xt}: {txt}")
        print(f"  [{tag}] " + ", ".join(lines))
