#!/usr/bin/env python3
"""
Transport Fever 2 Dynamic Map Analysis, Network Design & Optimization Framework
Exports SVG outputs for Passenger, Cargo, and Combined networks.
Also injects native Inkscape layers into the original SVG map export.
"""

import argparse
import base64
import io
import json
import math
import re
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from lxml import etree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.sparse.csgraph as csgraph
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from skimage.graph import route_through_array
import networkx as nx

DEFAULT_TF2_CHAINS = {
    'Quarry': {'product': 'Stone', 'consumer': 'Construction materials plant', 'car': 'gondola (dry bulk)'},
    'Construction materials plant': {'product': 'Construction materials', 'consumer': None, 'car': 'boxcar (break bulk)'},
    'Iron ore mine': {'product': 'Iron ore', 'consumer': 'Steel mill', 'car': 'gondola (dry bulk)'},
    'Coal mine': {'product': 'Coal', 'consumer': 'Steel mill', 'car': 'gondola (dry bulk)'},
    'Steel mill': {'product': 'Steel', 'consumer': ['Goods factory', 'Machines factory'], 'car': 'flatcar w/ stakes (bundled)'},
    'Forest': {'product': 'Logs', 'consumer': 'Saw mill', 'car': 'flatcar w/ stakes (bundled)'},
    'Saw mill': {'product': 'Planks', 'consumer': 'Tools factory', 'car': 'flatcar w/ stakes (bundled)'},
    'Farm': {'product': 'Grain', 'consumer': 'Food processing plant', 'car': 'gondola (dry bulk)'},
    'Food processing plant': {'product': 'Food', 'consumer': None, 'car': 'boxcar (break bulk)'},
    'Oil well': {'product': 'Crude oil', 'consumer': 'Oil refinery', 'car': 'tank car (liquid bulk)'},
    'Oil refinery': {'product': 'Refined oil', 'consumer': ['Fuel refinery', 'Chemical plant'], 'car': 'tank car (liquid bulk)'},
    'Chemical plant': {'product': 'Plastic', 'consumer': 'Goods factory', 'car': 'tank car (liquid bulk)'},
    'Fuel refinery': {'product': 'Fuel', 'consumer': None, 'car': 'tank car (liquid bulk)'},
    'Tools factory': {'product': 'Tools', 'consumer': 'Machines factory', 'car': 'boxcar (break bulk)'},
    'Machines factory': {'product': 'Machines', 'consumer': None, 'car': 'flatcar w/ stakes (bundled)'},
    'Goods factory': {'product': 'Goods', 'consumer': None, 'car': 'boxcar (break bulk)'},
}

def parse_svg_path_segments(d_str):
    """Robust SVG path parser handling M, m, L, l, H, h, V, v, C, c, Z, z."""
    tokens = re.findall(r'([a-zA-Z])|([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)', d_str)
    segments = []
    curr_x, curr_y = 0.0, 0.0
    start_x, start_y = 0.0, 0.0
    cmd = None
    
    raw_tokens = []
    for letter, num in tokens:
        if letter:
            raw_tokens.append(letter)
        else:
            raw_tokens.append(float(num))
            
    idx = 0
    n = len(raw_tokens)
    while idx < n:
        item = raw_tokens[idx]
        if isinstance(item, str):
            cmd = item
            idx += 1
            if idx >= n: break
            item = raw_tokens[idx]
            
        if cmd == 'M':
            curr_x, curr_y = raw_tokens[idx], raw_tokens[idx+1]
            start_x, start_y = curr_x, curr_y
            idx += 2
            cmd = 'L'
        elif cmd == 'm':
            curr_x += raw_tokens[idx]
            curr_y += raw_tokens[idx+1]
            start_x, start_y = curr_x, curr_y
            idx += 2
            cmd = 'l'
        elif cmd == 'L':
            nx_pt, ny_pt = raw_tokens[idx], raw_tokens[idx+1]
            segments.append(((curr_x, curr_y), (nx_pt, ny_pt)))
            curr_x, curr_y = nx_pt, ny_pt
            idx += 2
        elif cmd == 'l':
            nx_pt = curr_x + raw_tokens[idx]
            ny_pt = curr_y + raw_tokens[idx+1]
            segments.append(((curr_x, curr_y), (nx_pt, ny_pt)))
            curr_x, curr_y = nx_pt, ny_pt
            idx += 2
        elif cmd == 'H':
            nx_pt = raw_tokens[idx]
            segments.append(((curr_x, curr_y), (nx_pt, curr_y)))
            curr_x = nx_pt
            idx += 1
        elif cmd == 'h':
            nx_pt = curr_x + raw_tokens[idx]
            segments.append(((curr_x, curr_y), (nx_pt, curr_y)))
            curr_x = nx_pt
            idx += 1
        elif cmd == 'V':
            ny_pt = raw_tokens[idx]
            segments.append(((curr_x, curr_y), (curr_x, ny_pt)))
            curr_y = ny_pt
            idx += 1
        elif cmd == 'v':
            ny_pt = curr_y + raw_tokens[idx]
            segments.append(((curr_x, curr_y), (curr_x, ny_pt)))
            curr_y = ny_pt
            idx += 1
        elif cmd == 'C':
            nx_pt, ny_pt = raw_tokens[idx+4], raw_tokens[idx+5]
            segments.append(((curr_x, curr_y), (nx_pt, ny_pt)))
            curr_x, curr_y = nx_pt, ny_pt
            idx += 6
        elif cmd == 'c':
            nx_pt = curr_x + raw_tokens[idx+4]
            ny_pt = curr_y + raw_tokens[idx+5]
            segments.append(((curr_x, curr_y), (nx_pt, ny_pt)))
            curr_x, curr_y = nx_pt, ny_pt
            idx += 6
        elif cmd in ('Z', 'z'):
            if (curr_x, curr_y) != (start_x, start_y):
                segments.append(((curr_x, curr_y), (start_x, start_y)))
                curr_x, curr_y = start_x, start_y
        else:
            idx += 1
            
    return segments

def path_to_svg_d(pts):
    """Convert sequence of (x, y) coordinates into SVG path 'd' string."""
    if not pts: return ""
    d = [f"M{pts[0][0]:.2f},{pts[0][1]:.2f}"]
    for p in pts[1:]:
        d.append(f"L{p[0]:.2f},{p[1]:.2f}")
    return " ".join(d)

def optimize_svg_raster(svg_path, jpeg_quality=85):
    """
    Optimizes embedded raster images in SVG files.
    Matplotlib exports ax.imshow raster layers as raw, uncompressed 32-bit PNGs
    encoded in base64 on a single unbroken line (>10MB).
    Standard XML parsers and image viewers (libxml2, librsvg, glycin, GNOME Loupe,
    eog, Inkscape, QtSvg) enforce XML_MAX_TEXT_LENGTH = 10MB or crash with memory
    allocation failure, marking the SVG as corrupt.

    This function:
    1. Finds embedded base64 image data (data:image/png;base64,...).
    2. Re-encodes uncompressed PNGs as optimized JPEGs (reducing file size by ~80%).
    3. Wraps base64 output into 76-character lines, fully complying with standard XML parsers.
    JPEG has no alpha channel, so a semi-transparent layer (e.g. a uniform-alpha
    imshow overlay) would otherwise flatten to a fully opaque image and blot out
    everything beneath it -- if the source alpha is uniform, that value is carried
    over as the <image> element's own `opacity` attribute instead of being dropped.
    """
    svg_path = Path(svg_path)
    if not svg_path.exists():
        return

    ns_svg = "http://www.w3.org/2000/svg"
    ns_xlink = "http://www.w3.org/1999/xlink"

    with open(svg_path, 'rb') as f:
        tree = etree.parse(f)
    root = tree.getroot()

    for img_el in root.iter(f'{{{ns_svg}}}image'):
        href_attr = f'{{{ns_xlink}}}href' if img_el.get(f'{{{ns_xlink}}}href') is not None else 'href'
        href = img_el.get(href_attr, '')
        if not href.startswith('data:image/png;base64,'):
            continue
        raw_b64 = re.sub(r'\s+', '', href.split(',', 1)[1])
        try:
            raw_bytes = base64.b64decode(raw_b64)
            img = Image.open(io.BytesIO(raw_bytes))
            opacity = None
            if img.mode in ('RGBA', 'LA'):
                lo, hi = img.split()[-1].getextrema()
                if lo == hi and lo < 255:
                    opacity = lo / 255.0
                img_rgb = img.convert('RGB')
            else:
                img_rgb = img.convert('RGB')

            buf = io.BytesIO()
            img_rgb.save(buf, format='JPEG', quality=jpeg_quality, optimize=True)
            compressed_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
            wrapped = '\n'.join(compressed_b64[i:i+76] for i in range(0, len(compressed_b64), 76))
            img_el.set(href_attr, f'data:image/jpeg;base64,\n{wrapped}\n')
            if opacity is not None:
                img_el.set('opacity', f'{opacity:.3f}')
        except Exception:
            continue

    tree.write(str(svg_path), xml_declaration=True, encoding='utf-8')

def compute_normal_offsets(pts, offset_dist):
    """Offset an Nx2 polyline by offset_dist along its 2D normal vectors."""
    pts = np.asarray(pts, dtype=float)
    N = len(pts)
    if N < 2: return pts
    tangents = np.zeros_like(pts)
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    tangents[1:-1] = pts[2:] - pts[:-2]
    lens = np.hypot(tangents[:, 0], tangents[:, 1])
    lens[lens == 0] = 1.0
    tangents /= lens[:, None]
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    return pts + normals * offset_dist


class MapNetworkPlanner:
    """Dynamic network planner for Transport Fever 2 maps with SVG and PNG export."""
    
    def __init__(self, svg_path, custom_chains=None):
        self.svg_path = Path(svg_path)
        self.chains = custom_chains or DEFAULT_TF2_CHAINS
        
        self.towns = {}
        self.industries = []
        self.coastline = []
        self.rail_segments = []
        self.stations = []
        self.rail_tree = None
        self.contours_pts = []
        self.contour_segments = []
        
        self.map_bounds = (0.0, 0.0, 3456.33, 10369.0)
        self.relief_img = None
        self.scaled_relief = None
        self.elevation_formula = 'R-G'
        self.cost_surface = None
        self.grid_shape = (3072, 1024)
        
        self.cargo_plan = None
        self.pass_plan = None
        self.corridors = []
        self.missed_stops = []
        
        self._parse_map()
        self._build_terrain_surface()
        self._index_infrastructure()

    def _parse_map(self):
        print(f"[Init] Parsing SVG: {self.svg_path}")
        with open(self.svg_path, 'rb') as f:
            tree = etree.parse(f)
        root = tree.getroot()
        
        relief = root.xpath('//*[@id="relief"]')
        if relief:
            img_el = relief[0].xpath('.//svg:image', namespaces={'svg': 'http://www.w3.org/2000/svg'})[0]
            mx = float(img_el.attrib.get('x', 0.0))
            my = float(img_el.attrib.get('y', 0.0))
            mw = float(img_el.attrib.get('width', 3456.33))
            mh = float(img_el.attrib.get('height', 10369.0))
            self.map_bounds = (mx, my, mw, mh)
            
            href = img_el.attrib.get('{http://www.w3.org/1999/xlink}href') or img_el.attrib.get('href')
            b64 = href.split(',', 1)[1]
            self.relief_img = Image.open(io.BytesIO(base64.b64decode(b64)))
        else:
            raise ValueError("Relief layer with embedded image not found in SVG.")
            
        for el in root.xpath('//*[@id="towns"]//svg:text', namespaces={'svg': 'http://www.w3.org/2000/svg'}):
            t_name = el.text.strip()
            self.towns[t_name] = (float(el.attrib['x']), float(el.attrib['y']))
            
        for p in root.xpath('//*[@id="coastline"]//svg:path', namespaces={'svg': 'http://www.w3.org/2000/svg'}):
            self.coastline.extend(parse_svg_path_segments(p.attrib.get('d', '')))
            
        for p in root.xpath('//*[@id="rail"]//svg:path', namespaces={'svg': 'http://www.w3.org/2000/svg'}):
            self.rail_segments.extend(parse_svg_path_segments(p.attrib.get('d', '')))
            
        for el in root.xpath('//*[@id="stations"]//svg:use', namespaces={'svg': 'http://www.w3.org/2000/svg'}):
            t = el.attrib.get('transform', '')
            m = re.search(r'translate\(([^,]+),([^)]+)\)', t)
            if m:
                sx, sy = float(m.group(1)), float(m.group(2))
                href = el.attrib.get('{http://www.w3.org/1999/xlink}href') or el.attrib.get('href', '')
                self.stations.append({'type': href, 'x': sx, 'y': sy})
                
        for p in root.xpath('//*[@id="contours"]//svg:path', namespaces={'svg': 'http://www.w3.org/2000/svg'}):
            d = p.attrib.get('d', '')
            self.contour_segments.extend(parse_svg_path_segments(d))
            pts = [float(v) for v in re.findall(r'[-+]?\d*\.?\d+', d)]
            for i in range(0, min(len(pts), 40), 2):
                self.contours_pts.append((pts[i], pts[i+1]))
                
        t_sorted = sorted(self.towns.keys(), key=len, reverse=True)
        ind_idx = 0
        for el in root.xpath('//*[@id="industries"]//svg:text', namespaces={'svg': 'http://www.w3.org/2000/svg'}):
            text = el.text.strip()
            ix, iy = float(el.attrib['x']), float(el.attrib['y'])
            matched_town = None
            for tn in t_sorted:
                if text.startswith(tn + ' '):
                    matched_town = tn
                    break
            if not matched_town:
                matched_town = min(self.towns.keys(), key=lambda t: math.hypot(self.towns[t][0] - ix, self.towns[t][1] - iy))
                rem = text
            else:
                rem = text[len(matched_town):].strip()
                
            itype = re.sub(r'\s*#\d+\s*$', '', rem).strip()
            self.industries.append({
                'id': f"ind_{ind_idx}",
                'town': matched_town,
                'type': itype,
                'x': ix,
                'y': iy,
                'full_name': text
            })
            ind_idx += 1
            
        print(f"  Parsed {len(self.towns)} towns, {len(self.industries)} industries, {len(self.rail_segments)} rail segments.")

    def _to_grid(self, x, y):
        mx, my, mw, mh = self.map_bounds
        gh, gw = self.grid_shape
        col = int(np.clip((x - mx) / mw * gw, 0, gw - 1))
        row = int(np.clip((y - my) / mh * gh, 0, gh - 1))
        return col, row
        
    def _to_svg(self, col, row):
        mx, my, mw, mh = self.map_bounds
        gh, gw = self.grid_shape
        x = mx + (col + 0.5) / gw * mw
        y = my + (row + 0.5) / gh * mh
        return x, y

    def _index_infrastructure(self):
        pts = []
        for (x1, y1), (x2, y2) in self.rail_segments:
            pts.append((x1, y1))
            pts.append((x2, y2))
            pts.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        for st in self.stations:
            if 'rail' in st['type']:
                pts.append((st['x'], st['y']))
                
        if pts:
            self.rail_pts = np.array(pts)
            self.rail_tree = cKDTree(self.rail_pts)
            print(f"  Indexed {len(pts)} pre-built rail anchor waypoints.")
        else:
            self.rail_tree = None

    def _build_terrain_surface(self):
        print("[Terrain] Calibrating and building terrain cost surface...")
        mx, my, mw, mh = self.map_bounds
        aspect = mh / mw
        gw = 1024
        gh = int(round(gw * aspect))
        self.grid_shape = (gh, gw)
        
        self.scaled_relief = self.relief_img.resize((gw, gh), Image.Resampling.BILINEAR)
        arr = np.array(self.scaled_relief).astype(float)
        
        candidates = {
            'R-G': arr[:,:,0] - arr[:,:,1],
            'G-R': arr[:,:,1] - arr[:,:,0],
            'R-B': arr[:,:,0] - arr[:,:,2],
            'Gray': 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2],
            '(R+G)/2 - B': (arr[:,:,0] + arr[:,:,1])/2.0 - arr[:,:,2]
        }
        
        t_px = [self._to_grid(tx, ty)[::-1] for tx, ty in self.towns.values()]
        c_px = [self._to_grid(cx, cy)[::-1] for cx, cy in self.contours_pts] if self.contours_pts else [(0,0)]
        
        best_formula, best_score = 'R-G', -1e9
        for name, c_arr in candidates.items():
            t_vals = [c_arr[r, c] for r, c in t_px]
            c_vals = [c_arr[r, c] for r, c in c_px]
            diff = np.mean(c_vals) - np.mean(t_vals)
            pooled_std = np.sqrt((np.var(c_vals) + np.var(t_vals)) / 2.0)
            score = diff / (pooled_std + 1e-5)
            if score > best_score:
                best_score = score
                best_formula = name
                
        self.elevation_formula = best_formula
        raw_elev = candidates[best_formula]
        min_e, max_e = raw_elev.min(), raw_elev.max()
        norm_elev = (raw_elev - min_e) / (max_e - min_e + 1e-5)
        
        gy, gx = np.gradient(norm_elev)
        grad_mag = np.hypot(gx, gy)
        
        self.cost_surface = 1.0 + 2.0 * norm_elev + 25.0 * grad_mag
        print(f"  Auto-calibrated elevation proxy: '{best_formula}' (Separation score: {best_score:.2f}). Grid: {gh}x{gw}")

    def add_town(self, name, x, y):
        self.towns[name] = (float(x), float(y))
        print(f"[Dynamic] Added/Updated town '{name}' at ({x:.1f}, {y:.1f})")
        
    def remove_town(self, name):
        if name in self.towns:
            del self.towns[name]
            if self.towns:
                for ind in self.industries:
                    if ind['town'] == name:
                        nearest = min(self.towns.keys(), key=lambda t: math.hypot(self.towns[t][0] - ind['x'], self.towns[t][1] - ind['y']))
                        ind['town'] = nearest
            print(f"[Dynamic] Removed town '{name}'")
            
    def add_industry(self, itype, x, y, town=None):
        if town is None:
            town = min(self.towns.keys(), key=lambda t: math.hypot(self.towns[t][0] - x, self.towns[t][1] - y))
        new_id = f"ind_{len(self.industries)}"
        ind = {
            'id': new_id,
            'town': town,
            'type': itype,
            'x': float(x),
            'y': float(y),
            'full_name': f"{town} {itype}"
        }
        self.industries.append(ind)
        print(f"[Dynamic] Added industry '{itype}' at ({x:.1f}, {y:.1f}) assigned to '{town}'")
        return new_id
        
    def remove_industry(self, ind_id):
        self.industries = [ind for ind in self.industries if ind['id'] != ind_id]
        print(f"[Dynamic] Removed industry '{ind_id}'")

    def find_route(self, p1, p2, anchor_radius=400.0):
        c1, r1 = self._to_grid(p1[0], p1[1])
        c2, r2 = self._to_grid(p2[0], p2[1])
        
        if self.rail_tree is not None:
            d1, idx1 = self.rail_tree.query(p1)
            d2, idx2 = self.rail_tree.query(p2)

            anchor_idx = None
            if d1 < anchor_radius and d2 > anchor_radius:
                anchor_idx = idx1
            elif d2 < anchor_radius and d1 > anchor_radius:
                anchor_idx = idx2

            if anchor_idx is not None:
                anc = self.rail_pts[anchor_idx]
                ca, ra = self._to_grid(anc[0], anc[1])
                rt1, cst1 = route_through_array(self.cost_surface, (r1, c1), (ra, ca), fully_connected=True, geometric=True)
                rt2, cst2 = route_through_array(self.cost_surface, (ra, ca), (r2, c2), fully_connected=True, geometric=True)
                full_px = rt1 + rt2[1:]
                return [self._to_svg(p[1], p[0]) for p in full_px], cst1 + cst2

        rt, cst = route_through_array(self.cost_surface, (r1, c1), (r2, c2), fully_connected=True, geometric=True)
        return [self._to_svg(p[1], p[0]) for p in rt], cst

    def optimize_cargo_network(self, k_clusters=None):
        print("[Cargo] Optimizing regional hubs and supply chains...")
        if not self.industries:
            print("  [Notice] 0 industries found in SVG map. Regional freight hubs placed at multi-town cluster geometric centroids.")
        t_counts = Counter(ind['town'] for ind in self.industries)
        t_list = sorted(self.towns.keys())
        N = len(t_list)
        if N == 0: return None
        
        K = k_clusters or max(4, min(12, round(N / 3)))
        K = min(K, N)
        
        X = np.array([self.towns[t] for t in t_list])
        W = np.array([max(1, t_counts[t]) for t in t_list], dtype=float)
        
        np.random.seed(42)
        init_idx = np.random.choice(len(X), size=K, replace=False, p=W/W.sum())
        centers = X[init_idx].copy()
        
        for _ in range(100):
            dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_centers = np.zeros_like(centers)
            for k in range(K):
                mask = (labels == k)
                if np.sum(mask) == 0:
                    new_centers[k] = X[np.random.choice(len(X))]
                else:
                    new_centers[k] = np.sum(X[mask] * W[mask, None], axis=0) / np.sum(W[mask])
            if np.allclose(centers, new_centers, atol=1e-3):
                break
            centers = new_centers
            
        hubs = []
        for k in range(K):
            c_towns = [t_list[i] for i in range(N) if labels[i] == k]
            c_inds = [ind for ind in self.industries if ind['town'] in c_towns]
            
            if c_inds:
                cx = float(np.mean([ind['x'] for ind in c_inds]))
                cy = float(np.mean([ind['y'] for ind in c_inds]))
            else:
                cx = float(np.mean([self.towns[t][0] for t in c_towns]))
                cy = float(np.mean([self.towns[t][1] for t in c_towns]))
                
            nearest_t = min(self.towns.keys(), key=lambda t: math.hypot(self.towns[t][0] - cx, self.towns[t][1] - cy))
            offset = math.hypot(self.towns[nearest_t][0] - cx, self.towns[nearest_t][1] - cy)
            # Ensure regional hub yard is visibly offset from the town center (min 250m)
            if offset < 150.0:
                cx += 250.0
                cy += 250.0
                offset = math.hypot(self.towns[nearest_t][0] - cx, self.towns[nearest_t][1] - cy)
            primary_town = max(c_towns, key=lambda t: t_counts[t]) if c_towns else nearest_t
            
            hubs.append({
                'id': k,
                'name': f"Hub {k+1} ({primary_town} Yard)",
                'primary_town': primary_town,
                'towns': c_towns,
                'industries': c_inds,
                'industry_count': len(c_inds),
                'centroid': (cx, cy),
                'nearest_town': nearest_t,
                'town_offset': offset
            })
            
        hub_dist = np.zeros((K, K), dtype=float)
        for i in range(K):
            for j in range(i + 1, K):
                d = math.hypot(hubs[i]['centroid'][0] - hubs[j]['centroid'][0],
                                hubs[i]['centroid'][1] - hubs[j]['centroid'][1])
                hub_dist[i, j] = hub_dist[j, i] = d
        hub_mst = csgraph.minimum_spanning_tree(csr_matrix(hub_dist)).toarray() if K > 1 else np.zeros((K, K))
        trunk_edges = list(zip(*np.nonzero(np.triu(hub_mst))))

        trunk_routes = []
        for i, j in trunk_edges:
            h1, h2 = hubs[i], hubs[j]
            pts, cst = self.find_route(h1['centroid'], h2['centroid'])
            trunk_routes.append({'from_hub': h1['name'], 'to_hub': h2['name'], 'path_svg': pts, 'cost': cst})
            
        spur_routes = []
        for h in hubs:
            for t in h['towns']:
                pts, cst = self.find_route(self.towns[t], h['centroid'])
                spur_routes.append({'town': t, 'hub': h['name'], 'path_svg': pts, 'cost': cst})
                
        max_vol_hub = max(hubs, key=lambda h: h['industry_count']) if hubs else None
        flagged_hauls = []
        for h in hubs:
            ind_types = Counter(ind['type'] for ind in h['industries'])
            for fac, cnt in ind_types.items():
                info = self.chains.get(fac)
                if not info: continue
                consumer = info['consumer']
                if consumer is not None:
                    if isinstance(consumer, list):
                        has_consumer = any(ind_types[c] > 0 for c in consumer)
                        c_str = ' or '.join(consumer)
                    else:
                        has_consumer = ind_types[consumer] > 0
                        c_str = consumer
                    if not has_consumer:
                        flagged_hauls.append({
                            'origin_hub': h['name'],
                            'facility': fac,
                            'facility_count': cnt,
                            'cargo': info['product'],
                            'required_consumer': c_str,
                            'car_type': info['car'],
                            'is_priority_risk': (h['id'] == max_vol_hub['id']) if max_vol_hub else False
                        })
                        
        self.cargo_plan = {
            'hubs': hubs,
            'trunk_edges': trunk_edges,
            'trunk_routes': trunk_routes,
            'spur_routes': spur_routes,
            'flagged_hauls': flagged_hauls,
            'max_volume_hub': max_vol_hub
        }
        return self.cargo_plan

    def optimize_passenger_network(self):
        """
        Builds a clustered star network: reuses cargo's k-means town clusters as
        regional groups, picks each cluster's 1-median town as its hub (the town
        minimizing total terrain-cost distance to its own cluster-mates), gives
        every other member a direct spoke straight to that hub (strict star, not
        a cost-minimizing tree -- legibility over construction cost), then links
        the regional hubs with an intercity backbone (MST over hub towns, same
        technique as the cargo trunk).
        """
        print("[Passenger] Building clustered star network with intercity backbone...")
        if not self.cargo_plan:
            print("  [Notice] Passenger star clusters reuse cargo's k-means regions; run optimize_cargo_network() first.")
            return None

        t_list = sorted(self.towns.keys())
        N = len(t_list)
        if N < 2: return None
        t_idx = {t: i for i, t in enumerate(t_list)}

        adj = np.zeros((N, N), dtype=float)
        for i in range(N):
            p1 = self.towns[t_list[i]]
            c1, r1 = self._to_grid(p1[0], p1[1])
            for j in range(i + 1, N):
                p2 = self.towns[t_list[j]]
                c2, r2 = self._to_grid(p2[0], p2[1])
                steps = int(max(abs(r2 - r1), abs(c2 - c1), 10))
                rr = np.linspace(r1, r2, steps).astype(int)
                cc = np.linspace(c1, c2, steps).astype(int)
                cst = np.sum(self.cost_surface[rr, cc]) * (math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / steps)
                adj[i, j] = adj[j, i] = cst

        clusters = []
        for h in self.cargo_plan['hubs']:
            towns = h['towns']
            if not towns:
                continue
            if len(towns) == 1:
                hub_town = towns[0]
            else:
                local_idx = [t_idx[t] for t in towns]
                sub = adj[np.ix_(local_idx, local_idx)]
                hub_town = towns[int(np.argmin(sub.sum(axis=1)))]
            clusters.append({'hub': hub_town, 'members': [t for t in towns if t != hub_town]})

        hub_towns = [c['hub'] for c in clusters]

        spoke_routes = []
        for c in clusters:
            for t in c['members']:
                pts, cst = self.find_route(self.towns[t], self.towns[c['hub']])
                spoke_routes.append({'town': t, 'hub': c['hub'], 'path_svg': pts, 'cost': cst})

        backbone_routes = []
        K = len(hub_towns)
        if K > 1:
            hub_idx = [t_idx[h] for h in hub_towns]
            hub_mst = csgraph.minimum_spanning_tree(csr_matrix(adj[np.ix_(hub_idx, hub_idx)])).toarray()
            hub_mst = hub_mst + hub_mst.T
            for i, j in zip(*np.nonzero(np.triu(hub_mst))):
                h1, h2 = hub_towns[i], hub_towns[j]
                pts, cst = self.find_route(self.towns[h1], self.towns[h2])
                backbone_routes.append({'from_hub': h1, 'to_hub': h2, 'path_svg': pts, 'cost': cst})

        print(f"  Built {len(clusters)} regional star cluster(s), {len(spoke_routes)} spoke(s), {len(backbone_routes)} backbone link(s).")

        self.pass_plan = {
            'hub_towns': hub_towns,
            'clusters': clusters,
            'spoke_routes': spoke_routes,
            'backbone_routes': backbone_routes
        }
        return self.pass_plan

    def _reconcile_stretches(self, ref_arr, ref_tree, cand_pts, threshold, min_len, min_pass_span,
                              action, offset_dist, corridor_start_idx, label, tier='main'):
        """
        Finds stretches where `cand_pts` runs within `threshold` of the reference
        polyline (ref_arr/ref_tree) for at least `min_len`, and merges/offsets them.
        Returns (corridor_records, updated_cand_pts). Shared logic behind both the
        passenger-backbone-vs-cargo-trunk check and the spoke-vs-spoke check.
        """
        c_pts = np.array(cand_pts)
        if len(c_pts) < 3:
            return [], cand_pts

        N_orig = len(c_pts)
        orig_start, orig_end = tuple(c_pts[0]), tuple(c_pts[-1])

        dists, p_indices = ref_tree.query(c_pts)
        in_corr = (dists <= threshold)

        stretches = []
        start = None
        for i, ok in enumerate(in_corr):
            if ok and start is None:
                start = i
            elif not ok and start is not None:
                stretches.append((start, i - 1))
                start = None
        if start is not None:
            stretches.append((start, len(c_pts) - 1))

        records = []
        counter = corridor_start_idx
        # Process stretches in reverse order so array splicing does not invalidate preceding indices
        for s, e in reversed(stretches):
            seg = c_pts[s:e+1]
            diffs = np.diff(seg, axis=0)
            slen = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))

            p_s = int(p_indices[s])
            p_e = int(p_indices[e])
            p_min, p_max = min(p_s, p_e), max(p_s, p_e)
            p_seg = ref_arr[p_min:p_max+1]
            p_len = float(np.sum(np.hypot(np.diff(p_seg, axis=0)[:, 0], np.diff(p_seg, axis=0)[:, 1]))) if len(p_seg) > 1 else 0.0

            if slen >= min_len and p_len >= min_pass_span:
                counter += 1

                costs = []
                for pt in seg:
                    gc, gr = self._to_grid(pt[0], pt[1])
                    costs.append(float(self.cost_surface[gr, gc]))
                mean_cost = float(np.mean(costs))

                if action == 'merge':
                    chosen_act = 'MERGED_SHARED_ROW'
                elif action == 'offset':
                    chosen_act = 'DELIBERATE_OFFSET'
                elif action == 'auto':
                    chosen_act = 'MERGED_SHARED_ROW' if mean_cost >= 1.35 else 'DELIBERATE_OFFSET'
                else:
                    chosen_act = 'REPORT_ONLY'

                if p_s <= p_e:
                    shared_pts = [tuple(p) for p in ref_arr[p_s:p_e+1]]
                else:
                    shared_pts = [tuple(p) for p in ref_arr[p_e:p_s+1][::-1]]

                if chosen_act == 'MERGED_SHARED_ROW':
                    new_c_path = list(map(tuple, c_pts[:s])) + shared_pts + list(map(tuple, c_pts[e+1:]))
                    # A merge stretch touching the candidate's own start/end drops that anchor
                    # (town/hub) since the reference path doesn't necessarily reach it -- restore it.
                    if s == 0:
                        new_c_path.insert(0, orig_start)
                    if e == N_orig - 1:
                        new_c_path.append(orig_end)
                    c_pts = np.array(new_c_path)
                    corr_path = shared_pts
                elif chosen_act == 'DELIBERATE_OFFSET':
                    N_seg = len(seg)
                    idx_arr = np.arange(N_seg)
                    taper = np.sin(np.pi * idx_arr / max(1, N_seg - 1))
                    vecs = seg - ref_arr[p_indices[s:e+1]]
                    v_dists = np.hypot(vecs[:, 0], vecs[:, 1])
                    v_dists[v_dists == 0] = 1.0
                    u_vecs = vecs / v_dists[:, None]
                    needed_shift = np.maximum(0.0, offset_dist - v_dists)
                    offset_seg = seg + (taper * needed_shift)[:, None] * u_vecs
                    new_c_path = list(map(tuple, c_pts[:s])) + [tuple(p) for p in offset_seg] + list(map(tuple, c_pts[e+1:]))
                    if s == 0:
                        new_c_path.insert(0, orig_start)
                    if e == N_orig - 1:
                        new_c_path.append(orig_end)
                    c_pts = np.array(new_c_path)
                    corr_path = [tuple(p) for p in offset_seg]
                else:
                    corr_path = [tuple(p) for p in seg]

                est_savings = round(slen * mean_cost * 1.5, 1) if chosen_act == 'MERGED_SHARED_ROW' else 0.0

                record = {
                    'id': f"corridor_{counter}",
                    'name': f"Corridor {counter} ({label})",
                    'tier': tier,
                    'length_m': slen,
                    'pass_span_m': p_len,
                    'avg_separation_m': float(np.mean(dists[s:e+1])),
                    'mean_terrain_cost': mean_cost,
                    'action': chosen_act,
                    'shared_path': corr_path,
                    'entry_point': corr_path[0],
                    'exit_point': corr_path[-1],
                    'estimated_grading_saved': est_savings
                }
                records.append(record)

        return records, [tuple(p) for p in c_pts]

    def reconcile_corridors(self, threshold=250.0, min_len=350.0, min_pass_span=100.0, action='auto', offset_dist=150.0):
        """
        Reconciles overlapping track corridors in two places:
        1. Passenger intercity backbone vs. cargo trunk -- where both networks'
           hub-to-hub trunks run parallel, merge into a shared 4-track ROW.
        2. Spoke vs. spoke -- two star spokes from the same regional hub can run
           near-parallel close to the hub; merge/offset those too.

        Action policies:
        - 'merge': Unify into a shared 4-track Right-of-Way (ROW), sharing grading, tunnels, and bridges.
        - 'offset': Apply deliberate clearance displacement (offset_dist) to avoid collision/overlap.
        - 'auto': In hilly/mountain/canyon corridors (mean cost >= 1.35), merge into shared ROW.
                  In wide flat plains (mean cost < 1.35), apply deliberate offset.
        - 'none': Keep routes unchanged, just report.
        """
        if not self.pass_plan or not self.cargo_plan:
            print("[Corridors] Skipping reconciliation: both passenger and cargo plans required.")
            return []

        print(f"[Corridors] Analyzing trunk polyline proximity (threshold={threshold:.0f}m, min_len={min_len:.0f}m, strategy='{action}')...")
        self.corridors = []

        def log_committed(recs):
            for r in recs:
                print(f"  [{r['action']}] {r['name']}: {r['length_m']:.0f}m (Pass Span: {r['pass_span_m']:.0f}m, "
                      f"Avg Sep: {r['avg_separation_m']:.1f}m, Cost: {r['mean_terrain_cost']:.2f})")

        # 0. Cargo spur vs. cargo trunk. Unlike backbone/trunk or spoke/spoke, both sides here
        # are freight -- there's no reason to keep a dotted feeder spur running directly on top
        # of a solid trunk line just because a member town happens to sit almost on the trunk's
        # own path (e.g. Manokwari sitting on the Aceh<->Surabaya trunk). Trim the spur back to
        # where it first joins a trunk for a real stretch, snapped onto the trunk's own point so
        # it still visually touches it, instead of drawing a redundant duplicate track.
        for spur in self.cargo_plan['spur_routes']:
            s_pts = np.array(spur['path_svg'])
            if len(s_pts) < 3:
                continue
            for t_route in self.cargo_plan['trunk_routes']:
                t_pts = [tuple(p) for p in t_route['path_svg']]
                if len(t_pts) < 2:
                    continue
                t_arr = np.array(t_pts)
                dists, t_idx = cKDTree(t_arr).query(s_pts)
                in_corr = dists <= threshold
                i, cut = 0, None
                while i < len(in_corr):
                    if in_corr[i]:
                        j = i
                        while j < len(in_corr) and in_corr[j]:
                            j += 1
                        seg = s_pts[i:j]
                        slen = float(np.sum(np.hypot(np.diff(seg, axis=0)[:, 0], np.diff(seg, axis=0)[:, 1]))) if len(seg) > 1 else 0.0
                        if slen >= min_len:
                            cut = i
                            break
                        i = j
                    else:
                        i += 1
                if cut is not None:
                    snap_pt = tuple(t_arr[t_idx[cut]])
                    spur['path_svg'] = [tuple(p) for p in s_pts[:cut]] + [snap_pt]
                    print(f"  [SPUR_TRIMMED] {spur['town']} -> {spur['hub']}: redundant with "
                          f"{t_route['from_hub']} <-> {t_route['to_hub']} trunk past {cut} of {len(s_pts)} points")
                    break

        # 1. Passenger intercity backbone vs. cargo trunk. Each backbone_routes entry is its
        # own continuous polyline (an individual MST edge) -- unlike the old single main_line,
        # the backbone as a whole is a bag of disjoint edges, not one connected path, so each
        # edge must be checked against cargo trunks separately rather than concatenated into
        # one reference polyline (concatenating them would splice fake "jumps" between
        # unrelated edges into the shared-path geometry).
        #
        # Multiple backbone edges can share one vertex (a hub where 3+ lines fork), so a cargo
        # trunk passing near that fork is genuinely close to all of them at once. Matching it
        # against every backbone edge in turn -- and re-testing the already-spliced result each
        # time -- produced several overlapping MERGED_SHARED_ROW records stacked on the same
        # spot (redundant track near Surabaya, a corrupted stub near Mataram). Fixed by scoring
        # each cargo trunk against every backbone edge using its original, unmutated path, and
        # committing only the single best-overlapping match.
        for c_route in self.cargo_plan['trunk_routes']:
            best = None
            for bb_route in self.pass_plan['backbone_routes']:
                ref_pts = [tuple(p) for p in bb_route['path_svg']]
                if len(ref_pts) < 2:
                    continue
                ref_arr = np.array(ref_pts)
                ref_tree = cKDTree(ref_arr)
                label = f"{bb_route['from_hub']} ↔ {bb_route['to_hub']} Backbone / {c_route['from_hub']} ↔ {c_route['to_hub']} Cargo Trunk"
                recs, new_path = self._reconcile_stretches(ref_arr, ref_tree, c_route['path_svg'],
                                                             threshold, min_len, min_pass_span,
                                                             action, offset_dist, len(self.corridors), label)
                if recs:
                    score = sum(r['length_m'] for r in recs)
                    if best is None or score > best[2]:
                        best = (recs, new_path, score)
            if best:
                recs, new_path, _ = best
                c_route['path_svg'] = new_path
                log_committed(recs)
                self.corridors.extend(recs)

        # 2. Cargo spur vs. passenger spoke. Different traffic than the spur-vs-trunk trim
        # above (freight feeder vs. passenger feeder, not freight vs. freight), so a nearby
        # pair isn't pure redundancy -- merge them into a shared ROW the same way backbone and
        # trunk do. Same fork-vertex risk as step 1: score every spur against every spoke using
        # its original path and commit only the single best match.
        for spur in self.cargo_plan['spur_routes']:
            best = None
            for spoke in self.pass_plan['spoke_routes']:
                ref_pts = [tuple(p) for p in spoke['path_svg']]
                if len(ref_pts) < 2:
                    continue
                ref_arr = np.array(ref_pts)
                ref_tree = cKDTree(ref_arr)
                label = f"{spoke['town']} spoke @ {spoke['hub']} / {spur['town']} spur @ {spur['hub']}"
                recs, new_path = self._reconcile_stretches(ref_arr, ref_tree, spur['path_svg'],
                                                             threshold, min_len, min_pass_span,
                                                             action, offset_dist, len(self.corridors), label,
                                                             tier='feeder')
                if recs:
                    score = sum(r['length_m'] for r in recs)
                    if best is None or score > best[2]:
                        best = (recs, new_path, score)
            if best:
                recs, new_path, _ = best
                spur['path_svg'] = new_path
                log_committed(recs)
                self.corridors.extend(recs)

        # 3. Spoke vs. spoke within each regional star cluster. Same fork problem as above --
        # every spoke in a cluster starts at the same hub vertex, so with 3+ spokes a candidate
        # could get merged against more than one reference spoke in turn. Score all pairs from
        # the original (pristine) paths first, then commit only each spoke's single best match.
        for cluster in self.pass_plan['clusters']:
            hub = cluster['hub']
            spokes = [r for r in self.pass_plan['spoke_routes'] if r['hub'] == hub]
            if len(spokes) < 2:
                continue
            pristine = [list(map(tuple, s['path_svg'])) for s in spokes]
            best_for = {}
            for i in range(len(spokes)):
                ref_pts = pristine[i]
                if len(ref_pts) < 2:
                    continue
                ref_arr = np.array(ref_pts)
                ref_tree = cKDTree(ref_arr)
                for j in range(i + 1, len(spokes)):
                    label = f"{spokes[i]['town']} & {spokes[j]['town']} spokes @ {hub}"
                    recs, new_path = self._reconcile_stretches(ref_arr, ref_tree, pristine[j],
                                                                 threshold, min_len, min_pass_span,
                                                                 action, offset_dist, len(self.corridors), label)
                    if recs:
                        score = sum(r['length_m'] for r in recs)
                        if j not in best_for or score > best_for[j][2]:
                            best_for[j] = (recs, new_path, score)
            for j, (recs, new_path, _) in best_for.items():
                spokes[j]['path_svg'] = new_path
                log_committed(recs)
                self.corridors.extend(recs)

        total_shared_len = sum(c['length_m'] for c in self.corridors if c['action'] == 'MERGED_SHARED_ROW')
        print(f"[Corridors] Complete: {len(self.corridors)} corridors analyzed, {total_shared_len:.0f}m consolidated into shared ROW.")
        return self.corridors

    def flag_missed_stops(self, threshold=150.0):
        """
        Report-only check: flags towns that an intercity backbone (passenger) or
        trunk (cargo) route's terrain path runs within `threshold` meters of,
        without that town being one of the route's own hub endpoints. These are
        candidate stations -- track already runs past them -- but nothing is
        built or rerouted automatically; express lines are meant to skip towns
        that aren't their endpoints.
        """
        self.missed_stops = []
        if not self.pass_plan and not self.cargo_plan:
            print("[MissedStops] Skipping: no passenger or cargo plan to check.")
            return self.missed_stops

        def check(routes, kind, exclude_fn):
            for r in routes:
                path = np.array(r['path_svg'])
                if len(path) < 2:
                    continue
                tree = cKDTree(path)
                excluded = exclude_fn(r)
                for t_name, (tx, ty) in self.towns.items():
                    if t_name in excluded:
                        continue
                    dist, _ = tree.query((tx, ty))
                    if dist <= threshold:
                        self.missed_stops.append({
                            'kind': kind,
                            'route': f"{r['from_hub']} <-> {r['to_hub']}",
                            'town': t_name,
                            'distance_m': float(dist)
                        })
                        print(f"  [MISSED_STOP] {kind} {r['from_hub']} <-> {r['to_hub']} passes {dist:.0f}m from "
                              f"'{t_name}' without stopping -- candidate station.")

        if self.pass_plan:
            # Backbone endpoints are real town names (each cluster hub is a town).
            check(self.pass_plan['backbone_routes'], 'Backbone', lambda r: {r['from_hub'], r['to_hub']})
        if self.cargo_plan:
            # Trunk endpoints are hub yard names, not towns -- exclude every town
            # already in either endpoint hub's own cluster (it already has a spur there).
            hubs_by_name = {h['name']: h for h in self.cargo_plan['hubs']}
            check(self.cargo_plan['trunk_routes'], 'Trunk',
                  lambda r: set(hubs_by_name[r['from_hub']]['towns']) | set(hubs_by_name[r['to_hub']]['towns']))

        print(f"[MissedStops] {len(self.missed_stops)} pass-by town(s) flagged as candidate stations.")
        return self.missed_stops

    # --------------------------------------------------------------------------
    # RENDERING: STANDALONE SVG + PNG (PASSENGER, CARGO, COMBINED)
    # --------------------------------------------------------------------------
    def render(self, out_dir="."):
        print("[Render] Generating high-resolution composite maps in SVG and PNG...")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        mx, my, mw, mh = self.map_bounds

        # Helper to plot base layer
        COAST_COLOR = '#64748b'  # Muted slate gray for river context
        def plot_base(ax):
            ax.imshow(self.scaled_relief, extent=[mx, mx + mw, my + mh, my], aspect='equal')
            if self.contour_segments:
                # A separate ax.plot()/LineCollection entry per segment emits one <path>
                # per segment in the SVG output (tens of thousands of them) -- NaN-separated
                # single arrays collapse it to one Line2D artist / one <path> element.
                cxs, cys = [], []
                for (x1, y1), (x2, y2) in self.contour_segments:
                    cxs.extend([x1, x2, np.nan])
                    cys.extend([y1, y2, np.nan])
                ax.plot(cxs, cys, color='#78350f', linewidth=0.5, alpha=0.45, zorder=1)
            for (x1, y1), (x2, y2) in self.coastline:
                ax.plot([x1, x2], [y1, y2], color=COAST_COLOR, linewidth=0.8, alpha=0.5)
            for (x1, y1), (x2, y2) in self.rail_segments:
                ax.plot([x1, x2], [y1, y2], color='#000000', linewidth=3.0, alpha=0.9, zorder=3)
                ax.plot([x1, x2], [y1, y2], color='#fbbf24', linewidth=1.5, linestyle='--', zorder=4)

        # ----------------------------------------------------------------------
        # 1. PASSENGER NETWORK (GREEN) (SVG & PNG)
        # ----------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 30), dpi=150)
        plot_base(ax)
        
        # Consistent Passenger Hue: Emerald Green (#16a34a)
        PASS_COLOR = '#16a34a'
        PASS_CASING = '#14532d'
        SPOKE_COLOR = '#86e5ae'

        # Star spokes: light green, from each town straight to its regional hub
        for sp in self.pass_plan['spoke_routes']:
            xs, ys = [p[0] for p in sp['path_svg']], [p[1] for p in sp['path_svg']]
            ax.plot(xs, ys, color=SPOKE_COLOR, linewidth=2.5, alpha=0.9, zorder=5)

        # Intercity backbone: dark green, solid with casing (hub-to-hub trunk)
        for bb in self.pass_plan['backbone_routes']:
            xs, ys = [p[0] for p in bb['path_svg']], [p[1] for p in bb['path_svg']]
            ax.plot(xs, ys, color=PASS_CASING, linewidth=4.5, alpha=0.95, zorder=6)
            ax.plot(xs, ys, color=PASS_COLOR, linewidth=2.2, alpha=1.0, zorder=7)

        hub_set = set(self.pass_plan['hub_towns'])
        for t_name, (tx, ty) in self.towns.items():
            if t_name in hub_set:
                ax.scatter(tx, ty, s=180, facecolor=PASS_COLOR, edgecolor='#ffffff', linewidth=2.0, zorder=10)
                ax.text(tx, ty - 45, t_name, fontsize=9, fontweight='bold', ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffff', alpha=0.85, edgecolor=PASS_COLOR, lw=1.4), zorder=11)
            else:
                ax.scatter(tx, ty, s=110, facecolor=SPOKE_COLOR, edgecolor='#ffffff', linewidth=1.5, zorder=9)
                ax.text(tx, ty - 35, t_name, fontsize=8, ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffff', alpha=0.75, edgecolor=PASS_COLOR, lw=0.8), zorder=11)

        ax.set_xlim(mx, mx + mw)
        ax.set_ylim(my + mh, my)
        ax.set_title("Transport Fever 2 — Passenger Rail Network (Green)\nRegional Star Clusters & Intercity Backbone", fontsize=14, fontweight='bold', pad=15)
        ax.axis('off')

        custom_legend_pass = [
            plt.Line2D([0], [0], color=PASS_COLOR, lw=4, label='Intercity Backbone (Double Track - Solid Green)'),
            plt.Line2D([0], [0], color=SPOKE_COLOR, lw=2.5, label='Regional Star Spoke (Single Track - Light Green)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=PASS_COLOR, markeredgecolor='k', markersize=10, label='Regional Hub Station (Green)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=SPOKE_COLOR, markeredgecolor='k', markersize=7, label='Local Station (Light Green)'),
            plt.Line2D([0], [0], color=COAST_COLOR, lw=1.2, label='River / Coastline')
        ]
        ax.legend(handles=custom_legend_pass, loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
        plt.tight_layout()
        
        pass_svg = out_dir / "passenger_network.svg"
        plt.savefig(pass_svg, format='svg', bbox_inches='tight')
        plt.close()
        optimize_svg_raster(pass_svg)
        print(f"  Saved {pass_svg}")

        # ----------------------------------------------------------------------
        # 2. CARGO NETWORK (BLUE) (SVG & PNG)
        # ----------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 30), dpi=150)
        plot_base(ax)

        # Hub catchment zoning underlay: nearest-hub-centroid color per point (the
        # same straight-line metric optimize_cargo_network() used to assign towns
        # to hubs), so a future industry can be matched to a hub by its zone color.
        hubs = self.cargo_plan['hubs']
        hub_tree = cKDTree(np.array([h['centroid'] for h in hubs]))
        zgh, zgw = self.grid_shape
        zcols, zrows = np.meshgrid(np.arange(zgw), np.arange(zgh))
        zx = mx + (zcols + 0.5) / zgw * mw
        zy = my + (zrows + 0.5) / zgh * mh
        _, zone_idx = hub_tree.query(np.column_stack([zx.ravel(), zy.ravel()]))
        zone_grid = zone_idx.reshape(zgh, zgw)
        K = len(hubs)
        zone_cmap = plt.get_cmap('tab20', max(K, 1))
        ax.imshow(zone_grid, extent=[mx, mx + mw, my + mh, my], aspect='equal',
                  cmap=zone_cmap, vmin=-0.5, vmax=K - 0.5, alpha=0.35, interpolation='nearest', zorder=2)

        # Consistent Cargo Hue: Royal Blue (#2563eb)
        CARGO_COLOR = '#2563eb'
        CARGO_CASING = '#1e3a8a'

        for s in self.cargo_plan['spur_routes']:
            xs, ys = [p[0] for p in s['path_svg']], [p[1] for p in s['path_svg']]
            ax.plot(xs, ys, color=CARGO_COLOR, linewidth=2.2, linestyle=':', alpha=0.9, zorder=5)
        for t in self.cargo_plan['trunk_routes']:
            xs, ys = [p[0] for p in t['path_svg']], [p[1] for p in t['path_svg']]
            ax.plot(xs, ys, color=CARGO_CASING, linewidth=4.5, alpha=0.95, zorder=6)
            ax.plot(xs, ys, color=CARGO_COLOR, linewidth=2.2, alpha=1.0, zorder=7)
            
        for t_name, (tx, ty) in self.towns.items():
            ax.scatter(tx, ty, s=70, facecolor='#64748b', edgecolor='#ffffff', linewidth=1.2, zorder=8)
            ax.text(tx, ty - 30, t_name, fontsize=7.5, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='#ffffff', alpha=0.7, edgecolor='#cbd5e1', lw=0.6), zorder=9)
                    
        for i, h in enumerate(hubs):
            hx, hy = h['centroid']
            ax.scatter(hx, hy, s=120, marker='D', facecolor=CARGO_COLOR, edgecolor='#ffffff', linewidth=2.0, zorder=12)
            ax.text(hx, hy - 35, h['name'], fontsize=7.5, fontweight='bold', ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='#ffffff', alpha=0.85, edgecolor=CARGO_COLOR, lw=0.8), zorder=13)

        ax.set_xlim(mx, mx + mw)
        ax.set_ylim(my + mh, my)
        ax.set_title("Transport Fever 2 — Freight Trunk Network & Hub Yards (Blue)\nIndustry Centroid Yards (Offset from Towns) & Feeder Spurs\nShaded zones: nearest-hub catchment for future industries", fontsize=14, fontweight='bold', pad=15)
        ax.axis('off')

        custom_legend_cargo = [
            plt.Line2D([0], [0], color=CARGO_COLOR, lw=4, label='Freight Trunk Line (Solid Blue)'),
            plt.Line2D([0], [0], color=CARGO_COLOR, lw=2.2, ls=':', label='Freight Feeder Spur (Dotted Blue)'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=CARGO_COLOR, markeredgecolor='w', markersize=11, label='Regional Hub Yard (Blue Diamond)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#64748b', markeredgecolor='w', markersize=8, label='Town Center Dot'),
            plt.Line2D([0], [0], color=COAST_COLOR, lw=1.2, label='River / Coastline')
        ]
        custom_legend_cargo.extend([
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=zone_cmap(i), markeredgecolor='k',
                       markersize=10, label=f"Zone: {h['name']}") for i, h in enumerate(hubs)
        ])
        ax.legend(handles=custom_legend_cargo, loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
        plt.tight_layout()
        
        cargo_svg = out_dir / "cargo_network.svg"
        plt.savefig(cargo_svg, format='svg', bbox_inches='tight')
        plt.close()
        optimize_svg_raster(cargo_svg)
        print(f"  Saved {cargo_svg}")

        # ----------------------------------------------------------------------
        # 3. COMBINED NETWORK (PASSENGER GREEN + CARGO BLUE) (SVG & PNG)
        # ----------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 30), dpi=150)
        plot_base(ax)
        
        # Cargo spurs (dotted blue)
        for s in self.cargo_plan['spur_routes']:
            xs, ys = [p[0] for p in s['path_svg']], [p[1] for p in s['path_svg']]
            ax.plot(xs, ys, color=CARGO_COLOR, linewidth=2.0, linestyle=':', alpha=0.9, zorder=5)
            
        # Passenger star spokes (light green)
        for sp in self.pass_plan['spoke_routes']:
            xs, ys = [p[0] for p in sp['path_svg']], [p[1] for p in sp['path_svg']]
            ax.plot(xs, ys, color=SPOKE_COLOR, linewidth=2.5, alpha=0.9, zorder=6)

        # Cargo trunk (solid blue)
        for t in self.cargo_plan['trunk_routes']:
            xs, ys = [p[0] for p in t['path_svg']], [p[1] for p in t['path_svg']]
            ax.plot(xs, ys, color=CARGO_CASING, linewidth=4.5, alpha=0.95, zorder=7)
            ax.plot(xs, ys, color=CARGO_COLOR, linewidth=2.2, alpha=1.0, zorder=8)

        # Passenger intercity backbone (solid double-track dark green)
        for bb in self.pass_plan['backbone_routes']:
            xs, ys = [p[0] for p in bb['path_svg']], [p[1] for p in bb['path_svg']]
            ax.plot(xs, ys, color=PASS_CASING, linewidth=4.5, alpha=0.95, zorder=9)
            ax.plot(xs, ys, color=PASS_COLOR, linewidth=2.2, alpha=1.0, zorder=10)

        # Shared Multi-Track Corridors (Consolidated 4-Track ROW)
        FEEDER_CASING = '#581c87'
        FEEDER_CORE = '#a855f7'
        has_merged_main = any(c['action'] == 'MERGED_SHARED_ROW' and c.get('tier') != 'feeder' for c in self.corridors)
        has_merged_feeder = any(c['action'] == 'MERGED_SHARED_ROW' and c.get('tier') == 'feeder' for c in self.corridors)
        for c in self.corridors:
            if c['action'] != 'MERGED_SHARED_ROW':
                continue
            c_pts = np.array(c['shared_path'])
            xs, ys = c_pts[:, 0], c_pts[:, 1]
            if c.get('tier') == 'feeder':
                # Secondary passenger spoke / freight spur merger: dashed purple, distinct from main-line ROW
                ax.plot(xs, ys, color=FEEDER_CASING, linewidth=5.5, linestyle=(0, (6, 3)), alpha=0.98, zorder=11)
                if len(c_pts) >= 2:
                    p_offset = compute_normal_offsets(c_pts, -8.0)
                    c_offset = compute_normal_offsets(c_pts, 8.0)
                    ax.plot(p_offset[:, 0], p_offset[:, 1], color=PASS_COLOR, linewidth=1.8, linestyle=(0, (4, 2)), zorder=13)
                    ax.plot(c_offset[:, 0], c_offset[:, 1], color=CARGO_COLOR, linewidth=1.8, linestyle=(0, (4, 2)), zorder=13)
                else:
                    ax.plot(xs, ys, color=FEEDER_CORE, linewidth=2.5, zorder=13)
            else:
                # 4-track wide ballast embankment
                ax.plot(xs, ys, color='#0f172a', linewidth=7.5, alpha=0.98, zorder=11)
                ax.plot(xs, ys, color='#334155', linewidth=5.2, alpha=1.0, zorder=12)
                # Parallel dual-tone tracks (green passenger + blue freight)
                if len(c_pts) >= 2:
                    p_offset = compute_normal_offsets(c_pts, -12.0)
                    c_offset = compute_normal_offsets(c_pts, 12.0)
                    ax.plot(p_offset[:, 0], p_offset[:, 1], color=PASS_COLOR, linewidth=2.2, zorder=13)
                    ax.plot(c_offset[:, 0], c_offset[:, 1], color=CARGO_COLOR, linewidth=2.2, zorder=13)
                else:
                    ax.plot(xs, ys, color='#38bdf8', linewidth=3.0, zorder=13)

        # Towns (regional hubs in dark green, local stations in light green)
        for t_name, (tx, ty) in self.towns.items():
            if t_name in hub_set:
                ax.scatter(tx, ty, s=170, facecolor=PASS_COLOR, edgecolor='#ffffff', linewidth=2.0, zorder=12)
                ax.text(tx, ty - 40, t_name, fontsize=8.5, fontweight='bold', ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffff', alpha=0.85, edgecolor=PASS_COLOR, lw=1.3), zorder=14)
            else:
                ax.scatter(tx, ty, s=110, facecolor=SPOKE_COLOR, edgecolor='#ffffff', linewidth=1.5, zorder=11)
                ax.text(tx, ty - 32, t_name, fontsize=7.5, ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffff', alpha=0.75, edgecolor=PASS_COLOR, lw=0.8), zorder=14)

        # Cargo Hub Yards (Blue Diamonds)
        for h in self.cargo_plan['hubs']:
            hx, hy = h['centroid']
            ax.scatter(hx, hy, s=110, marker='D', facecolor=CARGO_COLOR, edgecolor='#ffffff', linewidth=2.0, zorder=13)

        ax.set_xlim(mx, mx + mw)
        ax.set_ylim(my + mh, my)
        ax.set_title("Transport Fever 2 — Combined Rail Network\nPassenger Star Clusters & Backbone (Green) + Freight Trunk & Hub Yards (Blue)", fontsize=14, fontweight='bold', pad=15)
        ax.axis('off')

        custom_legend_comb = [
            plt.Line2D([0], [0], color=PASS_COLOR, lw=4, label='Intercity Backbone (Double Track - Solid Green)'),
            plt.Line2D([0], [0], color=SPOKE_COLOR, lw=2.5, label='Regional Star Spoke (Single Track - Light Green)'),
            plt.Line2D([0], [0], color=CARGO_COLOR, lw=4, label='Freight Trunk Line (Solid Blue)'),
            plt.Line2D([0], [0], color=CARGO_COLOR, lw=2.2, ls=':', label='Freight Feeder Spur (Dotted Blue)'),
        ]
        if has_merged_main:
            custom_legend_comb.extend([
                plt.Line2D([0], [0], color='#0f172a', lw=6, label='Shared 4-Track Corridor (Main Line ROW - Green/Blue)'),
            ])
        if has_merged_feeder:
            custom_legend_comb.extend([
                plt.Line2D([0], [0], color=FEEDER_CASING, lw=5, linestyle=(0, (6, 3)), label='Shared Feeder Corridor (Secondary Spoke + Spur ROW)'),
            ])
        custom_legend_comb.extend([
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=PASS_COLOR, markeredgecolor='k', markersize=9, label='Regional Hub Station (Green)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=SPOKE_COLOR, markeredgecolor='k', markersize=7, label='Local Station (Light Green)'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=CARGO_COLOR, markeredgecolor='w', markersize=10, label='Cargo Hub Yard (Blue Diamond)'),
            plt.Line2D([0], [0], color=COAST_COLOR, lw=1.2, label='River / Coastline')
        ])
        ax.legend(handles=custom_legend_comb, loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
        plt.tight_layout()
        
        comb_svg = out_dir / "combined_network.svg"
        plt.savefig(comb_svg, format='svg', bbox_inches='tight')
        plt.close()
        optimize_svg_raster(comb_svg)
        print(f"  Saved {comb_svg}")

        # ----------------------------------------------------------------------
        # 4. INJECT NATIVE INKSCAPE LAYERS INTO ORIGINAL SVG
        # ----------------------------------------------------------------------
        self.export_layered_svg(out_dir / "map_export_with_networks.svg")

        return {
            'passenger_svg': pass_svg,
            'cargo_svg': cargo_svg,
            'combined_svg': comb_svg,
            'layered_svg': out_dir / "map_export_with_networks.svg"
        }

    def export_layered_svg(self, out_svg_path):
        """Inject planned networks into the original SVG as native Inkscape layers with consistent colors."""
        print(f"[Export] Injecting native Inkscape layers into {out_svg_path}...")
        with open(self.svg_path, 'rb') as f:
            tree = etree.parse(f)
        root = tree.getroot()
        
        ns_inkscape = "http://www.inkscape.org/namespaces/inkscape"
        PASS_COLOR = "#16a34a"
        PASS_CASING = "#14532d"
        SPOKE_COLOR = "#86e5ae"
        CARGO_COLOR = "#2563eb"
        CARGO_CASING = "#1e3a8a"

        # 1. Passenger Layer (Green)
        g_pass = etree.SubElement(root, "g")
        g_pass.attrib[f"{{{ns_inkscape}}}groupmode"] = "layer"
        g_pass.attrib[f"{{{ns_inkscape}}}label"] = "Planned Passenger Network (Green)"
        g_pass.attrib["id"] = "layer_planned_passenger"

        # Add star spoke paths (light green)
        for sp in self.pass_plan['spoke_routes']:
            p_el = etree.SubElement(g_pass, "path")
            p_el.attrib["d"] = path_to_svg_d(sp['path_svg'])
            p_el.attrib["stroke"] = SPOKE_COLOR
            p_el.attrib["stroke-width"] = "4"
            p_el.attrib["fill"] = "none"
            p_el.attrib["stroke-linecap"] = "round"
            p_el.attrib["stroke-linejoin"] = "round"
            p_el.attrib["opacity"] = "0.9"

        # Add intercity backbone paths (solid dark green with casing)
        for bb in self.pass_plan['backbone_routes']:
            p_casing = etree.SubElement(g_pass, "path")
            p_casing.attrib["d"] = path_to_svg_d(bb['path_svg'])
            p_casing.attrib["stroke"] = PASS_CASING
            p_casing.attrib["stroke-width"] = "8"
            p_casing.attrib["fill"] = "none"
            p_casing.attrib["stroke-linecap"] = "round"
            p_casing.attrib["stroke-linejoin"] = "round"
            p_casing.attrib["opacity"] = "0.95"

            p_core = etree.SubElement(g_pass, "path")
            p_core.attrib["d"] = path_to_svg_d(bb['path_svg'])
            p_core.attrib["stroke"] = PASS_COLOR
            p_core.attrib["stroke-width"] = "4"
            p_core.attrib["fill"] = "none"
            p_core.attrib["stroke-linecap"] = "round"
            p_core.attrib["stroke-linejoin"] = "round"

        # 2. Cargo Layer (Blue)
        g_cargo = etree.SubElement(root, "g")
        g_cargo.attrib[f"{{{ns_inkscape}}}groupmode"] = "layer"
        g_cargo.attrib[f"{{{ns_inkscape}}}label"] = "Planned Cargo Network (Blue)"
        g_cargo.attrib["id"] = "layer_planned_cargo"
        
        # Add spurs (Dotted blue)
        for s in self.cargo_plan['spur_routes']:
            p_el = etree.SubElement(g_cargo, "path")
            p_el.attrib["d"] = path_to_svg_d(s['path_svg'])
            p_el.attrib["stroke"] = CARGO_COLOR
            p_el.attrib["stroke-width"] = "3.5"
            p_el.attrib["stroke-dasharray"] = "4,4"
            p_el.attrib["fill"] = "none"
            p_el.attrib["stroke-linecap"] = "round"
            p_el.attrib["stroke-linejoin"] = "round"
            p_el.attrib["opacity"] = "0.85"
            
        # Add trunk (Solid blue with dark casing)
        for t in self.cargo_plan['trunk_routes']:
            p_casing = etree.SubElement(g_cargo, "path")
            p_casing.attrib["d"] = path_to_svg_d(t['path_svg'])
            p_casing.attrib["stroke"] = CARGO_CASING
            p_casing.attrib["stroke-width"] = "8"
            p_casing.attrib["fill"] = "none"
            p_casing.attrib["stroke-linecap"] = "round"
            p_casing.attrib["stroke-linejoin"] = "round"
            p_casing.attrib["opacity"] = "0.95"
            
            p_core = etree.SubElement(g_cargo, "path")
            p_core.attrib["d"] = path_to_svg_d(t['path_svg'])
            p_core.attrib["stroke"] = CARGO_COLOR
            p_core.attrib["stroke-width"] = "4"
            p_core.attrib["fill"] = "none"
            p_core.attrib["stroke-linecap"] = "round"
            p_core.attrib["stroke-linejoin"] = "round"

        # Add hub yards (Blue diamonds)
        for h in self.cargo_plan['hubs']:
            hx, hy = h['centroid']
            poly = etree.SubElement(g_cargo, "polygon")
            pts_str = f"{hx},{hy-18} {hx+18},{hy} {hx},{hy+18} {hx-18},{hy}"
            poly.attrib["points"] = pts_str
            poly.attrib["fill"] = CARGO_COLOR
            poly.attrib["stroke"] = "#ffffff"
            poly.attrib["stroke-width"] = "3"
            
            txt = etree.SubElement(g_cargo, "text")
            txt.attrib["x"] = f"{hx:.1f}"
            txt.attrib["y"] = f"{hy+35:.1f}"
            txt.attrib["text-anchor"] = "middle"
            txt.attrib["font-family"] = "sans-serif"
            txt.attrib["font-size"] = "18"
            txt.attrib["font-weight"] = "bold"
            txt.attrib["fill"] = CARGO_CASING
            txt.attrib["stroke"] = "#ffffff"
            txt.attrib["stroke-width"] = "4"
            txt.attrib["paint-order"] = "stroke"
            txt.text = f"{h['name']} (Offset: {h['town_offset']:.0f}m)"

        # 3. Shared Corridors Layer (Native Inkscape Layer - 4-Track ROW)
        has_merged = any(c['action'] == 'MERGED_SHARED_ROW' for c in self.corridors)
        if has_merged:
            g_corr = etree.SubElement(root, "g")
            g_corr.attrib[f"{{{ns_inkscape}}}groupmode"] = "layer"
            g_corr.attrib[f"{{{ns_inkscape}}}label"] = "Planned Shared Corridors (4-Track ROW)"
            g_corr.attrib["id"] = "layer_planned_shared_corridors"
            
            for c in self.corridors:
                if c['action'] != 'MERGED_SHARED_ROW':
                    continue
                is_feeder = c.get('tier') == 'feeder'
                # Embankment casing
                p_emb = etree.SubElement(g_corr, "path")
                p_emb.attrib["d"] = path_to_svg_d(c['shared_path'])
                p_emb.attrib["stroke"] = "#581c87" if is_feeder else "#0f172a"
                p_emb.attrib["stroke-width"] = "10" if is_feeder else "14"
                p_emb.attrib["fill"] = "none"
                p_emb.attrib["stroke-linecap"] = "round"
                p_emb.attrib["stroke-linejoin"] = "round"
                p_emb.attrib["opacity"] = "0.95"
                if is_feeder:
                    p_emb.attrib["stroke-dasharray"] = "12,6"

                # Inner track
                p_core = etree.SubElement(g_corr, "path")
                p_core.attrib["d"] = path_to_svg_d(c['shared_path'])
                p_core.attrib["stroke"] = "#a855f7" if is_feeder else "#38bdf8"
                p_core.attrib["stroke-width"] = "4" if is_feeder else "6"
                p_core.attrib["fill"] = "none"
                p_core.attrib["stroke-linecap"] = "round"
                p_core.attrib["stroke-linejoin"] = "round"
                if is_feeder:
                    p_core.attrib["stroke-dasharray"] = "12,6"

        with open(out_svg_path, 'wb') as f:
            f.write(etree.tostring(tree, pretty_print=True, xml_declaration=True, encoding="utf-8"))
        print(f"  Layered SVG written to {out_svg_path}")

    def _fan_place(self, G, node, parent, direction, pos, unit):
        """Recursively place node's subtree as straight rays; forks fan out within a 50-degree cone."""
        children = [n for n in G.neighbors(node) if n != parent and n not in pos]
        if not children:
            return
        base_angle = math.atan2(direction[1], direction[0])
        if len(children) == 1:
            angles = [base_angle]
        else:
            spread = math.radians(50)
            angles = [base_angle - spread / 2 + spread * i / (len(children) - 1) for i in range(len(children))]
        for child, ang in zip(children, angles):
            d = (math.cos(ang), math.sin(ang))
            pos[child] = (pos[node][0] + d[0] * unit, pos[node][1] + d[1] * unit)
            self._fan_place(G, child, node, d, pos, unit)

    def _spread_angles(self, count, blocked, resolution=24):
        """Pick `count` angles (radians) spread as far as possible from `blocked` directions and each other."""
        pool = [2 * math.pi * i / resolution for i in range(resolution)]

        def min_dist(a, others):
            if not others:
                return math.pi
            return min(min(abs(a - b), 2 * math.pi - abs(a - b)) for b in others)

        chosen = []
        for _ in range(count):
            best = max(pool, key=lambda a: min_dist(a, blocked + chosen))
            chosen.append(best)
            pool.remove(best)
        return chosen

    def _compute_schematic_layout(self, hub_unit=380.0, star_radius=130.0):
        """
        MRT-style "star of stars" schematic layout:
        - Regional hubs are laid out as their own small tree (the backbone MST's
          longest path becomes a horizontal spine, other backbone edges fan off
          it -- same technique as the old single-line layout, just one level up).
        - Each hub's cluster members are then placed radially around it, avoiding
          the direction(s) toward its backbone neighbors so spokes don't cross
          the trunk line unnecessarily.
        Not geographically accurate -- topology only, like a real transit diagram.
        """
        hub_towns = self.pass_plan['hub_towns']
        backbone_edges = [(r['from_hub'], r['to_hub']) for r in self.pass_plan['backbone_routes']]

        Gh = nx.Graph()
        Gh.add_nodes_from(hub_towns)
        Gh.add_edges_from(backbone_edges)

        if len(hub_towns) > 1:
            lengths = dict(nx.all_pairs_shortest_path_length(Gh))
            max_hops, pair = 0, (hub_towns[0], hub_towns[0])
            for u in Gh.nodes():
                for v in Gh.nodes():
                    if lengths[u][v] > max_hops:
                        max_hops, pair = lengths[u][v], (u, v)
            spine = nx.shortest_path(Gh, pair[0], pair[1])
        else:
            spine = hub_towns[:]

        pos = {h: (i * hub_unit, 0.0) for i, h in enumerate(spine)}

        up = True
        for h in spine:
            for nb in Gh.neighbors(h):
                if nb not in pos:
                    angle = math.radians(60 if up else -60)
                    up = not up
                    d = (math.cos(angle), math.sin(angle))
                    pos[nb] = (pos[h][0] + d[0] * hub_unit, pos[h][1] + d[1] * hub_unit)
                    self._fan_place(Gh, nb, h, d, pos, hub_unit)

        for cluster in self.pass_plan['clusters']:
            hub = cluster['hub']
            members = cluster['members']
            if not members:
                continue
            blocked = []
            for nb in Gh.neighbors(hub):
                dx, dy = pos[nb][0] - pos[hub][0], pos[nb][1] - pos[hub][1]
                blocked.append(math.atan2(dy, dx))
            angles = self._spread_angles(len(members), blocked)
            for m, ang in zip(members, angles):
                pos[m] = (pos[hub][0] + star_radius * math.cos(ang), pos[hub][1] + star_radius * math.sin(ang))

        return pos, Gh

    def render_mrt_map(self, out_dir="."):
        """MRT/subway-style schematic diagram: regional star clusters + intercity backbone (topology, not geography)."""
        print("[Render] Generating MRT-style schematic passenger map...")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        pos, Gh = self._compute_schematic_layout()
        hub_set = set(self.pass_plan['hub_towns'])

        PASS_COLOR = '#16a34a'
        PASS_CASING = '#14532d'
        SPOKE_COLOR = '#86e5ae'
        BG = '#fcfcfb'

        fig, ax = plt.subplots(figsize=(18, 14), dpi=150)
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)

        # Star spokes (light green) -- drawn first, underneath the backbone
        for cluster in self.pass_plan['clusters']:
            hub = cluster['hub']
            for m in cluster['members']:
                x1, y1 = pos[hub]
                x2, y2 = pos[m]
                ax.plot([x1, x2], [y1, y2], color=SPOKE_COLOR, linewidth=2.8, solid_capstyle='round', zorder=2)

        # Intercity backbone (dark green, thicker)
        for u, v in Gh.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color=PASS_CASING, linewidth=6.0, solid_capstyle='round', zorder=3)
            ax.plot([x1, x2], [y1, y2], color=PASS_COLOR, linewidth=3.5, solid_capstyle='round', zorder=4)

        member_hub = {m: c['hub'] for c in self.pass_plan['clusters'] for m in c['members']}
        for t, (x, y) in pos.items():
            is_hub = t in hub_set
            if is_hub:
                ax.scatter(x, y, s=280, facecolor='#ffffff', edgecolor=PASS_CASING, linewidth=3.2, zorder=6)
                xytext = (0, 16)
            else:
                ax.scatter(x, y, s=90, facecolor=SPOKE_COLOR, edgecolor='#ffffff', linewidth=1.4, zorder=5)
                # Label points radially outward along the spoke (away from its hub) so it
                # doesn't collide with the hub's own label when a member sits near the hub.
                hub = member_hub.get(t)
                if hub:
                    dx, dy = x - pos[hub][0], y - pos[hub][1]
                    dnorm = math.hypot(dx, dy) or 1.0
                    xytext = (14 * dx / dnorm, 14 * dy / dnorm)
                else:
                    xytext = (0, 12)
            ax.annotate(t, (x, y), textcoords="offset points", xytext=xytext,
                        ha='center', fontsize=8.0 if is_hub else 6.5, fontweight='bold' if is_hub else 'normal',
                        color=PASS_CASING, zorder=7)

        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title("Transport Fever 2 — Passenger Network (MRT-Style Schematic)\nRegional Star Clusters + Intercity Backbone, not to scale",
                     fontsize=14, fontweight='bold', pad=20)

        legend = [
            plt.Line2D([0], [0], color=PASS_COLOR, lw=4, label='Intercity Backbone'),
            plt.Line2D([0], [0], color=SPOKE_COLOR, lw=3, label='Regional Star Spoke'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffffff', markeredgecolor=PASS_CASING, markersize=10, label='Regional Hub'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=SPOKE_COLOR, markeredgecolor='w', markersize=9, label='Local Station'),
        ]
        ax.legend(handles=legend, loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=4,
                  frameon=True, facecolor='white', framealpha=0.9, fontsize=8)

        plt.tight_layout()
        mrt_svg = out_dir / "mrt_style_network.svg"
        plt.savefig(mrt_svg, format='svg', bbox_inches='tight')
        plt.close()
        print(f"  Saved {mrt_svg}")
        return {'mrt_svg': mrt_svg}

    @staticmethod
    def _path_length(pts):
        a = np.array(pts)
        if len(a) < 2:
            return 0.0
        return float(np.sum(np.hypot(np.diff(a[:, 0]), np.diff(a[:, 1]))))

    def export_route_list(self, out_dir="."):
        """Writes a markdown list of every planned passenger and freight route/line."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        L = self._path_length

        shared_count = sum(1 for c in self.corridors if c['action'] == 'MERGED_SHARED_ROW')

        lines = ["# Transport Fever 2 - Route List", "", "## Passenger Routes", "", "### Intercity Backbone", "",
                 "| Segment | Distance |", "| --- | --- |"]
        for bb in self.pass_plan['backbone_routes']:
            lines.append(f"| {bb['from_hub']} <-> {bb['to_hub']} | {L(bb['path_svg']):.0f} m |")

        lines += ["", "### Regional / Local Spokes", "", "| Hub | Serves |", "| --- | --- |"]
        for cluster in self.pass_plan['clusters']:
            hub = cluster['hub']
            served = [f"{sp['town']} ({L(sp['path_svg']):.0f} m)"
                      for sp in self.pass_plan['spoke_routes'] if sp['hub'] == hub]
            if served:
                lines.append(f"| {hub} | {', '.join(served)} |")

        lines += ["", "## Freight Routes", "", "### Trunk (hub yard <-> hub yard)", "",
                  "| Segment | Distance |", "| --- | --- |"]
        for t in self.cargo_plan['trunk_routes']:
            lines.append(f"| {t['from_hub']} <-> {t['to_hub']} | {L(t['path_svg']):.0f} m |")

        lines += ["", "### Feeder Spurs (town/industry <-> hub yard)", "",
                  "| Yard | Industries | Serves |", "| --- | --- | --- |"]
        for h in self.cargo_plan['hubs']:
            served = [f"{s['town']} ({L(s['path_svg']):.0f} m)"
                      for s in self.cargo_plan['spur_routes'] if s['hub'] == h['name'] and L(s['path_svg']) > 0]
            if served:
                lines.append(f"| {h['name']} | {h['industry_count']} | {', '.join(served)} |")

        n_pass = len(self.pass_plan['backbone_routes']) + len(self.pass_plan['spoke_routes'])
        n_freight = len(self.cargo_plan['trunk_routes']) + sum(
            1 for s in self.cargo_plan['spur_routes'] if L(s['path_svg']) > 0)
        lines += ["", "## Summary", "",
                  f"- **{n_pass}** passenger routes ({len(self.pass_plan['backbone_routes'])} intercity + {len(self.pass_plan['spoke_routes'])} regional)",
                  f"- **{n_freight}** freight routes ({len(self.cargo_plan['trunk_routes'])} trunk + "
                  f"{n_freight - len(self.cargo_plan['trunk_routes'])} feeder)",
                  f"- **{shared_count}** of these share a 4-track ROW with another route on at least one stretch "
                  f"(see `combined_network.svg`).",
                  f"- **{len(self.missed_stops)}** town(s) sit within track distance of a through-route "
                  f"without a stop -- candidate stations (report-only, see `network_summary.json`)."]

        route_md = out_dir / "route_list.md"
        route_md.write_text("\n".join(lines) + "\n")
        print(f"  Saved {route_md}")
        return {'route_list_md': route_md}

    def plan_all(self, out_dir=".", corridor_threshold=250.0, corridor_min_len=350.0, corridor_action='auto'):
        self.optimize_cargo_network()
        self.optimize_passenger_network()
        self.reconcile_corridors(threshold=corridor_threshold, min_len=corridor_min_len, action=corridor_action)
        self.flag_missed_stops()
        rendered = self.render(out_dir)
        rendered.update(self.render_mrt_map(out_dir))
        rendered.update(self.export_route_list(out_dir))
        return {
            'cargo': self.cargo_plan,
            'passenger': self.pass_plan,
            'corridors': self.corridors,
            'missed_stops': self.missed_stops,
            'files': rendered
        }


def main():
    parser = argparse.ArgumentParser(description="Transport Fever 2 Dynamic Network Planner")
    parser.add_argument("svg", nargs="?", default="map_export_20260920_133731.svg", help="Path to SVG map export")
    parser.add_argument("--out", default="./result", help="Output directory for generated maps and summary")
    parser.add_argument("--k", type=int, default=None, help="Number of cargo clusters (default: auto)")
    parser.add_argument("--corridor-threshold", type=float, default=250.0, help="Corridor proximity threshold in meters (default: 250)")
    parser.add_argument("--corridor-min-len", type=float, default=350.0, help="Minimum corridor stretch length in meters (default: 350)")
    parser.add_argument("--corridor-action", default="auto", choices=["auto", "merge", "offset", "none"], help="Corridor reconciliation action (default: auto)")
    args = parser.parse_args()
    
    planner = MapNetworkPlanner(args.svg)
    res = planner.plan_all(args.out, corridor_threshold=args.corridor_threshold, corridor_min_len=args.corridor_min_len, corridor_action=args.corridor_action)
    
    summary_path = Path(args.out) / "network_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            'elevation_proxy': planner.elevation_formula,
            'towns_count': len(planner.towns),
            'industries_count': len(planner.industries),
            'passenger_hub_towns': planner.pass_plan['hub_towns'],
            'passenger_clusters': [{
                'hub': c['hub'],
                'members': c['members']
            } for c in planner.pass_plan['clusters']],
            'cargo_hubs': [{
                'name': h['name'],
                'centroid': h['centroid'],
                'nearest_town': h['nearest_town'],
                'offset': h['town_offset'],
                'industries': h['industry_count'],
                'towns': h['towns']
            } for h in planner.cargo_plan['hubs']],
            'reconciled_corridors': [{
                'id': c['id'],
                'name': c['name'],
                'length_m': round(c['length_m'], 1),
                'pass_span_m': round(c['pass_span_m'], 1),
                'avg_separation_m': round(c['avg_separation_m'], 1),
                'mean_terrain_cost': round(c['mean_terrain_cost'], 2),
                'action': c['action'],
                'estimated_grading_saved': c['estimated_grading_saved']
            } for c in planner.corridors],
            'flagged_hauls_count': len(planner.cargo_plan['flagged_hauls']),
            'missed_stops': [{
                'kind': m['kind'],
                'route': m['route'],
                'town': m['town'],
                'distance_m': round(m['distance_m'], 1)
            } for m in planner.missed_stops],
            'outputs': {k: str(v) for k, v in res['files'].items()}
        }, f, indent=2)
    print(f"Summary written to {summary_path}")

if __name__ == "__main__":
    main()
