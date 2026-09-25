#!/usr/bin/env python3
"""
Plot a HuNavSim map in world coordinates, with the robot's line and the
human routes from labels.json, so routes can be checked against obstacles.

Usage:
  python3 plot_map.py                      # cafe map, all routes, saves ~/ask_or_act/map_routes.png
  python3 plot_map.py --map house --no-routes
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HOME = os.path.expanduser('~')
MAPS = os.path.join(HOME, 'hunav_ws', 'src', 'hunav_gazebo_wrapper', 'maps')
LABELS = os.path.join(HOME, 'ask_or_act', 'scenarios', 'labels.json')


def read_pgm(path):
    with open(path, 'rb') as f:
        assert f.readline().strip() == b'P5', 'expected a binary PGM (P5)'
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        w, h = map(int, line.split())
        maxval = int(f.readline())
        data = np.frombuffer(f.read(), dtype=np.uint8 if maxval < 256 else '>u2')
    return data.reshape(h, w)


def read_map_yaml(path):
    info = {}
    for line in open(path):
        if ':' in line:
            k, v = line.split(':', 1)
            info[k.strip()] = v.strip()
    origin = [float(x) for x in info['origin'].strip('[]').split(',')]
    return float(info['resolution']), origin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default='cafe')
    ap.add_argument('--no-routes', action='store_true')
    ap.add_argument('--out', default=os.path.join(HOME, 'ask_or_act', 'map_routes.png'))
    args = ap.parse_args()

    img = read_pgm(os.path.join(MAPS, args.map + '.pgm'))
    res, origin = read_map_yaml(os.path.join(MAPS, args.map + '.yaml'))
    h, w = img.shape
    extent = [origin[0], origin[0] + w * res, origin[1], origin[1] + h * res]
    print(f'Map {w}x{h} px, resolution {res} m/px, world x {extent[0]:.2f}..{extent[1]:.2f}, '
          f'y {extent[2]:.2f}..{extent[3]:.2f}')

    fig, ax = plt.subplots(figsize=(7, 12))
    ax.imshow(img, cmap='gray', extent=extent, origin='upper')

    # Robot line (must match line_controller.py)
    ax.plot([2.9, -2.9], [0.1, 0.0], 'b-', lw=2, label='robot line')
    ax.plot(2.9, 0.1, 'bo')

    if not args.no_routes and os.path.exists(LABELS):
        seen = set()
        for lab in json.load(open(LABELS)):
            if lab['env'] != args.map:
                continue
            key = (lab['family'], lab['intent'])
            if key in seen:
                continue
            seen.add(key)
            pts = [lab.get('start', [0.269, -3.204])] + lab['waypoints']
            xs, ys = zip(*pts)
            ax.plot(xs, ys, '--', lw=1.2, marker='.', label=f"{lab['family']} / {lab['intent']}")

    # Measured crossing from the test run, for comparison
    ax.plot(2.29, 0.0, 'rx', ms=10, mew=2, label='measured crossing (test run)')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(np.arange(np.floor(extent[0]), np.ceil(extent[1]) + 1, 1.0))
    ax.set_yticks(np.arange(np.floor(extent[2]), np.ceil(extent[3]) + 1, 1.0))
    ax.legend(loc='upper left', fontsize=7)
    ax.set_title(f'{args.map} map with robot line and human routes')
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
