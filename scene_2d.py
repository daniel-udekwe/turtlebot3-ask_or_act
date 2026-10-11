#!/usr/bin/env python3
"""
2D figure of one moment in an episode: the cafe map with only the robot and the
pedestrian, in the style of plot_map.py (world coordinates, 1 m grid).

The robot is drawn at its true footprint (TurtleBot3 Burger, radius ~0.105 m) and
the pedestrian as a 0.3 m body circle, each with a line showing the path ahead of
it from that moment on, all explained in a legend. Positions and paths come from a
recorded episode (trajectory.csv), or from the planned motion.

Usage:
  python3 scene_2d.py --scenario cafe_hs_x_7000 --time 4.5 \\
      --trajectory ~/ask_or_act/runs/validation/always_ask/cafe_hs_x_7000/trajectory.csv
  python3 scene_2d.py ... --crop -1 4 -3 2          # zoom in: x from -1 to 4, y from -3 to 2
  python3 scene_2d.py ... --legend-loc 'lower left'
"""
import argparse
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

from generate_scenarios import ROBOT, build
from ped_traj import HUMAN_RADIUS, build_trajectory, pose_at
from plot_map import read_map_yaml, read_pgm

HOME = os.path.expanduser('~')
MAPS = os.path.join(HOME, 'hunav_ws', 'src', 'hunav_gazebo_wrapper', 'maps')
OUT = os.path.join(HOME, 'ask_or_act', 'paper', 'figures')
ROBOT_RADIUS = 0.105
SHORT = {'cc': 'clear_cross', 'cn': 'clear_noncross', 'hs': 'hesitant', 'lt': 'late_turn',
         'sc': 'slow_cross', 'wf': 'wait_far'}
ROBOT_COLOR, PED_COLOR = '#333333', '#009E73'


def poses(scenario, t, trajectory):
    """Pedestrian and robot positions at time t, and the paths each follows from then on."""
    if trajectory:
        import csv
        rows = [dict((k, float(v)) for k, v in r.items()) for r in csv.DictReader(open(os.path.expanduser(trajectory)))]
        at = min(rows, key=lambda r: abs(r['t'] - t))
        ahead = [r for r in rows if r['t'] >= at['t']]
        ped_path = [(r['human_x'], r['human_y']) for r in ahead]
        robot_path = [(r['robot_x'], r['robot_y']) for r in ahead]
        return (at['human_x'], at['human_y']), (at['robot_x'], at['robot_y']), ped_path, robot_path
    parts = scenario.split('_')
    lab = build(parts[0], SHORT[parts[1]], 'crossing' if parts[2] == 'x' else 'not_crossing', int(parts[3]),
                parts[4] if len(parts) > 4 else 'nominal')
    traj = build_trajectory(lab['start'], lab['plan'], lab['human_speed'])
    ts = [s[0] for s in traj]
    rel = t - lab['release_delay']
    hx, hy, _, _ = pose_at(traj, ts, rel)
    ped_path = [(hx, hy)] + [(s[1], s[2]) for s in traj if s[0] > rel]
    rx = ROBOT['start'][0] - ROBOT['speed'] * max(0.0, t - lab['robot_delay'])
    return (hx, hy), (rx, ROBOT['start'][1]), ped_path, [(rx, ROBOT['start'][1]), ROBOT['goal']]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario', default='cafe_hs_x_7000')
    ap.add_argument('--time', type=float, default=4.5, help='seconds since episode start')
    ap.add_argument('--trajectory', default=None, help='trajectory.csv of a recorded episode')
    ap.add_argument('--map', default='cafe')
    ap.add_argument('--crop', nargs=4, type=float, default=None, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
                    help='show only this part of the map (default: whole map)')
    ap.add_argument('--human-scale', type=float, default=1.0)
    ap.add_argument('--no-legend', action='store_true')
    ap.add_argument('--legend-loc', default='upper left', help="matplotlib legend position, e.g. 'lower right'")
    ap.add_argument('--out', default=os.path.join(OUT, 'fig_moment'))
    args = ap.parse_args()

    img = read_pgm(os.path.join(MAPS, f'{args.map}.pgm'))
    res, origin = read_map_yaml(os.path.join(MAPS, f'{args.map}.yaml'))
    h, w = img.shape
    extent = [origin[0], origin[0] + w * res, origin[1], origin[1] + h * res]
    (hx, hy), (rx, ry), ped_path, robot_path = poses(args.scenario, args.time, args.trajectory)

    x0, x1, y0, y1 = args.crop if args.crop else extent
    width = 3.5 if args.crop else 3.0
    fig, ax = plt.subplots(figsize=(width, width * (y1 - y0) / (x1 - x0)))
    ax.imshow(img, cmap='gray', extent=extent, origin='upper')

    pr = HUMAN_RADIUS * args.human_scale
    ax.plot(*zip(*robot_path), '--', color=ROBOT_COLOR, lw=1.2, zorder=2)
    ax.plot(*zip(*ped_path), '--', color=PED_COLOR, lw=1.2, zorder=2)
    ax.add_patch(Circle((hx, hy), pr, color=PED_COLOR, zorder=3))
    ax.add_patch(Circle((rx, ry), ROBOT_RADIUS, color=ROBOT_COLOR, zorder=3))
    if not args.no_legend:
        handles = [Line2D([], [], ls='none', marker='o', ms=5, color=ROBOT_COLOR, label='Robot'),
                   Line2D([], [], ls='--', lw=1.2, color=ROBOT_COLOR, label='Robot path ahead'),
                   Line2D([], [], ls='none', marker='o', ms=9, color=PED_COLOR, label='Pedestrian'),
                   Line2D([], [], ls='--', lw=1.2, color=PED_COLOR, label='Pedestrian path ahead')]
        ax.legend(handles=handles, loc=args.legend_loc, fontsize=6, framealpha=0.9)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    step = 1.0
    ax.set_xticks(np.arange(math.ceil(x0), math.floor(x1) + 1, step))
    ax.set_yticks(np.arange(math.ceil(y0), math.floor(y1) + 1, step))
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (m)', fontsize=8)
    ax.set_ylabel('y (m)', fontsize=8)
    ax.tick_params(labelsize=7)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', dpi=300, bbox_inches='tight')
    print(f'Pedestrian at ({hx:.2f}, {hy:.2f}), robot at ({rx:.2f}, {ry:.2f}); saved {args.out}.pdf and .png')


if __name__ == '__main__':
    main()
