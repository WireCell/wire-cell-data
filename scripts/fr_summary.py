#!/usr/bin/env python3
"""Summarize a Wire-Cell Toolkit field-response file.

Reads a WCT field response JSON file (optionally .bz2/.gz compressed) and
prints a summary of the domain it covers: number of planes, origin, tstart,
sample period, drift speed, response axis, and per-plane pitch/time-tick
extents.

For each plane it also integrates the response current along each path to get
an induced/collected charge, expressed in units of electron count:

    charge[electrons] = period * sum(current) / eplus

In the WCT system of units nanosecond == 1 and eplus == 1, so the current
array (units of eplus/ns) integrated over the tick period yields charge in
units of the elementary charge, i.e. a count of electrons.

The per-path charges are partitioned by impact position into two regions:

    central     : pitchpos in [-pitch/2, +pitch/2)  (central-wire cell)
    off-center  : all other impacts               (neighbor-wire cells)

and [min, mean, max] are reported for each region of each plane.  This exposes
how well the field response satisfies the Ramo sum rules: a "collection" plane
should integrate to ~ -1 electron for central impacts and ~0 off-center, while
an "induction" plane (bipolar signal) should integrate to ~0 everywhere.

All domain quantities are in the WCT system of units as stored in the file
(lengths in mm, times in ns, speed in mm/ns).

Usage:
    fr_summary.py FILE.json[.bz2|.gz]
"""

import sys
import json
import bz2
import gzip
import argparse
import statistics

# WCT system of units (WireCellUtil/Units.h): eplus == 1 (elementary charge),
# nanosecond == 1.  Integrating current[eplus/ns] over period[ns] and dividing
# by eplus gives charge as an electron count.
EPLUS = 1.0


def load(path):
    """Load a (possibly compressed) JSON field-response file."""
    if path.endswith(".bz2"):
        opener = bz2.open
    elif path.endswith(".gz"):
        opener = gzip.open
    else:
        opener = open
    with opener(path, "rt") as fp:
        return json.load(fp)


def fmt(x, nd=4):
    """Compact number formatting."""
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def summarize(path):
    doc = load(path)
    fr = doc["FieldResponse"]

    planes = fr["planes"]
    origin = fr["origin"]
    tstart = fr["tstart"]
    period = fr["period"]
    speed = fr["speed"]
    axis = fr["axis"]

    print(f"File: {path}")
    print("Field response (global):")
    print(f"  number of planes : {len(planes)}")
    print(f"  response axis    : {axis}")
    print(f"  origin           : {fmt(origin)} mm  (drift plane position along axis)")
    print(f"  tstart           : {fmt(tstart)} ns")
    print(f"  sample period    : {fmt(period)} ns  "
          f"({fmt(period/1000.0)} us, {fmt(1e3/period)} MHz)")
    print(f"  drift speed      : {fmt(speed)} mm/ns  ({fmt(speed*1e3)} mm/us)")
    print()

    # Rows for the per-region charge table, accumulated across all planes.
    rows = []

    for plane in planes:
        pr = plane["PlaneResponse"]
        paths = pr["paths"]
        planeid = pr["planeid"]
        location = pr["location"]
        pitch = pr["pitch"]
        half = 0.5 * pitch

        # Per-path pitch positions and current arrays.
        pitchpos = [p["PathResponse"]["pitchpos"] for p in paths]
        arrays = [p["PathResponse"]["current"]["array"] for p in paths]

        # Number of time ticks (response samples) -- take from first path and
        # verify all paths agree.
        nticks = arrays[0]["shape"][-1]
        consistent = all(a["shape"][-1] == nticks for a in arrays)

        # Integrated charge per path, in electron count.
        # The central wire owns the half-open pitch cell [-pitch/2, +pitch/2):
        # the electron is collected by the central wire there.  The impact at
        # exactly +pitch/2 sits on the boundary and belongs to the neighbor.
        central = []
        offcenter = []
        for pp, a in zip(pitchpos, arrays):
            charge = period * sum(a["elements"]) / EPLUS
            if -half - 1e-9 <= pp < half - 1e-9:
                central.append(charge)
            else:
                offcenter.append(charge)

        rows.append((planeid, "central", central))
        rows.append((planeid, "off-center", offcenter))

        t_lo = tstart
        t_hi = tstart + nticks * period
        drift_lo = t_lo * speed
        drift_hi = t_hi * speed

        print(f"Plane {planeid}:")
        print(f"  wire-plane location : {fmt(location)} mm")
        print(f"  nominal pitch       : {fmt(pitch)} mm")
        print(f"  number of paths     : {len(paths)}")
        print(f"  pitch domain        : [{fmt(min(pitchpos))}, {fmt(max(pitchpos))}] mm "
              f"(span {fmt(max(pitchpos)-min(pitchpos))} mm)")
        print(f"  time ticks per path : {nticks}"
              + ("" if consistent else "  (WARNING: paths differ in length!)"))
        print(f"  time domain         : [{fmt(t_lo)}, {fmt(t_hi)}] ns "
              f"(span {fmt(nticks*period)} ns = {fmt(nticks*period/1000.0)} us)")
        print(f"  drift domain        : [{fmt(drift_lo)}, {fmt(drift_hi)}] mm")
        print()

    # Per-region integrated-charge table (electron count).
    print("Per-path integrated charge [electron count]  "
          "(charge = period * sum(current) / eplus)")
    print("  central    : impacts in [-pitch/2, +pitch/2) (central-wire cell)")
    print("  off-center : all other impacts (neighbor-wire cells)")
    print()
    hdr = f"  {'plane':>5}  {'region':<10}  {'N':>3}  " \
          f"{'min':>13}  {'mean':>13}  {'max':>13}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for planeid, region, vals in rows:
        if vals:
            lo, me, hi = min(vals), statistics.mean(vals), max(vals)
            print(f"  {planeid:>5}  {region:<10}  {len(vals):>3}  "
                  f"{lo:>+13.4e}  {me:>+13.4e}  {hi:>+13.4e}")
        else:
            print(f"  {planeid:>5}  {region:<10}  {0:>3}  "
                  f"{'-':>13}  {'-':>13}  {'-':>13}")
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="Field-response JSON file (.json, .json.bz2, .json.gz)")
    args = ap.parse_args(argv)
    summarize(args.file)


if __name__ == "__main__":
    main()
