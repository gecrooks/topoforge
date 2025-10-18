#!/usr/bin/env python

# Copyright 2023-2024, Gavin E. Crooks
#
# This source code is licensed under the MIT License
# found in the LICENSE file in the root directory of this source tree.
#
#
# landscape2stl: High resolution terrain models for 3D printing
#
# See README for more information
#
#
# Gavin E. Crooks 2023-2025
#

# Note to self:
# latitude is north-south
# longitude is west-east (long way around)

import argparse
import math
import os
import sys
import zipfile
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import py3dep
import requests
import trimesh
import us
import xarray as xr
from numpy.typing import ArrayLike
from typing_extensions import TypeAlias
import pyvista 

# We use many units and coordinate systems.
# Use TypeAlias's in desperate effort to
# keep everything straight
MM: TypeAlias = float  # millimeters
Meters: TypeAlias = float  # meters
Degrees: TypeAlias = float
ECEF: TypeAlias = tuple[
    Meters, Meters, Meters
]  # Earth-Centered, Earth-Fixed (ECEF) Cartesian coordinates
LLA: TypeAlias = tuple[
    Degrees, Degrees, Meters
]  # latitude, longitude, altitude (in meters) coordinates
ENU: TypeAlias = tuple[MM, MM, MM]  # East, North, Up model coordinates (millimeters)
BBox: TypeAlias = tuple[
    Degrees, Degrees, Degrees, Degrees
]  # Geographic bounding box: south, west, north, east




# standard_scales = [
#     15_625,  # about 4" to 1 mile
#     31_250,  # about 2" to 1 mile
#     62_500,  # about 1" to 1 mile
#     125_000,  # about 1" to 2 miles
#     250_000,  # about 1" to 4 miles
#     500_000,  # about 1" to 8 miles
#     1_000_000,  # about 1" to 16 miles
# ]

default_text = "TopoForge"

default_cache = "cache"


@dataclass
class STLParameters:
    """Parameters for the STL terrain models (apart from actual coordinates).
    Scale is the important parameter to set, the rest can generally be left at default.
    """

    scale: int = 62_500
    resolution: int = 0  # Auto set in __post_init__
    resolution_choices: tuple[int] = (10, 30)  # meters
    pitch: MM = 0.40  # Nozzle size

    min_altitude: Meters = -100.0  # Lowest point in US is -86 m

    drop_sea_level: bool = True
    sea_level: Meters = 0.8  # was 1.7
    sea_level_drop: MM = 0.48  # 6 layers
    exaggeration: float = 0.0  # Auto set in __post_init__

    base_height: MM = 10.0

    magnet_holes: bool = True
    magnet_spacing: Degrees = 0.0  # Auto set in __post_init__
    magnet_diameter: MM = 6.00
    magnet_padding: MM = 0.025
    magnet_depth: MM = 2.00
    magnet_recess: MM = 0.15
    magnet_sides: int = 24

    pin_holes: bool = True
    pin_length: MM = 9
    pin_diameter: MM = 1.75
    pin_padding: MM = 0.05 * 3
    pin_sides: int = 8

    bottom_holes: bool = False
    bottom_hole_offset: MM = 10
    bottom_hole_diameter: MM = 5.6
    bottom_hole_padding: MM = 0.2
    bottom_hole_depth: MM = 9.1
    bottom_hole_sides: int = 24


    def __post_init__(self):
        if not self.magnet_spacing:
            self.magnet_spacing = self.scale / 2_000_000

        if not self.resolution:
            if self.scale < 250_000:
                self.resolution = self.resolution_choices[0]
            else:
                self.resolution = self.resolution_choices[1]

        if not self.exaggeration:
            # Heuristic for vertical exaggeration
            # scale exaggeration
            # <= 62_500     1.0
            # 125_000       1.5
            # 250_000       2.0
            # 500_000       2.5
            # 1_000_000     3.0
            if self.scale <= 62_500:
                self.exaggeration = 1.0
            else:
                self.exaggeration = 3 - 0.5 * math.log2(1_000_000 / self.scale)


# end STLParameters


def main() -> int:
    default_params = STLParameters()
    parser = argparse.ArgumentParser(description="Create quadrangle landscape STLs")
    parser.add_argument(
        "coordinates",
        metavar="S W N E",
        type=float,
        nargs="*",
        help="Latitude/longitude coordinates for quadrangle (Order south edge, west edge, north edge, east edge)",
    )

    parser.add_argument("--quad", dest="quad", type=str)

    parser.add_argument("--state", dest="state", type=str, default="CA")

    parser.add_argument("--scale", dest="scale", type=int, help="Map scale")


    parser.add_argument(
        "--exaggeration",
        dest="exaggeration",
        type=float,
        # default=1.0,
        help="Vertical exaggeration",
    )

    parser.add_argument(
        "--magnets", dest="magnets", type=float, help="Magnet spacing (in degrees)"
    )

    parser.add_argument("--filename", dest="filename", type=str, help="Filename for model")

    parser.add_argument("-v", "--verbose", action="store_true")

    args = vars(parser.parse_args())
    name = None

    text = default_text

    if args["quad"] is not None:
  

        name = args["quad"].lower().replace(" ", "_")
        coords = quad_coordinates(name, args["state"])
        args["coordinates"] = coords

        label = usgs_quadrangle_label(coords[0], coords[3])

        text += '\n' + args["quad"] + ', ' + args["state"] + '\n'+label


        # args["scale"] = 62_500 # params default to this scale

        if args["filename"] is None:
            args["filename"] = "quad_" + args["state"].lower() + "_" + name

    if not args["coordinates"]:
        parser.print_help()
        return 0

    if args["scale"] is None:
        args["scale"] = default_params.scale


    params = STLParameters(
        scale=args["scale"],
        exaggeration=args["exaggeration"],
        magnet_spacing=args["magnets"],
        # resolution=args["resolution"],
        # projection=args["projection"],
    )


    text = text + '\n' + 'scale 1 : '+f"{params.scale:,}"


    create_stl(
        params, args["coordinates"], filename=args["filename"], text=text,verbose=args["verbose"]
    )

    return 0


def ustopo_current():
    url = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Maps/Metadata/ustopo_current.zip"
    zip_file_name = "ustopo_current.zip"
    csv_file_name = "ustopo_current.csv"
    directory = "cache"

    zip_file_path = os.path.join(directory, zip_file_name)

    if not os.path.exists(zip_file_path):
        os.makedirs(directory, exist_ok=True)
        response = requests.get(url)
        with open(zip_file_path, "wb") as f:
            f.write(response.content)

    with zipfile.ZipFile(zip_file_path, "r") as z:
        with z.open(csv_file_name) as csv_file:
            df = pd.read_csv(csv_file)

    return df


def quad_coordinates(quad_name, state="CA"):
    df = ustopo_current()

    quad_name = quad_name.lower().replace("_", " ")

    condition = (df["map_name"].str.lower() == quad_name) & df[
        "state_list"
    ].str.contains(state)

    row = df[condition]

    if len(row) == 0:
        raise ValueError("Quadrangle " + quad_name + " not found")

    southbc = row["southbc"].astype(float).iloc[0]
    westbc = row["westbc"].astype(float).iloc[0]

    return southbc, westbc, southbc + 1 / 8, westbc + 1 / 8


def quad_from_coordinates(lat, long):
    df = ustopo_current()

    condition = (
        (df["southbc"].astype(float) <= lat)
        & (lat < df["northbc"].astype(float))
        & (df["westbc"].astype(float) <= long)
        & (long < df["eastbc"].astype(float))
    )

    row = df[condition]
    if len(row) == 0:
        return (None, None)
    name = row["map_name"].astype(str).iloc[0]
    state_name = row["primary_state"].astype(str).iloc[0]
    state = us.states.lookup(state_name)

    return name, state.abbr


def create_quad_stl(name, state, filename=None, verbose=False):
    coords = quad_coordinates(name, state)
    if filename is None:
        filename = "quad_" + state.lower() + "_" + name.lower().replace(" ", "_")

    params = STLParameters(
        scale=62_500,
    )

    create_stl(params, coords, filename, verbose)


def create_stl(
    params: STLParameters,
    boundary: BBox,
    filename: Optional[str] = None,
    text: str = "",
    verbose: bool = False,
) -> None:
    if verbose:
        print(params)

    # Locate origin
    south, west, north, east = boundary
    origin = (south + north) / 2, (east + west) / 2, 0.0

    # Calculate steps...
    north_west_enu = lla_to_model((north, west, 0.0), origin, params)
    south_east_enu = lla_to_model((south, east, 0.0), origin, params)

    extent_ns = south_east_enu[0] - north_west_enu[0]
    extent_we = north_west_enu[1] - south_east_enu[1]

    ns_steps = int(round(extent_ns / params.pitch))
    we_steps = int(round(extent_we / params.pitch))
    steps = max(ns_steps, we_steps)

    elevation = download_elevation(boundary, steps, params.resolution, verbose)

    if verbose:
        print("Building terrain...")

    surface = elevation_to_surface(elevation, origin, params)

    if verbose:
        print("Triangulating surface...")

    model = triangulate_surface(surface, boundary, origin, params)
    model = add_base_holes(model, boundary, origin, params)
    model = add_base_text(model, text)


    if verbose:
        print("Faces:", len(model.faces))

    if verbose:
        print("Creating STL...")

    if filename is None:
        filename = "{:f}_{:f}_{:f}_{:f}.stl".format(*boundary)
    else:
        filename = filename + ".stl"

    if verbose:
        print(f"Saving {filename}")

    model.export(filename)


# end create_stl


def download_elevation(
    boundary: BBox,
    steps: int,
    resolution: int,
    verbose: bool = False,
) -> xr.Dataset | xr.DataArray:
    elevation: xr.Dataset | xr.DataArray
    south, west, north, east = boundary

    xcoords = np.linspace(west, east, steps)
    ycoords = np.linspace(south, north, steps)

    filename = "{}_{:.2f}_{:.2f}_{:.2f}_{:.2f}.nc".format(steps, *boundary)

    if not os.path.exists(default_cache):
        os.mkdir(default_cache)
    fname = os.path.join(default_cache, filename)

    if not os.path.exists(fname):
        if verbose:
            print("Downloading elevation data... ", end="", flush=True)
        elevation = py3dep.elevation_bygrid(
            tuple(xcoords), tuple(ycoords), crs="EPSG:4326", resolution=resolution
        )  # or 30

        elevation.to_netcdf(fname)
        if verbose:
            print("Done", flush=True)

    if verbose:
        print("Loading elevation data from cache...", end="", flush=True)
    elevation = xr.open_dataset(fname)
    if verbose:
        print("", flush=True)

    return elevation


def elevation_to_surface(
    elevation: xr.Dataset | xr.DataArray, origin: LLA, params: STLParameters
) -> np.ndarray:
    ycoords = np.asarray(elevation.coords["y"])
    xcoords = np.asarray(elevation.coords["x"])
    steps = len(ycoords)
    elevation_array = np.asarray(elevation.to_array()).reshape((steps, steps)).T

    # Missing data will be nan
    elevation_array = np.nan_to_num(elevation_array, nan=0.0)

    if params.drop_sea_level:

        near_death_valley = np.outer(
            (xcoords >= -118) & (xcoords <= -116), (ycoords <= 38) & (ycoords >= 34)
        )
        near_salton_trough = np.outer(
            (xcoords >= -116.5) & (xcoords <= -114), (ycoords <= 34) & (ycoords >= 31.9)
        )
        exclude = near_death_valley | near_salton_trough

        dropped_sea_level = (
            -(params.scale * params.sea_level_drop / 1000) / params.exaggeration
        )

        elevation_array = np.where(
            (elevation_array <= params.sea_level) & ~exclude,
            dropped_sea_level,
            elevation_array,
        )

    surface = np.zeros(shape=(steps, steps, 3))

    for x in range(steps):
        for y in range(steps):
            lat = ycoords[y]
            lon = xcoords[x]
            alt = elevation_array[x, y]
            surface[x, y] = lla_to_model((lat, lon, alt), origin, params)

    return surface


def triangulate_surface(
    surface: np.ndarray,
    boundary: BBox,
    origin: LLA,
    params: STLParameters,
) -> trimesh.Trimesh:
    steps = surface.shape[0]
    if steps < 2:
        raise ValueError("Surface resolution too low to build mesh.")

    top_vertices = surface.reshape(-1, 3)

    base_alt = params.min_altitude - (
        params.base_height * params.scale / (1000 * params.exaggeration)
    )

    magnets_alt = (base_alt + params.min_altitude) / 2

    bot_corners = corners_to_model(boundary, base_alt, origin, params)
    bot_height = bot_corners[0][2]

    bottom_grid = surface.copy()
    bottom_grid[:, :, 2] = bot_height
    bottom_vertices = bottom_grid.reshape(-1, 3)

    vertices = np.vstack([top_vertices, bottom_vertices])
    faces: list[list[int]] = []
    offset = steps * steps

    def top_idx(x: int, y: int) -> int:
        return x * steps + y

    def bottom_idx(x: int, y: int) -> int:
        return offset + x * steps + y

    # Top surface
    for x in range(steps - 1):
        for y in range(steps - 1):
            v00 = top_idx(x, y)
            v10 = top_idx(x + 1, y)
            v01 = top_idx(x, y + 1)
            v11 = top_idx(x + 1, y + 1)
            if ((x + y) % 2) == 0:
                faces.append([v00, v10, v11])
                faces.append([v00, v11, v01])
            else:
                faces.append([v01, v00, v10])
                faces.append([v10, v11, v01])

    # Bottom Surface

    for x in range(1, steps - 2):
        south0 = bottom_idx(x, 0)
        south1 = bottom_idx(x + 1, 0)
        north0 = bottom_idx(x, steps - 1)
        north1 = bottom_idx(x + 1, steps - 1)

        faces.append([south0, north0, south1])
        faces.append([south1, north0, north1])

    for y in range(0, steps - 1):
        south0 = bottom_idx(1, 0)
        west0 = bottom_idx(0, y)
        west1 = bottom_idx(0, y + 1)
        faces.append([west0, west1, south0])

    faces.append([bottom_idx(1, 0), bottom_idx(0, steps - 1), bottom_idx(1, steps - 1)])

    for y in range(0, steps - 1):
        north0 = bottom_idx(steps - 2, steps - 1)
        east0 = bottom_idx(steps - 1, y)
        east1 = bottom_idx(steps - 1, y + 1)
        faces.append([east0, north0, east1])

    faces.append(
        [
            bottom_idx(steps - 2, steps - 1),
            bottom_idx(steps - 1, 0),
            bottom_idx(steps - 2, 0),
        ]
    )

    # Sides

    for x in range(steps - 1):
        t0 = top_idx(x, 0)
        t1 = top_idx(x + 1, 0)
        b0 = bottom_idx(x, 0)
        b1 = bottom_idx(x + 1, 0)
        faces.append([t0, b0, b1])
        faces.append([t0, b1, t1])

    for x in range(steps - 1):
        t0 = top_idx(x, steps - 1)
        t1 = top_idx(x + 1, steps - 1)
        b0 = bottom_idx(x, steps - 1)
        b1 = bottom_idx(x + 1, steps - 1)
        faces.append([t0, b1, b0])
        faces.append([t0, t1, b1])

    for y in range(steps - 1):
        t0 = top_idx(0, y)
        t1 = top_idx(0, y + 1)
        b0 = bottom_idx(0, y)
        b1 = bottom_idx(0, y + 1)
        faces.append([t0, b0, b1])
        faces.append([t0, b1, t1])

    for y in range(steps - 1):
        t0 = top_idx(steps - 1, y)
        t1 = top_idx(steps - 1, y + 1)
        b0 = bottom_idx(steps - 1, y)
        b1 = bottom_idx(steps - 1, y + 1)
        faces.append([t0, b1, b0])
        faces.append([t0, t1, b1])

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )

    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    mesh.update_faces(mesh.unique_faces())


    return mesh


def add_base_holes(
    model,
    boundary: BBox,
    origin: LLA,
    params: STLParameters,
):

    south, west, north, east = boundary

    base_alt = params.min_altitude - (
        params.base_height * params.scale / (1000 * params.exaggeration)
    )

    magnets_alt = (base_alt + params.min_altitude) / 2

    top_corners = corners_to_model(boundary, params.min_altitude, origin, params)
    west_north_top, west_south_top, east_south_top, east_north_top = top_corners

    bot_corners = corners_to_model(boundary, base_alt, origin, params)
    west_north_bot, west_south_bot, east_south_bot, east_north_bot = bot_corners

    def make_hole(sides, depth, radius, center, axis):
        base_axis = np.array([0.0, 0.0, 1.0])
        target_axis = trimesh.util.unitize(np.asarray(axis, dtype=float))

        transform = trimesh.geometry.align_vectors(base_axis, target_axis)
        if transform is None:
            transform = np.eye(4)

        cylinder = trimesh.creation.cylinder(
            radius=radius, height=depth * 2, sections=sides
        )
        cylinder.apply_transform(transform)
        cylinder.apply_translation(center)
        return cylinder

    magnets = params.magnet_spacing
    long_steps = 1 + round((east - west) / magnets) * 2
    lat_steps = 1 + round((north - south) / magnets) * 2
    longs = np.linspace(west, east, long_steps)
    lats = np.linspace(south, north, lat_steps)

    south_normal = triangle_normal(east_south_bot, east_south_top, west_south_bot)
    north_normal = triangle_normal(east_north_bot, west_north_bot, east_north_top)
    west_normal = triangle_normal(west_south_bot, west_south_top, west_north_top)
    east_normal = triangle_normal(east_south_top, east_south_bot, east_north_top)

    holes = []

    if params.magnet_holes:
        mag_radius = (params.magnet_diameter) / 2 + params.magnet_padding
        mag_depth = params.magnet_depth + params.magnet_recess
        mag_sides = params.magnet_sides

        # south
        for i in range(1, long_steps, 2):
            mag_lla = (south, longs[i], magnets_alt)
            mag_enu = lla_to_model(mag_lla, origin, params)
            hole = make_hole(mag_sides, mag_depth, mag_radius, mag_enu, south_normal)
            holes.append(hole)

        # north
        for i in range(1, long_steps, 2):
            mag_lla = (north, longs[i], magnets_alt)
            mag_enu = lla_to_model(mag_lla, origin, params)
            hole = make_hole(mag_sides, mag_depth, mag_radius, mag_enu, north_normal)
            holes.append(hole)

        # west
        for i in range(1, lat_steps, 2):
            mag_lla = (lats[i], west, magnets_alt)
            mag_enu = lla_to_model(mag_lla, origin, params)
            hole = make_hole(mag_sides, mag_depth, mag_radius, mag_enu, west_normal)
            holes.append(hole)

        # east
        for i in range(1, lat_steps, 2):
            mag_lla = (lats[i], east, magnets_alt)
            mag_enu = lla_to_model(mag_lla, origin, params)
            hole = make_hole(mag_sides, mag_depth, mag_radius, mag_enu, east_normal)
            holes.append(hole)

    if params.pin_holes:
        pin_radius = (params.pin_diameter / 2) + params.pin_padding
        pin_length = params.pin_length
        pin_sides = params.pin_sides

        for i in range(2, long_steps - 1, 2):
            pin_lla = (south, longs[i], magnets_alt)
            pin_enu = lla_to_model(pin_lla, origin, params)
            hole = make_hole(pin_sides, pin_length, pin_radius, pin_enu, south_normal)
            holes.append(hole)

        for i in range(2, long_steps - 1, 2):
            pin_lla = (north, longs[i], magnets_alt)
            pin_enu = lla_to_model(pin_lla, origin, params)
            hole = make_hole(pin_sides, pin_length, pin_radius, pin_enu, north_normal)
            holes.append(hole)

        for i in range(2, lat_steps - 1, 2):
            pin_lla = (lats[i], west, magnets_alt)
            pin_enu = lla_to_model(pin_lla, origin, params)
            hole = make_hole(pin_sides, pin_length, pin_radius, pin_enu, west_normal)
            holes.append(hole)

        for i in range(2, lat_steps - 1, 2):
            pin_lla = (lats[i], east, magnets_alt)
            pin_enu = lla_to_model(pin_lla, origin, params)
            hole = make_hole(pin_sides, pin_length, pin_radius, pin_enu, east_normal)
            holes.append(hole)

    if params.bottom_holes:
        offset = params.bottom_hole_offset
        sides = params.bottom_hole_sides
        radius = params.bottom_hole_padding + params.bottom_hole_diameter / 2
        depth = params.bottom_hole_depth

        corner_centers = [
            np.asarray(west_north_bot) + np.array([offset, -offset, 0]),
            np.asarray(west_south_bot) + np.array([offset, offset, 0]),
            np.asarray(east_south_bot) + np.array([-offset, offset, 0]),
            np.asarray(east_north_bot) + np.array([-offset, -offset, 0]),
        ]
        bottom_normal = np.array([0.0, 0.0, -1.0])
        for center in corner_centers:
            hole = make_hole(sides, depth, radius, center, bottom_normal)
            holes.append(hole)

        center = (np.asarray(east_north_bot) + np.asarray(west_south_bot)) / 2.0
        hole = make_hole(sides, depth, radius, center, bottom_normal)
        holes.append(hole)

    model = trimesh.boolean.difference([model, *holes], backend="blender")
    model.remove_unreferenced_vertices()
    model.process(validate=True)


    model.remove_unreferenced_vertices()
    model.fix_normals()
    model.update_faces(model.unique_faces())


    return model


def add_base_text(model, text):
    lines = text.splitlines()

    height = 5
    depth = 1

    meshes = []

    z_min = model.bounds[0, 2]

    for i, line in enumerate(lines):
        text_mesh = pyvista.Text3D(line, height=height, depth=depth, center=[0,-i*height*1.5,z_min])
        trimesh_mesh = trimesh.Trimesh(
            vertices=text_mesh.points,
            faces=text_mesh.faces.reshape(-1, 4)[:, 1:4]
        )
        trimesh_mesh.remove_unreferenced_vertices()
        trimesh_mesh.process(validate=True)
        trimesh_mesh.fix_normals()
        trimesh_mesh.update_faces(trimesh_mesh.unique_faces())        
        meshes.append(trimesh_mesh)

    text_model = trimesh.boolean.union(meshes)
    text_model.apply_scale([-1, 1, 1])

    model = trimesh.boolean.difference([model, text_model])

    return model


def triangle_normal(A: ArrayLike, B: ArrayLike, C: ArrayLike) -> np.ndarray:
    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)
    normal = np.cross(B - A, C - A)
    return trimesh.util.unitize(normal)


def lla_to_model(
    lat_lon_alt: LLA, origin_lat_lon_alt: LLA, params: STLParameters
) -> ENU:
    """
    Convert latitude, longitude, and altitude (LLA) coordinates
    to model ENU Cartesian coordinates in millimeters
    """

    lat, lon, alt = lat_lon_alt
    origin_lat, origin_lon, origin_alt = origin_lat_lon_alt

    east, north = lambert_conformal_conic(lat, lon, center_meridian=origin_lon)
    center_east, center_north = lambert_conformal_conic(
        origin_lat, origin_lon, center_meridian=origin_lon
    )

    east = east - center_east
    north = north - center_north
    alt = alt - origin_alt

    up = alt * params.exaggeration
    enu_scaled = np.asarray([east, north, up])
    enu_scaled /= params.scale
    enu_scaled *= 1000  # meters to mm

    return (enu_scaled[0], enu_scaled[1], enu_scaled[2])


def corners_to_model(
    boundary: BBox,
    alt: Meters,
    origin: LLA,
    params: STLParameters,
) -> tuple[ENU, ENU, ENU, ENU]:
    south, west, north, east = boundary
    west_north: ENU = lla_to_model((north, west, alt), origin, params)
    east_north: ENU = lla_to_model((north, east, alt), origin, params)
    east_south: ENU = lla_to_model((south, east, alt), origin, params)
    west_south: ENU = lla_to_model((south, west, alt), origin, params)

    return west_north, west_south, east_south, east_north


def usgs_quadrangle_label(south, east):
    """
    Return the USGS 7.5-minute quadrangle label for given lat/lon.
    Format: DDDLLLrx where:
    - DDD = integer latitude of SE corner of 1° block
    - LLL = integer longitude of SE corner of 1° block
    - r = column (a–h), west to east
    - x = row (1–8), south to north
    """
    lat = south
    lon = east
    if not (-180 <= lon <= 0 and 0 <= lat <= 90):
        raise ValueError("Latitude must be in [-90, 90], Longitude in [-180, 180]")

    # Normalize to SE corner of 1° block
    block_lat = math.floor(lat)
    block_lon = math.ceil(lon)

    # Offset within block (0–1)
    lat_offset = lat - block_lat
    lon_offset = -(lon - block_lon)


    # Index within 8x8 grid (7.5 arcmin = 1/8 degree)
    row = int((lat_offset * 8) // 1)  # 0 (south) to 7 (north)
    col = int((lon_offset * 8) // 1)  # 0 (west) to 7 (east)

    # Convert column to letter (a–h), row to 1–8 (south to north)
    col_letter = chr(ord('a') + row)
    row_number = col + 1

    label = f"{block_lat:02d}{abs(block_lon):03d}{col_letter}{row_number}"
    return label


def lambert_conformal_conic(
    lat: float,
    lon: float,
    standard_parallel1: float = 33.0,
    standard_parallel2: float = 45.0,
    center_meridian: float = -96.0,
) -> Tuple[float, float]:
    """
    Convert latitude and longitude to Lambert Conformal Conic projection coordinates.

    :param lat: Latitude in degrees.
    :param lon: Longitude in degrees.
    :param standard_parallel1: First standard parallel.
    :param standard_parallel2: Second standard parallel.
    :param center_meridian: Longitude of the central meridian.
    :return: (x, y) coordinates in the Lambert Conformal Conic projection.
    """

    # Convert degrees to radians
    lat = math.radians(lat)
    lon = math.radians(lon)
    standard_parallel1 = math.radians(standard_parallel1)
    standard_parallel2 = math.radians(standard_parallel2)
    center_meridian = math.radians(center_meridian)

    # Ellipsoid parameters for WGS 84
    a = 6378137  # semi-major axis
    f = 1 / 298.257223563  # flattening
    e = math.sqrt(f * (2 - f))  # eccentricity

    # Calculate the scale factor at the standard parallels
    m1 = math.cos(standard_parallel1) / math.sqrt(
        1 - e**2 * math.sin(standard_parallel1) ** 2
    )
    m2 = math.cos(standard_parallel2) / math.sqrt(
        1 - e**2 * math.sin(standard_parallel2) ** 2
    )
    t = math.tan(math.pi / 4 - lat / 2) / (
        (1 - e * math.sin(lat)) / (1 + e * math.sin(lat))
    ) ** (e / 2)
    t1 = math.tan(math.pi / 4 - standard_parallel1 / 2) / (
        (1 - e * math.sin(standard_parallel1)) / (1 + e * math.sin(standard_parallel1))
    ) ** (e / 2)
    t2 = math.tan(math.pi / 4 - standard_parallel2 / 2) / (
        (1 - e * math.sin(standard_parallel2)) / (1 + e * math.sin(standard_parallel2))
    ) ** (e / 2)

    # Calculate the scale factor n
    n = math.log(m1 / m2) / math.log(t1 / t2)

    # Calculate the projection constants F and rho0
    F = m1 / (n * t1**n)
    rho = a * F * t**n
    rho0 = a * F * t1**n

    # Calculate the projected coordinates
    x = rho * math.sin(n * (lon - center_meridian))
    y = rho0 - rho * math.cos(n * (lon - center_meridian))

    return x, y


if __name__ == "__main__":
    sys.exit(main())
