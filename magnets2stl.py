#!/usr/bin/env python

# Copyright 2023-25, Gavin E. Crooks and contributors
#
# This source code is licensed under the MIT License
# found in the LICENSE file in the root directory of this source tree.
#
# This is a test for press-fit alignment magnets.
#
# Magnets (bought from amazon) are nominally 6mm x 2mm, but actual size should be checked with calipers.
# The size varies brand to brand and even batch to batch.
#
# Run script to create proto_magnets.stl, and print two copies.
#
# Press fit magnets into both (in opposite orientations). The trick to press fitting these magnets is to
# take a stack of 10 or more, hold over the opening, and give the top of the stack a tap with a
# mallet. If magnet fails to hold, a small drop of superglue can be added.
#
# Gavin E. Crooks 2023


import numpy as np
import trimesh
from trimesh import transformations


from landscape2stl import STLParameters

params = STLParameters()
magnet_radius = (params.magnet_diameter) / 2 + params.magnet_padding
magnet_depth = params.magnet_depth + params.magnet_recess
magnet_sides = params.magnet_sides

pin_length = params.pin_length
pin_radius = (params.pin_diameter / 2) + params.pin_padding
pin_sides = params.pin_sides


rotate_z_to_y = transformations.rotation_matrix(np.pi / 2, [1, 0, 0])

# Base block (20 mm x 20 mm x 10 mm) centered at origin.
base = trimesh.creation.box(extents=(20.0, 20.0, 10.0))

holes = []


cylinder = trimesh.creation.cylinder(
    radius=magnet_radius, height=magnet_depth * 2, sections=magnet_sides
)
cylinder.apply_transform(rotate_z_to_y)
cylinder.apply_translation([-5, 10.0, 0.0])
holes.append(cylinder)


cylinder = trimesh.creation.cylinder(
    radius=magnet_radius, height=magnet_depth * 2, sections=magnet_sides
)
cylinder.apply_transform(rotate_z_to_y)
cylinder.apply_translation([5, 10.0, 0.0])
holes.append(cylinder)


cylinder = trimesh.creation.cylinder(
    radius=pin_radius, height=pin_length * 2, sections=pin_sides
)
cylinder.apply_transform(rotate_z_to_y)
cylinder.apply_translation([-5, -10.0, 0.0])
holes.append(cylinder)

cylinder = trimesh.creation.cylinder(
    radius=pin_radius, height=pin_length * 2, sections=pin_sides
)
cylinder.apply_transform(rotate_z_to_y)
cylinder.apply_translation([5, -10.0, 0.0])
holes.append(cylinder)


model = trimesh.boolean.difference([base, *holes], backend="blender")
model.remove_unreferenced_vertices()
model.process(validate=True)


print(f"is closed surface? {model.is_watertight}")
print(f"is manifold? {model.is_winding_consistent}")
filename = "proto_magnets.stl"
model.export(filename)
