#!/usr/bin/env python

import topoforge


# https://livingatlas.arcgis.com/topomapexplorer/#maps=42127,41890,41897,42116,42072,40171,182252,39096,40325,39314,40917,40375,40663,40261,39332,39175,40102,40967&loc=-122.59,37.88&LoD=8.00
# https://livingatlas.arcgis.com/topomapexplorer/#maps=41619,41741,42048,41683,41862,41657,41767,41747,41498,41585,41475,41985,103073,41847,103029,102981,39710,41470,41263,41598,41485,41514,41547&loc=-119.71,39.19&LoD=6.30

description = """
Seirra Nevada (and south cascades) 

From Mt Shasta in the north to Tehachapi pass in the south

scale of 1: 250000
vertical exaggeration x2
Each tile is 30 minutes on a side.
"""

print(description)


tiles = {
    #
    "Dunsmuir": (41.0, -122.5, 41.5, -122.0),
    "Bartle": (41.0, -122, 41.5, -121.5),
    "McArthur East": (41.0, -121.5, 41.5, -121),
    #
    "Redding": (40.5, -122.5, 41, -122),  # ??
    "Burney": (40.5, -122, 41, -121.5),
    "Halls Flat": (40.5, -121.5, 41, -121),
    "Eagle Lake West": (40.5, -121, 41, -120.5),
    #
    "Red Bluff East": (40, -122.5, 40.5, -122),
    "Mineral": (40, -122, 40.5, -121.5),
    "Lake Almanor East": (40, -121.5, 40.5, -121),
    "Susanville West": (40, -121, 40.5, -120.5),
    "Susanville East": (40, -120.5, 40.5, -120),
    #
    "Chico": (39.5, -122, 40, -121.5),
    "Bidwell Bar": (39.5, -121.5, 40, -121),
    "Downieville": (39.5, -121, 40, -120.5),
    "Seirraville": (39.5, -120.5, 40, -120),
    "Reno": (39.5, -120, 40, -119.5),
    #
    "Smartsville": (39, -121.5, 39.5, -121),
    "Colfax": (39, -121, 39.5, -120.5),
    "Truckee": (39, -120.5, 39.5, -120),
    "Carson": (39, -120, 39.5, -119.5),
    #
    "Sacramento": (38.5, -121.5, 39, -121),
    "Placeville": (38.5, -121, 39, -120.5),
    "Pyramid Peak": (38.5, -120.5, 39, -120),
    "Markleeville": (38.5, -120, 39, -119.5),
    "Smith Valley East": (38.5, -119.5, 39, -119),
    #
    "Jackson": (38, -121, 38.5, -120.5),
    "Big Trees": (38, -120.5, 38.5, -120),
    "Dardanelles": (38, -120, 38.5, -119.5),
    "Bridgeport": (38, -119.5, 38.5, -119),
    "Excelsior Mountains West": (38, -119, 38.5, -118.5),  # NV
    #
    # "Oakdale": (37.5, -121, 38, -120.5),
    "Sonora": (37.5, -120.5, 38, -120),
    "Yosemite": (37.5, -120, 38, -119.5),
    "Mount Lyell": (37.5, -119.5, 38, -119),
    "Mount Morrison": (37.5, -119, 38, -118.5),
    "White Mountain": (37.5, -118.5, 38, -118),
    #
    "Merced East": (37, -120.5, 37.5, -120),
    "Mariposa": (37, -120, 37.5, -119.5),
    "Kaiser": (37, -119.5, 37.5, -119),
    "Mount Goddard": (37, -119, 37.5, -118.5),
    "Bishop": (37, -118.5, 37.5, -118),
    #
    "Fresno West": (36.5, -120, 37, -119.5),
    "Dinuba": (36.5, -119.5, 37, -119),
    "Tehipite": (36.5, -119, 37, -118.5),
    "Mount Whitney": (36.5, -118.5, 37, -118),
    "Saline Valley West": (36.5, -118, 37, -117.5),
    #
    "Visalia East": (36, -119.5, 36.5, -119),
    "Kaweah": (36, -119, 36.5, -118.5),
    "Olancha": (36, -118.5, 36.5, -118),
    "Darwin Hills West": (36, -118, 36.5, -117.5),
    #
    "Delano East": (35.5, -119.5, 36, -119),
    "Tobias Peak": (35.5, -119, 36, -118.5),
    "Kernville": (35.5, -118.5, 36, -118),
    "Ridgecrest West": (35.5, -118, 36, -117.5),
    #
    "Caliente": (35, -119, 35.5, -118.5),
    "Mojave": (35, -118.5, 35.5, -118),
}


params_250000 = topoforge.STLParameters(
    scale=250000,
    drop_sea_level=False,
)


for name, coords in tiles.items():
    text = "\n".join(
        ["TopoForge", name, str(coords), "scale 1:250,000", "vertical exaggeration x2"]
    )
    name = name.lower().replace(" ", "_")

    print(text)
    topoforge.create_stl(
        params_250000, coords, f"quad_30m_{name}", text=text, verbose=True
    )

    print()
