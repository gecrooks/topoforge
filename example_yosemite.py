#!/usr/bin/env python

import topoforge


# Names of 7.5 minute quadrangles, north to south, blocks east to west.
yosemite_quadrangles = (
    "Kibbie Lake",
    "Tiltill Mountain",
    "Piute Mountain",
    "Matterhorn Peak",
    "Dunderberg Peak",
    "Lundy",
    "Negit Island",
    "Lake Eleanor",
    "hetch_hetchy_reservoir",
    "ten_lakes",
    "falls_ridge",
    "tioga_pass",
    "mount_dana",
    "lee_vining",
    "Ackerson Mountain",
    "tamarack_flat",
    "yosemite_falls",
    "tenaya_lake",
    "vogelsang_peak",
    "koip_peak",
    "june_lake",
    "El Portal",
    "el_capitan",
    "half_dome",
    "merced_peak",
    "mount_lyell",
    "mount_ritter",
    "mammoth_mountain",
    "Buckingham Mountain",
    "Wawona",
    "Mariposa grove",
    "Sing Peak",
    "Timber Knob",
    "Cattle Mountain",
    "Crystal Crag",
    "Sulphur Pond",
    "Mono Mills",
    "Crestview",
    "Old Mammoth",
    "Bloody Mountain",
    "Cherry Lake North",
    "Cherry Lake South",
    "Ascension Mountain",
    "Kinsley",
    "Feliciana Mountain",
)


for name in yosemite_quadrangles:
    params = topoforge.STLParameters(
        scale=62_500,
    )
    coords = topoforge.quad_coordinates(name, "CA")
    topoforge.create_quad_stl(params, coords, verbose=True)
