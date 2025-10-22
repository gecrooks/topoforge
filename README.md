# TopoForge: Modular, high resolution terrain models for 3D printing


[Source](https://github.com/gecrooks/topoforge)


Gavin E. Crooks (2024-2025)


![Yosemite](images/yosemite.jpeg)
Yosemite valley and environs, scale about 1" to 1 mile, overall size 42"x43" (a little over 1 square meter). 

![Magnets](images/landscape_magnets.jpeg)
Yosemite was printed in 30 segments, held together by magnets installed along the base.


The python script topoforge.py generates large scale terrain models as an STL files suitable for 3D printing. I have
incorporated a number of innovations, notably a high resolution data source and base alignment magnets.

Python script is best managed with uv.

    > brew install uv

You also need to install blender, e.g.

    > brew install blender

Then create STL terrain models.

    > ./topoforge.py --quad "Half Dome"


The standard quadrangle is 7.5 minutes by 7.5 minutes (1/8 of a degree), at a scale of 1:65250 (about 1 inch to 1 mile). 


## Data Source: 3DEP 1/3rd arc-second

The elevation data is taken from the USGS 3D Elevation Program (3DEP) 1/3rd arc-second dataset (approximately 10 meter resolution). This
is a high quality, high resolution dataset. I've experimented printing with scales up to 1:15,625 (About 1" to 4 miles), and the models 
look great. (If the dataset resolution isn't fine enough the model starts looking like Minecraft.)

The downside is that this dataset is only available for the contiguous 48 states, Hawaii, and some parts of Alaska.


## Base alignment magnets

Since print beds are limited in size large landscapes have to be printed as a collection of segments. However, getting neighboring segments to line up satisfactorily can be tricky.

Inspired by the gridfinity project, I solved the alignment problem by adding holes along the base into which you can press-fit 6mm X 2mm
magnets. (I'm using DIYMAG brand from Amazon, which cost about 5¢ each.) Neighboring segments snap together with a satisfying click, with nigh on perfect alignment, and can be broken apart again for storage or transport. 

I've also added smaller holes along the sides for alignment pins (Use metal pins, or short lengths of filament).


## CLI: topoforge.py
topoforge
```
> python -m topoforge --help

usage: topoforge.py [-h] [--quad QUAD] [--state STATE] [--scale SCALE] [--exaggeration EXAGGERATION] [--magnets MAGNETS] [--filename FILENAME] [-v] [S W N E ...]

Create quadrangle landscape STLs

positional arguments:
  S W N E               Latitude/longitude coordinates for quadrangle (Order south edge, west edge, north edge, east edge)

options:
  -h, --help            show this help message and exit
  --quad QUAD
  --state STATE
  --scale SCALE         Map scale
  --exaggeration EXAGGERATION
                        Vertical exaggeration
  --magnets MAGNETS     Magnet spacing (in degrees)
  --filename FILENAME   Filename for model
  -v, --verbose
```

###  S W N E

Latitude/longitude coordinates for slab (Order south edge, west edge, north edge, east edge)


### Quad
Alternatively give the name of a USGS 7.5 minute quadrangle map. This will print at 1:62_500 scale.

One resource to find the names of quadrangle maps is https://livingatlas.arcgis.com/topomapexplorer/
(Select 1:24_000 scale for 7.5 minute maps.)

### Scale
At a scale of 1:1_000_000 we can fit a 2° x 2° tile of a standard 256mm x 256mm print bed. Each tile is about 220mm by 175mm  (Exact size varies a little due to the projection). From that scale we zoom in, reducing the latitude and longitude spanned appropriately. Good scales for tiles are the following.

* 1:1_000_000       Approx 1 inch: 16 miles       2° x 2°             144 x 112 miles
* 1:250_000         Approx 1 inch: 4 miles        1/2° x 1/2°         36 x 28 miles
* 1:62_500          Approx 1 inch: 1 mile         1/8° x 1/8°         9 x 7 miles
* 1:15_625          Approx 4 inches: 1 mile       1/32° x 1/32°       2 1/4 x 1 3/4 miles

Here the scales zoom in by factors of 4, so 16 tiles of a smaller scale fit in one tile of the next bigger scale. Also listed are the approximate scale in inches per mile, the latitude and longitude extend of the quadrangle that will fit on a 256mm x 256mm print bed, and the approximate scaled size in miles.


### Exaggeration

Low scale models require vertical exaggeration else the landscape looks flat and uninteresting. Exaggeration isn't needed at scales of 1:125_000 or higher. Opinions vary as to appropriate exaggeration, but at 1:1_000_000 an exaggeration of 3:1 looks OK. (Just don't go the way of NASA when showing pictures of Maat Mons) If not set explicitly, the exaggeration is set using a heuristic based on the scale.


### Magnet spacing
The spacing between magnets in degrees. If you use a standard scale a reasonable magnet spacing will be chosen for you, with 4 magnets per side. Thus at e.g. a scale of 1:1_000_000 the spacing between magnets is 1/2°.



## Slicing and Printing

* Printer: Bambu Lab P1S
* Nozzle: 0.4mm
* Layer height: 0.08mm
* Infill: 6% gyroid 
* Filament: eSUN PLA PRO (PLA+), Cool White (Sometimes described as Cold White)
* Brim (for bed adhesion)

The filament I used is a bright, opaque white that highlights details by throwing valleys and folds into shadow.

With a layer height of 0.08mm contour lines are approximately 15' apart at a scale of 1" : 1 mile.

After printing, use a deburring tool to clean up bottom edges, lightly sand the sides, and press-fit the magnets. Make sure the magnet orientations are consistent. I use the same orientation on North and East edges, and flip the orientation for the South and West edges. Make sure to use the same magnet orientations for each additional slab. 

To fit the magnets, take a stack of magnets, place over the magnet hole (in the correct orientation), and give the top of the stack a few lights taps with a mallet. If the magnet won't stay in its socket, add a small spot of superglue.


## Affixing to wall
Lighter tiles can be affixed to a wall using four Velcro 3/4" circles (one in each corner). Velcro has the advantages that tiles can be removed and re-positioned, and there is little give that allows the magnets to pull neighboring tiles into alignment.  However heavier tiles (62_500 scale for example) require a more robust adhesive. Large 3M Command strips work well, although getting tiles aligned is more tricky.


## Known Issues


### Limited data coverage

The 3DEP 1/3 arc second data set only covers the USA (And not even all of Alaska). Ideally we would fall back to other lower resolution datasets for the rest of the world.

### Projection

I'm using a Lambert conformal conic projection with standard parallels of 33 and 45 degrees. This projection works well for the contiguous United States, but other projections would be better for Hawaii and Alaska, and for other regions of the world.

### Shorelines

We drop ocean areas by a small amount to make shorelines more visible in the models. Unfortunately the 3DEP dataset does not make it clear where land ends and the sea begins. You might think that sea is at sea level, and everything above sea level is land, but that produces bad looking coast lines. Setting sea level to 1m  oddly produces better results, but not ideal. Also some areas of California, notable Death Valley, are below sea level. And the sea is not always at zero meters (for no not obviously good reason). I've added some heuristics to generate shores lines that seem to do an adequate job.


