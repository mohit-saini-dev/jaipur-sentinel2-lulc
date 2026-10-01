import os
import numpy as np
import planetary_computer as pc
from pystac_client import Client
import rasterio
from rasterio.windows import from_bounds

# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

print("Connecting to Planetary Computer STAC catalog...")
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)

# 1. Search parameters (Jaipur)
jaipur_lonlat = [75.72, 26.82, 75.92, 27.02]
utm_min_x, utm_min_y = 571500.0, 2966500.0
utm_max_x, utm_max_y = 591500.0, 2988500.0

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=jaipur_lonlat,
    datetime="2024-04-01/2024-05-30",
    query={"eo:cloud_cover": {"lt": 5}},
    limit=1,
)

scene = list(search.items())[0]
print(f"Streaming Scene ID: {scene.id} ({scene.datetime.date()})")

# 2. Bands for False-Color Composite (NIR, Red, Green)
fcc_bands = ["B08", "B04", "B03"]
fcc_stack = []
out_profile = None

for band in fcc_bands:
    asset_url = scene.assets[band].href
    with rasterio.open(asset_url) as src:
        window = from_bounds(utm_min_x, utm_min_y, utm_max_x, utm_max_y, transform=src.transform)
        # Read windowed data directly over HTTPS
        band_arr = src.read(1, window=window)
        fcc_stack.append(band_arr)
        
        # Prepare geospatial profile from the first read
        if out_profile is None:
            out_profile = src.profile.copy()
            out_transform = rasterio.windows.transform(window, src.transform)
            out_profile.update({
                "driver": "GTiff",
                "height": band_arr.shape[0],
                "width": band_arr.shape[1],
                "count": 3,
                "dtype": band_arr.dtype,
                "transform": out_transform,
                "compress": "lzw"
            })
        print(f"Streamed and cropped {band}...")

# 3. Stack into a 3D array (Bands, Height, Width)
fcc_array = np.stack(fcc_stack, axis=0)

# 4. Save georeferenced GeoTIFF
output_tif = "outputs/jaipur_fcc_premonsoon.tif"
with rasterio.open(output_tif, "w", **out_profile) as dst:
    dst.write(fcc_array)

print(f"Successfully saved georeferenced FCC to: {output_tif}")