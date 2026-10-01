import os
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer as pc
from pystac_client import Client
import rasterio
from rasterio.windows import from_bounds

# Ensure the output directory exists
os.makedirs("outputs", exist_ok=True)

# 1. Connect to Microsoft Planetary Computer catalog
print("Connecting to Planetary Computer STAC catalog...")
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)

# 2. Search bounding box in WGS84 [min_lon, min_lat, max_lon, max_lat]
jaipur_lonlat = [75.72, 26.82, 75.92, 27.02]

# Direct UTM Zone 43N bounds for Jaipur (20 km x 20 km sub-scene)
utm_min_x, utm_min_y = 571500.0, 2966500.0
utm_max_x, utm_max_y = 591500.0, 2988500.0

# 3. Query pre-monsoon scene (< 5% cloud cover)
print("Searching for cloud-free scene over Jaipur...")
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=jaipur_lonlat,
    datetime="2024-04-01/2024-05-30",
    query={"eo:cloud_cover": {"lt": 5}},
    limit=1,
)

items = list(search.items())
if not items:
    raise ValueError("No matching scene found within search criteria.")

scene = items[0]
print(f"Streaming Scene ID: {scene.id} (Date: {scene.datetime.date()})")

# 4. Stream and crop Red (B04) and NIR (B08) directly into memory
bands = ["B04", "B08"]
arrays = {}

for band in bands:
    asset_url = scene.assets[band].href
    with rasterio.open(asset_url) as src:
        # Read directly using the native UTM window
        window = from_bounds(utm_min_x, utm_min_y, utm_max_x, utm_max_y, transform=src.transform)
        raw = src.read(1, window=window).astype(np.float32)
        
        # Scale to surface reflectance
        arrays[band] = np.clip((raw - 1000) / 10000.0, 0.0, 1.0)
        print(f"Loaded {band} array with shape: {arrays[band].shape}")

# 5. Compute NDVI (Normalized Difference Vegetation Index)
b4 = arrays["B04"]
b8 = arrays["B08"]
denom = b8 + b4
ndvi = np.where(denom == 0, np.nan, (b8 - b4) / denom)

print(f"NDVI calculated! Min: {np.nanmin(ndvi):.2f}, Max: {np.nanmax(ndvi):.2f}")

# 6. Save visualization to outputs/
plt.figure(figsize=(9, 7))
plt.imshow(ndvi, cmap="RdYlGn", vmin=-0.1, vmax=0.7)
plt.colorbar(label="NDVI (Vegetation Index)")
plt.title(f"Jaipur Pre-Monsoon NDVI — {scene.datetime.date()}")
output_path = "outputs/ndvi_preview.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Success! Generated {output_path} with 0 full-scene downloads.")