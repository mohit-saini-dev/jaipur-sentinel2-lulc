import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from pyproj import Transformer

training_geojson = "data/jaipur_training.geojson"
raster_path = "outputs/jaipur_fcc_premonsoon.tif"

with open(training_geojson, "r", encoding="utf-8-sig") as f:
    geojson_data = json.load(f)

with rasterio.open(raster_path) as src:
    raster_profile = src.profile.copy()
    raw_img = src.read().astype(np.float32)  # Shape: (3, H, W) -> [NIR, Red, Green]
    transform = src.transform
    crs = src.crs
    res_x, res_y = abs(src.res[0]), abs(src.res[1])

    # Extract raw bands
    nir = raw_img[0]
    red = raw_img[1]
    green = raw_img[2]

    # Calculate Remote Sensing Indices (with epsilon to prevent division by zero)
    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    ratio_rb = (red - green) / (red + green + eps)

    # Stack into 6-channel feature tensor: [NIR, Red, Green, NDVI, NDWI, Ratio]
    feature_stack = np.stack([nir, red, green, ndvi, ndwi, ratio_rb], axis=0)
    n_features, height, width = feature_stack.shape

    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    X_list, y_list = [], []

    for feature in geojson_data["features"]:
        class_id = feature["properties"]["class_id"]
        geom_type = feature["geometry"]["type"]
        raw_coords = feature["geometry"]["coordinates"]

        if geom_type == "Polygon":
            projected_coords = [
                [transformer.transform(x, y) for x, y in ring]
                for ring in raw_coords
            ]
            proj_geom = {"type": "Polygon", "coordinates": projected_coords}
            mask = geometry_mask([proj_geom], out_shape=(height, width), transform=transform, invert=True)
            
            # Extract all 6 features for masked pixels
            pixels = feature_stack[:, mask].T

            if pixels.shape[0] > 0:
                max_samples = 3000
                if pixels.shape[0] > max_samples:
                    idx = np.random.choice(pixels.shape[0], max_samples, replace=False)
                    pixels = pixels[idx]

                X_list.append(pixels)
                y_list.append(np.full(pixels.shape[0], class_id))
                print(f"Class {class_id} ({feature['properties']['class_name']}): {pixels.shape[0]} training pixels.")

X = np.vstack(X_list)
y = np.concatenate(y_list)

# Split and train Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=150, max_depth=16, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("\n--- Feature-Engineered Classification Report ---")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importances
feature_names = ["NIR", "Red", "Green", "NDVI", "NDWI", "Soil/Built Ratio"]
print("\n--- Feature Importances ---")
for name, imp in zip(feature_names, rf.feature_importances_):
    print(f"{name:<18}: {imp*100:.2f}%")

# Predict Full Scene
flat_stack = feature_stack.reshape(n_features, -1).T
classified_flat = rf.predict(flat_stack)
lulc_map = classified_flat.reshape(height, width)

# Save Classified GeoTIFF
raster_profile.update(dtype=rasterio.uint8, count=1, nodata=0)
with rasterio.open("outputs/jaipur_lulc_2026.tif", "w", **raster_profile) as dst:
    dst.write(lulc_map.astype(rasterio.uint8), 1)

# Area Statistics
pixel_area_km2 = (res_x * res_y) / 1e6
class_names = {1: "Water", 2: "Vegetation", 3: "Built-up", 4: "Barren"}
total_pixels = lulc_map.size

print("\n--- Jaipur Land Cover Area Statistics ---")
for c_id, name in class_names.items():
    count = np.sum(lulc_map == c_id)
    area_km2 = count * pixel_area_km2
    pct = (count / total_pixels) * 100
    print(f"{name:<12}: {area_km2:>8.2f} km² ({pct:>5.1f}%)")

# Side-by-Side Thematic Visualisation
class_colors = ["#0077be", "#2ca02c", "#d62728", "#e6c280"]
cmap = ListedColormap(class_colors)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

fcc_display = np.stack([raw_img[0], raw_img[1], raw_img[2]], axis=-1)
for i in range(3):
    p2, p98 = np.percentile(fcc_display[:, :, i], (2, 98))
    fcc_display[:, :, i] = np.clip((fcc_display[:, :, i] - p2) / (p98 - p2 + 1e-6), 0, 1)

axes[0].imshow(fcc_display)
axes[0].set_title("Sentinel-2 False Color Composite (NIR-R-G)", fontsize=13)
axes[0].axis("off")

axes[1].imshow(lulc_map, cmap=cmap, vmin=1, vmax=4)
axes[1].set_title("LULC Map (Random Forest + Spectral Indices)", fontsize=13)
axes[1].axis("off")

legend_elements = [Patch(facecolor=class_colors[i], edgecolor='black', label=f"{class_names[i+1]}") for i in range(4)]
axes[1].legend(handles=legend_elements, loc="lower right", framealpha=0.9, fontsize=11)

plt.tight_layout()
plt.savefig("outputs/jaipur_lulc_preview.png", dpi=300)
print("\nPlot saved: outputs/jaipur_lulc_preview.png")