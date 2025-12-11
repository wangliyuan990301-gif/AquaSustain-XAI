# b1_tif_partition.py
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import Point
import os
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def run_partition(shp_file="流域投影SHAP/中国一级流域.shp",
                  tif_file="5.指标tif/SI/SI_2003.tif",
                  output_csv="栅格区域划分结果.csv",
                  output_npy="区域划分栅格.npy",
                  output_tif="区域划分结果.tif",
                  output_img="区域划分可视化.png"):
    print("正在读取shapefile...")
    if not os.path.exists(shp_file):
        raise FileNotFoundError(f"Shapefile未找到: {shp_file}")

    gdf = gpd.read_file(shp_file)
    print(f"读取到 {len(gdf)} 个区域")

    print("正在读取TIF文件...")
    if not os.path.exists(tif_file):
        raise FileNotFoundError(f"TIFFile未找到: {tif_file}")

    with rasterio.open(tif_file) as src:
        raster_data = src.read(1)
        transform = src.transform
        bounds = src.bounds
        rows, cols = raster_data.shape
        profile = src.profile.copy()

    print(f"栅格数据形状: {raster_data.shape}")

    x_coords, y_coords = np.meshgrid(
        np.linspace(bounds.left, bounds.right, cols),
        np.linspace(bounds.top, bounds.bottom, rows)
    )

    region_raster = np.full((rows, cols), -1)
    print("区域栅格已初始化")

    print("正在进行区域划分...")
    region_centers = []
    for geom in gdf.geometry:
        centroid = geom.centroid
        region_centers.append([centroid.x, centroid.y])
    region_centers = np.array(region_centers)

    for i in range(rows):
        for j in range(cols):
            if np.isnan(raster_data[i, j]):
                continue
            pixel_point = Point(x_coords[i, j], y_coords[i, j])
            containing_regions = []
            for region_idx, geometry in enumerate(gdf.geometry):
                if geometry.contains(pixel_point):
                    containing_regions.append(region_idx)
            if containing_regions:
                if len(containing_regions) == 1:
                    region_raster[i, j] = containing_regions[0]
                else:
                    areas = [gdf.geometry.iloc[i].area for i in containing_regions]
                    largest_region = containing_regions[np.argmax(areas)]
                    region_raster[i, j] = largest_region

    print("正在处理边界外像素...")
    unassigned_pixels = np.where(region_raster == -1)
    unassigned_count = len(unassigned_pixels[0])

    if unassigned_count > 0:
        print(f"发现 {unassigned_count} 个未分配像素")
        unassigned_points = np.column_stack((
            x_coords[unassigned_pixels],
            y_coords[unassigned_pixels]
        ))
        if unassigned_points.size > 0 and region_centers.size > 0:
            distances = np.sqrt(
                (unassigned_points[:, 0].reshape(-1, 1) - region_centers[:, 0]) ** 2 +
                (unassigned_points[:, 1].reshape(-1, 1) - region_centers[:, 1]) ** 2
            )
            nearest_regions = np.argmin(distances, axis=1)
            for idx, (i, j) in enumerate(zip(unassigned_pixels[0], unassigned_pixels[1])):
                region_raster[i, j] = nearest_regions[idx]
    else:
        print("没有未分配像素")

    print("正在创建数据集...")
    data_list = []
    for i in range(rows):
        for j in range(cols):
            if not np.isnan(raster_data[i, j]):
                data_list.append({
                    '值': raster_data[i, j],
                    '行数': i,
                    '列数': j,
                    '区域': int(region_raster[i, j])
                })
    dataset = pd.DataFrame(data_list)

    print("\n=== 区域划分统计 ===")
    print(f"总像素数: {len(dataset)}")
    region_counts = dataset['区域'].value_counts().sort_index()
    print("\n各区域像素数量:")
    for region, count in region_counts.items():
        print(f"区域 {int(region)}: {count} 个像素")

    dataset.to_csv(output_csv, index=False, encoding='utf-8-sig')
    np.save(output_npy, region_raster)

    try:
        profile.update(dtype=rasterio.int32, count=1)
        with rasterio.open(output_tif, 'w', **profile) as dst:
            dst.write(region_raster.astype(np.int32), 1)
        print("区域栅格已保存为GeoTIFF")
    except Exception as e:
        print(f"保存GeoTIFF时出错: {e}")

    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        ax[0].imshow(raster_data, cmap='viridis')
        ax[0].set_title('原始栅格数据')
        im = ax[1].imshow(region_raster, cmap='tab20')
        ax[1].set_title('区域划分结果')
        plt.colorbar(im, ax=ax[1], label='区域编号')
        plt.tight_layout()
        plt.savefig(output_img, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"可视化失败: {e}")

    print(" B1 分区完成")
    return True