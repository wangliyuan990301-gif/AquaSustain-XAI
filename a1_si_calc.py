# a1_si_calc.py
import os
import numpy as np
from rasterio import open as rio_open
from rasterio.crs import CRS
from osgeo import gdal, osr
import rasterio
import glob
import warnings

warnings.filterwarnings('ignore')

# ==============================
# 全局配置
# ==============================
BASE_GWS_PATH = './4.GWS/'
OUTPUT_INDEX_BASE = './5.指标tif/'
THRESHOLD = 0
VALID_THRESHOLD_MK = -10


# ==============================
# 工具函数
# ==============================

def create_output_folders(base_path):
    index_folders = ['REL', 'RES', 'VUL', 'SI']
    folder_paths = {}
    for folder in index_folders:
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        folder_paths[folder] = folder_path
    return folder_paths


def read_georaster(file_path):
    with rio_open(file_path) as src:
        data = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        if crs is None:
            crs = CRS.from_wkt(
                'GEOGCS["WGS 84",DATUM["World Geodetic System 1984",'
                'SPHEROID["WGS 84",6378137,298.257223563]],'
                'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,'
                'AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST]]')
    return data, transform, crs


def save_as_tif(data, transform, crs, file_path):
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(file_path, data.shape[1], data.shape[0], 1, gdal.GDT_Float32)

    if hasattr(transform, 'to_gdal'):
        gdal_transform = transform.to_gdal()
    else:
        gdal_transform = (
            transform.c, transform.a, transform.b,
            transform.f, transform.d, transform.e
        )
    out_ds.SetGeoTransform(gdal_transform)

    srs = osr.SpatialReference()
    wkt_string = ('GEOGCS["WGS 84",DATUM["World Geodetic System 1984",'
                  'SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
                  'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
                  'AXIS["Latitude",NORTH],AXIS["Longitude",EAST]]')
    srs.ImportFromWkt(wkt_string)
    out_ds.SetProjection(srs.ExportToWkt())

    nodata_value = -9999.0
    data_with_nodata = np.where(np.isnan(data), nodata_value, data)

    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(data_with_nodata)
    out_band.SetNoDataValue(nodata_value)
    out_band.FlushCache()
    out_ds = None


def normalize(data):
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    if max_val == min_val:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)


# ==============================
# A1: 指数计算
# ==============================

def calculate_indices():
    base_path = BASE_GWS_PATH
    out_path = OUTPUT_INDEX_BASE
    output_folders = create_output_folders(out_path)

    input_files = sorted([f for f in os.listdir(base_path) if f.endswith('.tif')])
    if not input_files:
        raise FileNotFoundError(f"No TIFF files found in {base_path}")
    n = len(input_files)
    print(f"Found {n} monthly GWS files.")

    first_file_path = os.path.join(base_path, input_files[0])
    data1, transform, crs = read_georaster(first_file_path)
    nan_positions = np.isnan(data1)
    wtd = np.full((data1.shape[0], data1.shape[1], n), np.nan, dtype=np.float32)

    for i, file_name in enumerate(input_files):
        file_path = os.path.join(base_path, file_name)
        try:
            data, _, _ = read_georaster(file_path)
            if data.shape != data1.shape:
                raise ValueError(f"Shape mismatch in {file_name}")
            data[data < -5000] = np.nan
            wtd[:, :, i] = data
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
            wtd[:, :, i] = np.nan

    avg_wtd = np.zeros((data1.shape[0], data1.shape[1], 12), dtype=np.float32)
    for month in range(12):
        month_data = wtd[:, :, month::12]
        avg_wtd[:, :, month] = np.nanmean(month_data, axis=2)

    wtd_gsd = wtd - np.take(avg_wtd, np.arange(n) % 12, axis=2)

    mean_anom = np.nanmean(wtd_gsd, axis=2, keepdims=True)
    std_anom = np.nanstd(wtd_gsd, axis=2, ddof=0, keepdims=True)
    std_anom[std_anom == 0] = np.nan
    ggdi = (wtd_gsd - mean_anom) / std_anom

    years_count = n // 12
    rel_yue = np.empty((data1.shape[0], data1.shape[1], years_count), dtype=np.float32)
    res_yue = np.empty_like(rel_yue)
    vul_yue = np.empty_like(rel_yue)

    for year_idx in range(years_count):
        start = year_idx * 12
        year_data = ggdi[:, :, start:start+12]

        non_drought = np.sum(year_data >= THRESHOLD, axis=2)
        rel_yue[:, :, year_idx] = non_drought / 12.0

        next_data = np.roll(year_data, -1, axis=2)
        recovery = np.sum((year_data < THRESHOLD) & (next_data >= THRESHOLD), axis=2)
        droughts = np.sum(year_data < THRESHOLD, axis=2)

        recovery = np.where(droughts == 0, 1, recovery)
        droughts = np.where(droughts == 0, 1, droughts)
        res_yue[:, :, year_idx] = recovery / droughts

        drought_mask = year_data < THRESHOLD
        vul_yue[:, :, year_idx] = np.nansum(np.abs(year_data) * drought_mask, axis=2) / 12.0

    rel_yue[nan_positions] = np.nan
    res_yue[nan_positions] = np.nan
    vul_yue[nan_positions] = np.nan

    rel_norm = normalize(rel_yue)
    res_norm = normalize(res_yue)
    vul_norm = normalize(vul_yue)
    si_yue = np.power(rel_norm * res_norm * (1 - vul_norm), 1/3)

    years = list(range(2003, 2003 + years_count))
    for year_idx, year in enumerate(years):
        save_as_tif(rel_norm[:, :, year_idx], transform, crs, os.path.join(output_folders['REL'], f'REL_{year}.tif'))
        save_as_tif(res_norm[:, :, year_idx], transform, crs, os.path.join(output_folders['RES'], f'RES_{year}.tif'))
        save_as_tif(vul_norm[:, :, year_idx], transform, crs, os.path.join(output_folders['VUL'], f'VUL_{year}.tif'))
        save_as_tif(si_yue[:, :, year_idx], transform, crs, os.path.join(output_folders['SI'], f'SI_{year}.tif'))

    print(" A1: 指标计算完成！")
    return years_count, output_folders, transform, crs, data1.shape


def calculate_pixel_average(input_dir, output_path):
    tif_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
    if not tif_files:
        print(f" No TIFF files in {input_dir}, skipping average.")
        return

    meta = None
    data_list = []
    for tif in tif_files:
        with rasterio.open(os.path.join(input_dir, tif)) as src:
            if meta is None:
                meta = src.meta.copy()
                profile = src.profile
                height, width = src.height, src.width
            data = src.read(1).astype(np.float64)
            data_list.append(data)

    data_cube = np.stack(data_list, axis=0)
    average_grid = np.nanmean(data_cube, axis=0)

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(average_grid.astype(profile['dtype']), 1)

    print(f" Average saved: {output_path}")


def mann_kendall_trend_analysis(input_folder, output_path, valid_threshold=-10):
    tif_files = sorted(glob.glob(os.path.join(input_folder, "*.tif")))
    if not tif_files:
        print(f" No TIFF files in {input_folder}, skipping trend analysis.")
        return

    print(f"MK: Found {len(tif_files)} files in {input_folder}")

    with rasterio.open(tif_files[0]) as src:
        profile = src.profile
        m, n = src.height, src.width

    cd = len(tif_files)
    datasum = np.full((m * n, cd), np.nan)

    for i, fp in enumerate(tif_files):
        with rasterio.open(fp) as src:
            data = src.read(1).astype(np.float32).reshape(-1)
            datasum[:, i] = data

    sresult = np.full(m * n, np.nan)
    result = np.full(m * n, np.nan)
    zc = np.full(m * n, np.nan)
    tread = np.full(m * n, -9999, dtype=np.int16)

    for i in range(m * n):
        if i % 100000 == 0:
            print(f"MK progress: {i}/{m*n} ({100*i/(m*n):.1f}%)")

        data = datasum[i, :]
        valid_mask = ~np.isnan(data) & (data > valid_threshold)
        if np.sum(valid_mask) < 3:
            continue

        valid_data = data[valid_mask]
        valid_idx = np.where(valid_mask)[0]
        n_valid = len(valid_data)

        sgnsum = []
        slopes = []
        for k in range(1, n_valid):
            for j in range(k):
                diff = valid_data[k] - valid_data[j]
                time_diff = valid_idx[k] - valid_idx[j]
                sgn = np.sign(diff)
                sgnsum.append(sgn)
                if time_diff > 0:
                    slopes.append(diff / time_diff)

        S = sum(sgnsum)
        sresult[i] = S
        if slopes:
            result[i] = np.median(slopes)

        vars_s = n_valid * (n_valid - 1) * (2 * n_valid + 5) / 18.0
        if S == 0:
            z = 0
        elif S > 0:
            z = (S - 1) / np.sqrt(vars_s)
        else:
            z = (S + 1) / np.sqrt(vars_s)
        zc[i] = z

    for i in range(m * n):
        if np.isnan(result[i]) or np.isnan(zc[i]):
            continue
        slope = result[i]
        z_abs = abs(zc[i])
        if slope > 0:
            if z_abs >= 2.58:   tread[i] = 4
            elif z_abs >= 1.96: tread[i] = 3
            elif z_abs >= 1.645:tread[i] = 2
            else:               tread[i] = 1
        elif slope < 0:
            if z_abs >= 2.58:   tread[i] = -4
            elif z_abs >= 1.96: tread[i] = -3
            elif z_abs >= 1.645:tread[i] = -2
            else:               tread[i] = -1
        else:
            tread[i] = 0

    tread_2d = tread.reshape(m, n)
    profile.update(dtype='int16', nodata=-9999)
    trend_out = output_path.replace('.tif', '_trend.tif')
    with rasterio.open(trend_out, 'w', **profile) as dst:
        dst.write(tread_2d.astype(np.int16), 1)

    print(f" Trend result saved: {trend_out}")


def main():
    try:
        gdal_data = os.path.join(os.path.dirname(os.path.abspath(gdal.__file__)), 'data')
        os.environ['PROJ_LIB'] = os.path.join(gdal_data, 'proj')
        gdal.SetConfigOption('GDAL_DATA', gdal_data)
    except Exception as e:
        print("Warning: Could not set GDAL paths:", e)

    years_count, output_folders, transform, crs, shape = calculate_indices()

    for idx in ['REL', 'RES', 'VUL', 'SI']:
        inp = output_folders[idx]
        out_avg = f"A1_{idx}_average.tif"
        calculate_pixel_average(inp, out_avg)

    for idx in ['REL', 'RES', 'VUL', 'SI']:
        inp = output_folders[idx]
        out_trend = f"{idx}_trend.tif"
        mann_kendall_trend_analysis(inp, out_trend, valid_threshold=VALID_THRESHOLD_MK)

    print("\n A1 阶段执行完毕！")
    return True