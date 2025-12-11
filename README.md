<h1 align="center">  An Interpretable Machine Learning Framework for Basin-Scale Groundwater Sustainability Assessment in China Using GRACE and Multi-Source Geospatial Data </h1>

## ✅ Groundwater Sustainability Mapping Product
You can retrieve our result data from `./data/`

---

## 📖 Abstract
This study develops an integrated framework to assess groundwater sustainability in China by combining hydroclimatic remote sensing data with interpretable machine learning. We compute annual Reliability (REL), Resilience (RES), Vulnerability (VUL), and a composite Sustainability Index (SI) from 2003 to 2023. Region-specific XGBoost models are trained for each major river basin, and SHAP (SHapley Additive exPlanations) is used to quantify feature importance and interactions among key drivers—such as precipitation, temperature, vegetation, surface water, and human activity proxies.
[image](assets/framework.png)


---

## 🛠️ Requirements
- python >= 3.10
- numpy >= 1.21
- pandas >= 1.4
- scikit-learn >= 1.2
- xgboost >= 1.7
- shap >= 0.40
- rasterio >= 1.3
- geopandas >= 0.12
- matplotlib >= 3.6
- tqdm >= 4.60
- scipy >= 1.9

---


