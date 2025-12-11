# c_shap_analysis.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
import shap
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK', 'Source Han Sans CN', 'WenQuanYi Zen Hei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

shap.initjs()
shap.explainers._tree.TreeExplainer = shap.explainers.Tree

def split_data_by_region():
    print("【步骤1】正在拆分原始数据...")
    try:
        df = pd.read_excel('总数据.xlsx')
    except Exception as e:
        print(f"无法读取 '总数据.xlsx'：{e}")
        return False

    features = ['可持续', '植被', '地表储水', '灯光', '耕地', '工业地', '工业用水', '降水', '人口', '温度', '蒸发', '行数', '列数']
    if not set(features).issubset(df.columns):
        missing = set(features) - set(df.columns)
        print(f"原始数据缺失列: {missing}")
        return False

    output_dir = "分区域数据"
    os.makedirs(output_dir, exist_ok=True)

    for region in range(10):
        region_df = df[df['区域'] == region].copy()
        if not region_df.empty:
            region_df = region_df[features]
            save_path = os.path.join(output_dir, f"{region}.csv")
            region_df.to_csv(save_path, index=False, encoding='utf-8-sig', float_format='%.6f')
            print(f"✓ 区域 {region} 已保存，包含 {len(region_df)} 行数据")
        else:
            print(f"区域 {region} 无数据")

    print("【步骤1】完成！\n")
    return True

def get_param_grid(sample_size):
    return {
        'colsample_bytree': [0.7, 0.8, 0.9],
        'learning_rate': [0.05, 0.10, 0.15],
        'max_depth': [7, 8, 9],
        'n_estimators': [800, 900, 1000],
        'subsample': [0.7, 0.8, 0.9, 1.0],
    }

def train_regional_models():
    print("【步骤2】开始训练区域模型...")

    regional_dir = "分区域数据"
    output_root = "分区分析结果3"
    os.makedirs(output_root, exist_ok=True)

    regional_files = [os.path.join(regional_dir, f) for f in os.listdir(regional_dir) if f.lower().endswith('.csv')]
    print(f"找到 {len(regional_files)} 个分区数据文件")

    results = []
    features = ['植被', '地表储水', '灯光', '耕地', '工业地', '工业用水', '降水', '人口', '温度', '蒸发']
    target = '可持续'

    for file_path in regional_files:
        region_name = os.path.splitext(os.path.basename(file_path))[0]
        region_dir = os.path.join(output_root, region_name)
        os.makedirs(region_dir, exist_ok=True)

        print(f"\n{'=' * 40}\n正在处理区域: {region_name}\n{'=' * 40}")

        try:
            data = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8-sig')
        except UnicodeDecodeError:
            data = pd.read_csv(file_path, sep=None, engine='python', encoding='gbk')

        if data.empty or not set(features + [target]).issubset(data.columns):
            print(f"! 跳过 {region_name}：数据为空或列缺失")
            continue

        X = data[features]
        y = data[target]
        sample_size = len(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

        param_grid = get_param_grid(sample_size)

        grid_search = GridSearchCV(
            estimator=xgb.XGBRegressor(random_state=40, verbosity=0),
            param_grid=param_grid,
            scoring='r2',
            cv=3,
            n_jobs=-1,
            verbose=0,
        )

        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        model_path = os.path.join(region_dir, f"{region_name}_xgboost_model.pkl")
        joblib.dump(best_model, model_path)
        print(f"模型已保存至: {model_path}")

        train_pred = best_model.predict(X_train)
        test_pred = best_model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)

        print(f"训练集 RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
        print(f"测试集 RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

        results.append({
            'region': region_name,
            'sample_size': sample_size,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'best_params': str(best_params)
        })

    if results:
        summary_df = pd.DataFrame(results)
        summary_df = summary_df[['region', 'sample_size', 'train_rmse', 'train_r2', 'test_rmse', 'test_r2', 'best_params']]
        summary_file = os.path.join(output_root, "分区模型性能汇总.csv")
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"\n【步骤2】完成！汇总文件: {summary_file}")
    else:
        print("\n【步骤2】未生成任何模型")

    return True

def perform_shap_analysis():
    print("【步骤3】开始SHAP可解释性分析...")

    data_dir = "分区域数据"
    model_dir = "分区分析结果3"
    output_root = "分区shap分析结果"
    os.makedirs(output_root, exist_ok=True)

    features = ['植被', '地表储水', '灯光', '耕地', '工业地', '工业用水', '降水', '人口', '温度', '蒸发']
    target = '可持续'

    processed_count = 0
    skipped = []

    for data_file in os.listdir(data_dir):
        if not data_file.lower().endswith('.csv'):
            continue

        region_id = os.path.splitext(data_file)[0]
        data_path = os.path.join(data_dir, data_file)
        model_path = os.path.join(model_dir, region_id, f"{region_id}_xgboost_model.pkl")

        if not os.path.exists(model_path):
            skipped.append(region_id)
            continue

        region_dir = os.path.join(output_root, region_id)
        os.makedirs(region_dir, exist_ok=True)

        try:
            data = pd.read_csv(data_path, sep=None, engine='python', encoding='utf-8-sig')
        except UnicodeDecodeError:
            data = pd.read_csv(data_path, sep=None, engine='python', encoding='gbk')

        if data.empty or not set(features + [target]).issubset(data.columns):
            skipped.append(region_id)
            continue

        X = data[features]
        y = data[target]

        try:
            model = joblib.load(model_path)
        except Exception as e:
            skipped.append(region_id)
            continue

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X)
        except Exception as e:
            skipped.append(region_id)
            continue

        shap_df = pd.DataFrame(shap_values.values, columns=features)
        shap_df.insert(0, '样本ID', range(1, 1 + len(shap_df)))
        shap_df.to_csv(os.path.join(region_dir, f'{region_id}_shap_values.csv'), index=False, encoding='utf-8-sig')

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, show=False)
        plt.title(f'{region_id} - SHAP特征重要性')
        plt.tight_layout()
        plt.savefig(os.path.join(region_dir, f'{region_id}_shap_summary.png'), dpi=120)
        plt.close()

        importance = np.abs(shap_values.values).mean(axis=0)
        feature_importance = pd.DataFrame({'特征': features, '重要性': importance}).sort_values('重要性', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['特征'], feature_importance['重要性'], color='#1f77b4')
        plt.title(f'{region_id} - 平均特征重要性')
        plt.xlabel('SHAP值平均绝对值')
        plt.tight_layout()
        plt.savefig(os.path.join(region_dir, f'{region_id}_feature_importance.png'), dpi=120)
        plt.close()

        with open(os.path.join(region_dir, f'{region_id}_report.txt'), 'w', encoding='utf-8') as f:
            f.write(f"区域分析报告: {region_id}\n")
            f.write("=" * 40 + "\n")
            f.write("特征重要性排序:\n")
            for _, row in feature_importance.iterrows():
                f.write(f"  {row['特征']}: {row['重要性']:.4f}\n")

        processed_count += 1
        print(f"✓ 完成 {region_id} 的 SHAP 分析")

    print(f"\n【步骤3】完成！成功分析 {processed_count} 个区域。")
    if skipped:
        print(f"跳过的区域: {skipped}")

    return True

def main():
    if not split_data_by_region():
        return False
    if not train_regional_models():
        return False
    if not perform_shap_analysis():
        pass
    print("\n C 阶段执行完毕！")
    return True