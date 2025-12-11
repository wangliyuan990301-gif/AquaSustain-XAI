# main_pipeline.py
import os
import sys

# 确保当前目录在路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from a1_si_calc import main as run_a1
except ImportError as e:
    print(f" 无法导入 a1_si_calc: {e}")
    exit(1)

try:
    from b1_tif_partition import run_partition
except ImportError as e:
    print(f" 无法导入 b1_tif_partition: {e}")
    exit(1)

try:
    from c_shap_analysis import main as run_c
except ImportError as e:
    print(f" 无法导入 c_shap_analysis: {e}")
    exit(1)


def main():
    print(" 启动全流程分析管道...\n")

    # 阶段 A
    print("【阶段 A】计算 SI 指标...")
    if not run_a1():
        print(" 阶段 A 失败")
        return

    # 阶段 B
    print("\n【阶段 B】栅格分区...")
    if not run_partition():
        print(" 阶段 B 失败")
        return

    # 阶段 C
    print("\n【阶段 C】SHAP 可解释建模...")
    if not run_c():
        print(" 阶段 C 部分失败")

    print("\n 全流程成功完成！")


if __name__ == "__main__":
    main()