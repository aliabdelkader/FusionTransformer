from pathlib import Path

save_runs = {
"lidar": ["MONTH_11_DAY_22_HOUR_02_MIN_43_SEC_46"],
"earlyfusion":["MONTH_12_DAY_02_HOUR_06_MIN_51_SEC_52"],
"image": ["MONTH_11_DAY_17_HOUR_23_MIN_14_SEC_54"],
"imageBilinear": ["MONTH_11_DAY_25_HOUR_02_MIN_42_SEC_29", "MONTH_12_DAY_24_HOUR_17_MIN_56_SEC_16"],
"latefusion": ["MONTH_12_DAY_02_HOUR_07_MIN_00_SEC_25"],
"middlefusion": ["MONTH_12_DAY_02_HOUR_06_MIN_56_SEC_17"],
}

root_path = Path("/home/user/logs/semantic_kitti")

for dir, runs in save_runs.items():
    stored = ( root_path / dir).glob("*")
    print(stored)