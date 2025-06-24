import numpy as np
import pandas as pd
import os


def ensure_numpy(data):
    """
    Chuyển đổi data thành ndarray float, bất kể là DataFrame, Series hay list.
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.to_numpy(dtype=float)
    return np.array(data, dtype=float)


def save_metrics_to_csv(filename, model_name, dataset_type, metrics_dict):
    """
    Lưu các chỉ số đánh giá vào file CSV.

    Parameters:
    - filename: tên file CSV (ví dụ: "reports/regression_metrics.csv")
    - model_name: tên mô hình, ví dụ: "KNN"
    - dataset_type: "train" hoặc "test"
    - metrics_dict: dict chứa các chỉ số đánh giá, ví dụ: {"MSE": ..., "MAE": ...}
    """

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)

        # Ghi header nếu file chưa tồn tại
        if not file_exists:
            header = ["Model", "Dataset"] + list(metrics_dict.keys())
            writer.writerow(header)

        row = [model_name, dataset_type] + list(metrics_dict.values())
        writer.writerow(row)
