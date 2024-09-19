import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Tuple

def load_data(train_path: str, valid_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data = pd.read_csv(train_path)
    validation_data = pd.read_csv(valid_path)
    return train_data, validation_data

def train_and_evaluate_regressor(X_train, y_train, X_val, y_val, model_name="RandomForest"):
    # モデルを訓練する（RandomForestRegressorに変更）
    regressor = RandomForestRegressor(random_state=42)
    regressor.fit(X_train, y_train)

    # バリデーションデータで予測を行う
    y_pred = regressor.predict(X_val)

    # 回帰向けのメトリクス計算
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mse)

    print(f"■{model_name} Regression")
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("Root Mean Squared Error:", rmse)

    # 予測結果のDataFrameを作成
    predictions_df = pd.DataFrame({
        'label': y_val,
        'predicted_label': y_pred,
        **X_val.to_dict('series')
    })

    # 予測結果をCSVに保存
    predictions_df.to_csv("results/regression/results_num.csv", index=False)

def predict():
    train_data, validation_data = load_data('dataset/train_num.csv', 'dataset/validation_num.csv')
    
    X_train = train_data.drop(columns=['target'])
    y_train = train_data['target']
    X_val = validation_data.drop(columns=['target'])
    y_val = validation_data['target']
    
    train_and_evaluate_regressor(X_train, y_train, X_val, y_val)
