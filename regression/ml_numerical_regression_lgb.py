import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import shap
import matplotlib.pyplot as plt

def train_and_evaluate_model(folder, train_path, valid_path):

    # データのロードと前処理
    train_df = pd.read_csv(folder+"/"+train_path)
    valid_df = pd.read_csv(folder+"/"+valid_path)

    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_valid = valid_df.drop(columns=['target'])
    y_valid = valid_df['target']
    
    # モデルのトレーニング
    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # 予測と評価
    y_pred = model.predict(X_valid)
    
    # MSE, MAE, RMSEの計算
    mse = mean_squared_error(y_valid, y_pred)
    mae = mean_absolute_error(y_valid, y_pred)
    rmse = np.sqrt(mse)
    
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("Root Mean Squared Error (RMSE):", rmse)
    
    # 結果を保存
    predictions_df = pd.DataFrame({
        'actual_target': y_valid,
        'predicted_target': y_pred,
        **valid_df.drop(columns=['target']).to_dict('series')
    })
    predictions_df.to_csv('results/regression/result_num_lgb.csv', index=False)
    
    # SHAP解析とプロット
    explainer = shap.Explainer(model)
    shap_values = explainer(X_valid)
    
    # SHAP値の形状を確認
    print(f"SHAP values shape: {len(shap_values)}")  # 回帰問題の場合、shap_valuesはリストになる

    feature_names = X_train.columns
    
    # SHAPプロット
    shap.summary_plot(shap_values, X_valid, feature_names=feature_names)
    
    plt.show()
