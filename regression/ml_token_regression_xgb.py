import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import shap
import matplotlib.pyplot as plt

def train_and_evaluate_model(folder, train_path, valid_path, model_name):

    # データのロードと前処理
    train_df = pd.read_csv(folder+"/"+train_path)
    valid_df = pd.read_csv(folder+"/"+valid_path)
    all_sentences = train_df['sentence'].tolist() + valid_df['sentence'].tolist()
    all_labels = train_df['label'].tolist() + valid_df['label'].tolist()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_sentences = [' '.join(tokenizer.tokenize(sentence)) for sentence in all_sentences]
    
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    X = vectorizer.fit_transform(tokenized_sentences)
    X_train, X_valid = X[:len(train_df)].toarray(), X[len(train_df):].toarray()
    y_train, y_valid = np.array(all_labels[:len(train_df)]), np.array(all_labels[len(train_df):])
    
    # モデルのトレーニング
    model = XGBRegressor(use_label_encoder=False, eval_metric='rmse', random_state=42)
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
        'actual_label': y_valid,
        'predicted_label': y_pred,
        'sentence': valid_df['sentence']
    })
    predictions_df.to_csv('results/regression/result_text_xgb.csv', index=False)
    
    # SHAP解析とプロット
    explainer = shap.Explainer(model)
    shap_values = explainer(X_valid)
    
    # SHAP値の形状を確認
    print(f"SHAP values shape: {shap_values.shape}")
    
    # SHAPプロット
    shap.summary_plot(shap_values, X_valid, feature_names=vectorizer.get_feature_names_out())
    
    plt.show()
