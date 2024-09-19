
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
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

    all_labels = train_df['target'].tolist() + valid_df['target'].tolist()
    unique_labels = np.unique(all_labels)
    
    # ラベルエンコーディングの設定
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_valid_encoded = label_encoder.transform(y_valid)
    
    # モデルのトレーニングとSHAP解析
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train_encoded)
    
    # 予測と評価
    predictions_df = pd.DataFrame({
        'label': y_valid,
        'predicted_label': label_encoder.inverse_transform(model.predict(X_valid)),
        **valid_df.drop(columns=['target']).to_dict('series')
    })
    predictions_df.to_csv('results/classification/result_num_rf.csv', index=False)
    conf_matrix_df = pd.DataFrame(confusion_matrix(predictions_df['label'], predictions_df['predicted_label']),
                                  columns=sorted(set(predictions_df['label']) | set(predictions_df['predicted_label'])),
                                  index=sorted(set(predictions_df['label']) | set(predictions_df['predicted_label'])))
    conf_matrix_df.to_csv('results/classification/confusion_matrix_num_rf.csv')
    
    # メトリクスの計算と表示
    print("Accuracy:", accuracy_score(predictions_df['label'], predictions_df['predicted_label']))
    print("Precision:", precision_score(predictions_df['label'], predictions_df['predicted_label'], average='macro'))
    print("Recall:", recall_score(predictions_df['label'], predictions_df['predicted_label'], average='macro'))

    
    # SHAP解析とプロット
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_valid)
    
    # SHAP値の形状を確認
    print(f"SHAP values shape: {shap_values.shape}")

    feature_names=X_train.columns
    
    # 3クラス以上の場合のSHAPプロット
     # # 2クラスのとき
    if len(unique_labels)==2:
        shap.summary_plot(shap_values[:, :, 1], X_valid, feature_names=feature_names)
    else:
        # 3クラス以上のとき
        for i in range(len(unique_labels)):
            print("Class:" + str(i))  # 修正: iを文字列に変換して出力
            shap.summary_plot(shap_values[:, :, i], X_valid, feature_names=feature_names)
        
    plt.show()


# train_path = 'dataset/train_1.csv'
# valid_path = 'dataset/validation_1.csv'

# model, X_train, X_valid, feature_names = train_and_evaluate_model_rf(train_path, valid_path, "cl-tohoku/bert-base-japanese-v3")
# plot_shap_rf(model, X_train, X_valid, feature_names )


