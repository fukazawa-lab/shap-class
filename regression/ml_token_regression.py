import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from typing import Tuple
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datasets import Dataset

def load_data(train_path: str, valid_path: str) -> Tuple[Dataset, Dataset]:
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    return train_dataset, valid_dataset

def preprocess_data(train_dataset: Dataset, valid_dataset: Dataset) -> Tuple[np.ndarray, np.ndarray, list, list]:
    all_sentences = train_dataset['sentence'] + valid_dataset['sentence']
    all_labels = train_dataset['label'] + valid_dataset['label']
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v3")
    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in all_sentences]
    tokenized_sentences = [' '.join(tokens) for tokens in tokenized_sentences]
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    X = vectorizer.fit_transform(tokenized_sentences)
    num_train_samples = len(train_dataset)
    X_train = X[:num_train_samples]
    X_valid = X[num_train_samples:]
    train_labels = all_labels[:num_train_samples]
    valid_labels = all_labels[num_train_samples:]
    return X_train, X_valid, train_labels, valid_labels

def train_and_evaluate_regressor(regressor, X_train, train_labels, X_valid, valid_labels, model_name):
    regressor.fit(X_train, train_labels)
    valid_predictions = regressor.predict(X_valid)
    mse = mean_squared_error(valid_labels, valid_predictions)
    mae = mean_absolute_error(valid_labels, valid_predictions)
    rmse = np.sqrt(mse)
    print(f"■{model_name} Regression")
    print("MSE:", mse)
    print("MAE:", mae)
    print("RMSE:", rmse)
    predictions_df = pd.DataFrame({
        'label': valid_labels,
        'sentence': valid_dataset['sentence'],
        'predicted_label': valid_predictions
    })
    predictions_df.to_csv(f"results/regression/results_{model_name.lower()}.csv", index=False)

def predict():
    train_dataset, valid_dataset = load_data('dataset/train.csv', 'dataset/validation.csv')
    X_train, X_valid, train_labels, valid_labels = preprocess_data(train_dataset, valid_dataset)
    
    rf_regressor = RandomForestRegressor()
    train_and_evaluate_regressor(rf_regressor, X_train, train_labels, X_valid, valid_labels, "RandomForest")

    xgb_regressor = XGBRegressor()
    train_and_evaluate_regressor(xgb_regressor, X_train, train_labels, X_valid, valid_labels, "XGBoost")

    lgbm_params = {'verbose': -1}
    lgbm_regressor = LGBMRegressor(**lgbm_params)
    train_and_evaluate_regressor(lgbm_regressor, X_train, train_labels, X_valid, valid_labels, "LightGBM")
