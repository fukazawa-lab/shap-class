import torch
from transformers.trainer_utils import set_seed
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from transformers import Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, r2_score
from pprint import pprint
from datasets import Dataset
from sklearn.preprocessing import MinMaxScaler
from transformers import DataCollatorWithPadding, TrainingArguments
from collections import Counter
from typing import Union
from transformers import BatchEncoding
import importlib
import os

from transformers import pipeline, AutoModelForSequenceClassification, BertJapaneseTokenizer
import torch
import pandas as pd
import shap
import japanize_matplotlib
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import seaborn as sns



def set_random_seed(seed: int = 42):
    set_seed(seed)
    print("乱数シード設定完了")

def load_data(train_path: str, valid_path: str):
    original_train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    train_dataset = Dataset.from_pandas(original_train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    pprint(train_dataset[0])
    return train_dataset, valid_dataset


def load_and_preprocess_data(train_path: str, valid_path: str):
    original_train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)

    scaler = MinMaxScaler()
    original_train_df['label'] = scaler.fit_transform(original_train_df[['label']])
    valid_df['label'] = scaler.transform(valid_df[['label']])

    train_dataset = Dataset.from_pandas(original_train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    pprint(train_dataset[0])
    return train_dataset, valid_dataset, scaler

def tokenize_data(train_dataset, valid_dataset, model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(type(tokenizer).__name__)
    tokens = tokenizer.tokenize(train_dataset[0]['sentence'])
    print(tokens)

    def preprocess_text_classification(example: dict[str, Union[str, int]]) -> BatchEncoding:
        encoded_example = tokenizer(example["sentence"], max_length=512)
        input_tokens = tokenizer.convert_ids_to_tokens(encoded_example["input_ids"])
        encoded_example["labels"] = float(example["label"])
        return encoded_example

    encoded_train_dataset = train_dataset.map(preprocess_text_classification, remove_columns=train_dataset.column_names)
    encoded_valid_dataset = valid_dataset.map(preprocess_text_classification, remove_columns=valid_dataset.column_names)
    print(encoded_train_dataset[0])
    return encoded_train_dataset, encoded_valid_dataset, tokenizer



def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_model_with_contiguous_tensors(model, output_dir):
    state_dict = model.state_dict()
    contiguous_state_dict = {k: v.contiguous() for k, v in state_dict.items()}
    ensure_directory_exists(output_dir)  # Ensure that the directory exists
    file_path = os.path.join(output_dir, "pytorch_model.bin")  # Save as pytorch_model.bin
    torch.save(contiguous_state_dict, file_path)
    
    # Save model config
    config = model.config
    config_file_path = os.path.join(output_dir, "config.json")
    config.to_json_file(config_file_path)


class CustomTrainer(Trainer):
    def _save(self, output_dir, state_dict=None):
        # デフォルトの保存処理をスキップし、カスタム保存関数を使用
        save_model_with_contiguous_tensors(self.model, output_dir)
        
        # Save additional files if needed
        # For example, save tokenizer or other artifacts
        # self.tokenizer.save_pretrained(output_dir)
        # Additional files can be saved here if required

def prepare_model_and_trainer(model_name: str, encoded_train_dataset, encoded_valid_dataset, tokenizer, output_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        num_train_epochs=5,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,  # 最新の1つだけを保存
        evaluation_strategy="epoch",  # Updated to eval_strategy
        load_best_model_at_end=True,
        metric_for_best_model="1/mse",
        fp16=False,
        fp16_full_eval=False,
    )

    def compute_metrics_for_regression(eval_pred):
        logits, labels = eval_pred
        labels = labels.reshape(-1, 1)
        mse = mean_squared_error(labels, logits)
        mae = mean_absolute_error(labels, logits)
        r2 = r2_score(labels, logits)
        single_squared_errors = ((logits - labels).flatten()**2).tolist()
        accuracy = 1 / mse
        return {"mse": mse, "mae": mae, "r2": r2, "1/mse": accuracy}

    trainer = CustomTrainer(
        model=model,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_valid_dataset,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics_for_regression,
    )


    return trainer

def train_and_evaluate(trainer):
    trainer.train()
    latest_eval_metrics = trainer.evaluate()
    print(latest_eval_metrics)
    return trainer

def save_model(trainer, output_dir: str):
    model_save_path = os.path.join(output_dir, "best_model")
    trainer.save_model(model_save_path)
    print(f"モデルを {model_save_path} に保存しました")

def predict_and_save_results(trainer, encoded_valid_dataset, valid_dataset, scaler, output_file: str):
    predictions = trainer.predict(encoded_valid_dataset)
    predictions_df = pd.DataFrame({
        'label': valid_dataset["label"],
        'sentence': valid_dataset["sentence"],
        'predicted_value': predictions.predictions.flatten()
    })

    original_label = scaler.inverse_transform(predictions_df[['label']])
    original_predicted_labels = scaler.inverse_transform(predictions_df[['predicted_value']])
    predictions_df_2 = pd.DataFrame({
        'label': original_label.flatten(),
        'sentence': valid_dataset["sentence"],
        'predicted_value': original_predicted_labels.flatten()
    })

    predictions_df_2.to_csv(output_file, index=False)

    mse_original_scale = mean_squared_error(original_label, original_predicted_labels)
    mae_original_scale = mean_absolute_error(original_label, original_predicted_labels)
    rmse_original_scale = np.sqrt(mse_original_scale)

    print("MSE:", mse_original_scale)
    print("MAE:", mae_original_scale)
    print("RMSE:", rmse_original_scale)

def llm_regression(folder, train_file_name, valid_file_name, model_name):
    set_random_seed(42)
    train_dataset, valid_dataset, scaler = load_and_preprocess_data(folder+"/"+train_file_name, folder+"/"+valid_file_name)
    encoded_train_dataset, encoded_valid_dataset, tokenizer = tokenize_data(train_dataset, valid_dataset, model_name)
    trainer = prepare_model_and_trainer("cl-tohoku/bert-base-japanese-v3", encoded_train_dataset, encoded_valid_dataset, tokenizer, "output_wrime")
    trainer = train_and_evaluate(trainer)
    predict_and_save_results(trainer, encoded_valid_dataset, valid_dataset, scaler, "results/regression/results_lmm.csv")


def shap_values_to_df(shap_values, feature_names):
    num_samples, dummy, num_classes = shap_values.shape

    print("Converting shap values to Dataframe. It takes a few minutes")

    # SHAP値を格納するリスト
    data = []
    for i in range(num_samples):
        print(f"Converting {i}/{num_samples} to Dataframe")
        num_tokens=shap_values[i].data.shape[0]
        for j in range(num_tokens):
            # print(shap_values[i].data)
            for k in range(num_classes):
                # print(j)
                # print(shap_values[i].data[j])
                # print(k)
                data.append({
                    'sample': i,
                    'token': shap_values[i].data[j],
                    'class': 0,
                    'shap_value': shap_values[i].values[j][k]
                })
    print("Done!!")

    # データフレームを作成
    return pd.DataFrame(data)
def calculate_shap(folder, file_name, model_name, num):
    # モデルとトークナイザーの読み込み
    output_dir = "output_wrime"
    best_checkpoint = max([d for d in os.listdir(output_dir) if d.startswith("checkpoint-")], key=lambda x: int(x.split('-')[1]))
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(output_dir, best_checkpoint)).to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # # validation.csvからユニークなラベルのリストを取得
    df = pd.read_csv(folder+"/"+file_name)
    # # unique_labels = df['label'].unique().tolist()
    # unique_labels = sorted(df['label'].unique().tolist())  # ラベルを昇順にソート

    # SHAPのExplainerの設定とSHAP値の計算
    def f(x):
        tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=128, truncation=True) for v in x]).to(model.device)
        attention_mask = (tv != 0).type(torch.int64).to(model.device)
        outputs = model(tv, attention_mask=attention_mask)[0].detach().cpu().numpy()
        return outputs  # 出力を直接返す
        
    sentences = df['sentence'].head(num)
    shap_values = shap.Explainer(f, tokenizer)(sentences.tolist())
    
    # SHAP値の特徴量名を取得
    feature_names = shap_values.feature_names

    # # # SHAP値の構造を出力
    # print("SHAP values type:", type(shap_values))
    # print("SHAP values shape:", shap_values.shape)
    # print("SHAP values sample (first element):", shap_values[0])
    
    # #     # SHAP値の形状を確認
    # print("SHAP values shape:", shap_values.shape)
    # print("Feature names length:", len(shap_values.feature_names))
    # print("Feature names sample:", shap_values.feature_names[:10])

    # SHAP値を DataFrame に変換する例
    feature_names_list = feature_names  # トークンのリスト
    # unique_labels_list = unique_labels  # クラスラベルのリスト
    
    shap_df = shap_values_to_df(shap_values, feature_names_list)
    
    return shap_df



def plot_shap(shap_file, num_appear, top_n):
    # shap_value.csvからデータを読み込む
    shap_df = pd.read_csv(shap_file)
    
    # tokenカラムの半角スペースを削除
    shap_df['token'] = shap_df['token'].str.replace(' ', '', regex=False)
    
    # クラスごとにプロットを作成
    classes = shap_df['class'].unique()
    for cls in classes:

        df_cls = shap_df[shap_df['class'] == cls]
        
        # トークンごとの平均、標準偏差、カウントを計算
        grouped = df_cls.groupby('token')['shap_value'].agg(['mean', 'std', 'count']).reset_index()
        
        # 点の数が3個以上のトークンのみを抽出
        filtered_grouped = grouped[grouped['count'] >= num_appear]
        
        # 平均値の絶対値が大きい順にトークンを並べ替え
        sorted_grouped = filtered_grouped.reindex(filtered_grouped['mean'].abs().sort_values(ascending=False).index)
        
        # top_n トークンのみを選択
        top_tokens = sorted_grouped.head(top_n)['token']
        
        # フィルタリング後のデータを元に df_cls も並べ替え
        df_cls = df_cls[df_cls['token'].isin(top_tokens)]
        df_cls['token'] = pd.Categorical(df_cls['token'], categories=top_tokens, ordered=True)
        
        plt.figure(figsize=(6, 12))
        
        # 散布図を描く
        sns.scatterplot(data=df_cls, x='shap_value', y='token', color='red', alpha=0.5)
        
        # エラーバー付きの横棒を描く（トークンごとに）
        top_grouped = sorted_grouped[sorted_grouped['token'].isin(top_tokens)]
        plt.errorbar(x=top_grouped['mean'], y=top_grouped['token'], xerr=top_grouped['std'], fmt='s', color='blue', capsize=5, label='Mean ± Std')
        
        # x=0 の実線を引く
        plt.axvline(x=0, color='black', linestyle='-')

        
        # プロットのラベルとタイトルを設定
        plt.title(f'SHAP values')
        plt.ylabel('Token')
        plt.xlabel('SHAP Value')
        
        # グリッド表示
        plt.grid(True)
        
        plt.show()


