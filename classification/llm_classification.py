import torch
from transformers.trainer_utils import set_seed
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from pprint import pprint
from datasets import Dataset
from typing import Union
from transformers import BatchEncoding
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, precision_recall_fscore_support

import os
from transformers import pipeline, AutoModelForSequenceClassification, BertJapaneseTokenizer
import torch
import pandas as pd
import shap
import japanize_matplotlib
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


import os
import pandas as pd

import os
from transformers import pipeline, AutoModelForSequenceClassification, BertJapaneseTokenizer
import torch
import pandas as pd
import shap
import japanize_matplotlib
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



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

def tokenize_data(train_dataset, valid_dataset, model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(type(tokenizer).__name__)
    tokens = tokenizer.tokenize(train_dataset[0]['sentence'])
    print(tokens)

    def preprocess_text_classification(example: dict[str, Union[str, int]]) -> BatchEncoding:
        encoded_example = tokenizer(example["sentence"], max_length=512)
        encoded_example["labels"] = example["label"]
        return encoded_example

    encoded_train_dataset = train_dataset.map(preprocess_text_classification, remove_columns=train_dataset.column_names)
    encoded_valid_dataset = valid_dataset.map(preprocess_text_classification, remove_columns=valid_dataset.column_names)
    print(encoded_train_dataset[0])
    return encoded_train_dataset, encoded_valid_dataset, tokenizer

def prepare_model(model_name: str, num_labels: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (AutoModelForSequenceClassification
             .from_pretrained(model_name, num_labels=num_labels)
             .to(device))
    return model




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

def prepare_trainer(model, encoded_train_dataset, encoded_valid_dataset, tokenizer, output_dir: str):
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
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='macro')
        recall = recall_score(labels, predictions, average='macro')
        f1 = f1_score(labels, predictions, average='macro')
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    trainer = CustomTrainer(
        model=model,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_valid_dataset,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
    )
    return trainer

def train_and_evaluate(trainer):
    trainer.train()
    return trainer

def save_predictions(trainer, encoded_valid_dataset, valid_dataset, output_file: str):
    predictions = trainer.predict(encoded_valid_dataset)
    predictions_df = pd.DataFrame({
        'label': predictions.label_ids,
        'predicted_label': predictions.predictions.argmax(axis=1),
        'sentence': valid_dataset["sentence"]
    })
    predictions_df.to_csv(output_file, index=False)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, precision_recall_fscore_support
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, precision_recall_fscore_support
import pandas as pd
import numpy as np

def evaluate_predictions(predictions_df, output_file: str):
    conf_matrix = confusion_matrix(predictions_df['label'], predictions_df['predicted_label'])
    unique_labels = sorted(set(predictions_df['label'].unique()) | set(predictions_df['predicted_label'].unique()))
    conf_matrix_df = pd.DataFrame(conf_matrix, columns=unique_labels, index=unique_labels)
    conf_matrix_df.to_csv(output_file)

    accuracy = accuracy_score(predictions_df['label'], predictions_df['predicted_label'])
    precision = precision_score(predictions_df['label'], predictions_df['predicted_label'], average='macro')
    recall = recall_score(predictions_df['label'], predictions_df['predicted_label'], average='macro')


    # メトリクスの計算と表示
    print("Accuracy:",accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

    # # クラスごとのPrecisionとRecall
    # precisions, recalls, _, _ = precision_recall_fscore_support(predictions_df['label'], predictions_df['predicted_label'], labels=unique_labels)
    # for label, precision, recall in zip(unique_labels, precisions, recalls):
    #     print(f"bert, {label}, non, {precision}, {recall}")

    # # ランダム予測のベースラインの計算
    # label_counts = predictions_df['label'].value_counts(normalize=True).sort_index()
    # baseline_precisions = label_counts.values
    # baseline_recalls = label_counts.values

    # baseline_accuracy = np.mean([label_counts[label] for label in predictions_df['label']])

    # print(f"baseline, all, {baseline_accuracy}, {np.mean(baseline_precisions)}, {np.mean(baseline_recalls)}")

    # # ベースラインのクラスごとのPrecisionとRecall
    # for label, baseline_precision, baseline_recall in zip(unique_labels, baseline_precisions, baseline_recalls):
    #     print(f"baseline, {label}, non, {baseline_precision}, {baseline_recall}")

# 使用例
# predictions_df = pd.DataFrame({
#     'label': [0, 1, 2, 2, 1],
#     'predicted_label': [0, 0, 2, 2, 1]
# })
# evaluate_predictions(predictions_df, 'conf_matrix.csv')

def llm_classification(folder, train_file_name, valid_file_name, model_name):
    set_random_seed(42)
    train_dataset, valid_dataset = load_data(folder+"/"+train_file_name, folder+"/"+valid_file_name)
    encoded_train_dataset, encoded_valid_dataset, tokenizer = tokenize_data(train_dataset, valid_dataset, model_name)
    # encoded_train_dataset, encoded_valid_dataset, tokenizer = tokenize_data(train_dataset, valid_dataset, "cl-tohoku/bert-base-japanese-v3")
    labels = [example["label"] for example in train_dataset]
    num_labels = np.max(labels) + 1
    model = prepare_model(model_name, num_labels)
    trainer = prepare_trainer(model, encoded_train_dataset, encoded_valid_dataset, tokenizer, "output_wrime")
    trainer = train_and_evaluate(trainer)
    save_predictions(trainer, encoded_valid_dataset, valid_dataset, "results/classification/result_llm.csv")

    predictions_df = pd.read_csv( "results/classification/result_llm.csv")
    evaluate_predictions(predictions_df,  "results/classification/confusion_matrix_llm.csv")






def shap_values_to_df(shap_values, feature_names, unique_labels):
    num_samples, dummy, num_classes = shap_values.shape

    print("Converting shap values to Dataframe. It takes a few minutes")

    # SHAP値を格納するリスト
    data = []
    for i in range(num_samples):
        print(f"Converting {i}/{num_samples} to Dataframe")
        num_tokens=shap_values[i].data.shape[0]
        for j in range(num_tokens):
            # print(shap_values[i].data)
            for k in range(len(unique_labels)):
                # print(j)
                # print(shap_values[i].data[j])
                # print(k)
                data.append({
                    'sample': i,
                    'token': shap_values[i].data[j],
                    'class': unique_labels[k],
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
    
    # validation.csvからユニークなラベルのリストを取得
    df = pd.read_csv(folder+"/"+file_name)
    # unique_labels = df['label'].unique().tolist()
    unique_labels = sorted(df['label'].unique().tolist())  # ラベルを昇順にソート

    # SHAPのExplainerの設定とSHAP値の計算
    def f(x):
        tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=128, truncation=True) for v in x]).to(model.device)
        attention_mask = (tv != 0).type(torch.int64).to(model.device)
        outputs = model(tv, attention_mask=attention_mask)[0].detach().cpu().numpy()
        return sp.special.logit((np.exp(outputs).T / np.exp(outputs).sum(-1)).T)
    
    sentences = df['sentence'].head(num)
    shap_values = shap.Explainer(f, tokenizer, output_names=unique_labels)(sentences.tolist())
    
    # SHAP値の特徴量名を取得
    feature_names = shap_values.feature_names

    # # SHAP値の構造を出力
    # print("SHAP values type:", type(shap_values))
    # print("SHAP values shape:", shap_values.shape)
    # print("SHAP values sample (first element):", shap_values[0])
    
    #     # SHAP値の形状を確認
    # print("SHAP values shape:", shap_values.shape)
    # print("Feature names length:", len(shap_values.feature_names))
    # print("Feature names sample:", shap_values.feature_names[:10])

    # SHAP値を DataFrame に変換する例
    feature_names_list = feature_names  # トークンのリスト
    unique_labels_list = unique_labels  # クラスラベルのリスト
    
    shap_df = shap_values_to_df(shap_values, feature_names_list, unique_labels_list)
    
    return shap_df



def plot_shap(shap_file, num_appear, top_n):
    # shap_value.csvからデータを読み込む
    shap_df = pd.read_csv(shap_file)
    
    # tokenカラムの半角スペースを削除
    shap_df['token'] = shap_df['token'].str.replace(' ', '', regex=False)
    
    # 除外したいトークンをリストで指定
    exclude_tokens = ['the','a']

    # 除外するトークンをフィルタリング
    shap_df = shap_df[~shap_df['token'].isin(exclude_tokens)]
    
    # クラスごとにプロットを作成
    classes = shap_df['class'].unique()
    for cls in classes:

        print(f"Class={cls}")
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

