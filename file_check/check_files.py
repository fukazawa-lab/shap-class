from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter
import japanize_matplotlib
import matplotlib.pyplot as plt
from datasets import Dataset
from tqdm import tqdm
from datasets import Dataset, ClassLabel
from datasets import load_dataset
import numpy as np
import pandas as pd


from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset

plt.rcParams["font.size"] = 12  # 文字サイズを大きくする

def visualize_text_length(dataset: Dataset):
    # モデル名を指定してトークナイザを読み込む
    model_name = "cl-tohoku/bert-base-japanese-v3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    """データセット中のテキストのトークン数の分布をグラフとして描画"""
    # データセット中のテキストの長さを数える
    length_counter = Counter()
    for data in tqdm(dataset):
        length = len(tokenizer.tokenize(data["sentence"]))
        length_counter[length] += 1
    # length_counterの値から棒グラフを描画する
    # グラフのサイズを設定
    plt.figure(figsize=(4, 3))  # 幅 8 インチ、高さ 6 インチ

    plt.bar(length_counter.keys(), length_counter.values(), width=1.0)
    plt.xlabel("トークン数")
    plt.ylabel("事例数")
    plt.show()



import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

def visualize_labels(file_name, column_name):
    """指定されたファイルのカラム内のラベル分布をグラフとして描画"""
    # ファイルを読み込む
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"エラー: ファイル {file_name} が見つかりません。")
        return
    except Exception as e:
        print(f"エラー: ファイル {file_name} を読み込む際にエラーが発生しました: {e}")
        return

    # 指定されたカラムが存在するかを確認
    if column_name not in df.columns:
        print(f"エラー: ファイル {file_name} に {column_name} カラムが含まれていません。")
        return

    # データセット中のラベルの数を数える
    label_counter = Counter(df[column_name])

    # ラベルIDをラベル名に変換するための辞書（任意で変更可能）
    label_id_to_name = {label_id: f"ラベル{label_id}" for label_id in label_counter.keys()}

    # グラフのサイズを設定
    plt.figure(figsize=(len(label_counter) * 1.5, 3))  # 幅がラベルの数に応じて調整される

    # label_counterを棒グラフとして描画する
    label_names = [label_id_to_name[label_id] for label_id in label_counter.keys()]
    label_counts = list(label_counter.values())
    plt.bar(label_names, label_counts, width=0.6)
    plt.xlabel("ラベル")
    plt.ylabel("事例数")
    plt.show()

