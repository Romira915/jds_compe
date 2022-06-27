import MeCab
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import pipeline

# 感情分析
nlp = pipeline("sentiment-analysis")

# 推論
print(nlp("I love you"))  # ポジティブ
print(nlp("I hate you"))  # ネガティブ

train_df = pd.read_csv("./train.csv")
test_df = pd.read_csv("./test.csv")
compe_df = pd.read_csv("./compe.csv")

train_text = train_df["text"].values.astype('U')
test_text = test_df["text"].values.astype('U')
y = train_df["label"].values.astype("int8")
test_y = test_df["label"].values.astype("int8")
compe_text = compe_df["text"].values.astype('U')

y_pred = np.zeros(len(test_text))
wakati = MeCab.Tagger(
    "-r /home/linuxbrew/.linuxbrew/etc/mecabrc -d /home/linuxbrew/.linuxbrew/lib/mecab/dic/ipadic -O wakati")
for i, text in enumerate(test_text):
    if nlp(wakati.parse(text))[0]["label"] == "NEGATIVE":
        y_pred[i] = 1

print("score", accuracy_score(test_y, y_pred))
