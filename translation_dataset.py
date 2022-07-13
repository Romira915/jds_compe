import codecs
import csv
import time

import deepl
import pandas as pd
import pyppeteer
from deepl import deepl

translater = deepl.DeepLCLI("en", "ja")

names = ("target", "ids", "date", "flag", "user", "text")

with codecs.open("training.1600000.processed.noemoticon.csv", "r", "utf-8", "ignore") as f:
    df = pd.read_csv(f, names=names)

start = 1848
end = 800000
df = df[start:end]
size = df.shape[0]

with open("training.1600000.processed.noemoticon-ja-0-800000.csv", mode="a") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)

    for index, target, ids, date, flag, user, text in zip(range(start, end), df[names[0]], df[names[1]], df[names[2]], df[names[3]], df[names[4]], df[names[5]]):
        ja = ""
        try:
            ja = translater.translate(text)
        except pyppeteer.errors.NetworkError:
            pass
        except:
            f.flush()
            err = f"error: index {index}, ids {ids}"
            print(err)
            with open("logs/error.log", mode="a") as log:
                log.write(err)
            exit(-1)

        print(f"{index}, {ids}: {ja}")
        writer.writerow([target, ids, date, flag, user, ja])
        f.flush()


# batch_size = 50000
# iter = int(df.shape[0] / batch_size)

# for i in range(0, iter):
#     df[i * batch_size:(i + 1) * batch_size].to_csv(f".tmp/only_text_{i}.txt",
#                                                    columns=["text"], index=False, header=False)
