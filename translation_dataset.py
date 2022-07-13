import codecs
import csv
import time
import urllib.parse

import deepl
import lxml
import pandas as pd
import pyperclip
import pyppeteer
import requests
from bs4 import BeautifulSoup
from deepl import deepl
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# # Chrome のオプションを設定する
# options = webdriver.ChromeOptions()
# options.add_argument('--headless')

# # Selenium Server に接続する
# driver = webdriver.Remote(
#     command_executor='http://localhost:4444/wd/hub',
#     desired_capabilities=options.to_capabilities(),
#     options=options,
# )

# # Selenium 経由でブラウザを操作する
# driver.get('https://qiita.com')
# print(driver.current_url)

# # ブラウザを終了する
# driver.quit()

translater = deepl.DeepLCLI("en", "ja")

names = ("target", "ids", "date", "flag", "user", "text")

with codecs.open("training.1600000.processed.noemoticon.csv", "r", "utf-8", "ignore") as f:
    df = pd.read_csv(f, names=names)

size = df.shape[0]

with open("training.1600000.processed.noemoticon-ja.csv", mode="a") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    for index, target, ids, date, flag, user, text in zip(range(size), df[names[0]], df[names[1]], df[names[2]], df[names[3]], df[names[4]], df[names[5]]):
        try:
            ja = translater.translate(text)
        except pyppeteer.errors.NetworkError:
            pass
        except:
            f.flush()
            print(f"error: index {index}")

        print(f"{index} {ja}")
        time.sleep(1)
        writer.writerow([target, ids, date, flag, user, ja])
        f.flush()


# batch_size = 50000
# iter = int(df.shape[0] / batch_size)

# for i in range(0, iter):
#     df[i * batch_size:(i + 1) * batch_size].to_csv(f".tmp/only_text_{i}.txt",
#                                                    columns=["text"], index=False, header=False)
