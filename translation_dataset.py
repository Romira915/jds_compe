import codecs

import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver

# Chrome のオプションを設定する
options = webdriver.ChromeOptions()
options.add_argument('--headless')

# Selenium Server に接続する
driver = webdriver.Remote(
    command_executor='http://localhost:4444/wd/hub',
    desired_capabilities=options.to_capabilities(),
    options=options,
)

# Selenium 経由でブラウザを操作する
driver.get('https://qiita.com')
print(driver.current_url)

# ブラウザを終了する
driver.quit()

url = r"https://www.deepl.com/translator#en/ja/I%20couldn't%20bear%20to%20watch%20it.%20%20And%20I%20thought%20the%20UA%20loss%20was%20embarrassing"
res = requests.get(url)
print(res.text.find("見るに耐えなかった"))

soup = BeautifulSoup(res.text, "html.parser")
elems = soup.select("#dl_translator > div.lmt__text > div.lmt__sides_container > div.lmt__side_container.lmt__side_container--target > div.lmt__textarea_container.lmt__textarea_container_no_shadow > div.lmt__translations_as_text > p.lmt__translations_as_text__item.lmt__translations_as_text__main_translation > button.lmt__translations_as_text__text_btn")

# names = ("target", "ids", "date", "flag", "user", "text")

# with codecs.open("training.1600000.processed.noemoticon.csv", "r", "utf-8", "ignore") as f:
#     df = pd.read_csv(f, names=names)

# print(df.shape)
# df[0:50000].to_csv("only_text0.csv", columns=["text"], index=False, header=False)

# batch_size = 50000
# iter = int(df.shape[0] / batch_size)

# for i in range(0, iter):
#     df[i * batch_size:(i + 1) * batch_size].to_csv(f".tmp/only_text_{i}.txt",
#                                                    columns=["text"], index=False, header=False)
