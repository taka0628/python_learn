# 数値計算やデータフレーム操作に関するライブラリをインポートする
from pandas import plotting
import numpy as np
import pandas as pd
# 図やグラフを図示するためのライブラリをインポートする。
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sklearn  # 機械学習のライブラリ
from sklearn.decomposition import PCA  # 主成分分析器

df = pd.read_csv("univ-power.csv", sep=",")
name = ["num-books", "num-rooms", "female-ratio",
        "num-faculties", "num-students", "num-doctors"]
# 主成分分析の実行
pca = PCA()
pca.fit(df)

# データを主成分空間に写像
feature = pca.transform(df)

# PCA の固有ベクトル
print(pd.DataFrame(pca.components_, columns=df.columns[:], index=[
      "PC{}".format(x+1) for x in range(len(df.columns))]))

# 第一主成分と第二主成分でプロットする
plt.figure(figsize=(6, 6))
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.9)
plt.grid()
plt.xlabel("principal component 1")
plt.ylabel("principal component 2")
plt.show()

# 第一主成分と第二主成分における観測変数の寄与度をプロットする
plt.figure(figsize=(6, 6))
for x, y, name in zip(pca.components_[0], pca.components_[1], df.columns[:]):
    plt.text(x, y, name)
plt.scatter(pca.components_[0], pca.components_[1], alpha=0.8)
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# 寄与率
print(pd.DataFrame(pca.explained_variance_ratio_, index=[
    "PC{}".format(x + 1) for x in range(len(df.columns))]))

# 累積寄与率を図示する
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution rate")
plt.grid()
plt.show()
