
import matplotlib.pyplot as plt
import mpl_finance as mpf
import pandas as pd
#pip install mpl_finance

#読み込みファイル
df = pd.read_csv(r"FilePath", header=0,index_col=0,parse_dates=True)
df
#時間軸の変更　　 ↓ここを、5T,15T,30T,H,4H,D,W,M　に書き換え　　T分H時D日W週M月
df = df.resample("D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})

#保存するファイル名を指定
df.to_csv(r"FilePath")

# matplotlibでローソク足作成
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
mpf.candlestick2_ohlc(ax,opens=df.open,closes=df.close,lows=df.low,highs=df.high,colorup='r',colordown='b',width=1)
ax.grid()

plt.show()