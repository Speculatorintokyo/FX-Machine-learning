#棒読みちゃんを起動します。必要ないかたは削除・コメントアウトしてください。
import subprocess
subprocess.Popen(r"FilePath \BouyomiChan.exe",shell=True)

import pandas_datareader.data as web
import matplotlib.pyplot as plt
from datetime import timedelta
import datetime
import talib
import pyperclip
#必要なライブラリをインストールしてください。
#pip install pandas-datareader
#pip install matplotlib
#pip install TA-Lib
#pip install pyperclip


#データを取得する期間
today =  datetime.datetime.now()
#yesterday = today - timedelta(days=1)
#start = datetime.datetime(2015, 1, 1)
end = datetime.datetime(today.year, today.month, today.day)
start = end - timedelta(days=365)

def chart(symbol,name,ro):
    data = web.DataReader(symbol, 'fred', start, end)
    data = data.dropna(axis=0, how='any')

    #テクニカル指標
    data["MA"] = talib.EMA(data[symbol],timeperiod=21)
    data["RSI"] = talib.RSI(data[symbol],timeperiod=7)
    data["MACD"],data["macdsignal"],data["macdhist"]  = talib.MACD(data[symbol], fastperiod=12, slowperiod=26, signalperiod=9)

    #チャートを描写
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 4), sharex=True,gridspec_kw={'height_ratios': [5, 1,1]})
    axes[0].plot(data[symbol])
    axes[0].plot(data["MA"])
    axes[1].plot(data["RSI"])
    axes[2].plot(data["MACD"])
    axes[2].plot(data["macdsignal"])
    axes[0].grid()
    axes[1].grid()
    axes[2].grid()
    plt.get_current_fig_manager().full_screen_toggle()

    #読み上げる文章
    beforeratio = round(data[symbol][-1] - data[symbol][-2],ro+1)
    if beforeratio > 0:
        trnd = "の上昇です"
    else:
        trnd = "の下落です"

    if data.iloc[-1,3] > data.iloc[-1,4]:
        macstr = "上向きです"
    else:
        macstr = "下向きです"


    tit = name+str(round(data[symbol][-1],ro))+ "　前日比 "+str(beforeratio)+"  " +str(round((beforeratio/data[symbol][-1])*100,2))+"%"+trnd
    pyperclip.copy("、　　　　　　　　　　、"+tit
    +"　あーるえすあいは"+str(round(data.iloc[-1,2],1))
    +"　マックディーは"+ macstr )

    plt.text( 0.05, 0.9, tit,horizontalalignment='left', verticalalignment='top', family='monospace'
    , transform=axes[0].transAxes,fontname="Yu Gothic", fontsize=18)


    plt.draw()
    plt.pause(13)#13秒間停止
    plt.close()


chart("NIKKEI225","日経平均株価、",0)
chart("DJIA","ダウ平均株価、",0)
chart("VIXCLS","ビックス、",2)
chart("GOLDAMGBD228NLBM","金、",1)
chart("DGS10","米10年債利回り、",4)
chart("T10Y2Y","米10年債と2年債の金利差、",2)

if today.weekday() == 0:
    chart("SP500","Ｓ＆Ｐ500、",0)
    chart("WALCL","フェッド資産額、",3)
    chart("BAMLH0A3HYC","ハイイールドインデックス、",3)
    chart("DCOILWTICO","WTI原油、",1)
    chart("DEXJPUS","ドル円、",2)
    chart("DEXUSEU","ユーロドル、",4)
    chart("DGS2","米2年債利回り、",4)
    chart("DGS30","米30年債利回り、",4)


#棒読みちゃんを終了。
subprocess.call("taskkill /IM BouyomiChan.exe")