import pandas_datareader.data as web
import matplotlib.pyplot as plt
from datetime import timedelta
import datetime
import talib
import pyperclip



today =  datetime.datetime.now()
#yesterday = today - timedelta(days=1)
#start = datetime.datetime(2015, 1, 1)
end = datetime.datetime(today.year, today.month, today.day)
start = end - timedelta(days=365)

def chart(symbol,name,ro):
    data = web.DataReader(symbol, 'fred', start, end)
    data = data.dropna(axis=0, how='any')

    data["MA"] = talib.EMA(data[symbol],timeperiod=21)
    data["RSI"] = talib.RSI(data[symbol],timeperiod=7)
    data["MACD"],data["macdsignal"],data["macdhist"]  = talib.MACD(data[symbol], fastperiod=12, slowperiod=26, signalperiod=9)

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

    beforeratio = round(data[symbol][-1] - data[symbol][-2],ro+1)
    if beforeratio > 0:
        trnd = "の上昇です"
    else:
        trnd = "の下落です"

    if data.iloc[-1,3] > data.iloc[-1,4]:
        macstr = "上向きです"
    else:
        macstr = "下向きです"

#data.iloc[-1,1]
    tit = name+str(round(data[symbol][-1],ro))+ "　前日比　"+str(beforeratio)+trnd
    pyperclip.copy("、　　　　　　　　　　、"+tit
    +"　あーるえすあいは"+str(round(data.iloc[-1,2],1))
    +"　マックディーは"+ macstr )

    plt.text( 0.1, 0.9, tit,horizontalalignment='left', verticalalignment='top', family='monospace'
    , transform=axes[0].transAxes,fontname="Yu Gothic", fontsize=18)

    plt.draw()
    plt.pause(13)
    plt.close()


chart("NIKKEI225","日経平均株価、",0)
chart("DJIA","ダウ平均株価、",0)
chart("VIXCLS","ビックス、",2)
chart("DGS2","米2年債利回り、",4)
chart("GOLDAMGBD228NLBM","金、",1)


if today.weekday() == 0:
    chart("SP500","Ｓ＆Ｐ500、",0)
    chart("WALCL","フェッド資産、",3)
    chart("BAMLH0A3HYC","ジャンク債、",3)
    chart("DEXJPUS","ドル円、",2)
    chart("DEXUSEU","ユーロドル、",4)
    chart("DCOILWTICO","原油、",1)