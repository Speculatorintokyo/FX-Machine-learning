
import pandas_datareader.data as web
import datetime
from datetime import timedelta
import pandas as pd


today =  datetime.datetime.now()
end = datetime.datetime(today.year, today.month, today.day)
start = end - timedelta(days=1825)


def chart(symbol):
    data = web.DataReader(symbol, 'fred', start, end)
    data.index = pd.to_datetime(data.index)
    return data

df =  pd.DataFrame(chart("DEXJPUS"))
df.columns = ['USDJPY']

list = ["NIKKEI225","DJIA","SP500","VIXCLS","GOLDAMGBD228NLBM","DCOILWTICO","DGS2","DGS10","T10Y2Y","BAMLH0A3HYC","DEXUSEU"]
for i in list:
    df[i] =  pd.DataFrame(chart(i))


df.tail()

#相関係数
print(df.corr())


#df.to_csv('FilePath')


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#pip install tensorflow
#pip install tensorflow-gpu
#pip install keras
#pip install scikit-learn


#df = pd.read_csv(r"FilePath", header=0,index_col=0,parse_dates=True)

#ドル円レートを同じ「0」上昇「1」下降「2」へデータを変換
def targeting(df):
    mask1 = df['diff'] > 0
    mask2 = df['diff'] < 0
    mask3 = df['diff'] == 0
    column_name = 'target'
    df.loc[mask1, column_name] = 1
    df.loc[mask2, column_name] = 2
    df.loc[mask3, column_name] = 0
    return df

df['diff'] = df['USDJPY'].diff()
targeting(df)
del df['diff']
del df['USDJPY']

#欠損値削除
df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
print(df)

# データをトレーニング用、評価用に分割
X_train, X_test, y_train, y_test = train_test_split(df[df.columns[df.columns != 'target']], df.target, test_size=0.3,shuffle=False)


# 正規化（Normarization）
from sklearn.preprocessing import MinMaxScaler
scaler_x = MinMaxScaler()
X_train_n = scaler_x.fit_transform(X_train)
X_test_n = scaler_x.transform(X_test)

print('train size:', X_train_n.shape[0])
print('test size:', X_test_n.shape[0])


# 正解データを数値からダミー変数の形式に変換
y_train_n = np_utils.to_categorical(y_train)
y_test_n = np_utils.to_categorical(y_test)

print('train data:', X_train_n)
print('train target:', X_test_n)
print('train data:', y_train_n)
print('train target:', y_test_n)


# モデルの定義
model = Sequential([
        Dense(64, input_shape=(11,)),
        Activation('sigmoid'),
        Dense(32,),Activation('sigmoid'),
        Dense(32,),Activation('sigmoid'),
        Dense(3),Activation('softmax')
    ])


# 損失関数、 最適化アルゴリズムなどを設定
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#過学習の防止
es_cb = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
# 学習処理の実行
fit = model.fit(X_train_n, y_train_n, batch_size=16, verbose=1, epochs=500, validation_split=0.1,callbacks=[es_cb])
# 予測
score = model.evaluate(X_test_n, y_test_n, verbose=1)

print('Test score:', score[0])
print('test accuracy : ', score[1])


#学習曲線の可視化
import matplotlib.pyplot as plt
fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

def plot_history_loss(fit):
    # Plot the loss in the history
    axL.plot(fit.history['loss'],label="loss for training")
    axL.plot(fit.history['val_loss'],label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

def plot_history_acc(fit):
    # Plot the loss in the history
    axR.plot(fit.history['accuracy'],label="loss for training")
    axR.plot(fit.history['val_accuracy'],label="loss for validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='upper right')

plot_history_loss(fit)
plot_history_acc(fit)
plt.show()
#fig.savefig('./mnist-tutorial.png')
#plt.close()


'''
"NIKKEI225" ,"日経平均株価"
"DJIA","ダウ平均株価、
"SP500","Ｓ＆Ｐ500、
"VIXCLS","ビックス、
"GOLDAMGBD228NLBM","金、
"DCOILWTICO","WTI原油、
"DGS2","米2年債利回り、
"DGS10","米10年債利回り、
"T10Y2Y","米10年債と2年債の金利差、
"BAMLH0A3HYC","ハイイールドインデックス、
"DEXUSEU","ユーロドル、
'''