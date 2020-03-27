import pandas as pd

#ＭＴ４のヒストリカルデータcsv形式を読み込み
df = pd.read_csv(r"FilePath", names=('day', 'time2', 'open', 'high', 'low', 'close', 'volume'))
df
df.dtypes
df['data'] = df['day'].str.cat(df['time2'], sep='.')
df['time'] = pd.to_datetime(df['data'], format='%Y.%m.%d.%H:%M')
df = df.set_index(df['time'])

del df['day']
del df['time2']
del df['data']
del df['time']

df
#保存するファイル名を指定
df.to_csv(r"FilePath", encoding="utf-8", float_format='%.3f')

