# This script filter data from log.csv and convert to datetime
# This script also helps determine the threshold of road quality scores
# by visualizing the iriZ - time relationship on that data collected by teamWM
import pandas as pd
import datetime
import matplotlib.pyplot as plt

phone_id = 'e3c058a7bc07c878'
log_filepath = "/var/lib/cdsw/share/log.csv"
df_log = pd.read_csv(log_filepath)
df_log = df_log[df_log['phoneID']==phone_id]
df_log = df_log[df_log['iriZ']<1]
df_log

#convert mills to datetime
base_datetime = datetime.datetime( 1970, 1, 1 )
df_log['time'] = df_log['time'].apply(lambda x:datetime.timedelta( 0, 0, 0,x) + base_datetime)

df_log.to_csv('teamWM_log.csv', index=False)
plt.hist(df_log[df_log['iriZ']<0.06]['iriZ'])

csv_filepath = "teamWM_log.csv"
df = pd.read_csv(csv_filepath)
print(df)
df = df[df['iriZ']<1]
df['time'] = df['time'].apply(lambda x: str(x).split('.')[0])
df['time'] = df['time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
df.sort_values(by='time', ascending=False)
print(df.head())
plt.hist(df[df['iriZ']<0.06]['iriZ'])

df_sub = df[df['time'].dt.date.astype(str)=='2020-11-04']
df_sub['hour'] = df_sub['time'].dt.time.apply(lambda x: int(str(x).split(':')[0]))
df_sub = df_sub[df_sub['hour']==20]
#df_sub = df_sub[df_sub['hour']<21]
df_sub
plt.hist(df_sub['iriZ'])
plt.scatter(df_sub['time'].dt.minute, df_sub['iriZ'])
#plt.xlabel("minute")
#plt.ylabel('iriZ')
plt.show()



