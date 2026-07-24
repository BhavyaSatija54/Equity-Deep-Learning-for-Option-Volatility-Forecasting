
# imports
import pandas as pd


grid_path = '/content/drive/MyDrive/bcmocccogeuhqddt_grid_filtered.csv'
features_path = '/content/drive/MyDrive/bcmocccogeuhqddt_features.csv'


df = pd.read_csv(grid_path, parse_dates=['date'])


print("Unique tenors (days):", sorted(df['days'].unique()))



iv = df.pivot_table(
       index='date',
       columns=['days','delta'],
       values='impl_volatility'
).sort_index(axis=1)


iv_weekly = iv.rolling(window=5, min_periods=5).mean().add_suffix('_w5')
iv_monthly = iv.rolling(window=22, min_periods=22).mean().add_suffix('_w22')



features = pd.concat([iv, iv_weekly, iv_monthly], axis=1)


features = features.dropna()


features.to_csv(features_path, index=True)
print(f"Features saved to {features_path}")
print("Feature matrix shape:", features.shape)

import pandas as pd


grid_path = '/content/drive/MyDrive/bcmocccogeuhqddt_grid_filtered.csv'
features_path = '/content/drive/MyDrive/bcmocccogeuhqddt_features_flat.csv'


df = pd.read_csv(grid_path, parse_dates=['date'])


iv = df.pivot_table(
   index='date',
   columns=['days','delta'],
   values='impl_volatility'
).sort_index(axis=1)


iv.columns = [f"{t}_{d}" for t,d in iv.columns]


iv_w5 = iv.rolling(window=5, min_periods=5).mean().add_suffix('_w5')
iv_w22 = iv.rolling(window=22, min_periods=22).mean().add_suffix('_w22')


features = pd.concat([iv, iv_w5, iv_w22], axis=1).dropna()


features.to_csv(features_path, index=True)
print("Flat features saved to:", features_path)
print("Shape:", features.shape)

import pandas as pd

features_path = '/content/drive/MyDrive/bcmocccogeuhqddt_features_flat.csv'
df = pd.read_csv(features_path, index_col=0, parse_dates=True, low_memory=False)
print("Index type:", type(df.index))
print("First 5 dates:", df.index[:5])
print("Feature matrix shape:", df.shape)


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K


features_path = '/content/drive/MyDrive/bcmocccogeuhqddt_features_flat.csv'
df = pd.read_csv(features_path, index_col=0, parse_dates=True, low_memory=False)
print("Loaded features:", df.shape, "from", df.index.min(), "to", df.index.max() )


total_cols = df.shape[1]
n_grid = total_cols // 3

iv_daily = df.iloc[:, :n_grid].values
iv_weekly = df.iloc[:, n_grid:2*n_grid].values
iv_monthly = df.iloc[:, 2*n_grid:3*n_grid].values


X_all = np.stack([iv_daily, iv_weekly, iv_monthly], axis=1)
y_all = iv_daily[1:]
X_all = X_all[:-1]
dates = df.index[:-1]


cutoff = df.index.min() + pd.DateOffset(years=8)
train_mask = dates < cutoff
val_mask = dates >= cutoff

X_train, y_train = X_all[train_mask], y_all[train_mask]
X_val, y_val = X_all[val_mask], y_all[val_mask]

print("Train shapes:", X_train.shape, y_train.shape)
print("Val shapes: ", X_val.shape, y_val.shape)


inp = layers.Input(shape=(3, n_grid), name='input')


h1 = layers.LSTM(135, return_sequences=True, name='lstm1')(inp)
h1 = layers.Dropout(0.2, name='drop1')(h1)


score = layers.Dense(1, name='score')(h1)


attn_weights = layers.Softmax(axis=1, name='attn_weights')(score)


context = layers.Lambda(lambda x: K.sum(x[0] * x[1], axis=1),
                        name='context')([attn_weights, h1])


context_seq = layers.RepeatVector(3, name='context_seq')(context)


h2 = layers.LSTM(135, name='lstm2')(context_seq)
h2 = layers.Dropout(0.2, name='drop2')(h2)


out = layers.Dense(n_grid, name='output')(h2)

model = models.Model(inp, out)
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
model.summary()


ckpt_path = '/content/drive/MyDrive/iv_attlstm_ckpt.h5'
cp_cb = tf.keras.callbacks.ModelCheckpoint(
    ckpt_path, save_best_only=True, monitor='val_loss'
)


history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=50,
    callbacks=[cp_cb]
)


import pickle
with open('/content/drive/MyDrive/attlstm_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
model.save('/content/drive/MyDrive/iv_attlstm_final.h5')
print(" Training finished. Checkpoint, model, and history are on Drive.")

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout
import pickle


baseline_histories = {}


mlp_in = Input(shape=(3, n_grid), name='mlp_input')
x = Flatten(name='mlp_flat')(mlp_in)
x = Dense(256, activation='relu', name='mlp_dense1')(x)
x = Dropout(0.2, name='mlp_drop1')(x)
x = Dense(128, activation='relu', name='mlp_dense2')(x)
x = Dropout(0.2, name='mlp_drop2')(x)
x = Dense(64, activation='relu', name='mlp_dense3')(x)
out_mlp = Dense(n_grid, name='mlp_output')(x)

mlp = Model(mlp_in, out_mlp, name='MLP')
mlp.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
hist_mlp = mlp.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=50,
    verbose=2
)
baseline_histories['MLP'] = hist_mlp.history


lstm_in = Input(shape=(3, n_grid), name='lstm_input')
h = LSTM(135, return_sequences=True, name='lstm1_base')(lstm_in)
h = Dropout(0.2, name='drop1_base')(h)
h = LSTM(135, name='lstm2_base')(h)
h = Dropout(0.2, name='drop2_base')(h)
out_lstm = Dense(n_grid, name='lstm_output')(h)

vanilla_lstm = Model(lstm_in, out_lstm, name='Vanilla_LSTM')
vanilla_lstm.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
hist_lstm = vanilla_lstm.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=50,
    verbose=2
)
baseline_histories['Vanilla_LSTM'] = hist_lstm.history


with open('/content/drive/MyDrive/baseline_histories.pkl', 'wb') as f:
    pickle.dump(baseline_histories, f)

print("Baseline (MLP & Vanilla LSTM) training complete.")

import pickle
import matplotlib.pyplot as plt


with open('/content/drive/MyDrive/attlstm_history.pkl', 'rb') as f:
   hist_att = pickle.load(f)
with open('/content/drive/MyDrive/baseline_histories.pkl', 'rb') as f:
   hist_base = pickle.load(f)


epochs_att = range(1, len(hist_att['loss'])+1)
mse_att_tr = hist_att['loss']
mse_att_val= hist_att['val_loss']

epochs_mlp = range(1, len(hist_base['MLP']['loss'])+1)
mse_mlp_tr = hist_base['MLP']['loss']
mse_mlp_val= hist_base['MLP']['val_loss']

epochs_lstm = range(1, len(hist_base['Vanilla_LSTM']['loss'])+1)
mse_lstm_tr = hist_base['Vanilla_LSTM']['loss']
mse_lstm_val= hist_base['Vanilla_LSTM']['val_loss']


plt.figure(figsize=(10,6))
plt.plot(epochs_att, mse_att_tr, label='Att-LSTM Train')
plt.plot(epochs_att, mse_att_val, label='Att-LSTM Val', linestyle='--')
plt.plot(epochs_lstm, mse_lstm_tr, label='LSTM Train')
plt.plot(epochs_lstm, mse_lstm_val,label='LSTM Val', linestyle='--')
plt.plot(epochs_mlp, mse_mlp_tr, label='MLP Train')
plt.plot(epochs_mlp, mse_mlp_val, label='MLP Val', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training & Validation MSE for MLP, LSTM, Att-LSTM')
plt.legend()
plt.show()

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


model.load_weights('/content/drive/MyDrive/iv_attlstm_ckpt.h5')


yhat_att = model.predict(X_val)
yhat_mlp = mlp.predict(X_val)
yhat_lstm = vanilla_lstm.predict(X_val)
# func qlike


def qlike(y, yhat):
   return np.mean(np.log(yhat / y) + y / yhat)


results = {}
for name, yhat in [
        ('Att-LSTM', yhat_att),
        ('Vanilla_LSTM', yhat_lstm),
        ('MLP', yhat_mlp)
   ]:
   mse = mean_squared_error(y_val.flatten(), yhat.flatten())
   mae = mean_absolute_error(y_val.flatten(), yhat.flatten())
   ql = qlike(y_val.flatten(), yhat.flatten())
   results[name] = {'MSE': mse, 'MAE': mae, 'QLIKE': ql}


import pandas as pd
pd.DataFrame(results).T

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error
# func build attlstm


def build_attlstm(n_grid):
   inp = layers.Input((3, n_grid))
   h1 = layers.LSTM(135, return_sequences=True)(inp)
   h1 = layers.Dropout(0.2)(h1)

   score = layers.Dense(1)(h1)
   attn = layers.Softmax(axis=1)(score)
   context = layers.Lambda(lambda x: K.sum(x[0]*x[1], axis=1))([attn, h1])
   seq = layers.RepeatVector(3)(context)
   h2 = layers.LSTM(135)(seq)
   h2 = layers.Dropout(0.2)(h2)
   out = layers.Dense(n_grid)(h2)
   m = models.Model(inp, out)
   m.compile('rmsprop', 'mse')
   return m
# func build vanilla lstm


def build_vanilla_lstm(n_grid):
   inp = layers.Input((3, n_grid))
   h = layers.LSTM(135, return_sequences=True)(inp)
   h = layers.Dropout(0.2)(h)
   h = layers.LSTM(135)(h)
   h = layers.Dropout(0.2)(h)
   out = layers.Dense(n_grid)(h)
   m = models.Model(inp, out)
   m.compile('rmsprop', 'mse')
# func build mlp
   return m

def build_mlp(n_grid):
   inp = layers.Input((3, n_grid))
   x = layers.Flatten()(inp)
   x = layers.Dense(256, activation='relu')(x); x = layers.Dropout(0.2)(x)
   x = layers.Dense(128, activation='relu')(x); x = layers.Dropout(0.2)(x)
   x = layers.Dense(64, activation='relu')(x); x = layers.Dropout(0.2)(x)
   out = layers.Dense(n_grid)(x)
   m = models.Model(inp, out)
   m.compile('rmsprop', 'mse')
   return m





window_train = pd.DateOffset(years=3)
window_test = pd.DateOffset(years=2)
start_date = dates.min()
end_date = dates.max()


roll_results = []
current_start = start_date

while True:
   train_start = current_start
   train_end = train_start + window_train
   test_start = train_end
   test_end = test_start + window_test
   if test_end > end_date:
        break


   tr_mask = (dates >= train_start) & (dates < train_end)
   te_mask = (dates >= test_start) & (dates < test_end)

   X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
   X_te, y_te = X_all[te_mask], y_all[te_mask]

   metrics = {}

   for name, builder in [
        ('Att-LSTM', build_attlstm),
        ('Vanilla_LSTM', build_vanilla_lstm),
        ('MLP', build_mlp)
   ]:
        m = builder(n_grid)
        m.fit(X_tr, y_tr, epochs=50, batch_size=50, verbose=0)

        y_tr_hat = m.predict(X_tr)
        mse_tr = mean_squared_error(y_tr.flatten(), y_tr_hat.flatten())
        mae_tr = mean_absolute_error(y_tr.flatten(), y_tr_hat.flatten())
        ql_tr = np.mean(np.log(y_tr_hat/y_tr) + y_tr/y_tr_hat)

        y_te_hat = m.predict(X_te)
        mse_te = mean_squared_error(y_te.flatten(), y_te_hat.flatten())
        mae_te = mean_absolute_error(y_te.flatten(), y_te_hat.flatten())
        ql_te = np.mean(np.log(y_te_hat/y_te) + y_te/y_te_hat)

        metrics[name] = {
           'in_MSE': mse_tr, 'in_MAE': mae_tr, 'in_QLIKE': ql_tr,
               'out_MSE': mse_te, 'out_MAE': mae_te, 'out_QLIKE': ql_te
          }

     roll_results.append({
          'window_start': train_start.strftime('%Y-%m-%d'),
          **{f'{model}_{k}': v for model, met in metrics.items() for k, v in met. items()}
     })

     current_start += pd.DateOffset(years=1)


df_roll = pd.DataFrame(roll_results)
summary = df_roll.filter(like='in_MSE').mean().to_frame('Mean').join(
              df_roll.filter(like='out_MSE').mean().to_frame('Mean'), rsuffix='_out')
print("Average rolling-window metrics (in- and out-sample):")
print(summary)




import pandas as pd




models = ['Att-LSTM', 'Vanilla_LSTM', 'MLP']
metrics = ['MSE', 'MAE', 'QLIKE']


summary = pd.DataFrame(index=models, columns=[
     'in_MSE','out_MSE','in_MAE','out_MAE','in_QLIKE','out_QLIKE'
])


for mod in models:
     for m in metrics:
          summary.loc[mod, f'in_{m}'] = df_roll[f'{mod}_in_{m}'].mean()
          summary.loc[mod, f'out_{m}'] = df_roll[f'{mod}_out_{m}'].mean()

print("Rollingwindow averages:")
print(summary)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


feat_path = '/content/drive/MyDrive/bcmocccogeuhqddt_features_flat.csv'
df_feat = pd.read_csv(feat_path, index_col=0, parse_dates=True, low_memory=False )


n_grid = df_feat.shape[1] // 3
cols = df_feat.columns[:n_grid]

tenors = sorted({int(c.split('_')[0]) for c in cols})
deltas = sorted({int(c.split('_')[1]) for c in cols})


pos = [(int(c.split('_')[0]), int(c.split('_')[1])) for c in cols]

idx_map = { (t,d): i for i,(t,d) in zip(range(n_grid), pos) }


iv_daily = df_feat.iloc[:, :n_grid].values
iv_weekly = df_feat.iloc[:, n_grid:2*n_grid].values
iv_monthly = df_feat.iloc[:,2*n_grid:3*n_grid].values
X_all = np.stack([iv_daily, iv_weekly, iv_monthly], axis=1)
y_all = iv_daily[1:]
X_all = X_all[:-1]
dates = df_feat.index[:-1]



cutoff = df_feat.index.min() + pd.DateOffset(years=8)
test_dates= dates[dates >= cutoff]
sample_dt = test_dates[0]
idx = np.where(dates == sample_dt)[0][0]



X_input = X_all[idx:idx+1]
y_pred = model.predict(X_input)[0]
y_true = y_all[idx]


Z_pred = np.zeros((len(tenors), len(deltas)))
Z_true = np.zeros_like(Z_pred)
for i,(t,d) in enumerate(pos):
   ti = tenors.index(t)
   di = deltas.index(d)
   Z_pred[ti,di] = y_pred[i]
   Z_true[ti,di] = y_true[i]


fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,5), sharey=True)
im1 = ax1.imshow(Z_true, origin='lower', aspect='auto')
ax1.set_title(f'Actual IV surface\n{sample_dt.date()}')
ax1.set_xlabel('Delta')
ax1.set_ylabel('Tenor (days)')
ax1.set_xticks(range(len(deltas))); ax1.set_xticklabels(deltas)
ax1.set_yticks(range(len(tenors))); ax1.set_yticklabels(tenors)
fig.colorbar(im1, ax=ax1, label='IV')

im2 = ax2.imshow(Z_pred, origin='lower', aspect='auto')
ax2.set_title('Predicted IV surface')
ax2.set_xlabel('Delta')
ax2.set_xticks(range(len(deltas))); ax2.set_xticklabels(deltas)
fig.colorbar(im2, ax=ax2, label='IV')

plt.suptitle('Figure 8Actual vs. Predicted IV Surface')
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import LinearNDInterpolator
# func bs price


def bs_price(call_put_flag, S, K, T, r, q, sigma):
   """
   call_put_flag: 'C' or 'P'
   S: spot index level
   K: strike
   T: time to maturity in years
   r: riskfree rate (annual)
   q: dividend yield (annual)
   sigma: implied vol
   """
   d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
   d2 = d1 - sigma*np.sqrt(T)
   if call_put_flag.upper()=='C':
         return np.exp(-q*T)*S*norm.cdf(d1) - np.exp(-r*T)*K*norm.cdf(d2)
   else:
         return np.exp(-r*T)*K*norm.cdf(-d2) - np.exp(-q*T)*S*norm.cdf(-d1)



pred_iv = model.predict(X_all)
dates = df_feat.index[:-1]


tenors = np.array(sorted({int(c.split('_')[0]) for c in df_feat.columns[:n_grid ]}))
deltas = np.array(sorted({int(c.split('_')[1]) for c in df_feat.columns[:n_grid ]}))




# func make iv interpolator
points = np.array([[t/252, d] for t in tenors for d in deltas])

def make_iv_interpolator(iv_vector):
   """Given flat n_grid IVs, return a function f(T_days, delta) -> sigma."""
   values = iv_vector
   interp = LinearNDInterpolator(points, values, fill_value=np.nan)
   return lambda T, d: interp(T/252, d)


f_iv = make_iv_interpolator(pred_iv[idx])







import seaborn as sns


example_date = pd.to_datetime('2020-01-01')
i0 = dates.get_loc(example_date)


iv_act0 = iv_actual[i0]
iv_pred0 = pred_iv[i0]


tenors = sorted({t for t,d in pos})
deltas = sorted({d for t,d in pos})


Z_act = np.zeros((len(tenors), len(deltas)))
Z_pred = np.zeros_like(Z_act)
for (t,d), idx in idx_map.items():
   r = tenors .index(t)
   c = deltas .index(d)
   Z_act [r,c] = iv_act0 [idx]
   Z_pred[r,c] = iv_pred0[idx]


vmin, vmax = Z_act.min(), Z_act.max()


fig, (ax1,ax2) = plt.subplots(1,2, figsize=(14,5), sharey=True)
sns.heatmap(Z_act, xticklabels=deltas, yticklabels=tenors,
          vmin=vmin, vmax=vmax, cmap='viridis', ax=ax1)
ax1.set_title(f'Actual IV surface\n{example_date.date()}')
ax1.set_xlabel('Delta'); ax1.set_ylabel('Tenor (days)')

sns.heatmap(Z_pred, xticklabels=deltas, yticklabels=False,
             vmin=vmin, vmax=vmax, cmap='viridis', ax=ax2)
ax2.set_title(f'Predicted IV surface\n{example_date.date()}')
ax2.set_xlabel('Delta')

plt.tight_layout()
plt.show()
# func qlike


def qlike(sig_true, sig_pred):
   return np.log(sig_pred**2/sig_true**2) + sig_true**2/sig_pred**2 - 1

cut = pd.to_datetime('2013-02-01')
mask = df.index <= cut

mse_in = np.mean((pred_iv[mask] - iv_actual[mask])**2)
mae_in = np.mean(np.abs(pred_iv[mask] - iv_actual[mask]))
qlike_in = np.mean(qlike(iv_actual[mask], pred_iv[mask]))

mse_out = np.mean((pred_iv[~mask] - iv_actual[~mask])**2)
mae_out = np.mean(np.abs(pred_iv[~mask] - iv_actual[~mask]))
qlike_out = np.mean(qlike(iv_actual[~mask], pred_iv[~mask]))

print("Forecast metrics:")
print(f"in-sample MSE={mse_in:.6f}, MAE={mae_in:.4f}, QLIKE={qlike_in:.4f}")
print(f"out-sample MSE={mse_out:.6f}, MAE={mae_out:.4f}, QLIKE={qlike_out:.4f}")





iv_act_df = pd.DataFrame(iv_actual, index=df.index, columns=cols)
iv_pred_df = pd.DataFrame(pred_iv, index=df.index, columns=cols)


cut = pd.to_datetime('2013-02-01')
in_mask = iv_act_df.index <= cut
out_mask = iv_act_df.index > cut

results_delta = []
for d in sorted({d for t,d in pos}):

   cols_d = [c for c in cols if int(c.split('_')[1])==d]

   mse_in_d = ((iv_pred_df.loc[in_mask, cols_d] - iv_act_df.loc[in_mask, cols_d ])**2).mean().mean()
   mae_in_d = ( np.abs(iv_pred_df.loc[in_mask, cols_d] - iv_act_df.loc[in_mask, cols_d]) ).mean().mean()
   ql_in_d = qlike(iv_act_df.loc[in_mask, cols_d], iv_pred_df.loc[in_mask, cols_d]).mean().mean()
   mse_out_d = ((iv_pred_df.loc[out_mask, cols_d] - iv_act_df.loc[out_mask, cols_d])**2).mean().mean()
   mae_out_d = ( np.abs(iv_pred_df.loc[out_mask, cols_d] - iv_act_df.loc[ out_mask, cols_d]) ).mean().mean()
   ql_out_d = qlike(iv_act_df.loc[out_mask,cols_d], iv_pred_df.loc[out_mask, cols_d]).mean().mean()
   results_delta.append({
        'delta': d,
        'MSE_in': mse_in_d, 'MAE_in': mae_in_d, 'QLIKE_in': ql_in_d,
        'MSE_out': mse_out_d, 'MAE_out': mae_out_d, 'QLIKE_out': ql_out_d
   })

df_delta = pd.DataFrame(results_delta).set_index('delta').sort_index()
print("Table 4: Error by Delta")
print(df_delta.round(6))




window = 252*5
step = 21

roll_results = []
for start in range(0, len(df.index) - window, step):
   idx0 = df.index[start]
   idx1 = df.index[start + window]
   mask_w = (df.index >= idx0) & (df.index < idx1)


   act_sub = iv_act_df.loc[mask_w].values
   pred_sub = iv_pred_df.loc[mask_w].values


   mse_w = np.mean((pred_sub - act_sub)**2)
   mae_w = np.mean(np.abs(pred_sub - act_sub))
   ql_w = np.mean(qlike(act_sub, pred_sub))

   roll_results.append({
        'window_start': idx0,
        'window_end': idx1,
        'MSE': mse_w,
        'MAE': mae_w,
        'QLIKE': ql_w
   })

df_roll = pd.DataFrame(roll_results).set_index('window_start')


plt.figure(figsize=(12,4))
plt.plot(df_roll.index, df_roll['MSE'], label='MSE')
plt.plot(df_roll.index, df_roll['MAE'], label='MAE')
plt.plot(df_roll.index, df_roll['QLIKE'], label='QLIKE')
plt.title('5-Year Rolling Forecast Errors')
plt.ylabel('Error')
plt.legend()
plt.show()


import pandas as pd





df = pd.read_csv('/content/drive/MyDrive/pred_iv_surface.csv')


pd.set_option('display.max_columns', None)



import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K


features_path = '/content/drive/MyDrive/bcmocccogeuhqddt_features_flat.csv'
df = pd.read_csv(features_path, index_col=0, parse_dates=True, low_memory=False)
print("Loaded features:", df.shape, "from", df.index.min(), "to", df.index.max() )


total_cols = df.shape[1]
n_grid = total_cols // 3

iv_daily = df.iloc[:, :n_grid].values
iv_weekly = df.iloc[:, n_grid:2*n_grid].values
iv_monthly = df.iloc[:, 2*n_grid:3*n_grid].values


X_all = np.stack([iv_daily, iv_weekly, iv_monthly], axis=1)
y_all = iv_daily[1:]
X_all = X_all[:-1]
dates = df.index[:-1]


cutoff = df.index.min() + pd.DateOffset(years=8)
train_mask = dates < cutoff
val_mask = dates >= cutoff

X_train, y_train = X_all[train_mask], y_all[train_mask]
X_val, y_val = X_all[val_mask], y_all[val_mask]

print("Train shapes:", X_train.shape, y_train.shape)
print("Val shapes: ", X_val.shape, y_val.shape)


inp = layers.Input(shape=(3, n_grid), name='input')


h1 = layers.LSTM(135, return_sequences=True, name='lstm1')(inp)
h1 = layers.Dropout(0.2, name='drop1')(h1)


score = layers.Dense(1, name='score')(h1)


attn_weights = layers.Softmax(axis=1, name='attn_weights')(score)


context = layers.Lambda(lambda x: K.sum(x[0] * x[1], axis=1),
                        name='context')([attn_weights, h1])


context_seq = layers.RepeatVector(3, name='context_seq')(context)


h2 = layers.LSTM(135, name='lstm2')(context_seq)
h2 = layers.Dropout(0.2, name='drop2')(h2)


out = layers.Dense(n_grid, name='output')(h2)

model = models.Model(inp, out)
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
model.summary()


ckpt_path = '/content/drive/MyDrive/iv_attlstm_ckpt.h5'
cp_cb = tf.keras.callbacks.ModelCheckpoint(
    ckpt_path, save_best_only=True, monitor='val_loss'
)


history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[cp_cb]
)


import pickle
with open('/content/drive/MyDrive/attlstm_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
model.save('/content/drive/MyDrive/iv_attlstm_final.h5')
print(" Training finished. Checkpoint, model, and history are on Drive.")


last_date = df.index[-1]
last_input = np.stack([iv_daily[-1:], iv_weekly[-1:], iv_monthly[-1:]], axis=0). reshape(1, 3, n_grid)

forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 21)]
forecasted_iv = []

for _ in range(20):
    next_pred = model.predict(last_input, verbose=0)
    forecasted_iv.append(next_pred[0])
    last_input = np.roll(last_input, -1, axis=1)
    last_input[0, -1] = next_pred

forecasted_iv = np.array(forecasted_iv)


forecast_df = pd.DataFrame(
    forecasted_iv,
    index=forecast_dates,
    columns=[str(i) for i in range(n_grid)]
)


forecast_df.to_csv('/content/drive/MyDrive/pred_iv_surface_t20.csv')
print(" 20-day forecast saved to /content/drive/MyDrive/pred_iv_surface_t20.csv" )

















import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K


features_path = '/content/drive/MyDrive/bcmocccogeuhqddt_features_flat.csv'
df = pd.read_csv(features_path, index_col=0, parse_dates=True, low_memory=False)
print("Loaded features:", df.shape, "from", df.index.min(), "to", df.index.max() )


total_cols = df.shape[1]
n_grid = total_cols // 3

iv_daily = df.iloc[:, :n_grid].values
iv_weekly = df.iloc[:, n_grid:2*n_grid].values
iv_monthly = df.iloc[:, 2*n_grid:3*n_grid].values


TIMESTEPS = 30
HORIZON = 20
WINDOW_YEARS = 3
EPOCHS = 50
BATCH_SIZE = 50
# func build model


def build_model(n_grid: int,
                 timesteps: int = TIMESTEPS,
                 horizon: int = HORIZON) -> tf.keras.Model:
   inp = layers.Input(shape=(timesteps, n_grid), name='input')


   h1 = layers.LSTM(135, return_sequences=True, name='lstm1')(inp)
   h1 = layers.Dropout(0.2, name='drop1')(h1)


   score = layers.Dense(1, name='score')(h1)
   attn_weights = layers.Softmax(axis=1, name='attn_weights')(score)
   context = layers.Lambda(lambda x: K.sum(x[0] * x[1], axis=1),
                                  name='context')([attn_weights, h1])


   context_seq = layers.RepeatVector(timesteps, name='context_seq')(context)
   h2 = layers.LSTM(135, name='lstm2')(context_seq)
   h2 = layers.Dropout(0.2, name='drop2')(h2)


   out_flat = layers.Dense(n_grid * horizon, name='output_flat')(h2)
   out = layers.Reshape((horizon, n_grid), name='output')(out_flat)

   model = models.Model(inp, out, name='iv_attlstm_rolling')
   model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
   return model
# func make xy


def make_xy(daily_arr: np.ndarray,
          start_idx: int,
          end_idx: int,
          timesteps: int = TIMESTEPS,
          horizon: int = HORIZON):
   """Build X (N, timesteps, n_grid) and y (N, horizon, n_grid)."""
   X, y = [], []
   for idx in range(start_idx, end_idx - timesteps - horizon + 1):
       X.append(daily_arr[idx: idx + timesteps])
       y.append(daily_arr[idx + timesteps: idx + timesteps + horizon])
   return np.array(X), np.array(y)


results = {}
all_history = {}
dates = df.index


first_train_end = (dates.min() + pd.DateOffset(years=WINDOW_YEARS)).to_period('M ').end_time

last_train_end = dates.max() - pd.DateOffset(days=TIMESTEPS + HORIZON)

current_end = first_train_end

while current_end <= last_train_end:

   train_start = current_end - pd.DateOffset(years=WINDOW_YEARS)
   train_mask = (dates >= train_start) & (dates < current_end)
   train_idx_start = np.searchsorted(dates, train_start, side='left')
   train_idx_end = np.searchsorted(dates, current_end, side='left')


   X_train, y_train = make_xy(iv_daily, train_idx_start, train_idx_end)

   print(f" {current_end.date()} | Train samples: {X_train.shape[0]}"
        f" ({train_start.date()}   {current_end.date()})")

   if X_train.shape[0] == 0:
       current_end += pd.DateOffset(months=1)
       continue


   tf.keras.backend.clear_session()
   model = build_model(n_grid)

   ckpt_path = f'/content/drive/MyDrive/iv_attlstm_ckpt_{current_end:%Y%m%d}. keras'
   hist_path = f'/content/drive/MyDrive/attlstm_history_{current_end:%Y%m%d}. pkl'
   model_path = f'/content/drive/MyDrive/iv_attlstm_final_{current_end:%Y%m%d}. keras'

  cp_cb = tf.keras.callbacks.ModelCheckpoint(
      ckpt_path, save_best_only=True, monitor='val_loss'
  )

  history = model.fit(
      X_train, y_train,
      validation_split=0.1,
      epochs=EPOCHS,
      batch_size=BATCH_SIZE,
      callbacks=[cp_cb],
      verbose=0
  )


  with open(hist_path, 'wb') as f:
      pickle.dump(history.history, f)
  model.save(model_path)




  last30_idx_end = np.searchsorted(dates, current_end, side='left')
  last30_idx_start = last30_idx_end - TIMESTEPS


  if last30_idx_start < 0:
      print(" Skipping", current_end.date(), " not enough prior rows for 30- day context")
      current_end = (current_end + pd.DateOffset(months=1)).to_period('M'). end_time
      continue

  X_last30 = iv_daily[last30_idx_start:last30_idx_end].reshape(1, TIMESTEPS, n_grid)

  preds = model.predict(X_last30, verbose=0)[0]


  pred_dates = dates[last30_idx_end: last30_idx_end + HORIZON]

  if pred_dates.size < HORIZON:
       preds = preds[:pred_dates.size]

   results[current_end] = pd.DataFrame(preds, index=pred_dates, columns=df. columns[:n_grid])

   all_history[current_end] = history.history
   print(f" Finished {current_end.date()} | Saved .keras model, history, predictions")


   current_end = (current_end + pd.DateOffset(months=1)).to_period('M'). end_time


preds_concat = pd.concat(results.values())
pd.concat(results.values())
preds_path = '/content/drive/MyDrive/iv_attlstm_rolling_preds.csv'
preds_concat.to_csv(preds_path)
print(" All monthly forecasts saved to", preds_path)






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob, pickle, re
from datetime import datetime as dt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


features_path = '/content/drive/MyDrive/bcmocccogeuhqddt_features_flat.csv'
preds_path = '/content/drive/MyDrive/iv_attlstm_rolling_preds.csv'
history_glob = '/content/drive/MyDrive/attlstm_history_*.pkl'
# func mape


def mape(y_true, y_pred):
    mask = y_true != 0
# func smape
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
# func pkl date
    return np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask])

def pkl_date(fname):
    m = re.search(r'(\\d{8})', Path(fname).stem)
    return dt.strptime(m.group(1), '%Y%m%d').date() if m else None


for _p in [features_path, preds_path]:
    if not Path(_p).exists():
       raise FileNotFoundError(f'{_p} not found    check path.')

df_feat = pd.read_csv(features_path, index_col=0, parse_dates=True)
df_preds = pd.read_csv(preds_path, index_col=0, parse_dates=True)



df_preds = (
    df_preds
     .sort_index()
     .groupby(level=0)
     .last()
)




n_grid = df_feat.shape[1] // 3
df_actual = df_feat.iloc[:, :n_grid]


df_actual = df_actual.groupby(level=0).last()


common_idx = df_preds.index.intersection(df_actual.index)
df_preds = df_preds.loc[common_idx]
df_actual = df_actual.loc[common_idx]

print(f'Aligned shapes   preds: {df_preds.shape} , actual: {df_actual.shape}')


y_pred = df_preds.values.ravel()
y_true = df_actual.values.ravel()

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mape_val = mape(y_true, y_pred)
smape_val = smape(y_true, y_pred)

metrics_global = pd.DataFrame(
   {"Metric": ["MSE", "RMSE", "MAE", "R2", "MAPE", "SMAPE", "Sample_Pts"],
    "Value" : [mse, rmse, mae, r2, mape_val, smape_val, y_true.size]}
).set_index("Metric")

print("\\n============= GLOBAL OOS METRICS =============")
print(metrics_global)
print("================================================\\n")


per_date_rmse = np.sqrt(((df_preds - df_actual) ** 2).mean(axis=1))


hist_files = sorted(glob.glob(history_glob))
records = []
for hf in hist_files:
   d = pkl_date(hf)
   with open(hf, 'rb') as f:
       h = pickle.load(f)
   be = int(np.argmin(h['val_loss']))
   records.append({
       "Train_End": d,
       "Train_RMSE": np.sqrt(h['loss'][be]),
       "Val_RMSE": np.sqrt(h['val_loss'][be]),
        "Train_MAE": h.get('mae', [np.nan])[be],
        "Val_MAE": h.get('val_mae', [np.nan])[be]
   })
df_tv = pd.DataFrame(records).set_index('Train_End').sort_index()

print("----------- PER-MONTH TRAIN / VAL BEST-EPOCH -----------")
print(df_tv.head())
print("... ({} rows total)".format(len(df_tv)))
print("---------------------------------------------------------\\n")

print("Average Train RMSE :", df_tv['Train_RMSE'].mean())
print("Average Val RMSE :", df_tv['Val_RMSE'].mean())
print("Average Train MAE :", df_tv['Train_MAE'].mean())
print("Average Val MAE :", df_tv['Val_MAE'].mean())


plt.figure(figsize=(10, 4))
per_date_rmse.plot()
plt.title('Daily RMSE of IV-Surface Forecasts')
plt.ylabel('RMSE')
plt.xlabel('Date')
plt.tight_layout()
plt.show()












import numpy as np
import pandas as pd
from math import log, sqrt, exp, pi
from pathlib import Path
from scipy.stats import norm


root = '/content/drive/MyDrive'
iv_path = f'{root}/bcmocccogeuhqddt_features_flat.csv'
spot_path = f'{root}/kg4snxuhemmtl4nx.csv'
vega_out = f'{root}/vega_grid.csv'


df_iv = pd.read_csv(iv_path, index_col=0, parse_dates=True)
df_spot = (pd.read_csv(spot_path, usecols=['DlyCalDt', 'DlyPrcInd'],
                      index_col='DlyCalDt', parse_dates=True)
            .rename(columns={'DlyPrcInd': 'S'}))


n_grid = df_iv.shape[1] // 3
df_iv = df_iv.iloc[:, :n_grid]


df_spot = df_spot.reindex(df_iv.index).ffill()
# func parse header


def parse_header(colname):
    tenor_str, delta_str = colname.split('_')
    return int(tenor_str), float(delta_str) / 100.0

col_meta = pd.DataFrame(
    [parse_header(c) for c in df_iv.columns],
    columns=['T_days', 'Delta'],
    index=df_iv.columns
)
# func inv delta to strike


def inv_delta_to_strike(S, delta, T, sigma):
    """
    Invert Black-Scholes delta to strike K.
    Uses r = q = 0, sigma in decimals, T in years.
    Vectorised on S & sigma (pandas Series).
    """

    call_mask = delta > 0
   put_mask = delta < 0


   d1 = np.empty_like(delta, dtype=float)
   d1[call_mask] = norm.ppf(delta[call_mask])
   d1[put_mask] = -norm.ppf(-delta[put_mask])


   sqrtT = np.sqrt(T)
   numer = S * np.exp( (0 + 0.5 * sigma**2) * T )
   denom = np.exp(d1 * sigma * sqrtT)
# func bs vega
   return numer / denom

def bs_vega(S, K, T, sigma):
   """Black-Scholes vega per 1-vol-point (sigma in decimals)."""
   d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
   return S * np.sqrt(T) * np.exp(-0.5 * d1**2) / np.sqrt(2 * pi)


vega_grid = pd.DataFrame(index=df_iv.index, columns=df_iv.columns, dtype=float)

for col in df_iv.columns:
   T_days = col_meta.loc[col, 'T_days']
   delta = col_meta.loc[col, 'Delta']
   T_yrs = T_days / 365.0

   S_series = df_spot['S']
   sigma = df_iv[col]


   K_series = inv_delta_to_strike(S_series.values,
                               np.full_like(S_series.values, delta),
                               T_yrs,
                               sigma.values)

   vega_grid[col] = bs_vega(S_series.values, K_series, T_yrs, sigma.values)


vega_grid.to_csv(vega_out)
print(f" vega_grid.csv written: {vega_grid.shape}    {vega_out}")







"""-----------------"""



















import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from datetime import datetime


try:
   tf.config.optimizer.set_jit(True)
except Exception:
   pass

try:
   from tensorflow.keras import mixed_precision
   mixed_precision.set_global_policy("mixed_float16")
   MIXED = True
except Exception:
   MIXED = False


ROOT = "/content/drive/MyDrive"
FEATURES_FP = f"{ROOT}/bcmocccogeuhqddt_features_flat.csv"
OUT_WIDE = f"{ROOT}/iv_convlstm_rolling_preds.csv"
OUT_LONG = f"{ROOT}/conv_lstm_20d_preds.csv"
OUT_METRICS = f"{ROOT}/iv_convlstm_metrics.csv"
OUT_DATES = f"{ROOT}/iv_dates.npy"


TIMESTEPS = 30
HORIZON = 20
WINDOW_YEARS = 3
RETRAIN_FREQ = "M"
EPOCHS = 20
BATCH_SIZE = 50
VAL_SPLIT = 0.10
FILTERS = 16
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


df = pd.read_csv(FEATURES_FP, index_col=0, parse_dates=True, low_memory=False)
print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded: {df.shape} | {df.index. min().date()}     {df.index.max().date()}")


n_grid = df.shape[1] // 3
grid_cols = df.columns[:n_grid]
# func parse col


def parse_col(c):
   m, d = str(c).split("_"); return int(m), int(d)

pairs = [parse_col(c) for c in grid_cols]
maturities = sorted({m for m, _ in pairs})
deltas = sorted({d for _, d in pairs})

grid_order = []
for d in deltas:
   row = []
   for m in maturities:
       name = f"{m}_{d:+d}".replace("+","")
       if name not in df.columns:
          raise RuntimeError(f"Missing grid column: {name}")
       row.append(name)
   grid_order.append(row)

rows = len(deltas)
cols = len(maturities)

flat = df[[c for r in grid_order for c in r]].to_numpy(np.float32)
T = flat.shape[0]
surf = flat.reshape(T, rows, cols)
dates = df.index

print(f"[{datetime.now().strftime('%H:%M:%S')}] Grid rows={rows}, cols={cols}, n_grid={n_grid}, T={T}")
# func make xy


def make_xy(Zg, start_idx, end_idx, timesteps=TIMESTEPS, horizon=HORIZON):
   Xs, Ys = [], []
   lim = end_idx - timesteps - horizon + 1
   for t in range(start_idx, max(start_idx, lim)):
       Xs.append(Zg[t: t+timesteps])
       Ys.append(Zg[t+timesteps: t+timesteps+horizon])
   if not Xs:
       return (np.zeros((0, timesteps, rows, cols, 1), np.float32),
                 np.zeros((0, horizon, rows, cols, 1), np.float32))
   X = np.expand_dims(np.stack(Xs), -1)
   Y = np.expand_dims(np.stack(Ys), -1)
# func build convlstm
   return X, Y

def build_convlstm(rows, cols, horizon=HORIZON, filters=FILTERS):
   inp = layers.Input(shape=(TIMESTEPS, rows, cols, 1))
   x = layers.ConvLSTM2D(filters, (3,3), padding="same", return_sequences=True, activation="tanh")(inp)
   x = layers.BatchNormalization()(x)
   x = layers.ConvLSTM2D(filters, (3,3), padding="same", return_sequences=False , activation="tanh")(x)
   x = layers.BatchNormalization()(x)
   x_tiled = layers.Lambda(lambda t: tf.tile(tf.expand_dims(t, 1), [1, horizon, 1, 1, 1]))(x)
   y = layers.TimeDistributed(layers.Conv2D(filters, (3,3), padding="same", activation="tanh"))(x_tiled)
   y = layers.TimeDistributed(layers.BatchNormalization())(y)
   out = layers.TimeDistributed(layers.Conv2D(1, (1,1), padding="same", activation=None))(y)
   if MIXED:
       out = layers.Activation('linear', dtype='float32', name='fp32_head')(out )
   model = models.Model(inp, out, name="iv_convlstm_rolling_full")
   model.compile(optimizer='rmsprop', loss='mse',
                  metrics=['mae','mape','accuracy'])
   return model


first_train_end = (dates.min() + pd.DateOffset(years=WINDOW_YEARS)).to_period('M ').end_time
last_train_end = (dates.max() - pd.Timedelta(days=HORIZON)).to_period('M'). end_time
freq = 'M' if RETRAIN_FREQ.upper().startswith('M') else 'Q'
periods = pd.period_range(first_train_end, last_train_end, freq=freq). to_timestamp('M')
print(f"[{datetime.now().strftime('%H:%M:%S')}] Retrain cadence: {freq} | cycles : {len(periods)} "
     f"({periods[0].date()}    {periods[-1].date()})")


results_wide = {}
metrics_log = []
long_records = []
origin_dates = []
sample_idx = 0
model = None


mat_to_bucket = {m:i for i,m in enumerate(maturities)}

if len(deltas) == 23 and min(deltas) <= -55 and max(deltas) >= 55 and all(d%5==0 for d in deltas):
   delta_to_bucket = {d:int(d/5 + 11) for d in deltas}
else:
   delta_to_bucket = {d:i for i,d in enumerate(deltas)}

for k, train_end in enumerate(periods, start=1):
   train_start = train_end - pd.DateOffset(years=WINDOW_YEARS)
   s_idx = np.searchsorted(dates, train_start, side='left')
   e_idx = np.searchsorted(dates, train_end, side='right')

   X_tr, Y_tr = make_xy(surf, s_idx, e_idx)
   if X_tr.shape[0] == 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {k}/{len(periods)} { train_end.date()}   skip (no samples)")
        continue

   if model is None:
        model = build_convlstm(rows, cols)

   es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
   rl = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0)

   print(f"[{datetime.now().strftime('%H:%M:%S')}] {k}/{len(periods)} "
         f"train_end={train_end.date()} window={dates[s_idx].date()}{dates[e_idx -1].date()} "
         f"samples={X_tr.shape[0]}")

   hist = model.fit(
        X_tr, Y_tr,
      validation_split=VAL_SPLIT,
      epochs=EPOCHS,
      batch_size=BATCH_SIZE,
      callbacks=[es, rl],
      shuffle=False,
      verbose=0
  )


  for ep in range(len(hist.history['loss'])):
      row = {'train_end': train_end.date(), 'epoch': ep+1}
      for key, vals in hist.history.items():
         row[key] = vals[ep]
      metrics_log.append(row)


  end_pos = np.searchsorted(dates, train_end, side='left')
  start_pos = end_pos - TIMESTEPS
  if start_pos < 0:
      print(f"[{datetime.now().strftime('%H:%M:%S')}] {k}/{len(periods)} forecast skipped (insufficient lookback)")
      continue

  X_last = surf[start_pos:end_pos].reshape(1, TIMESTEPS, rows, cols, 1)
  pred = model.predict(X_last, verbose=0)[0]
  pred = pred[..., 0]
  pred_flat = pred.reshape(HORIZON, rows*cols)
  f_dates = dates[end_pos : end_pos + HORIZON]
  if f_dates.size == 0:
      break


  block = pd.DataFrame(pred_flat[:len(f_dates)],
                       index=f_dates,
                       columns=[c for r in grid_order for c in r])
  results_wide[train_end] = block.loc[:, grid_cols]



  origin_dates.append(np.datetime64(train_end.normalize().date()))
  steps = min(HORIZON, len(f_dates))
   for h in range(1, steps+1):
       dt = f_dates[h-1]

       for ri, d in enumerate(deltas):
          d_bucket = delta_to_bucket[d]
          for cj, m in enumerate(maturities):
              m_bucket = mat_to_bucket[m]
              val = float(pred[h-1, ri, cj])
              long_records.append({
                   "sample_idx": sample_idx,
                   "step_ahead": h,
                   "delta_bucket": d_bucket,
                   "maturity_bucket": m_bucket,
                   "iv_pred": val,
                   "iv_pred_raw": val,
                   "origin_date": pd.Timestamp(train_end).date(),
                   "target_date": pd.Timestamp(dt).date()
              })
   sample_idx += 1


if not results_wide:
   raise RuntimeError("No predictions produced.")


preds_concat = pd.concat(results_wide.values()).sort_index()
preds_concat = preds_concat.groupby(level=0).last()
preds_concat.to_csv(OUT_WIDE)


long_df = pd.DataFrame(long_records)
long_df.to_csv(OUT_LONG, index=False)


pd.DataFrame(metrics_log).to_csv(OUT_METRICS, index=False)


np.save(OUT_DATES, np.array(origin_dates, dtype="datetime64[D]"))

print(f"[{datetime.now().strftime('%H:%M:%S')}]   Wide   {OUT_WIDE}")
print(f"[{datetime.now().strftime('%H:%M:%S')}]   Long   {OUT_LONG}")
print(f"[{datetime.now().strftime('%H:%M:%S')}]   Metrics   {OUT_METRICS}")
print(f"[{datetime.now().strftime('%H:%M:%S')}]   Dates   {OUT_DATES}")

"""CALENDAR STRADDLE - CONVLSTM"""















import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from math import sqrt
from datetime import datetime


DECIMALS = 6
np.set_printoptions(suppress=True,
                 formatter={'float_kind': lambda x: f"{x:.{DECIMALS}f}"})
pd.options.display.float_format = lambda x: f"{x:.{DECIMALS}f}"
CSV_FLOAT_FMT = f"%.{DECIMALS}f"
FMT = lambda x: f"{x:.{DECIMALS}f}"


ROOT = "/content/drive/MyDrive"
IV_CSV = f"{ROOT}/bcmocccogeuhqddt_features_flat.csv"
PRED_WIDE = f"{ROOT}/iv_convlstm_rolling_preds.csv"
PRED_LONG = f"{ROOT}/conv_lstm_20d_preds.csv"
METRICS_LOG = f"{ROOT}/iv_convlstm_metrics.csv"
DATES_NPY = f"{ROOT}/iv_dates.npy"
OUT_DIR = Path(ROOT)






HEATMAP_DATE = None
# func parse grid


def parse_grid(cols):
   def _split(c):
       m, d = str(c).split('_'); return int(m), int(d)
   pairs = [_split(c) for c in cols]
   mats = sorted({m for m,_ in pairs})
   dels = sorted({d for _,d in pairs})
   grid = [[f"{m}_{d:+d}".replace("+","") for m in mats] for d in dels]
# func wide from long
   return mats, dels, grid

def wide_from_long(long_path, actual_df):
   raw = pd.read_csv(long_path)
   if 'step_ahead' not in raw.columns:
       raise RuntimeError("Long preds need 'step_ahead'. Re-export.")
   STEP_AHEAD = int(raw['step_ahead'].mode().iloc[0])
   raw = raw[raw['step_ahead']==STEP_AHEAD].copy()

   if 'origin_date' in raw.columns:
       raw['date'] = pd.to_datetime(raw['origin_date'])
   elif Path(DATES_NPY).exists():
       cal = np.load(DATES_NPY).astype('datetime64[D]')
       raw['date'] = pd.to_datetime(cal[raw['sample_idx'].to_numpy()])
   else:
       cal = actual_df.index.to_numpy(dtype="datetime64[D]")
       raw['date'] = pd.to_datetime(cal[raw['sample_idx'].clip(0,len(cal)-1)])

   ivcol = 'iv_pred_raw' if 'iv_pred_raw' in raw.columns else 'iv_pred'
   if ivcol not in raw.columns:
       raise RuntimeError("Long preds need iv_pred or iv_pred_raw.")

   mats_sorted = sorted({int(c.split('_')[0]) for c in actual_df.columns})
   dels_sorted = sorted({int(c.split('_')[1]) for c in actual_df.columns})

   def infer_map(series, universe_sorted):
       s = series.astype(int)
       k = s.max() + 1
       if k == len(universe_sorted) and set(s.unique()) <= set(range(k)):
          return {i: universe_sorted[i] for i in range(k)}
       if s.nunique() == 23 and set(s.unique()) <= set(range(23)):
          return {i: (i-11)*5 for i in range(23)}
       if s.nunique() == 3 and set(s.unique()) <= {0,1,2}:
          return {0:-10,1:0,2:10}
       uniq = sorted(s.unique())
       return {b: universe_sorted[i % len(universe_sorted)] for i,b in enumerate(uniq)}

   mm = infer_map(raw['maturity_bucket'], mats_sorted)
   dm = infer_map(raw['delta_bucket'], dels_sorted)

   def mkcol(r):
       m = mm[int(r.maturity_bucket)]; d = dm[int(r.delta_bucket)]
       return f"{m}_{d:+d}".replace("+","")
   raw['grid_col'] = raw.apply(mkcol, axis=1)

   raw.sort_values(['date','maturity_bucket','delta_bucket'], inplace=True)
   raw = raw.drop_duplicates(subset=['date','grid_col'], keep='last')
   wide = (raw.pivot(index='date', columns='grid_col', values=ivcol)
            .sort_index())
# func align wide pred actual
   return wide

def align_wide_pred_actual(pred_wide, actual_df):
   pred_cols = [c[:-2] if c.endswith('_f') else c for c in pred_wide.columns]
   pred_wide = pred_wide.copy()
   pred_wide.columns = pred_cols

   keep_cols = [c for c in actual_df.columns if c in pred_wide.columns]
   if not keep_cols:
       raise RuntimeError("No overlapping columns between preds and actual.")
   common_idx = actual_df.index.intersection(pred_wide.index)
   A = actual_df.loc[common_idx, keep_cols].astype(np.float32)
   P = pred_wide.loc[common_idx, keep_cols].astype(np.float32)
# func r2 score
   return common_idx, keep_cols, P, A

def r2_score(y, yhat):
   sse = np.square(y - yhat).sum()
   sst = np.square(y - y.mean()).sum()
# func mape
   return 1.0 - sse/max(sst, 1e-12)

def mape(y, yhat, eps=1e-8):
   mask = np.abs(y) > eps
# func smape
   return np.abs((yhat[mask]-y[mask]) / (y[mask]+np.sign(y[mask])*eps)).mean()

def smape(y, yhat, eps=1e-8):
# func qlike
   return (2.0*np.abs(yhat - y) / (np.abs(yhat)+np.abs(y)+eps)).mean()

def qlike(y, yhat, eps=1e-10):
   y = np.maximum(y, eps)
   yhat = np.maximum(yhat, eps)
   ratio = y / yhat
   return np.mean(ratio - np.log(ratio) - 1.0)


if not Path(IV_CSV).exists():
   raise FileNotFoundError(IV_CSV)

actual_full = (pd.read_csv(IV_CSV, parse_dates=['date'])
               .set_index('date')
               .sort_index())
n_grid = actual_full.shape[1]//3
actual = actual_full.iloc[:, :n_grid]
mats, dels, grid_order = parse_grid(actual.columns)

if Path(PRED_WIDE).exists():
   pred_wide = (pd.read_csv(PRED_WIDE, parse_dates=['date'])
                 .set_index('date')
                 .sort_index())
else:
   if not Path(PRED_LONG).exists():
       raise FileNotFoundError("Missing both wide and long ConvLSTM prediction files.")
    pred_wide = wide_from_long(PRED_LONG, actual)


idx, cols, P, A = align_wide_pred_actual(pred_wide, actual)
Yt = A.to_numpy()
Yp = P.to_numpy()

N, D = Yt.shape
print(f"Aligned shapes    preds: ({N}, {D}) , actual: ({N}, {D})")

err = Yp - Yt
mse = float(np.mean(err**2))
rmse = float(sqrt(mse))
mae = float(np.mean(np.abs(err)))
r2 = float(r2_score(Yt, Yp))
mape_v = float(mape(Yt, Yp))
smape_v = float(smape(Yt, Yp))
qlike_v = float(qlike(Yt, Yp))

global_tbl = pd.DataFrame(
    {"Value":[mse, rmse, mae, r2, mape_v, smape_v, qlike_v, N*D]},
    index=["MSE","RMSE","MAE","R2","MAPE","SMAPE","QLIKE","Sample_Pts"]
)

print("\n============ GLOBAL OOS METRICS ============\n")
print(global_tbl.to_string(float_format=FMT))


OUT_DIR.mkdir(parents=True, exist_ok=True)
global_tbl.to_csv(OUT_DIR/"convlstm_report_global_metrics.csv",
                  float_format=CSV_FLOAT_FMT)


daily_rmse = pd.Series(
    np.sqrt(np.mean((Yp - Yt)**2, axis=1)),
    index=idx, name="RMSE"
)
daily_rmse.to_csv(OUT_DIR/"convlstm_report_daily_rmse.csv",
                  float_format=CSV_FLOAT_FMT)

plt.figure(figsize=(14,3.8))
plt.plot(daily_rmse.index, daily_rmse.values)
plt.ylabel("RMSE"); plt.title("Daily RMSE of IV-Surface Forecasts")
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig(OUT_DIR/"convlstm_daily_rmse.png", dpi=160)
plt.show()



if HEATMAP_DATE is None:
   pick_date = idx[len(idx)//2]
elif isinstance(HEATMAP_DATE, str) and HEATMAP_DATE.lower() in {"best","worst"}:
   pick_date = daily_rmse.idxmin() if HEATMAP_DATE.lower()=="best" else daily_rmse.idxmax()
else:
   pick_date = pd.to_datetime(HEATMAP_DATE)


if pick_date not in idx:
   nearest_pos = idx.get_indexer([pick_date], method='nearest')[0]
# func row to surface
   pick_date = idx[nearest_pos]

def row_to_surface(row_vals):
   return row_vals.to_numpy().reshape(len(dels), len(mats))

a_row = actual.loc[pick_date, cols]
p_row = P.loc[pick_date, cols]
A_surf = row_to_surface(a_row)
P_surf = row_to_surface(p_row)

fig, axes = plt.subplots(1,2, figsize=(12,4.2), constrained_layout=True)
im0 = axes[0].imshow(A_surf, aspect='auto', origin='lower',
                     extent=[min(mats),max(mats),min(dels),max(dels)])
axes[0].set_title("Actual IV surface\n" + str(pd.to_datetime(pick_date).date()))
axes[0].set_xlabel("Tenor (days)"); axes[0].set_ylabel("Delta")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(P_surf, aspect='auto', origin='lower',
                     extent=[min(mats),max(mats),min(dels),max(dels)])
axes[1].set_title("Predicted IV surface\n" + str(pd.to_datetime(pick_date).date ()))
axes[1].set_xlabel("Tenor (days)")
fig.colorbar(im1, ax=axes[1])

mse_d = float(np.mean((P_surf - A_surf)**2))
mae_d = float(np.mean(np.abs(P_surf - A_surf)))
qlike_d = float(qlike(A_surf, P_surf))
caption = (f"Forecast metrics:\n"
           f"in-sample MSEn/a (rolling)\n"
           f"out-sample MSE={FMT(mse_d)}, MAE={FMT(mae_d)}, QLIKE={FMT(qlike_d)}" )
fig.text(0.02, -0.02, caption, fontsize=9)
plt.savefig(OUT_DIR/"convlstm_heatmaps_actual_vs_pred.png", dpi=160, bbox_inches ="tight")
plt.show()


pm_best = None
if Path(METRICS_LOG).exists():
   m = pd.read_csv(METRICS_LOG)
   m.columns = [c.lower() for c in m.columns]
   if not {'train_end','epoch','loss','val_loss'}.issubset(m.columns):
       print("\n(Training metrics log found but missing columns; skipping best- epoch table.)")
   else:
       best_idx = m.groupby('train_end')['val_loss'].idxmin()
       best = m.loc[best_idx, ['train_end','epoch','loss','val_loss','mae',' val_mae']]
       best['train_rmse'] = np.sqrt(best['loss'])
       best['val_rmse'] = np.sqrt(best['val_loss'])
       best = best.drop(columns=['loss','val_loss']).sort_values('train_end')
       pm_best = best.rename(columns={'mae':'train_mae','val_mae':'val_mae'})

       print("\n----------- PER-MONTH TRAIN / VAL BEST-EPOCH -----------")
       print(pm_best.head().to_string(float_format=FMT))
       pm_best.to_csv(OUT_DIR/"convlstm_report_per_month_best.csv",
                      index=False, float_format=CSV_FLOAT_FMT)


         agg = (m.groupby('epoch')[['loss','val_loss']]
                      .mean()
                      .assign(train_rmse=lambda d: np.sqrt(d['loss']),
                                val_rmse=lambda d: np.sqrt(d['val_loss'])))
         plt.figure(figsize=(8.5,4))
         plt.plot(agg.index, agg['train_rmse'], label='ConvLSTM Train')
         plt.plot(agg.index, agg['val_rmse'], label='ConvLSTM Val', linestyle='-- ')
         plt.xlabel("Epoch"); plt.ylabel("RMSE"); plt.title("Mean Train/Val Loss (across retrains)")
         plt.grid(alpha=.3); plt.legend()
         plt.tight_layout(); plt.savefig(OUT_DIR/"convlstm_mean_loss_curves.png", dpi=160)
         plt.show()

         print("----------------------------------------------------------\n")
         print("Average Train RMSE :", FMT(agg['train_rmse'].iloc[-1]))
         print("Average Val RMSE :", FMT(agg['val_rmse'].iloc[-1]))
         print("Average Train MAE :", FMT(m.groupby('epoch')['mae'].mean().iloc [-1]
                                                if 'mae' in m.columns else np.nan))
         print("Average Val MAE :", FMT(m.groupby('epoch')['val_mae'].mean().iloc [-1]
                                                if 'val_mae' in m.columns else np.nan))
else:
   print("\n(No training metrics log found            skipped per-month best-epoch and loss curves.)")

print(f"\nSaved to:\n           {OUT_DIR/'convlstm_report_global_metrics.csv'}"
        f"\n    {OUT_DIR/'convlstm_report_daily_rmse.csv'}"
        f"\n    {OUT_DIR/'convlstm_report_per_month_best.csv' if pm_best is not None else '(no per-month file)'}"
        f"\n    {OUT_DIR/'convlstm_daily_rmse.png'}"
        f"\n    {OUT_DIR/'convlstm_heatmaps_actual_vs_pred.png'}"
        f"\n    {OUT_DIR/'convlstm_mean_loss_curves.png'}")









import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from math import log
from datetime import datetime


ROOT = "/content/drive/MyDrive"
SPOT_CSV = f"{ROOT}/kg4snxuhemmtl4nx.csv"
IV_REAL_CSV = f"{ROOT}/bcmocccogeuhqddt_features_flat.csv"
CONV_WIDE_CSV = f"{ROOT}/iv_convlstm_rolling_preds.csv"
CONV_LONG_CSV = f"{ROOT}/conv_lstm_20d_preds.csv"
IV_MU_NPY = f"{ROOT}/iv_mu.npy"
IV_STD_NPY = f"{ROOT}/iv_std.npy"
DATES_NPY = f"{ROOT}/iv_dates.npy"


HORIZ = 20
FRONT_T = 30
BACK_T = 60
DELTA_LIST = list(range(-90, 100, 10))
SPLIT_DATE = "2004-01-01"
HEATMAP_DATE = "2020-01-01"
ROLL_WIN_DAYS = 252*5
SEED = 42

# func col
np.random.seed(SEED)

def col(t, d):
   return f"{t}_{d:+d}".replace("+", "")


hdr = pd.read_csv(IV_REAL_CSV, nrows=0)
n_grid = hdr.shape[1] // 3
iv_real = (pd.read_csv(IV_REAL_CSV, parse_dates=["date"])
              .set_index("date")
              .iloc[:, :n_grid])


keep_cols = [col(FRONT_T, d) for d in DELTA_LIST] + [col(BACK_T, d) for d in DELTA_LIST]
keep_cols = [c for c in keep_cols if c in iv_real.columns]
iv_real = iv_real[keep_cols].sort_index()


pred_wide_path = Path(CONV_WIDE_CSV)
pred_long_path = Path(CONV_LONG_CSV)

if pred_wide_path.exists():

   conv_pred = (pd.read_csv(CONV_WIDE_CSV, parse_dates=["date"])
                 .set_index("date"))

   conv_pred = conv_pred[[c for c in iv_real.columns if c in conv_pred.columns ]].sort_index()


   common = iv_real.index.intersection(conv_pred.index)
   iv_real = iv_real.loc[common]
   conv_pred = conv_pred.loc[common]






else:
   if not pred_long_path.exists():
        raise FileNotFoundError("No ConvLSTM predictions found. Provide CONV_WIDE_CSV or CONV_LONG_CSV.")


   long_df = pd.read_csv(CONV_LONG_CSV)

   if "iv_pred_raw" in long_df.columns:
        long_df["iv_use"] = long_df["iv_pred_raw"]
   else:

        if not (Path(IV_MU_NPY).exists() and Path(IV_STD_NPY).exists()):
          raise RuntimeError("conv_lstm_20d_preds.csv seems z-scored; iv_mu.npy & iv_std.npy required.")
      mu = np.load(IV_MU_NPY)
      std = np.load(IV_STD_NPY)
      def de_z(r):
          db = int(r["delta_bucket"]); mb = int(r["maturity_bucket"])
          return r["iv_pred"] * std[db, mb] + mu[db, mb]
      long_df["iv_use"] = long_df.apply(de_z, axis=1)



  maturity_map = {0: 30, 1: 60, 2: 91}
  to_delta = lambda b: (int(b) - 9) * 10
  long_df["grid_col"] = long_df.apply(
      lambda r: col(maturity_map.get(int(r["maturity_bucket"]), 30), to_delta( r["delta_bucket"])),
      axis=1
  )


  if Path(DATES_NPY).exists():
      cal = np.load(DATES_NPY).astype("datetime64[D]")
  else:
      cal = iv_real.index.to_numpy(dtype="datetime64[D]")

  n_windows = int(long_df["sample_idx"].max()) + 1
  base = 60
  long_df["origin_idx"] = base + long_df["sample_idx"].astype(int)
  long_df = long_df[(long_df["origin_idx"] >= 0) & (long_df["origin_idx"] < len(cal))]
  long_df["origin_date"] = cal[long_df["origin_idx"].to_numpy()]
  long_df["eval_date"] = long_df["origin_date"] + np.timedelta64(HORIZ, "D")


  long_tH = long_df[long_df["step_ahead"].astype(int) == HORIZ]


  conv_pred = (long_tH
               .pivot_table(index="eval_date", columns="grid_col", values=" iv_use", aggfunc="mean")
               .sort_index())
   conv_pred.index.name = "date"


   conv_pred = conv_pred[[c for c in iv_real.columns if c in conv_pred.columns ]]
   common = iv_real.index.intersection(conv_pred.index)
   iv_real = iv_real.loc[common]
   conv_pred = conv_pred.loc[common]
# func qlike


def qlike(a, f):

   y = np.clip(a, 1e-6, None)**2
   g = np.clip(f, 1e-6, None)**2
   r = np.clip(y/g, 1e-12, 1e12)
# func mse
   return r - np.log(r) - 1.0
# func mae

# func rmse
def mse(a, f): return np.nanmean((a - f)**2)
def mae(a, f): return np.nanmean(np.abs(a - f))
# func r2
def rmse(a, f): return np.sqrt(mse(a, f))

def r2(a, f):
   a = a.astype(float); f = f.astype(float)
   ss_res = np.nansum((a - f)**2)
   ss_tot = np.nansum((a - np.nanmean(a))**2)
   return 1 - ss_res/ss_tot if ss_tot > 0 else np.nan


split_dt = pd.Timestamp(SPLIT_DATE)
iv_in, pred_in = iv_real[iv_real.index < split_dt], conv_pred[conv_pred.index < split_dt]
iv_out, pred_out = iv_real[iv_real.index >= split_dt], conv_pred[conv_pred.index >= split_dt]


rows = []
for d in DELTA_LIST:
   for t in [FRONT_T, BACK_T]:
       c = col(t, d)
       if c not in iv_real.columns:
          continue

       if c in iv_in.columns and c in pred_in.columns and len(iv_in) > 0:
          mse_in = mse(iv_in[c].values, pred_in[c].values)
          mae_in = mae(iv_in[c].values, pred_in[c].values)
          qlk_in = np.nanmean(qlike(iv_in[c].values, pred_in[c].values))
       else:
          mse_in = mae_in = qlk_in = np.nan

       if c in iv_out.columns and c in pred_out.columns and len(iv_out) > 0:
          mse_out = mse(iv_out[c].values, pred_out[c].values)
          mae_out = mae(iv_out[c].values, pred_out[c].values)
          qlk_out = np.nanmean(qlike(iv_out[c].values, pred_out[c].values))
       else:
          mse_out = mae_out = qlk_out = np.nan

       rows.append([d, t, mse_in, mae_in, qlk_in, mse_out, mae_out, qlk_out])

tbl_delta = pd.DataFrame(rows, columns=[
   "delta","tenor","MSE_in","MAE_in","QLIKE_in","MSE_out","MAE_out","QLIKE_out"
]).set_index(["delta","tenor"])
pd.set_option("display.float_format", lambda x: f"{x:0.6f}")
print("\nTable A: Error by Delta (split at", SPLIT_DATE, ")\n")
print(tbl_delta.groupby(level=0).mean().to_string())


A = iv_out.values.flatten()
F = pred_out.values.flatten()
mask = np.isfinite(A) & np.isfinite(F)
A, F = A[mask], F[mask]

global_tbl = pd.DataFrame({
   "Metric": ["MSE","RMSE","MAE","R2","MAPE","SMAPE","Sample_Pts"],
   "Value": [
       mse(A,F),
       rmse(A,F),
       mae(A,F),
         r2(pd.Series(A), pd.Series(F)),
         np.nanmean(np.abs((A-F)/np.clip(A,1e-6,None))),
         2*np.nanmean(np.abs(A-F)/(np.abs(A)+np.abs(F)+1e-12)),
         A.size
     ]
})
print("\nTable B: Global OOS Metrics\n")
print(global_tbl.to_string(index=False))


daily_err = pd.DataFrame({
     "MSE": (iv_out - pred_out).pow(2).mean(axis=1),
     "MAE": (iv_out - pred_out).abs().mean(axis=1),
     "QLIKE": qlike(iv_out, pred_out).mean(axis=1)
})
roll_err = daily_err.rolling(ROLL_WIN_DAYS, min_periods=60).mean()

plt.figure(figsize=(11,4))
plt.plot(roll_err.index, roll_err["MSE"], label="MSE")
plt.plot(roll_err.index, roll_err["MAE"], label="MAE")
plt.plot(roll_err.index, roll_err["QLIKE"],label="QLIKE")
plt.title("5-Year Rolling Forecast Errors (ConvLSTM)")
plt.legend(); plt.grid(True, alpha=.3); plt.tight_layout(); plt.show()


rmse_daily = (iv_out - pred_out).pow(2).mean(axis=1).pow(0.5)
plt.figure(figsize=(11,4))
plt.plot(rmse_daily.index, rmse_daily.values)
plt.title("Daily RMSE of IV-Surface Forecasts (ConvLSTM)")
plt.ylabel("RMSE"); plt.grid(True, alpha=.3); plt.tight_layout(); plt.show()


key = col(FRONT_T, 10)
if key in iv_out.columns and key in pred_out.columns:
     plt.figure(figsize=(5.8,5.2))
     plt.scatter(iv_out[key], pred_out[key], s=8, alpha=.35)
     mn = min(iv_out[key].min(), pred_out[key].min())
     mx = max(iv_out[key].max(), pred_out[key].max())
     plt.plot([mn,mx],[mn,mx], linestyle="--")
     plt.xlabel("Actual IV"); plt.ylabel("Predicted IV")
     plt.title(f"Actual vs Predicted   {key} (ConvLSTM, OOS)")
   plt.grid(True, alpha=.3); plt.tight_layout(); plt.show()



hdate = pd.Timestamp(HEATMAP_DATE)
if hdate not in iv_real.index:

   hdate = iv_real.index[(np.abs(iv_real.index - hdate)).argmin()]


tenors = sorted({int(c.split("_")[0]) for c in iv_real.columns})
# func to matrix
deltas = sorted({int(c.split("_")[1]) for c in iv_real.columns})

def to_matrix(df, dt):
   M = np.full((len(tenors), len(deltas)), np.nan)
   for i,t in enumerate(tenors):
       for j,d in enumerate(deltas):
          name = col(t,d)
          if name in df.columns and dt in df.index:
               M[i,j] = df.at[dt, name]
   return M

A_mat = to_matrix(iv_real, hdate)
P_mat = to_matrix(conv_pred, hdate)

fig, axes = plt.subplots(1,2, figsize=(12,4), constrained_layout=True)
im0 = axes[0].imshow(A_mat, aspect="auto")
axes[0].set_title(f"Actual IV surface\n{hdate.date()}")
axes[0].set_ylabel("Tenor (days)")
axes[0].set_xlabel("Delta")
axes[0].set_yticks(range(len(tenors))); axes[0].set_yticklabels([str(t) for t in tenors])
axes[0].set_xticks(range(len(deltas))); axes[0].set_xticklabels([str(d) for d in deltas])
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(P_mat, aspect="auto")
axes[1].set_title(f"Predicted IV surface (ConvLSTM)\n{hdate.date()}")
axes[1].set_xlabel("Delta")
axes[1].set_yticks(range(len(tenors))); axes[1].set_yticklabels([str(t) for t in tenors])
axes[1].set_xticks(range(len(deltas))); axes[1].set_xticklabels([str(d) for d in deltas])
fig.colorbar(im1, ax=axes[1])

plt.show()

"""# **bold text**

> Add blockquote

1. List item

1. List item

1. List item

1. List item

1. List item

1. List item

1. List item

1. List item
2. List item


2. List item


2. List item


2. List item


2. List item


2. List item


2. List item


2. List item




"""





















import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from math import log, sqrt
from scipy.stats import norm
from typing import Optional


ROOT = "/content/drive/MyDrive"


SPOT_FILE = f"{ROOT}/kg4snxuhemmtl4nx.csv"
ACTUAL_IV_FILE = f"{ROOT}/bcmocccogeuhqddt_features_flat.csv"


CONV_WIDE_FILE = f"{ROOT}/iv_convlstm_rolling_preds.csv"
ATTN_WIDE_FILE = f"{ROOT}/iv_attlstm_rolling_preds.csv"


PLOT_DIR = f"{ROOT}/plots_calendar_unified"
Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)


FRONT_T = 30
BACK_T = 60
DELTA_BUCKET = 10
NOTIONAL = 1_000_000
YEAR_DAYS = 365.25


VIX_MIN = 0.0
VIX_MAX = 40.0
TERM_SLOPE_MIN = 0.0025
MIN_PREMIUM_PCT = 0.005
MAX_LEVERAGE = 10.0


STOP_LOSS_PCT = 0.03
MIN_EXPIRY_DAYS = 5


HEDGE_SMOOTH_ALPHA = 0.60
MAX_HEDGE_TURNOVER_X = 3.0


OPT_TC_BP_PER_LEG = 5
OPT_SLIP_BP_PER_LEG = 5
HEDGE_TC_BP = 1
HEDGE_SLIP_BP = 1
N_OPTION_LEGS = 4


CONV_FALLBACK_USE_REALIZED_SLOPE = True
CONV_FALLBACK_SLOPE_MIN = 0.0010
LSTM_ATTN_FALLBACK_USE_REALIZED_SLOPE = True
LSTM_ATTN_FALLBACK_SLOPE_MIN = 0.0010


COMPARE_START = "2004-01-01"
# func print header


def print_header(txt):
   bar = "=" * max(88, len(txt) + 4)
# func col
   print("\n" + bar + f"\n{txt}\n" + bar)

def col(t, d):
# func dedup index
   return f"{t}_{d:+d}".replace("+", "")

def _dedup_index(df):
   if not df.index.is_unique:
       df = df.groupby(level=0).last()
   return df
# func bs price


def bs_price(S, K, T, sig, cp, rate=0.0):
   S = float(S); K = float(K); T = float(T); sig = float(sig)
   if (T <= 0.0) or (sig <= 0.0) or (not np.isfinite(sig)):
       return max(0.0, S-K) if cp == 'c' else max(0.0, K-S)
   d1 = (log(S/K) + (rate + 0.5*sig*sig)*T)/(sig*sqrt(T))
   d2 = d1 - sig*sqrt(T)
   if cp == 'c':
       return S*norm.cdf(d1) - K*np.exp(-rate*T)*norm.cdf(d2)
   else:
# func bs delta
       return K*np.exp(-rate*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def bs_delta(S, K, T, sig, cp, rate=0.0):
   S = float(S); K = float(K); T = float(T); sig = float(sig)
   if (T <= 0.0) or (sig <= 0.0) or (not np.isfinite(sig)):
       return 0.0
   d1 = (log(S/K) + (rate + 0.5*sig*sig)*T)/(sig*sqrt(T))
   return norm.cdf(d1) if cp == 'c' else -norm.cdf(-d1)
# func load vix series


def load_vix_series(idx: pd.DatetimeIndex) -> pd.Series:
   fail_count = 0

   try:
       v = (pd.read_csv("https://stooq.com/q/d/l/?s=^vix&i=d")
             .rename(columns={"Date":"date","Close":"VIX"}))
       v["date"] = pd.to_datetime(v["date"])
       v = v.set_index("date").sort_index()[["VIX"]]
       v = v.reindex(idx).ffill()
       if v["VIX"].notna().sum() > 0:
           print("VIX source: Stooq")
           return v["VIX"]
   except Exception:
       fail_count += 1


   try:
       from pandas_datareader import data as pdr
       v = pdr.DataReader("VIXCLS", "fred", idx.min(), idx.max()).rename( columns={"VIXCLS":"VIX"})
       v = v.reindex(idx).ffill()
       print("VIX source: FRED")
       return v["VIX"]
   except Exception:
       fail_count += 1


   try:
       v = (pd.read_csv("https://raw.githubusercontent.com/datasets/finance-vix /master/data/vix-daily.csv",
                         parse_dates=["Date"])
             .set_index("Date").sort_index()[["VIXClose"]]
             .rename(columns={"VIXClose":"VIX"}))
       v = v.reindex(idx).ffill()
       print("VIX source: GitHub mirror")
       return v["VIX"]
   except Exception:
       fail_count += 1

   print(f"VIX load failed after {fail_count} attempts     using constant 20.")
# func load rfr series
   return pd.Series(20.0, index=idx, name="VIX")

def load_rfr_series(idx: pd.DatetimeIndex) -> pd.Series:
   try:
       from pandas_datareader import data as pdr
       r = pdr.DataReader("DGS3MO", "fred", idx.min(), idx.max())
       r = r.rename(columns={"DGS3MO": "rfr"})
       r["rfr"] /= 100.0
       r = r.reindex(idx).ffill().bfill()
       if r["rfr"].notna().sum() > 0:
          print("RFR source: FRED DGS3MO")
          return r["rfr"]
   except Exception as e:
       print(f"RFR load failed: {e}    using constant 0.02")
# func load spot and iv
       return pd.Series(0.02, index=idx, name="rfr")

def load_spot_and_iv():
   if not Path(SPOT_FILE).exists():
       raise FileNotFoundError(SPOT_FILE)
   if not Path(ACTUAL_IV_FILE).exists():
       raise FileNotFoundError(ACTUAL_IV_FILE)

   spot = (pd.read_csv(SPOT_FILE, parse_dates=["DlyCalDt"])
            .rename(columns={"DlyCalDt":"date","DlyPrcInd":"spot"})
            .set_index("date").sort_index().groupby(level=0).last())
   spot["spot"] = spot["spot"].astype(float)

   hdr = pd.read_csv(ACTUAL_IV_FILE, nrows=0)
   n_grid = hdr.shape[1] // 3
   iv = (pd.read_csv(ACTUAL_IV_FILE, parse_dates=["date"])
          .set_index("date").sort_index()
          .iloc[:, :n_grid]
          .groupby(level=0).last())

   needed = [col(FRONT_T, +DELTA_BUCKET), col(FRONT_T, -DELTA_BUCKET),
              col(BACK_T, +DELTA_BUCKET), col(BACK_T, -DELTA_BUCKET)]
   miss = [c for c in needed if c not in iv.columns]
   if miss:
       raise RuntimeError(f"Actual IV missing columns: {miss}")
   iv = iv[needed].copy()
# func load forecast wide
   return spot, iv

def load_forecast_wide(path, actual_cols):
   if not Path(path).exists():
       return None
   df = (pd.read_csv(path, parse_dates=["date"])
          .set_index("date").sort_index().groupby(level=0).last())
   miss = [c for c in actual_cols if c not in df.columns]
   if miss:
       print(f"Forecast file {Path(path).name} missing columns (skipping): { miss}")
       return None
   return df[actual_cols].add_suffix("_f")
# func option trade cost


def option_trade_cost(q, V_abs):
   bps = (OPT_TC_BP_PER_LEG + OPT_SLIP_BP_PER_LEG) / 1e4
# func hedge trade cost
   return abs(q) * abs(V_abs) * bps * N_OPTION_LEGS

def hedge_trade_cost(delta_shares_abs_notional):
   bps = (HEDGE_TC_BP + HEDGE_SLIP_BP) / 1e4
   return delta_shares_abs_notional * bps
# func run calendar engine


def run_calendar_engine(spot: pd.DataFrame,
                       realized_iv: pd.DataFrame,
                       forecast_iv: Optional[pd.DataFrame],
                       tag: str,
                       use_forecast_for_signal_and_hedge: bool):
   """
   If use_forecast_for_signal_and_hedge:
          Signals (term slope) and hedge deltas use forecast_iv (..._f)
   Else:
          Signals and hedge deltas use realized_iv
   In ALL cases:
          Pricing & MTM use realized_iv
          RFR applied in BS and daily equity accrual while a position is open
   """
   df = spot.join(realized_iv, how="inner").sort_index()
   if use_forecast_for_signal_and_hedge:
         if forecast_iv is None:
            raise RuntimeError(f"[{tag}] Forecast dataframe required but missing. ")
         df = df.join(forecast_iv, how="left")
   df = _dedup_index(df)


   vix = load_vix_series(df.index)
   df = df.join(vix.rename("VIX"))


   rfr = load_rfr_series(df.index)
   df = df.join(rfr.rename("rfr"))


   needed_real = list(realized_iv.columns)
   needed_fore = [c + "_f" for c in realized_iv.columns] if use_forecast_for_signal_and_hedge else []
   req = ["spot", "VIX", "rfr"] + needed_real + needed_fore
   df = df.dropna(subset=req)

   fF, pF = col(FRONT_T, +DELTA_BUCKET), col(FRONT_T, -DELTA_BUCKET)
   fB, pB = col(BACK_T, +DELTA_BUCKET), col(BACK_T, -DELTA_BUCKET)

  dates, pnl_mtm, pnl_hedge, pnl_costs, pnl_total, eq_series = [], [], [], [], [], []
  equity = NOTIONAL
  pos = None

  for i in range(1, len(df)):
     d_prev, d = df.index[i-1], df.index[i]
     S_prev, S = float(df.at[d_prev, "spot"]), float(df.at[d, "spot"])
     dS = S - S_prev
     vx = float(df.at[d, "VIX"])
     r_prev = float(df.at[d_prev, "rfr"])
     r_now = float(df.at[d, "rfr"])
     delta_t = (d - d_prev).days / YEAR_DAYS

     mtm_day = hedge_day = cost_day = rf_day = 0.0


     if pos is not None:
         pos["T1"] = max(0.0, pos["T1"] - delta_t)
         pos["T2"] = max(0.0, pos["T2"] - delta_t)


         s1c = float(df.at[d, fF]); s1p = float(df.at[d, pF])
         s2c = float(df.at[d, fB]); s2p = float(df.at[d, pB])

         V1_now = bs_price(S, pos["K"], pos["T1"], s1c, 'c', r_now) + bs_price (S, pos["K"], pos["T1"], s1p, 'p', r_now)
         V2_now = bs_price(S, pos["K"], pos["T2"], s2c, 'c', r_now) + bs_price (S, pos["K"], pos["T2"], s2p, 'p', r_now)
         V_now = V2_now - V1_now

         if np.isfinite(V_now):
               mtm_day = (V_now - pos["V_prev"]) * pos["q"]
               pos["V_prev"] = V_now

         hedge_day = pos["hedge_shares"] * dS


         if use_forecast_for_signal_and_hedge:
               fc1c = float(df.at[d, fF + "_f"]); fc1p = float(df.at[d, pF + "_f "])
         fc2c = float(df.at[d, fB + "_f"]); fc2p = float(df.at[d, pB + "_f "])
         if not all(np.isfinite(x) for x in [fc1c, fc1p, fc2c, fc2p]):
             fc1c, fc1p, fc2c, fc2p = s1c, s1p, s2c, s2p
         f1c, f1p, f2c, f2p = fc1c, fc1p, fc2c, fc2p
     else:
         f1c, f1p, f2c, f2p = s1c, s1p, s2c, s2p

     d_front = bs_delta(S, pos["K"], pos["T1"], f1c, 'c', r_now) + bs_delta(S, pos["K"], pos["T1"], f1p, 'p', r_now)
     d_back = bs_delta(S, pos["K"], pos["T2"], f2c, 'c', r_now) + bs_delta (S, pos["K"], pos["T2"], f2p, 'p', r_now)
     delta_target = (d_back - d_front)
     hedge_target = - pos["q"] * delta_target


     hedge_new = (1.0 - HEDGE_SMOOTH_ALPHA) * pos["hedge_shares"] + HEDGE_SMOOTH_ALPHA * hedge_target
     d_shares = hedge_new - pos["hedge_shares"]

     min_cap = 0.1 * INITIAL_CAPITAL
     effective_cap = max(min_cap, eq[-1], 0.1 * INITIAL_CAPITAL)
     max_turn_notional = MAX_HEDGE_TURNOVER_X * effective_cap
     d_notional = abs(d_shares) * S
     if d_notional > max_turn_notional and d_notional > 0:
         scale = max_turn_notional / d_notional
         d_shares *= scale
         hedge_new = pos["hedge_shares"] + d_shares

     cost_day += hedge_trade_cost(abs(d_shares) * S)
     pos["hedge_shares"] = hedge_new


     rf_day = equity * r_prev * delta_t


  if pos is None:
     if VIX_MIN <= vx <= VIX_MAX:
         S0 = S; K = S0; T1 = FRONT_T / YEAR_DAYS; T2 = BACK_T / YEAR_DAYS

         s1c = float(df.at[d, fF]); s1p = float(df.at[d, pF])
  s2c = float(df.at[d, fB]); s2p = float(df.at[d, pB])

  V1 = bs_price(S0, K, T1, s1c, 'c', r_now) + bs_price(S0, K, T1, s1p, 'p', r_now)
  V2 = bs_price(S0, K, T2, s2c, 'c', r_now) + bs_price(S0, K, T2, s2p, 'p', r_now)
  V0 = V2 - V1

  min_premium = MIN_PREMIUM_PCT * S0
  if np.isfinite(V0) and (V0 > 0.0) and (abs(V0) >= min_premium):


     if use_forecast_for_signal_and_hedge:
         fc1c = float(df.at[d, fF + "_f"]); fc1p = float(df.at[d, pF + "_f"])
         fc2c = float(df.at[d, fB + "_f"]); fc2p = float(df.at[d, pB + "_f"])
         term_edge_fore = 0.5 * ((fc2c + fc2p) - (fc1c + fc1p))
         signal_ok = np.isfinite(term_edge_fore) and ( term_edge_fore >= TERM_SLOPE_MIN)


         tag_l = tag.lower()
         term_edge_real = 0.5 * ((s2c + s2p) - (s1c + s1p))
         if (not signal_ok) and np.isfinite(term_edge_real):
             if tag_l.startswith("convlstm") and CONV_FALLBACK_USE_REALIZED_SLOPE:
                 signal_ok = (term_edge_real >= CONV_FALLBACK_SLOPE_MIN)
             elif tag_l.startswith("lstm-attn") and LSTM_ATTN_FALLBACK_USE_REALIZED_SLOPE:
                 signal_ok = (term_edge_real >= LSTM_ATTN_FALLBACK_SLOPE_MIN)


         if all(np.isfinite(x) for x in [fc1c, fc1p, fc2c, fc2p]):
             f1c, f1p, f2c, f2p = fc1c, fc1p, fc2c, fc2p
         else:
             f1c, f1p, f2c, f2p = s1c, s1p, s2c, s2p
     else:
         term_edge_real = 0.5 * ((s2c + s2p) - (s1c + s1p))
                 signal_ok = np.isfinite(term_edge_real) and ( term_edge_real >= TERM_SLOPE_MIN)
                 f1c, f1p, f2c, f2p = s1c, s1p, s2c, s2p

              if signal_ok:
                 q_base = NOTIONAL / abs(V0)
                 q_max = (MAX_LEVERAGE * NOTIONAL) / abs(V0)
                 q = min(q_base, q_max)

                 d_front = bs_delta(S0, K, T1, f1c, 'c', r_now) + bs_delta( S0, K, T1, f1p, 'p', r_now)
                 d_back = bs_delta(S0, K, T2, f2c, 'c', r_now) + bs_delta( S0, K, T2, f2p, 'p', r_now)
                 delta0 = (d_back - d_front)
                 hedge0 = - q * delta0


                 cost_entry = option_trade_cost(q, abs(V0)) + hedge_trade_cost(abs(hedge0) * S0)
                 cost_day += cost_entry

                 pos = {
                      "K": K, "T1": T1, "T2": T2,
                      "q": q,
                      "V_prev": V0,
                      "hedge_shares": hedge0,
                      "pnl_since_entry": -cost_entry,
                      "entry_notional": q * abs(V0),
                 }


  pnl_day = mtm_day + hedge_day - cost_day + rf_day
  equity += pnl_day

  if pos is not None:
     pos["pnl_since_entry"] += pnl_day
     exit_reason = None
     if pos["T1"] * YEAR_DAYS <= MIN_EXPIRY_DAYS:
         exit_reason = "auto_expiry"
     elif pos["pnl_since_entry"] <= - STOP_LOSS_PCT * pos["entry_notional" ]:
                exit_reason = "stop_loss"
          elif not (VIX_MIN <= vx <= VIX_MAX):
                exit_reason = "vix_gate"

          if exit_reason is not None:
                close_cost = option_trade_cost(pos["q"], abs(pos["V_prev"])) + hedge_trade_cost(abs(pos["hedge_shares"]) * S)
                equity -= close_cost
                cost_day += close_cost
                pos = None

       dates.append(d)
       pnl_mtm.append(mtm_day)
       pnl_hedge.append(hedge_day)
       pnl_costs.append(cost_day - rf_day)
       pnl_total.append(pnl_day)
       eq_series.append(equity)

   out = pd.DataFrame({
       "pnl_mtm": pnl_mtm,
       "pnl_hedge": pnl_hedge,
       "pnl_costs": pnl_costs,
       "pnl": pnl_total,
       "cum_pnl": eq_series
   }, index=pd.Index(dates, name="date"))
   return out
# func performance report


def performance_report(series, initial=NOTIONAL, title="PERFORMANCE"):
   eq = series.dropna()
   rets = eq.pct_change().dropna()
   total_return = float(eq.iloc[-1]/initial - 1.0)
   n_days = len(rets)
   base = eq.iloc[-1]/initial if n_days else 1.0
   ann_ret = (np.sign(base)*(np.abs(base)**(YEAR_DAYS/max(n_days,1)) - 1.0)) if n_days else 0.0
   ann_vol = float(rets.std()*np.sqrt(YEAR_DAYS)) if n_days>1 else 0.0
   sharpe = ann_ret/ann_vol if ann_vol>0 else 0.0
   cmax = eq.cummax(); dd = (cmax-eq)/cmax; max_dd=float(dd.max()) if len(dd) else 0.0
   dneg = rets[rets<0]; ddev = float(dneg.std()*np.sqrt(YEAR_DAYS)) if len(dneg )>1 else 0.0
   sortino = ann_ret/ddev if ddev>0 else 0.0
   win_rate = len(rets[rets>0])/n_days if n_days else 0.0
   gp=float(rets[rets>0].sum()); gl=abs(float(rets[rets<0].sum())); pf=gp/gl if gl>0 else float("inf")
   var_95=float(np.percentile(rets,5)) if n_days else 0.0
   es_95=float(rets[rets<=var_95].mean()) if n_days else 0.0
   avg=float(rets.mean()) if n_days else 0.0
   best=float(rets.max()) if n_days else 0.0
   worst=float(rets.min()) if n_days else 0.0

   print("\n" + "="*100)
   print(title)
   print("="*100)
   print(f"{'Metric':<45}{'Value':>22}{'Details':>20}")
   print("-"*100)
   print(f"{'Initial Capital:':<45}${initial:>21,.0f}")
   print(f"{'Final Equity:':<45}${eq.iloc[-1]:>21,.2f}")
   print(f"{'Total Return:':<45}{total_return:>21.2%}")
   print(f"{'Annualized Return (CAGR):':<45}{ann_ret:>21.2%}")
   print(f"{'Annualized Volatility:':<45}{ann_vol:>21.2%}")
   print(f"{'Sharpe Ratio (r=0):':<45}{sharpe:>21.2f}")
   print(f"{'Sortino Ratio:':<45}{sortino:>21.2f}")
   print(f"{'Max Drawdown:':<45}{max_dd:>21.2%}{('('+format(max_dd*100,'.1f') +'%)'):>20}")
   print(f"{'Win Rate:':<45}{win_rate:>21.2%}")
   print(f"{'Profit Factor:':<45}{pf:>21.2f}")
   print(f"{'Daily 95% VaR:':<45}{var_95:>21.2%}")
   print(f"{'Daily 95% Expected Shortfall:':<45}{es_95:>21.2%}")
   print(f"{'Avg Daily Return:':<45}{avg:>21.2%}")
   print(f"{'Best Daily Gain:':<45}{best:>21.2%}")
   print(f"{'Worst Daily Loss:':<45}{worst:>21.2%}")
   print(f"{'Trading Days:':<45}{n_days:>21}")
   print(f"{'Years:':<45}{n_days/YEAR_DAYS:>21.1f}")
   print("="*100)
# func stationary bootstrap summaries


def stationary_bootstrap_summaries(equity: pd.Series, initial: float,
                                runs=1000, block_mean=10, seed=42) -> pd. DataFrame:
  """Stationary bootstrap over daily returns; robust to NaN/Inf; strictly real output."""
  rets = equity.astype(float).pct_change().replace([np.inf, -np.inf], np.nan). dropna().values
  if len(rets) == 0:
     return pd.DataFrame()
  n = len(rets)
  rng = np.random.default_rng(seed)
  p = 1.0/max(block_mean,1)

  def _pick_indices():
     idx = np.empty(n, dtype=int)
     idx[0] = rng.integers(0, n)
     for t in range(1, n):
         if rng.random() < p:
             idx[t] = rng.integers(0, n)
         else:
             idx[t] = (idx[t-1] + 1) % n
     return idx

  finals, totals, maxdds, sharpes = [], [], [], []
  for _ in range(runs):
     idx = _pick_indices()
     boot = rets[idx].astype(float)
     path = initial * np.cumprod(1.0 + boot)
     fin = float(path[-1])
     tot = fin/initial - 1.0


     if n > 1:
         ann_vol = float(np.std(boot, ddof=1) * np.sqrt(YEAR_DAYS))
     else:
         ann_vol = np.nan
     if initial > 0 and fin > 0 and n > 0:
         ann_ret = (fin/initial)**(YEAR_DAYS/n) - 1.0
     else:
         ann_ret = np.nan
     sharpe = ann_ret/ann_vol if (np.isfinite(ann_ret) and np.isfinite( ann_vol) and ann_vol > 0) else np.nan

        s = pd.Series(path)
        dd = (s.cummax() - s) / s.cummax()
        mdd = float(dd.max()) if len(dd) else np.nan

        finals.append(fin); totals.append(tot); maxdds.append(mdd); sharpes. append(sharpe)

   def pct_safe(a, q):
        arr = np.asarray(a)
        if np.iscomplexobj(arr):
           arr = np.real(arr)
        arr = arr.astype(float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
           return float('nan')
        return float(np.percentile(arr, q))

   return pd.DataFrame({
        'metric': ['final_equity','total_return','max_dd','sharpe_0'],
        'p05': [pct_safe(finals,5), pct_safe(totals,5), pct_safe(maxdds,5), pct_safe(sharpes,5)],
        'p50': [pct_safe(finals,50),pct_safe(totals,50),pct_safe(maxdds,50), pct_safe(sharpes,50)],
        'p95': [pct_safe(finals,95),pct_safe(totals,95),pct_safe(maxdds,95), pct_safe(sharpes,95)],
        'runs': [runs]*4,
        'block_mean':[block_mean]*4,
        'seed': [seed]*4
# func print mc block
   })

def print_mc_block(name: str, mc_df: pd.DataFrame):
   if mc_df is None or mc_df.empty:
        print(f"[MC][{name}] No MC summary (empty).")
        return
   print(f"\n[MC][{name}] Stationary-bootstrap summary")
   for _, row in mc_df.iterrows():
        m = row['metric']
        if m == 'final_equity':
          print(f" Final Equity : P5={row['p05']:,.0f} P50={row['p50']:,.0f} P95={row['p95']:,.0f}")
       elif m == 'total_return':
          print(f" Total Return : P5={row['p05']:.2%} P50={row['p50']:.2%} P95 ={row['p95']:.2%}")
       elif m == 'max_dd':
          print(f" Max DD : P5={row['p05']:.2%} P50={row['p50']:.2%} P95={row[' p95']:.2%}")
       elif m == 'sharpe_0':
# func plot equity drawdown
          print(f" Sharpe (r=0) : P5={row['p05']:.2f} P50={row['p50']:.2f} P95 ={row['p95']:.2f}")

def plot_equity_drawdown(df_eq, title_prefix, png_name):
   plt.style.use("ggplot")
   fig, (ax1, ax2) = plt.subplots(2,1, figsize=(14,10), sharex=True, gridspec_kw={"height_ratios":[3,1]})
   final_equity = float(df_eq["cum_pnl"].iloc[-1])
   ax1.plot(df_eq.index, df_eq["cum_pnl"], linewidth=2, label=f"{title_prefix} (Final: ${final_equity:,.0f})")
   ax1.axhline(NOTIONAL, color="gray", linestyle="--", alpha=0.7, label=" Initial Capital")
   ax1.set_title(title_prefix, fontsize=16); ax1.set_ylabel("Equity ($)")
   ax1.grid(True, linestyle="--", alpha=0.7); ax1.legend(loc="best", fontsize =12)

   cmax = df_eq["cum_pnl"].cummax(); dd = (cmax - df_eq["cum_pnl"]) / cmax
   max_dd_pct = float(dd.max()*100.0) if len(dd) else 0.0
   ax2.fill_between(df_eq.index, dd*100.0, 0, alpha=0.3, label=f"Max Drawdown: {max_dd_pct:.1f}%")
   ax2.set_ylabel("Drawdown (%)"); ax2.grid(True, linestyle="--", alpha=0.7); ax2.legend(loc="best")
   ax2.xaxis.set_major_locator(mdates.YearLocator()); ax2.xaxis. set_major_formatter(mdates.DateFormatter("%Y"))
   plt.xlabel("Date"); plt.tight_layout()
   out_png = os.path.join(PLOT_DIR, png_name); plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.show()
# func plot pnl components
   print(f"Saved plot   {out_png}")

def plot_pnl_components(df, title_prefix, png_name):
   plt.figure(figsize=(14,4))
   plt.plot(df.index, df["pnl_mtm"], label="MTM")
   plt.plot(df.index, df["pnl_hedge"], label="Hedge")
   plt.plot(df.index, df["pnl_costs"], label="Costs (net of RFR)")
   plt.title(f"{title_prefix}   Daily PnL Components"); plt.xlabel("Date"); plt. ylabel("PnL ($)")
   plt.legend(); out_png = os.path.join(PLOT_DIR, png_name)
   plt.tight_layout(); plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt. show()
# func comparison plot
   print(f"Saved plot    {out_png}")

def comparison_plot(*curves_and_labels, png_name=" calendar_equity_comparison_common.png"):

   plt.style.use("ggplot"); plt.figure(figsize=(14,6))
   for df, lab in curves_and_labels:
       if df is not None and not df.empty:
          reb = df["cum_pnl"].iloc[0] - NOTIONAL
          plt.plot(df.index, df["cum_pnl"] - reb, label=lab, linewidth=2)
   plt.title("30/60 Calendar Straddle    Equity Curves (Rebased Visually, Common Window)")
   plt.axhline(NOTIONAL, color="gray", linestyle="--", alpha=0.6)
   plt.ylabel("Equity ($)"); plt.xlabel("Date"); plt.legend(); plt.grid(True, linestyle="--", alpha=0.7)
   out_png = os.path.join(PLOT_DIR, png_name)
   plt.tight_layout(); plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt. show()
   print(f"Saved plot    {out_png}")


if __name__ == "__main__":

   spot, realized_iv = load_spot_and_iv()
   need_cols = realized_iv.columns.tolist()


   conv_wide = load_forecast_wide(CONV_WIDE_FILE, need_cols)
   attn_wide = load_forecast_wide(ATTN_WIDE_FILE, need_cols)

   base_idx = spot.index.intersection(realized_iv.index)

  def prep_seeded_forecast(fore_df):
      if fore_df is None:
         return None
      ff = fore_df.reindex(base_idx).ffill()
      seeded = ff.combine_first(realized_iv.reindex(base_idx).add_suffix("_f") )
      return seeded

  conv_seeded = prep_seeded_forecast(conv_wide)
  attn_seeded = prep_seeded_forecast(attn_wide)


  common_idx = base_idx.copy()
  if conv_seeded is not None:
      common_idx = common_idx.intersection(conv_seeded.index)
  if attn_seeded is not None:
      common_idx = common_idx.intersection(attn_seeded.index)
  if COMPARE_START:
      common_idx = common_idx[common_idx >= pd.Timestamp(COMPARE_START)]

  if len(common_idx) < 3:
      raise RuntimeError("Common intersection too small. Check inputs/ forecasts.")


  print_header("Running all strategies strictly on the COMMON intersection")


  res_rule = run_calendar_engine(
      spot=spot.loc[common_idx],
      realized_iv=realized_iv.loc[common_idx],
      forecast_iv=None,
      tag="RULE",
      use_forecast_for_signal_and_hedge=False
  )


  res_conv = None
  if conv_seeded is not None:
      res_conv = run_calendar_engine(
         spot=spot.loc[common_idx],
         realized_iv=realized_iv.loc[common_idx],
         forecast_iv=conv_seeded.loc[common_idx],
         tag="ConvLSTM",
         use_forecast_for_signal_and_hedge=True
     )


  res_attn = None
  if attn_seeded is not None:
     res_attn = run_calendar_engine(
         spot=spot.loc[common_idx],
         realized_iv=realized_iv.loc[common_idx],
         forecast_iv=attn_seeded.loc[common_idx],
         tag="LSTM-ATTN",
         use_forecast_for_signal_and_hedge=True
     )


  print_header("Equity + Drawdown (Common Window Only)")
  plot_equity_drawdown(res_rule, "Rule-based (Realized Only; Costs & RFR)", " rule_realized_equity_dd.png")
  if res_conv is not None:
     plot_equity_drawdown(res_conv, "ConvLSTM (Seeded Forecasts; Costs & RFR) ", "convlstm_equity_dd.png")
  if res_attn is not None:
     plot_equity_drawdown(res_attn, "LSTM-ATTN (Seeded Forecasts; Costs & RFR )", "attnlstm_equity_dd.png")

  print_header("Daily PnL components (Common Window Only)")
  plot_pnl_components(res_rule, "Rule-based (Realized Only)", " rule_realized_pnl_components.png")
  if res_conv is not None:
     plot_pnl_components(res_conv, "ConvLSTM", "convlstm_pnl_components.png")
  if res_attn is not None:
     plot_pnl_components(res_attn, "LSTM-ATTN", "attnlstm_pnl_components.png" )


  named_views_cw = [("Rule-based (Costs & RFR)", res_rule)]
  if res_conv is not None:
         named_views_cw.append(("ConvLSTM (Seeded, Costs & RFR)", res_conv))
      if res_attn is not None:
         named_views_cw.append(("LSTM-ATTN (Seeded, Costs & RFR)", res_attn))

      comparison_plot(*[(df, name) for name, df in named_views_cw],
                    png_name="calendar_equity_comparison_common.png")


      print_header("PERFORMANCE    Common Intersection Only")
      for name, df in named_views_cw:
         performance_report(df["cum_pnl"], title=f"[Common Window] {name} PERFORMANCE")

         mc = stationary_bootstrap_summaries(df["cum_pnl"], NOTIONAL, runs=1000, block_mean=10, seed=42)
         print_mc_block(name, mc)

      print_header("DONE    All outputs use the SAME common intersection of dates.")


"""
Unified Short-30d Straddle Backtest Suite (Start: 2004, dynamic r from FRED)
============================================================================

Changes in this version:
       Added DAILY equity accrual based on the risk-free rate (r from FRED),
       applied on days when a position is open (included in pnl and equity).
       Replaced spot-move stop with a PnL-based stop:
         stop if cumulative PnL since entry <= -3% of NOTIONAL (after costs and rf accrual).

Strategies (calendar-style dynamic hedging):
      1) RULE-BASED (ATM)    strikes & MTM from realised IV, hedge   from realised IV (daily re-hedge).
      2) ATTN-LSTM Hedge- (ATM)    strikes & MTM from realised IV, hedge   from ATTN preds (ffill; fallback realised; daily).
      3) CONV-LSTM Hedge- (ATM)    strikes & MTM from realised IV, hedge   from CONV preds (ffill; fallback realised; daily).

Risk-free rate:
      - Fetched for the run window using FRED (DGS1MO, fallback DGS3MO) via pandas_datareader.
       Values are annualized (decimal). If internet fetch fails, fallback is zeros (still runs).

VIX gate:
      - VIX fetched from FRED (fallback-safe). Gate: block entries at/above 40; resume after 10 days < 39; force exit if VIX   40.

Start date:
      - All strategies and plots run from **2004-01-01** or later (strictly enforced).

Strict common-date intersection:
      - After running each strategy (2004+), intersect indices and **REBASE equity ** on the common window:
         equity_common = INITIAL + cumsum(pnl_common)

Costs:
      - Options: (commission_bps + slippage_bps)   |units|    option value (entry & exit).
      - Underlying hedge: (commission_bps + slippage_bps)     |shares change|   Spot ( entry, re-hedges, exit).

Author: (you)
"""





import os, glob, math, warnings
from math import log, sqrt
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
   from pandas_datareader import data as pdr
   _PDR_OK = True
except Exception:
   _PDR_OK = False
   pdr = None
   print("WARNING: pandas_datareader missing. FRED fetch will fall back to NaNs /zeros.")





ROOT = '/content/drive/MyDrive'
SPOT_CSV = f'{ROOT}/kg4snxuhemmtl4nx.csv'
IV_REAL_CSV = f'{ROOT}/bcmocccogeuhqddt_features_flat.csv'
IV_ATTN_CSV = f'{ROOT}/iv_attlstm_rolling_preds.csv'
IV_CONV_CSV = f'{ROOT}/iv_convlstm_rolling_preds.csv'

FAMILY_NAME = 'straddle_atm'


HEDGE_SMOOTH_ALPHA = 0.80
MAX_HEDGE_TURNOVER_X = 3.0

INITIAL_CAPITAL = 1_000_000.0
VOL_FLOOR = 1e-6
CALENDAR_YEAR = 365.25



# func dedup index


def _dedup_index(df: pd.DataFrame) -> pd.DataFrame:
   if not df.index.is_unique:
       df = df.groupby(level=0).last()
# func to numeric
   return df

def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
# func cell
   return df.apply(pd.to_numeric, errors='coerce')

def _cell(df: pd.DataFrame, date: pd.Timestamp, col: str) -> float:
   v = df.loc[date, col]
   if isinstance(v, (pd.Series, pd.Index, np.ndarray, list, tuple)):
       v = np.asarray(v).ravel()[-1]
# func pct change
   return float(v)

def _pct_change(series: pd.Series) -> pd.Series:
   try:
       return series.pct_change(fill_method=None)
   except TypeError:
# func reconcile equity
       return series.pct_change()

def _reconcile_equity(eq_series, pnl_series, initial):
   try:
       if len(eq_series) != len(pnl_series): return False
       cs = pnl_series.cumsum() + initial
       diff = (eq_series.values - cs.values)
       return (np.nanmax(np.abs(diff)) <= 1e-6 * max(initial,1.0))
   except Exception:
# func drawdown
       return False

def _drawdown(series: pd.Series) -> pd.Series:
   cmax = series.cummax()
   return (cmax - series)/cmax



# func fetch rf and vix


def fetch_rf_and_vix(idx: pd.DatetimeIndex) -> Tuple[pd.DataFrame, int, int]:
  """Fetch risk-free rate and VIX only for available trading days"""
  df = pd.DataFrame(index=idx)
  df['rf'] = 0.0
  df['vix'] = np.nan

  if len(idx) == 0:
     return df, 0, 0

  start_date = str(idx.min().date())
  end_date = str(idx.max().date())


  rf = None
  rf_fallback = False
  if _PDR_OK and pdr is not None:
     try:
          rf1 = pdr.DataReader('DGS1MO', 'fred', start_date, end_date)
          rf = rf1['DGS1MO']/100.0
     except Exception:
          try:
               rf3 = pdr.DataReader('DGS3MO', 'fred', start_date, end_date)
               rf = rf3['DGS3MO']/100.0
          except Exception:
               rf = None
               rf_fallback = True
  else:
     rf_fallback = True
  if rf is None:
     rf = pd.Series(0.0, index=pd.date_range(start_date, end_date, freq='D'))


  vix = None
  vix_fallback = False
  if _PDR_OK and pdr is not None:
     try:
          vix = pdr.DataReader('VIXCLS', 'fred', start_date, end_date)['VIXCLS' ]
     except Exception:
          vix = None
          vix_fallback = True
  if vix is None:
       vix = pd.Series(np.nan, index=pd.date_range(start_date, end_date, freq=' D'))

   out = pd.DataFrame(index=pd.date_range(start_date, end_date, freq='D'))
   out['rf'] = rf.reindex(out.index).ffill()
   out['vix'] = vix.reindex(out.index)
   out = out.reindex(idx).ffill()

   rf_fallback_days = len(out) if rf_fallback else 0
   vix_nan_days = out['vix'].isna().sum()

   if rf_fallback:
       print("RF fetch failed, using fallback 0 for all days.")
   if vix_fallback:
       print("VIX fetch failed, using NaN fallback.")

   return out, rf_fallback_days, vix_nan_days



# func bs price


def bs_price(S, K, T, sig, cp, r):
   S = float(S); K = float(K); T = float(T); sig = float(sig); r = float(r)
   if T <= 0.0 or sig <= 0.0:
       return max(0.0, S-K) if cp == "c" else max(0.0, K-S)
   d1 = (log(S/K) + (r + 0.5*sig*sig)*T)/(sig*sqrt(T))
   d2 = d1 - sig*sqrt(T)
   if cp == "c":
       return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
# func bs delta
   return K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def bs_delta(S, K, T, sig, cp, r):
   S = float(S); K = float(K); T = float(T); sig = float(sig); r = float(r)
   if T <= 0.0 or sig <= 0.0: return 0.0
   d1 = (log(S/K) + (r + 0.5*sig*sig)*T)/(sig*sqrt(T))
# func strike from delta
   return norm.cdf(d1) if cp == "c" else -norm.cdf(-d1)

def strike_from_delta(S, T, sig, target_delta, cp, r):
   S = float(S); T = float(T); sig = float(sig); r = float(r)
   if cp == 'c':
       if not (0.0 < target_delta < 1.0): raise ValueError("Call delta must be (0,1)")
       p = target_delta
   else:
       if not (-1.0 < target_delta < 0.0): raise ValueError("Put delta must be (-1,0)")
       p = 1.0 + target_delta
   d1 = norm.ppf(p)
   return S * math.exp(-d1*sig*sqrt(T) + (r + 0.5*sig*sig)*T)





CALL_DELTAS = [10,20,30,40,50]
# func get iv for strike
PUT_DELTAS = [-90,-80,-70,-60,-50,-40,-30,-20,-10]

def get_iv_for_strike(K: float, S: float, iv_df: pd.DataFrame, date: pd. Timestamp,
                      front_t: int, is_call: bool, rate: float) -> float:
   T_surface = front_t / CALENDAR_YEAR
   pts = []
   if is_call:
       for dd in CALL_DELTAS:
           col = f"{front_t}_{dd}"
           sig = max(VOL_FLOOR, _cell(iv_df, date, col))
           d1 = norm.ppf(dd/100.0)
           Kx = S * math.exp(-d1*sig*sqrt(T_surface) + (rate + 0.5*sig*sig)* T_surface)
           pts.append((Kx, sig))
   else:
       for dd in PUT_DELTAS:
           col = f"{front_t}_{dd}"
           sig = max(VOL_FLOOR, _cell(iv_df, date, col))
           d1 = norm.ppf(1.0 + dd/100.0)
           Kx = S * math.exp(-d1*sig*sqrt(T_surface) + (rate + 0.5*sig*sig)* T_surface)
          pts.append((Kx, sig))
   pts.sort(key=lambda x: x[0])
   Ks, sigs = zip(*pts)
   return float(interp1d(Ks, sigs, kind='linear', fill_value="extrapolate", bounds_error=False)(K))



# func stationary bootstrap summaries


def stationary_bootstrap_summaries(equity: pd.Series, initial: float,
                                 runs=1000, block_mean=10, seed=42) -> pd. DataFrame:
   rets = _pct_change(equity).dropna().values
   if len(rets) == 0:
       return pd.DataFrame()
   n = len(rets)
   rng = np.random.default_rng(seed)
   p = 1.0/max(block_mean,1)

   def _pick_indices():
       idx = np.empty(n, dtype=int)
       idx[0] = rng.integers(0, n)
       for t in range(1, n):
          if rng.random() < p:
              idx[t] = rng.integers(0, n)
          else:
              idx[t] = (idx[t-1] + 1) % n
       return idx

   f, tr, dd, sh = [], [], [], []
   for _ in range(runs):
       idx = _pick_indices()
       boot = rets[idx]
       eq = initial * np.cumprod(1.0 + boot)
       s = pd.Series(eq)
       tot = float(s.iloc[-1]/initial - 1.0)
       ann_ret = (1+tot)**(252.0/n) - 1.0 if n>0 else 0.0
       ann_vol = float(pd.Series(boot).std()*np.sqrt(252)) if n>1 else 0.0
       sharpe = ann_ret/ann_vol if ann_vol>0 else 0.0
        draw = (s.cummax()-s)/s.cummax()
        mdd = float(draw.max()) if len(draw) else 0.0

        f.append(float(s.iloc[-1])); tr.append(tot); dd.append(mdd); sh.append( sharpe)

   def pct(col, p): return float(np.percentile(col, p))

   return pd.DataFrame({
        'metric': ['final_equity','total_return','max_dd','sharpe_0'],
        'p05': [pct(f,5), pct(tr,5), pct(dd,5), pct(sh,5)],
        'p50': [pct(f,50),pct(tr,50),pct(dd,50),pct(sh,50)],
        'p95': [pct(f,95),pct(tr,95),pct(dd,95),pct(sh,95)],
        'runs': [runs]*4,
        'block_mean':[block_mean]*4,
        'seed': [seed]*4
   })



# func performance report


def performance_report(series_equity: pd.Series, initial: float, title: str) -> pd.DataFrame:
   rets = _pct_change(series_equity).dropna()
   total_return = float(series_equity.iloc[-1]/initial - 1.0)
   n = len(rets)
   ann_ret = (1+total_return)**(252/n) - 1 if n>0 else 0.0
   ann_vol = float(rets.std()*np.sqrt(252)) if n>1 else 0.0
   sharpe = ann_ret/ann_vol if ann_vol>0 else 0.0
   cmax = series_equity.cummax(); dd = (cmax-series_equity)/cmax; mdd = float( dd.max()) if len(dd) else 0.0
   dneg = rets[rets<0]; ddev = float(dneg.std()*np.sqrt(252)) if len(dneg)>1 else 0.0
   sortino = ann_ret/ddev if ddev>0 else 0.0
   win = len(rets[rets>0])/n if n>0 else 0.0
   gp = float(rets[rets>0].sum())
   gl = abs(float(rets[rets<0].sum()))
   pf = gp/gl if gl>0 else float('inf')
   var_95 = float(np.percentile(rets,5)) if n>0 else 0.0
   es_95 = float(rets[rets<=var_95].mean()) if n>0 else 0.0

   print("\n"+"="*70); print(f"{title}"); print("="*70)
   print(f"{'Initial Capital:':<35}${initial:>24,.0f}")
   print(f"{'Final Equity:':<35}${series_equity.iloc[-1]:>24,.2f}")
   print(f"{'Total Return:':<35}{total_return:>24.2%}")
   print(f"{'Annualized Return:':<35}{ann_ret:>24.2%}")
   print(f"{'Annualized Volatility:':<35}{ann_vol:>24.2%}")
   print(f"{'Sharpe Ratio (r=0):':<35}{sharpe:>24.2f}")
   print(f"{'Sortino Ratio:':<35}{sortino:>24.2f}")
   print(f"{'Max Drawdown:':<35}{mdd:>24.2%}")
   print(f"{'Win Rate:':<35}{win:>24.2%}")
   print(f"{'Profit Factor:':<35}{pf:>24.2f}")
   print(f"{'Daily 95% VaR:':<35}{var_95:>24.2%}")
   print(f"{'Daily 95% Expected Shortfall:':<35}{es_95:>24.2%}")
   print(f"{'Trading Days:':<35}{n:>24}")
   print(f"{'Years:':<35}{n/252:>24.1f}")
   print("="*70)

   return pd.DataFrame([{
         "Strategy":title,
         "Initial": initial,
         "FinalEquity": float(series_equity.iloc[-1]),
         "TotalReturn": total_return,
         "AnnReturn": ann_ret,
         "AnnVol": ann_vol,
         "Sharpe0": sharpe,
         "Sortino": sortino,
         "MaxDD": mdd,
         "WinRate": win,
         "ProfitFactor": pf,
         "VaR95": var_95,
         "ES95": es_95,
         "TradingDays": n,
         "Years": n/252.0
   }])



# func load spot


def load_spot(start_year: int) -> pd.DataFrame:
   spot = (pd.read_csv(SPOT_CSV, parse_dates=["DlyCalDt"])
             .rename(columns={"DlyCalDt":"date","DlyPrcInd":"spot"})
             .set_index("date"))
   spot = _dedup_index(spot)
   spot['spot'] = pd.to_numeric(spot['spot'], errors='coerce')
   spot = spot[['spot']]
   spot = spot[spot.index >= f"{start_year}-01-01"]
# func load iv real
   return spot

def load_iv_real(front_t: int, start_year: int) -> pd.DataFrame:
   need = [f"{front_t}_{d}" for d in [-90,-80,-70,-60,-50,-40,-30,-20,-10,10,20,30,40,50]]
   ivr = (pd.read_csv(IV_REAL_CSV, parse_dates=["date"]).set_index("date"))
   ivr = _dedup_index(ivr)
   missing = [c for c in need if c not in ivr.columns]
   if missing:
       raise KeyError(f"Realised IV missing columns: {missing}")
   ivr = _to_numeric(ivr[need]).clip(lower=VOL_FLOOR)
   ivr = ivr[ivr.index >= f"{start_year}-01-01"]
# func align trading idx
   return ivr

def align_trading_idx(spot: pd.DataFrame, iv_real: pd.DataFrame) -> Tuple[pd. DataFrame,pd.DataFrame,pd.DatetimeIndex]:
   idx = spot['spot'].dropna().index.intersection(iv_real.dropna().index)
   spot = spot.loc[idx]
   iv_real = iv_real.loc[idx]
   return spot, iv_real, idx




# class CostModel

@dataclass
class CostModel:
   option_comm_bps: float
   option_slip_bps: float
     und_comm_bps: float
     und_slip_bps: float

     def option_cost(self, traded_value: float) -> float:
           bps = self.option_comm_bps + self.option_slip_bps
           return abs(traded_value) * bps

     def underlying_cost(self, traded_notional: float) -> float:
           bps = self.und_comm_bps + self.und_slip_bps
           return abs(traded_notional) * bps



# func run rule strategy


def run_rule_strategy(*,
     front_t=30, DELTA=10, MAX_HOLD=20, NOTIONAL=1_000_000,
     STOP_PCT=0.03, START_YEAR=2004, VIX_HALT=40.0, VIX_RESUME=39.0, COOLDOWN_D =10,
     cost: CostModel = CostModel(0.0050,0.0025, 0.0008,0.0007)
):
     """
     STOP_PCT: PnL-based stop. Exit if cumulative PnL since entry <= -STOP_PCT * NOTIONAL.
     Daily RFR accrual applied while a position is open.
     """
     print("[INIT] RULE module loaded. Dynamic hedge (realised IV). START_YEAR =2004. Dynamic r used.")
     spot = load_spot(START_YEAR)
     iv_real = load_iv_real(front_t, START_YEAR)
     spot, iv_real, dates = align_trading_idx(spot, iv_real)
     rf_vix, rf_fallback_days, vix_nan_days = fetch_rf_and_vix(dates)
     print(f"[FRED] RF fallback days: {rf_fallback_days} | VIX NaN days: { vix_nan_days}")

     T1 = front_t / CALENDAR_YEAR
     pnl, eq, idx = [], [INITIAL_CAPITAL], []
     pos = []
     blocked = False; below39 = 0

  for i, d in enumerate(dates):
     S = _cell(spot, d, "spot")
     r = float(rf_vix.loc[d,'rf']) if 'rf' in rf_vix.columns else 0.0
     v = float(rf_vix.loc[d,'vix']) if 'vix' in rf_vix.columns else np.nan


     if not np.isnan(v):
         if v >= VIX_HALT: blocked = True; below39 = 0
         elif blocked:
            if v < VIX_RESUME:
                below39 += 1
                if below39 >= COOLDOWN_D: blocked = False; below39 = 0
            else:
                below39 = 0


     if not pos:
         if (not blocked) and (np.isnan(v) or v < VIX_HALT):
            sig_c = _cell(iv_real, d, f"{front_t}_{DELTA}")
            sig_p = _cell(iv_real, d, f"{front_t}_{-DELTA}")
            Kc = Kp = S

            prem = bs_price(S,Kc,T1,sig_c,'c',r) + bs_price(S,Kp,T1,sig_p,'p' ,r)
            if np.isfinite(prem) and prem>0:
                units = -NOTIONAL/prem

                d_call = bs_delta(S,Kc,T1,sig_c,'c',r)
                d_put = bs_delta(S,Kp,T1,sig_p,'p',r)
                hedge_target = -units*(d_call + d_put)
                hedge_shares = hedge_target
                entry_opt_cost = cost.option_cost(abs(units)*prem)
                entry_und_cost = cost.underlying_cost(abs(hedge_shares)*S)
                entry_costs = entry_opt_cost + entry_und_cost

                pnl.append(-entry_costs)
                eq.append(eq[-1] - entry_costs); idx.append(d)

                pos = [{
                      "Kc":Kc,"Kp":Kp,"T":T1,"units":units,
                      "V0": -units*prem,
                   "hedge_shares": hedge_shares,
                   "prev_S": S, "entry_S": S, "entry_day": d, "prev_d": d,
                   "pnl_since_entry": -entry_costs
              }]
         continue
     continue


  S_now = S
  r_now = r
  days_elapsed = (d - pos[0]["prev_d"]).days if i > 0 else 0
  T_rem = max(0.0, pos[0]["T"] - days_elapsed / CALENDAR_YEAR)


  sig_c_now = get_iv_for_strike(pos[0]["Kc"], S_now, iv_real, d, front_t, True, r_now)
  sig_p_now = get_iv_for_strike(pos[0]["Kp"], S_now, iv_real, d, front_t, False, r_now)
  prem_now = bs_price(S_now,pos[0]["Kc"],T_rem,sig_c_now,'c',r_now) +\
              bs_price(S_now,pos[0]["Kp"],T_rem,sig_p_now,'p',r_now)
  price_now = -pos[0]["units"] * prem_now

  option_pnl = pos[0]["V0"] - price_now
  dS = S_now - pos[0]["prev_S"]
  hedge_pnl = pos[0]["hedge_shares"] * dS


  d_call_h = bs_delta(S_now,pos[0]["Kc"],T_rem,sig_c_now,'c',r_now)
  d_put_h = bs_delta(S_now,pos[0]["Kp"],T_rem,sig_p_now,'p',r_now)
  target = -pos[0]["units"] * (d_call_h + d_put_h)

  hedge_new = (1.0 - HEDGE_SMOOTH_ALPHA)*pos[0]["hedge_shares"] + HEDGE_SMOOTH_ALPHA*target
  d_shares = hedge_new - pos[0]["hedge_shares"]
  max_turn_notional = MAX_HEDGE_TURNOVER_X * INITIAL_CAPITAL
  traded_notional = abs(d_shares) * S_now
  if traded_notional > max_turn_notional and traded_notional > 0:
     scale = max_turn_notional / traded_notional
     d_shares *= scale
     hedge_new = pos[0]["hedge_shares"] + d_shares
     traded_notional = abs(d_shares) * S_now

     rehedge_cost = cost.underlying_cost(traded_notional)


     rf_day = eq[-1] * (r_now / CALENDAR_YEAR) * days_elapsed


     pnl_day_core = option_pnl + hedge_pnl - rehedge_cost + rf_day
     pnl_tmp = pos[0]["pnl_since_entry"] + pnl_day_core


     days_live = (d - pos[0]["entry_day"]).days
     force_vix = (not np.isnan(v)) and (v >= VIX_HALT)
     final_day = (i == len(dates)-1)
     stop_loss = (pnl_tmp <= -STOP_PCT * NOTIONAL)
     will_exit = force_vix or stop_loss or (days_live>=MAX_HOLD) or final_day

     pnl_day = pnl_day_core
     if will_exit:
         exit_opt_cost = cost.option_cost(abs(pos[0]["units"])*prem_now)
         exit_und_cost = cost.underlying_cost(abs(hedge_new)*S_now)
         pnl_day -= (exit_opt_cost + exit_und_cost)

     eq.append(eq[-1] + pnl_day); pnl.append(pnl_day); idx.append(d)

     pos[0]["V0"] = price_now
     pos[0]["prev_S"] = S_now
     pos[0]["prev_d"] = d
     pos[0]["T"] = T_rem
     pos[0]["hedge_shares"] = hedge_new
     pos[0]["pnl_since_entry"] = pnl_tmp if not will_exit else 0.0

     if will_exit:
         pos.clear()
         if force_vix: blocked = True; below39 = 0

  res = pd.DataFrame({"pnl": pnl, "equity": eq[1:]}, index=idx)
  res['cum_pnl'] = res['equity']
  res['dd_pct'] = _drawdown(res['equity'])
  res['ret'] = _pct_change(res['equity'])

     ok = _reconcile_equity(res['equity'], res['pnl'], INITIAL_CAPITAL)
     print(f"\n[SANITY][RULE] (recon from equity) {'PASS' if ok else 'FAIL'}.\n")
     return res



# func load iv attn


def _load_iv_attn(front_t: int, start_year: int, trading_idx: pd.DatetimeIndex) -> pd.DataFrame:
     need = [f"{front_t}_-10", f"{front_t}_10"]
     attn = (pd.read_csv(IV_ATTN_CSV, parse_dates=["date"]).set_index("date"))
     attn = _dedup_index(attn)
     missing = [c for c in need if c not in attn.columns]
     if missing:
           raise KeyError(f"ATTN predicted IV missing columns: {missing}")
     attn = _to_numeric(attn[need]).clip(lower=VOL_FLOOR)
     attn = attn[attn.index >= f"{start_year}-01-01"]
     attn = attn.reindex(trading_idx).ffill()
# func run attn strategy
     return attn

def run_attn_strategy(*,
     front_t=30, DELTA=10, MAX_HOLD=20, NOTIONAL=1_000_000,
     STOP_PCT=0.03, START_YEAR=2004, VIX_HALT=40.0, VIX_RESUME=39.0, COOLDOWN_D =10,
     cost: CostModel = CostModel(0.0050,0.0025, 0.0008,0.0007)
):
     """
     STOP_PCT: PnL-based stop. Exit if cumulative PnL since entry <= -STOP_PCT * NOTIONAL.
     Daily RFR accrual applied while a position is open.
     """
     print("[INIT] ATTN module loaded. Dynamic hedge (pred IV; fallback realised) . START_YEAR=2004. Dynamic r used.")
     spot = load_spot(START_YEAR)
     iv_real = load_iv_real(front_t, START_YEAR)
     spot, iv_real, dates = align_trading_idx(spot, iv_real)
     rf_vix, rf_fallback_days, vix_nan_days = fetch_rf_and_vix(dates)
  print(f"[FRED] RF fallback days: {rf_fallback_days} | VIX NaN days: { vix_nan_days}")
  iv_attn = _load_iv_attn(front_t, START_YEAR, dates)

  T1 = front_t / CALENDAR_YEAR
  pnl, eq, idx = [], [INITIAL_CAPITAL], []
  pos = []
  blocked = False; below39 = 0

  for i, d in enumerate(dates):
     S = _cell(spot, d, "spot")
     r = float(rf_vix.loc[d,'rf']) if 'rf' in rf_vix.columns else 0.0
     v = float(rf_vix.loc[d,'vix']) if 'vix' in rf_vix.columns else np.nan


     if not np.isnan(v):
         if v >= VIX_HALT: blocked = True; below39 = 0
         elif blocked:
             if v < VIX_RESUME:
                below39 += 1
                if below39 >= COOLDOWN_D: blocked = False; below39 = 0
             else:
                below39 = 0


     if not pos:
         if (not blocked) and (np.isnan(v) or v < VIX_HALT):
             sig_c_r = _cell(iv_real, d, f"{front_t}_{DELTA}")
             sig_p_r = _cell(iv_real, d, f"{front_t}_{-DELTA}")
             Kc = Kp = S

             prem = bs_price(S,Kc,T1,sig_c_r,'c',r) + bs_price(S,Kp,T1,sig_p_r ,'p',r)
             if np.isfinite(prem) and prem>0:
                units = -NOTIONAL/prem


                sig_c_p = _cell(iv_attn, d, f"{front_t}_{DELTA}")
                sig_p_p = _cell(iv_attn, d, f"{front_t}_{-DELTA}")
                if not np.isfinite(sig_c_p) or not np.isfinite(sig_p_p):
                     sig_c_p, sig_p_p = sig_c_r, sig_p_r

              d_call_p = bs_delta(S,Kc,T1,sig_c_p,'c',r)
              d_put_p = bs_delta(S,Kp,T1,sig_p_p,'p',r)
              hedge_target = -units*(d_call_p+d_put_p)
              hedge_shares = hedge_target

              entry_opt_cost = cost.option_cost(abs(units)*prem)
              entry_und_cost = cost.underlying_cost(abs(hedge_shares)*S)
              entry_costs = entry_opt_cost + entry_und_cost

              pnl.append(-entry_costs)
              eq.append(eq[-1] - entry_costs); idx.append(d)

              pos = [{
                   "Kc":Kc,"Kp":Kp,"T":T1,"units":units,
                   "V0": -units*prem,
                   "hedge_shares": hedge_shares,
                   "prev_S": S, "entry_S": S, "entry_day": d, "prev_d": d,
                   "pnl_since_entry": -entry_costs
              }]
         continue
     continue


  S_now = S
  r_now = r
  days_elapsed = (d - pos[0]["prev_d"]).days if i > 0 else 0
  T_rem = max(0.0, pos[0]["T"] - days_elapsed / CALENDAR_YEAR)


  sig_c_now_r = get_iv_for_strike(pos[0]["Kc"], S_now, iv_real, d, front_t , True, r_now)
  sig_p_now_r = get_iv_for_strike(pos[0]["Kp"], S_now, iv_real, d, front_t , False, r_now)
  prem_now = bs_price(S_now,pos[0]["Kc"],T_rem,sig_c_now_r,'c',r_now) +\
                bs_price(S_now,pos[0]["Kp"],T_rem,sig_p_now_r,'p',r_now)
  price_now = -pos[0]["units"]*prem_now

  option_pnl = pos[0]["V0"] - price_now
  dS = S_now - pos[0]["prev_S"]
  hedge_pnl = pos[0]["hedge_shares"] * dS


  sig_c_p = _cell(iv_attn, d, f"{front_t}_{DELTA}")
  sig_p_p = _cell(iv_attn, d, f"{front_t}_{-DELTA}")
  if not np.isfinite(sig_c_p) or not np.isfinite(sig_p_p):
     sig_c_p, sig_p_p = sig_c_now_r, sig_p_now_r

  d_call_h = bs_delta(S_now,pos[0]["Kc"],T_rem,sig_c_p,'c',r_now)
  d_put_h = bs_delta(S_now,pos[0]["Kp"],T_rem,sig_p_p,'p',r_now)
  target = -pos[0]["units"] * (d_call_h + d_put_h)

  hedge_new = (1.0 - HEDGE_SMOOTH_ALPHA)*pos[0]["hedge_shares"] + HEDGE_SMOOTH_ALPHA*target
  d_shares = hedge_new - pos[0]["hedge_shares"]
  max_turn_notional = MAX_HEDGE_TURNOVER_X * INITIAL_CAPITAL
  traded_notional = abs(d_shares) * S_now
  if traded_notional > max_turn_notional and traded_notional > 0:
     scale = max_turn_notional / traded_notional
     d_shares *= scale
     hedge_new = pos[0]["hedge_shares"] + d_shares
     traded_notional = abs(d_shares) * S_now

  rehedge_cost = cost.underlying_cost(traded_notional)


  rf_day = eq[-1] * (r_now / CALENDAR_YEAR) * days_elapsed


  pnl_day_core = option_pnl + hedge_pnl - rehedge_cost + rf_day
  pnl_tmp = pos[0]["pnl_since_entry"] + pnl_day_core


  days_live = (d - pos[0]["entry_day"]).days
  force_vix = (not np.isnan(v)) and (v >= VIX_HALT)
  final_day = (i == len(dates)-1)
  stop_loss = (pnl_tmp <= -STOP_PCT * NOTIONAL)
  will_exit = force_vix or stop_loss or (days_live>=MAX_HOLD) or final_day

  pnl_day = pnl_day_core
  if will_exit:
     exit_opt_cost = cost.option_cost(abs(pos[0]["units"])*prem_now)
            exit_und_cost = cost.underlying_cost(abs(hedge_new)*S_now)
            pnl_day -= (exit_opt_cost + exit_und_cost)

         eq.append(eq[-1] + pnl_day); pnl.append(pnl_day); idx.append(d)

         pos[0]["V0"] = price_now
         pos[0]["prev_S"] = S_now
         pos[0]["prev_d"] = d
         pos[0]["T"] = T_rem
         pos[0]["hedge_shares"] = hedge_new
         pos[0]["pnl_since_entry"] = pnl_tmp if not will_exit else 0.0

         if will_exit:
            pos.clear()
            if force_vix: blocked = True; below39 = 0

   res = pd.DataFrame({"pnl": pnl, "equity": eq[1:]}, index=idx)
   res['cum_pnl'] = res['equity']
   res['dd_pct'] = _drawdown(res['equity'])
   res['ret'] = _pct_change(res['equity'])

   ok = _reconcile_equity(res['equity'], res['pnl'], INITIAL_CAPITAL)
   print(f"\n[SANITY][ATTN] (recon from equity) {'PASS' if ok else 'FAIL'}.\n")
   return res



# func conv preds wide or long to wide


def conv_preds_wide_or_long_to_wide(raw: pd.DataFrame, iv_real_cols, step_ahead =20) -> pd.DataFrame:
   """
   Accept either:
      wide: already date-indexed with '30_-10','30_10', ...
      long: {'date' or 'origin_date','maturity_bucket','delta_bucket',' step_ahead','iv_pred'/'iv_pred_raw'}
   Return wide (index=date, columns matching iv_real_cols subset).
   """
   if {'maturity_bucket','delta_bucket','step_ahead'}.issubset(raw.columns):
         if 'date' in raw.columns:
          raw['date'] = pd.to_datetime(raw['date'])
     elif 'origin_date' in raw.columns:
          raw['date'] = pd.to_datetime(raw['origin_date'])
     else:
          raise ValueError("Conv preds long-format needs 'date' or 'origin_date '.")

     raw = raw[raw['step_ahead'] == step_ahead].copy()
     valcol = 'iv_pred_raw' if 'iv_pred_raw' in raw.columns else ('iv_pred' if 'iv_pred' in raw.columns else None)
     if valcol is None:
          raise ValueError("Conv preds need an 'iv_pred' or 'iv_pred_raw' column.")

     mats = sorted({int(c.split('_')[0]) for c in iv_real_cols})
     dels = sorted({int(c.split('_')[1]) for c in iv_real_cols})

     def _infer_map(series, universe_sorted):
          series = series.astype(int)
          k = series.max()+1
          if k == len(universe_sorted) and set(series.unique()) <= set(range(k) ):
             return {i: universe_sorted[i] for i in range(k)}
          uniq = sorted(series.unique())
          if set(uniq) == set(range(23)):
             return {i:(i-11)*5 for i in range(23)}
          if set(uniq) == {0,1,2}:
             return {0:-10,1:0,2:10}
          return {b: universe_sorted[i%len(universe_sorted)] for i,b in enumerate(uniq)}

     m_map = _infer_map(raw['maturity_bucket'], mats)
     d_map = _infer_map(raw['delta_bucket'], dels)
     raw['grid_col'] = raw.apply(lambda r: f"{m_map[int(r.maturity_bucket)]}_ {d_map[int(r.delta_bucket)]}", axis=1)
     raw = raw.sort_values(['date','maturity_bucket','delta_bucket']). drop_duplicates(['date','grid_col'], keep='last')
     wide = raw.pivot(index='date', columns='grid_col', values=valcol). sort_index()
     return wide
  else:
           if 'date' in raw.columns:
                raw = raw.assign(date=pd.to_datetime(raw['date'])).set_index('date')
                raw = raw.drop(columns=['date'], errors='ignore')
# func load iv conv
           return raw

def _load_iv_conv(front_t: int, start_year: int, trading_idx: pd.DatetimeIndex, iv_real_cols) -> pd.DataFrame:
     need = [f"{front_t}_-10", f"{front_t}_10"]
     raw = pd.read_csv(IV_CONV_CSV)
     if 'date' in raw.columns or {'maturity_bucket','delta_bucket','step_ahead'}. issubset(raw.columns):
           wide = conv_preds_wide_or_long_to_wide(raw, iv_real_cols, step_ahead=20)
     else:
           wide = raw

     if 'date' in wide.columns:
           wide = wide.assign(date=pd.to_datetime(wide['date'])).set_index('date'). drop(columns=['date'], errors='ignore')

     missing = [c for c in need if c not in wide.columns]
     if missing:
           raise KeyError(f"CONV-LSTM predicted IV missing columns: {missing}")

     wide = _to_numeric(wide[need]).clip(lower=VOL_FLOOR)
     wide = wide[wide.index >= f"{start_year}-01-01"]
     wide = wide.reindex(trading_idx).ffill()

     usable = wide[need].notna().all(axis=1).sum()
     print(f"[CONV] usable days with both 10 preds: {usable} / {len(trading_idx)} ")
# func run conv strategy
     return wide

def run_conv_strategy(*,
     front_t=30, DELTA=10, MAX_HOLD=20, NOTIONAL=1_000_000,
     STOP_PCT=0.03, START_YEAR=2004, VIX_HALT=40.0, VIX_RESUME=39.0, COOLDOWN_D =10,
     cost: CostModel = CostModel(0.0050,0.0025, 0.0008,0.0007)
):
     """
  STOP_PCT: PnL-based stop. Exit if cumulative PnL since entry <= -STOP_PCT * NOTIONAL.
  Daily RFR accrual applied while a position is open.
  """
  print("[INIT] CONV module loaded. Dynamic hedge (pred IV; fallback realised) . START_YEAR=2004. Dynamic r used.")
  spot = load_spot(START_YEAR)
  iv_real = load_iv_real(front_t, START_YEAR)
  spot, iv_real, dates = align_trading_idx(spot, iv_real)
  rf_vix, rf_fallback_days, vix_nan_days = fetch_rf_and_vix(dates)
  print(f"[FRED] RF fallback days: {rf_fallback_days} | VIX NaN days: { vix_nan_days}")
  iv_conv = _load_iv_conv(front_t, START_YEAR, dates, iv_real.columns)

  T1 = front_t / CALENDAR_YEAR
  pnl, eq, idx = [], [INITIAL_CAPITAL], []
  pos = []
  blocked = False; below39 = 0

  for i, d in enumerate(dates):
        S = _cell(spot, d, "spot")
        r = float(rf_vix.loc[d,'rf']) if 'rf' in rf_vix.columns else 0.0
        v = float(rf_vix.loc[d,'vix']) if 'vix' in rf_vix.columns else np.nan


        if not np.isnan(v):
           if v >= VIX_HALT: blocked = True; below39 = 0
           elif blocked:
               if v < VIX_RESUME:
                    below39 += 1
                    if below39 >= COOLDOWN_D: blocked = False; below39 = 0
               else:
                    below39 = 0


        if not pos:
           if (not blocked) and (np.isnan(v) or v < VIX_HALT):
               sig_c_r = _cell(iv_real, d, f"{front_t}_{DELTA}")
               sig_p_r = _cell(iv_real, d, f"{front_t}_{-DELTA}")
               Kc = Kp = S

         prem = bs_price(S,Kc,T1,sig_c_r,'c',r) + bs_price(S,Kp,T1,sig_p_r ,'p',r)
         if np.isfinite(prem) and prem>0:

              sig_c_p = _cell(iv_conv, d, f"{front_t}_{DELTA}") if d in iv_conv.index else np.nan
              sig_p_p = _cell(iv_conv, d, f"{front_t}_{-DELTA}") if d in iv_conv.index else np.nan
              if not np.isfinite(sig_c_p) or not np.isfinite(sig_p_p):
                   sig_c_p, sig_p_p = sig_c_r, sig_p_r

              units = -NOTIONAL/prem
              d_call_p = bs_delta(S,Kc,T1,sig_c_p,'c',r)
              d_put_p = bs_delta(S,Kp,T1,sig_p_p,'p',r)
              hedge_target = -units*(d_call_p+d_put_p)
              hedge_shares = hedge_target

              entry_opt_cost = cost.option_cost(abs(units)*prem)
              entry_und_cost = cost.underlying_cost(abs(hedge_shares)*S)
              entry_costs = entry_opt_cost + entry_und_cost

              pnl.append(-entry_costs)
              eq.append(eq[-1] - entry_costs); idx.append(d)

              pos = [{
                   "Kc":Kc,"Kp":Kp,"T":T1,"units":units,
                   "V0": -units*prem,
                   "hedge_shares": hedge_shares,
                   "prev_S": S, "entry_S": S, "entry_day": d, "prev_d": d,
                   "pnl_since_entry": -entry_costs
              }]
         continue
     continue


  S_now = S
  r_now = r
  days_elapsed = (d - pos[0]["prev_d"]).days if i > 0 else 0
  T_rem = max(0.0, pos[0]["T"] - days_elapsed / CALENDAR_YEAR)


  sig_c_now_r = get_iv_for_strike(pos[0]["Kc"], S_now, iv_real, d, front_t , True, r_now)
  sig_p_now_r = get_iv_for_strike(pos[0]["Kp"], S_now, iv_real, d, front_t , False, r_now)
  prem_now = bs_price(S_now,pos[0]["Kc"],T_rem,sig_c_now_r,'c',r_now) +\
              bs_price(S_now,pos[0]["Kp"],T_rem,sig_p_now_r,'p',r_now)
  price_now = -pos[0]["units"]*prem_now

  option_pnl = pos[0]["V0"] - price_now
  dS = S_now - pos[0]["prev_S"]
  hedge_pnl = pos[0]["hedge_shares"] * dS


  sig_c_p = _cell(iv_conv, d, f"{front_t}_{DELTA}") if d in iv_conv.index else np.nan
  sig_p_p = _cell(iv_conv, d, f"{front_t}_{-DELTA}") if d in iv_conv.index else np.nan
  if not np.isfinite(sig_c_p) or not np.isfinite(sig_p_p):
     sig_c_p, sig_p_p = sig_c_now_r, sig_p_now_r

  d_call_h = bs_delta(S_now,pos[0]["Kc"],T_rem,sig_c_p,'c',r_now)
  d_put_h = bs_delta(S_now,pos[0]["Kp"],T_rem,sig_p_p,'p',r_now)
  target = -pos[0]["units"] * (d_call_h + d_put_h)

  hedge_new = (1.0 - HEDGE_SMOOTH_ALPHA)*pos[0]["hedge_shares"] + HEDGE_SMOOTH_ALPHA*target
  d_shares = hedge_new - pos[0]["hedge_shares"]
  max_turn_notional = MAX_HEDGE_TURNOVER_X * INITIAL_CAPITAL
  traded_notional = abs(d_shares) * S_now
  if traded_notional > max_turn_notional and traded_notional > 0:
     scale = max_turn_notional / traded_notional
     d_shares *= scale
     hedge_new = pos[0]["hedge_shares"] + d_shares
     traded_notional = abs(d_shares) * S_now

  rehedge_cost = cost.underlying_cost(traded_notional)


  rf_day = eq[-1] * (r_now / CALENDAR_YEAR) * days_elapsed


       pnl_day_core = option_pnl + hedge_pnl - rehedge_cost + rf_day
       pnl_tmp = pos[0]["pnl_since_entry"] + pnl_day_core


       days_live = (d - pos[0]["entry_day"]).days
       force_vix = (not np.isnan(v)) and (v >= VIX_HALT)
       final_day = (i == len(dates)-1)
       stop_loss = (pnl_tmp <= -STOP_PCT * NOTIONAL)
       will_exit = force_vix or stop_loss or (days_live>=MAX_HOLD) or final_day

       pnl_day = pnl_day_core
       if will_exit:
          exit_opt_cost = cost.option_cost(abs(pos[0]["units"])*prem_now)
          exit_und_cost = cost.underlying_cost(abs(hedge_new)*S_now)
          pnl_day -= (exit_opt_cost + exit_und_cost)

       eq.append(eq[-1] + pnl_day); pnl.append(pnl_day); idx.append(d)

       pos[0]["V0"] = price_now
       pos[0]["prev_S"] = S_now
       pos[0]["prev_d"] = d
       pos[0]["T"] = T_rem
       pos[0]["hedge_shares"] = hedge_new
       pos[0]["pnl_since_entry"] = pnl_tmp if not will_exit else 0.0

       if will_exit:
          pos.clear()
          if force_vix: blocked = True; below39 = 0

   res = pd.DataFrame({"pnl": pnl, "equity": eq[1:]}, index=idx)
   res['cum_pnl'] = res['equity']
   res['dd_pct'] = _drawdown(res['equity'])
   res['ret'] = _pct_change(res['equity'])

   ok = _reconcile_equity(res['equity'], res['pnl'], INITIAL_CAPITAL)
   print(f"\n[SANITY][CONV] (recon from equity) {'PASS' if ok else 'FAIL'}.\n")
   return res



# func save results and reports


def save_results_and_reports(res_df: pd.DataFrame, *, name_prefix: str,
                            title: str, tag: str, initial: float):
   """Emit results.csv + metrics.csv + mc.csv with tag suffix (RAW or COMMON)." ""
   out_csv = f"{ROOT}/{name_prefix}_results_{tag}.csv"
   res_df.to_csv(out_csv)
   metrics = performance_report(res_df['equity'], initial, title=f"{title} [{ tag}]")
   metrics_csv = f"{ROOT}/{name_prefix}_metrics_{tag}.csv"
   metrics.to_csv(metrics_csv, index=False)
   mc = stationary_bootstrap_summaries(res_df['equity'], initial, runs=1000, block_mean=10, seed=42)
   if mc is not None and not mc.empty:
       mc_csv = f"{ROOT}/{name_prefix}_mc_{tag}.csv"
       mc.to_csv(mc_csv, index=False)
       print(f"Saved MC summary: {mc_csv}")
   print(f"Saved results: {out_csv}")
# func plot equity dd single
   print(f"Saved metrics: {metrics_csv}")

def plot_equity_dd_single(df: pd.DataFrame, title: str, out_png: str):
   """Single strategy: equity curve + drawdown."""
   plt.style.use("ggplot")
   fig, (ax1, ax2) = plt.subplots(2,1, figsize=(16,10), sharex=True, gridspec_kw={"height_ratios":[3,1]})

   ax1.plot(df.index, df['equity'], linewidth=2, label=f"{title} (Final: ${df[' equity'].iloc[-1]:,.0f})")
   ax1.axhline(INITIAL_CAPITAL, linestyle='--', alpha=0.6, color='gray')
   ax1.set_title(title); ax1.set_ylabel("Equity ($)")
   ax1.grid(True, linestyle="--", alpha=0.7); ax1.legend(loc="best")

   dd = (df['equity'].cummax() - df['equity'])/df['equity'].cummax()
   ax2.fill_between(df.index, dd*100.0, 0, alpha=0.3, label=f"Max DD: {dd.max() *100:.1f}%")
   ax2.set_ylabel("Drawdown (%)"); ax2.grid(True, linestyle="--", alpha=0.7); ax2.legend(loc="best")
   ax2.xaxis.set_major_locator(mdates.YearLocator()); ax2.xaxis. set_major_formatter(mdates.DateFormatter("%Y"))
   plt.xlabel("Date"); plt.tight_layout(); plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.show()
# func plot combined common
   print(f"Saved plot    {out_png}")

def plot_combined_common(curves: Dict[str,pd.DataFrame], title: str, out_png: str):
   """Combined plot on common window (all series are already common & rebased). """
   plt.style.use("ggplot")
   fig, (ax1, ax2) = plt.subplots(2,1, figsize=(16,10), sharex=True, gridspec_kw={"height_ratios":[3,1]})

   for label, df in curves.items():
       ax1.plot(df.index, df['equity'], linewidth=2, label=f"{label} (Final: ${ df['equity'].iloc[-1]:,.0f})")
   ax1.axhline(INITIAL_CAPITAL, linestyle='--', alpha=0.6, color='gray')
   ax1.set_title(f"{title}    Combined (Strict Common Window)")
   ax1.set_ylabel("Equity ($)"); ax1.grid(True, linestyle="--", alpha=0.7); ax1 .legend(loc="best")

   for label, df in curves.items():
       dd = (df['equity'].cummax() - df['equity'])/df['equity'].cummax()
       ax2.plot(df.index, dd*100.0, linewidth=1.8, label=f"{label} DD (max {dd. max()*100:.1f}%)")
   ax2.set_ylabel("Drawdown (%)"); ax2.grid(True, linestyle="--", alpha=0.7); ax2.legend(loc="best")
   ax2.xaxis.set_major_locator(mdates.YearLocator()); ax2.xaxis. set_major_formatter(mdates.DateFormatter("%Y"))
   plt.xlabel("Date"); plt.tight_layout(); plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.show()
   print(f"Saved plot    {out_png}")



# func enforce common and rebase


def enforce_common_and_rebase(rule_res: pd.DataFrame,
                             attn_res: pd.DataFrame,
                             conv_res: pd.DataFrame,
                             initial: float = INITIAL_CAPITAL):
   """Intersect indices and REBASE equity so equity = initial + cumsum(pnl) within the window."""
   common_idx = rule_res.index.intersection(attn_res.index).intersection( conv_res.index)
   if common_idx.empty:
       raise RuntimeError("Common date intersection is empty      check your inputs .")

   def _view(df):
       d = df.loc[common_idx].copy()

       d['equity'] = initial + d['pnl'].cumsum()
       d['cum_pnl'] = d['equity']
       d['dd_pct'] = _drawdown(d['equity'])
       d['ret'] = _pct_change(d['equity'])
       return d

   return _view(rule_res), _view(attn_res), _view(conv_res), common_idx





if __name__ == "__main__":

   rule_raw = run_rule_strategy(START_YEAR=2004)
   attn_raw = run_attn_strategy(START_YEAR=2004)
   conv_raw = run_conv_strategy(START_YEAR=2004)


   save_results_and_reports(rule_raw,
       name_prefix="straddle_rule", title="RULE-BASED Short Straddle     Dynamic Hedge (dynamic r)",
       tag="RAW", initial=INITIAL_CAPITAL
   )
   save_results_and_reports(attn_raw,
       name_prefix="straddle_attn_hedgeonly", title="ATTN-LSTM Hedge-     Dynamic Hedge (dynamic r)",
       tag="RAW", initial=INITIAL_CAPITAL
   )
  save_results_and_reports(conv_raw,
      name_prefix="straddle_convlstm_hedgeonly", title="CONV-LSTM Hedge- Dynamic Hedge (dynamic r)",
      tag="RAW", initial=INITIAL_CAPITAL
  )


  rule_c, attn_c, conv_c, common_idx = enforce_common_and_rebase(rule_raw, attn_raw, conv_raw, INITIAL_CAPITAL)
  print(f"[COMMON] Window: {common_idx.min().date()}   {common_idx.max().date() } | days: {len(common_idx)}")


  for name, df in [("RULE", rule_c), ("ATTN", attn_c), ("CONV", conv_c)]:
      ok = _reconcile_equity(df['equity'], df['pnl'], INITIAL_CAPITAL)
      print(f"[SANITY][{name}][COMMON] (recon) {'PASS' if ok else 'FAIL'}")


  save_results_and_reports(rule_c,
      name_prefix="straddle_rule", title="RULE-BASED Short Straddle    Dynamic Hedge (dynamic r)",
      tag="COMMON", initial=INITIAL_CAPITAL
  )
  save_results_and_reports(attn_c,
      name_prefix="straddle_attn_hedgeonly", title="ATTN-LSTM Hedge-   Dynamic Hedge (dynamic r)",
      tag="COMMON", initial=INITIAL_CAPITAL
  )
  save_results_and_reports(conv_c,
      name_prefix="straddle_convlstm_hedgeonly", title="CONV-LSTM Hedge- Dynamic Hedge (dynamic r)",
      tag="COMMON", initial=INITIAL_CAPITAL
  )


  plot_equity_dd_single(rule_raw, "Rule-based Short Straddle (2004+; RAW window)",
                      os.path.join(ROOT, f"{FAMILY_NAME}_rule_equity_dd_RAW. png"))
  plot_equity_dd_single(attn_raw, "ATTN-LSTM Hedge- (2004+; RAW window)",
                        os.path.join(ROOT, f"{FAMILY_NAME}_attn_equity_dd_RAW. png"))
  plot_equity_dd_single(conv_raw, "CONV-LSTM Hedge- (2004+; RAW window)",
                        os.path.join(ROOT, f"{FAMILY_NAME}_conv_equity_dd_RAW. png"))


  plot_equity_dd_single(rule_c, "Rule-based Short Straddle (COMMON window)",
                        os.path.join(ROOT, f"{FAMILY_NAME}_rule_equity_dd_COMMON .png"))
  plot_equity_dd_single(attn_c, "ATTN-LSTM Hedge- (COMMON window)",
                        os.path.join(ROOT, f"{FAMILY_NAME}_attn_equity_dd_COMMON .png"))
  plot_equity_dd_single(conv_c, "CONV-LSTM Hedge- (COMMON window)",
                        os.path.join(ROOT, f"{FAMILY_NAME}_conv_equity_dd_COMMON .png"))



  plot_equity_dd_single(rule_c, "RULE-BASED Strategy", f"{ROOT}/{FAMILY_NAME} _rule_equity_dd.png")
  plot_equity_dd_single(attn_c, "ATTN-LSTM Strategy", f"{ROOT}/{FAMILY_NAME} _attn_equity_dd.png")
  plot_equity_dd_single(conv_c, "CONV-LSTM Strategy", f"{ROOT}/{FAMILY_NAME} _conv_equity_dd.png")


  curves = {
      "RULE": rule_c,
      "ATTN-LSTM": attn_c,
      "CONV-LSTM": conv_c
  }
  if benchmark_equity is not None:
      curves["SP500TR"] = benchmark_equity
      plot_combined_common(curves, "Short 30d 30 Strangle (dynamic r + rf accrual)", os.path.join(ROOT, f"{FAMILY_NAME}_combined_RAW.png"))



  plot_combined_common({
      "Rule": rule_c,
       "ATTN-LSTM": attn_c,
       "CONV-LSTM": conv_c
   }, "Short 30d ATM Straddle (dynamic r + rf accrual + pnl stop)", os.path. join(ROOT, f"{FAMILY_NAME}_combined_COMMON.png"))


   performance_report(rule_c['equity'], INITIAL_CAPITAL, "RULE-BASED PERFORMANCE [COMMON, dynamic r, rf accrual, pnl stop]")
   performance_report(attn_c['equity'], INITIAL_CAPITAL, "ATTN-LSTM PERFORMANCE [COMMON, dynamic r, rf accrual, pnl stop]")
   performance_report(conv_c['equity'], INITIAL_CAPITAL, "CONV-LSTM PERFORMANCE [COMMON, dynamic r, rf accrual, pnl stop]")

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

from google.colab import drive
drive.mount('/content/drive')
























import os
from pathlib import Path
from math import log, sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm, t as t_dist


ROOT = "/content/drive/MyDrive"


SPOT_FILE = f"{ROOT}/kg4snxuhemmtl4nx.csv"
REAL_IV_FILE = f"{ROOT}/bcmocccogeuhqddt_features_flat.csv"
CONV_FILE = f"{ROOT}/iv_convlstm_rolling_preds.csv"
ATTN_FILE = f"{ROOT}/iv_attlstm_rolling_preds.csv"
ATTN_WEIGHTS_FILE = None

OUT_DIR = f"{ROOT}/iv_eval"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


START_DATE_STR = "2004-01-01"
RECENT_REP_DATE = None
ROLL_WIN = 60
VOL_FLOOR = 1e-6
Q = 0.0
FIT_SVI_BASELINE = False


VIX_BUCKETS = [
    ("<15", -np.inf, 15.0),
    ("1525", 15.0, 25.0),
    ("2540", 25.0, 40.0),
    (">40", 40.0, np.inf),
]
# func col


def col(t, d):
# func dedup index
   return f"{t}_{int(d):+d}".replace("+","")

def _dedup_index(df):
   if not df.index.is_unique:
       df = df.groupby(level=0).last()
# func print block
   return df

def print_block(title):
   bar = "=" * max(90, len(title)+4)
   print("\n" + bar + f"\n{title}\n" + bar)
# func bs d1


def bs_d1(S, K, T, sig, r=0.0, q=0.0):
# func bs price
   return (np.log(S/K) + (r - q + 0.5*sig*sig)*T)/(sig*np.sqrt(T))

def bs_price(S, K, T, sig, cp, r=0.0, q=0.0):
   S=float(S);K=float(K);T=float(T);sig=float(sig);r=float(r);q=float(q)
   if T<=0 or sig<=0 or not np.isfinite(sig):
       return max(0.0, S-K) if cp=='c' else max(0.0, K-S)
   d1 = bs_d1(S,K,T,sig,r,q); d2 = d1 - sig*np.sqrt(T)
   if cp=='c':
       return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
   else:
# func bs delta
       return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

def bs_delta(S, K, T, sig, cp, r=0.0, q=0.0):
   if T<=0 or sig<=0 or not np.isfinite(sig): return 0.0
   d1 = bs_d1(S,K,T,sig,r,q)
# func bs vega
   return np.exp(-q*T)*norm.cdf(d1) if cp=='c' else -np.exp(-q*T)*norm.cdf(-d1)

def bs_vega(S, K, T, sig, r=0.0, q=0.0):
   if T<=0 or sig<=0 or not np.isfinite(sig): return 0.0
   d1 = bs_d1(S,K,T,sig,r,q)
# func strike from delta
   return S*np.exp(-q*T)*norm.pdf(d1)*np.sqrt(T)

def strike_from_delta(S, T, sig, target_delta, cp, r=0.0, q=0.0):
   S=float(S);T=float(T);sig=float(sig);r=float(r);q=float(q)
   if cp=='c':
       if not (0.0 < target_delta < 1.0): raise ValueError("Call delta must be (0,1)")
       p = target_delta
   else:
       if not (-1.0 < target_delta < 0.0): raise ValueError("Put delta must be (-1,0)")
       p = 1.0 + target_delta
   d1 = norm.ppf(p)
   return S*np.exp(-(d1*sig*np.sqrt(T)) + (r - q + 0.5*sig*sig)*T)
# func fetch rf and vix


def fetch_rf_and_vix(idx: pd.DatetimeIndex) -> pd.DataFrame:
   from pandas_datareader import data as pdr
   start, end = str(idx.min().date()), str(idx.max().date())
   out = pd.DataFrame(index=pd.date_range(start, end, freq="D"))
   try:
       rf = pdr.DataReader("DGS1MO","fred", start, end)["DGS1MO"]/100.0
   except Exception:
       rf = None
   if rf is None:
       try:
           rf = pdr.DataReader("DGS3MO","fred", start, end)["DGS3MO"]/100.0
       except Exception:
           rf = pd.Series(0.0, index=out.index)

   try:
       vix = pdr.DataReader("VIXCLS","fred", start, end)["VIXCLS"]
   except Exception:
       vix = pd.Series(np.nan, index=out.index)

   out["rf"] = rf.reindex(out.index).ffill()
   out["VIX"] = vix.reindex(out.index).ffill()
# func load spot
   return out.reindex(idx).ffill()

def load_spot():
   spot = (pd.read_csv(SPOT_FILE, parse_dates=["DlyCalDt"])
               .rename(columns={"DlyCalDt":"date","DlyPrcInd":"spot"})
               .set_index("date").sort_index())
   spot = _dedup_index(spot)[["spot"]].astype(float)
# func load surface csv
   return spot

def load_surface_csv(path):
   df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
   df = _dedup_index(df)
   df = df.apply(pd.to_numeric, errors="coerce")
   return df
# func align data


def align_data():
   spot = load_spot()
   true = load_surface_csv(REAL_IV_FILE)


   def is_bucket(c):
       parts = str(c).split("_")
       if len(parts)!=2: return False
       try:
          int(parts[0]); int(parts[1])
          return True
       except:
          return False
   grid_cols = [c for c in true.columns if is_bucket(c)]
   true = true[grid_cols].clip(lower=VOL_FLOOR)

   conv = load_surface_csv(CONV_FILE)
   attn = load_surface_csv(ATTN_FILE)
   keep_cols = [c for c in grid_cols if (c in conv.columns) and (c in attn. columns)]
   if not keep_cols:
       raise RuntimeError("No overlapping tenordelta columns across realized/ Conv/ATTN.")

   true = true[keep_cols]
   conv = conv[keep_cols]
   attn = attn[keep_cols]


   idx = spot.index.intersection(true.dropna().index).intersection(conv.dropna( how="all").index).intersection(attn.dropna(how="all").index)
   idx = idx[idx >= pd.Timestamp(START_DATE_STR)]
   spot = spot.reindex(idx)
   true = true.reindex(idx)
   conv = conv.reindex(idx).ffill()
   attn = attn.reindex(idx).ffill()

   aux = fetch_rf_and_vix(idx)
   return spot, true, conv, attn, aux
# func parse buckets


def parse_buckets(cols):
   pairs = sorted([(int(c.split("_")[0]), int(c.split("_")[1])) for c in cols])
   tenors = sorted({t for t,_ in pairs})
   deltas = sorted({d for _,d in pairs})
# func to long
   return tenors, deltas, pairs

def to_long(df_wide, name):
   out = (df_wide.stack()
           .rename("sigma")
           .reset_index()
           .rename(columns={"level_1":"bucket"}))
   out["tenor"] = out["bucket"].str.split("_").str[0].astype(int)
   out["delta"] = out["bucket"].str.split("_").str[1].astype(int)
   out = out.drop(columns=["bucket"])
   out["which"] = name
   return out
# func compute errors


def compute_errors(spot, true_w, pred_w, aux):
   """
   Returns a LONG dataframe keyed by (date, tenor, delta) with:
   true_sigma, pred_sigma, price_true_call/put, price_pred_call/put, vega, qlike
  """

  T = to_long(true_w, "true")
  P = to_long(pred_w, "pred")
  df = T.merge(P, on=["date","tenor","delta"], suffixes=("_true","_pred"))
  df = df.sort_values(["date","tenor","delta"]).reset_index(drop=True)


  df = df.merge(spot.reset_index(), on="date", how="left")
  df = df.merge(aux.reset_index().rename(columns={"index":"date"}), on="date", how="left")


  Ks, CPs, Ts = [], [], []
  for _, r in df.iterrows():
        S = float(r["spot"])
        T_yrs = float(r["tenor"])/252.0
        rfr = float(r["rf"])
        sig = float(max(VOL_FLOOR, r["sigma_true"]))
        dlt = int(r["delta"])
        if dlt >= 0:
           cp = 'c'
           targ = dlt/100.0 if dlt>0 else 0.5
        else:
           cp = 'p'
           targ = dlt/100.0
        K = strike_from_delta(S, T_yrs, sig, targ, cp, rfr, Q)
        Ks.append(K); CPs.append(cp); Ts.append(T_yrs)
  df["K"] = Ks
  df["cp"] = CPs
  df["T_yrs"] = Ts


  df["p_true"] = df.apply(lambda r: bs_price(r["spot"], r["K"], r["T_yrs"], max(VOL_FLOOR, r["sigma_true"]), r["cp"], r["rf"], Q), axis=1)
  df["p_pred"] = df.apply(lambda r: bs_price(r["spot"], r["K"], r["T_yrs"], max(VOL_FLOOR, r["sigma_pred"]), r["cp"], r["rf"], Q), axis=1)


   df["vega"] = df.apply(lambda r: bs_vega(r["spot"], r["K"], r["T_yrs"], max( VOL_FLOOR, r["sigma_true"]), r["rf"], Q), axis=1)


   df["err_sigma"] = (df["sigma_pred"] - df["sigma_true"]).astype(float)
   df["ae_sigma_bp"] = (np.abs(df["err_sigma"])*1e4).astype(float)
   df["se_sigma_bp2"] = (df["err_sigma"]*1e4)**2
   df["err_price"] = (df["p_pred"] - df["p_true"]).astype(float)


   vhat = np.maximum(VOL_FLOOR**2, df["sigma_pred"].values**2)
   vtru = np.maximum(VOL_FLOOR**2, df["sigma_true"].values**2)
   ratio = vhat / vtru
   df["qlike"] = ratio - np.log(ratio) - 1.0


   df = df.sort_values(["tenor","delta","date"])
   df["sigma_true_lag"] = df.groupby(["tenor","delta"])["sigma_true"].shift(1)
   df["truth_move"] = np.sign(df["sigma_true"] - df["sigma_true_lag"])
   df["pred_move"] = np.sign(df["sigma_pred"] - df["sigma_true_lag"])
   df["dir_hit"] = (df["truth_move"] == df["pred_move"]).astype(float)
   df.loc[~np.isfinite(df["dir_hit"]), "dir_hit"] = np.nan

   return df.dropna(subset=["sigma_true","sigma_pred","spot","rf","K","T_yrs"])
# func agg core


def agg_core(df, label):
   res = {}
   def safe_mean(x):
       x = pd.to_numeric(x, errors="coerce")
       return float(np.nanmean(x.values)) if len(x) else np.nan

   res["model"] = label
   res["RMSE_sigma_bp"] = np.sqrt(safe_mean(df["se_sigma_bp2"]))
   res["MAE_sigma_bp"] = safe_mean(df["ae_sigma_bp"])

   vw = df["vega"].values
   w = np.where(np.isfinite(vw) & (vw>0), vw, 0.0)
   if w.sum() > 0:
       rmse_vw = np.sqrt(np.nansum(df["err_sigma"].values**2 * w) / np.nansum(w )) * 1e4
   else:
       rmse_vw = np.nan
   res["RMSE_sigma_bp_vegaW"] = float(rmse_vw)

   res["RMSE_price"] = np.sqrt(safe_mean((df["err_price"]**2)))

   res["QLIKE"] = safe_mean(df["qlike"])

   res["DirAcc_overall"] = safe_mean(df["dir_hit"])
# func agg by bucket
   return pd.DataFrame([res])

def agg_by_bucket(df, label):
   g = (df.groupby(["tenor","delta"])
           .apply(lambda d: pd.Series({
              "RMSE_sigma_bp": np.sqrt(np.nanmean(d["se_sigma_bp2"])),
              "MAE_sigma_bp": np.nanmean(d["ae_sigma_bp"]),
              "RMSE_sigma_bp_vegaW":
                   (np.sqrt(np.nansum((d["err_sigma"]**2)*d["vega"]) / np.nansum( d["vega"])) * 1e4) if np.nansum(d["vega"])>0 else np.nan,
              "RMSE_price": np.sqrt(np.nanmean((d["err_price"]**2))),
              "QLIKE": np.nanmean(d["qlike"]),
              "DirAcc": np.nanmean(d["dir_hit"])
           }))).reset_index()
   g.insert(0, "model", label)
# func rolling metrics
   return g

def rolling_metrics(df, win=60):
   df_d = (df.groupby("date")
              .apply(lambda d: pd.Series({
                "RMSE_bp": np.sqrt(np.nanmean((d["err_sigma"]*1e4)**2)),
                "RMSE_bp_vegaW": np.sqrt(np.nansum((d["err_sigma"]**2)*d["vega" ]) / np.nansum(d["vega"])) * 1e4 if np.nansum(d["vega"])>0 else np.nan
              }))).reset_index()
   df_d["RMSE_bp_roll"] = df_d["RMSE_bp"].rolling(win, min_periods=max(5, win //5)).mean()
   df_d["RMSE_bp_vegaW_roll"] = df_d["RMSE_bp_vegaW"].rolling(win, min_periods= max(5, win//5)).mean()
   return df_d
# func check no arb


def check_no_arb(df, tenors, deltas):
   out = []
   for d in df["date"].unique():
       dd = df[df["date"]==d]

       if 30 in tenors and 60 in tenors:
          for dl in deltas:
              a = dd[(dd["tenor"]==30) & (dd["delta"]==dl)]
              b = dd[(dd["tenor"]==60) & (dd["delta"]==dl)]
              if len(a)==1 and len(b)==1:
                 K = float(a["K"])
                 S = float(a["spot"]); r = float(a["rf"])
                 cp = 'c' if dl>=0 else 'p'
                 sig1 = float(a["sigma_true"]); sig2 = float(b["sigma_true"])
                 p1 = bs_price(S,K,30/252.0,max(VOL_FLOOR,sig1),cp,r,0.0)
                 p2 = bs_price(S,K,60/252.0,max(VOL_FLOOR,sig2),cp,r,0.0)
                 out.append(("calendar", d, dl, max(0.0, p1-p2), (p2-p1)))

       for T in tenors:
          dT = dd[dd["tenor"]==T].copy()
          if dT.empty: continue
          Ceq = []
          for _,r in dT.iterrows():
              if r["cp"]=='c':
                 Ceq.append(float(r["p_true"]))
              else:
                 Ceq.append(float(r["p_true"] + r["spot"]*np.exp(-Q*r["T_yrs"]) - r["K"]*np.exp(-r["rf"]*r["T_yrs"])))
          dT = dT.assign(Ceq=Ceq).sort_values("K")
          diffs = np.diff(dT["Ceq"].values)
          viol_mono = np.sum(diffs>1e-10)
          mag_mono = np.max(diffs[diffs>1e-10]) if np.any(diffs>1e-10) else 0.0
          out.append(("monoK", d, T, viol_mono, mag_mono))
          if len(dT)>=3:
              secdiff = np.diff(dT["Ceq"].values, n=2)
                 viol_conv = np.sum(secdiff< -1e-10)
                 mag_conv = np.min(secdiff) if np.any(secdiff< -1e-10) else 0.0
             else:
                 viol_conv = 0; mag_conv = 0.0
             out.append(("convK", d, T, viol_conv, mag_conv))
   res = pd.DataFrame(out, columns=["type","date","key","violations","magnitude "])
   return res
# func baseline naive last


def baseline_naive_last(true_series):
# func baseline ar1 per bucket
   return true_series.shift(1)

def baseline_ar1_per_bucket(df_long):
   out = []
   for (T, D), d in df_long.groupby(["tenor","delta"]):
       y = d["sigma_true"].astype(float)
       x = d["sigma_true"].shift(1).astype(float)
       mask = x.notna() & y.notna()
       if mask.sum() < 20:
             pred = x
       else:
             X = np.column_stack([np.ones(mask.sum()), x[mask].values])
             beta = np.linalg.lstsq(X, y[mask].values, rcond=None)[0]
             pred_vals = beta[0] + beta[1]*x.values
             pred = pd.Series(pred_vals, index=d.index)
       dd = d[["date","tenor","delta"]].copy()
       dd["sigma_pred_ar1"] = pred.values
       out.append(dd)
# func dm test
   return pd.concat(out).sort_values(["tenor","delta","date"])

def dm_test(loss_model, loss_base, h=1):
   d = (loss_model - loss_base).dropna()
   if len(d) < 10:
       return np.nan, np.nan
   T = len(d)
   dbar = d.mean()
   gamma0 = np.var(d, ddof=1)
   if h<=1:
       var = gamma0
   else:
       var = gamma0
       for k in range(1, h):
           gk = np.cov(d[k:], d[:-k], ddof=1)[0,1]
           var += 2*(1 - k/h)*gk
   DM = dbar / np.sqrt(var/T)
   p = 2*(1 - t_dist.cdf(abs(DM), df=T-1))
   return float(DM), float(p)
# func mincer zarnowitz


def mincer_zarnowitz(y, yhat):
   m = pd.concat([y, yhat], axis=1).dropna()
   if len(m) < 20:
       return np.nan, np.nan, np.nan, np.nan, np.nan
   Y = m.iloc[:,0].values
   X = np.column_stack([np.ones(len(m)), m.iloc[:,1].values])
   beta = np.linalg.lstsq(X, Y, rcond=None)[0]
   resid = Y - X@beta
   s2 = (resid**2).sum()/(len(Y)-2)
   cov = s2 * np.linalg.inv(X.T@X)
   se = np.sqrt(np.diag(cov))
   a, b = beta
   ta, tb = a/se[0], b/se[1]
   return float(a), float(b), float(ta), float(tb), float(np.corrcoef(Y, m.iloc [:,1].values)[0,1])
# func heatmaps for date


def heatmaps_for_date(df_pred, df_true, label, date, tenors, deltas):
   def gridify(wide):
       M = []
       for T in tenors:
           row = []
           for D in deltas:
                c = col(T,D)
                row.append(float(wide.at[date, c]) if (date in wide.index and c in wide.columns) else np.nan)
           M.append(row)
       return np.array(M, dtype=float)

   P = gridify(df_pred)
   R = gridify(df_true)
   E = (R - P)

   fig, axs = plt.subplots(1,3, figsize=(18,5), constrained_layout=True)
   vmin = np.nanmin(R); vmax = np.nanmax(R)
   im0 = axs[0].imshow(R, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
   axs[0].set_title(f"True      {date.date()}")
   im1 = axs[1].imshow(P, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
   axs[1].set_title(f"Pred      {label}")
   er = np.nanmax(np.abs(E))
   im2 = axs[2].imshow(E, aspect="auto", cmap="coolwarm", vmin=-er, vmax=er)
   axs[2].set_title("Error (True        Pred)")
   for ax in axs:
       ax.set_yticks(range(len(tenors))); ax.set_yticklabels([f"{t}d" for t in tenors])
       ax.set_xticks(range(len(deltas))); ax.set_xticklabels([f"{d:+d}".replace ("+","") for d in deltas], rotation=45)
   for im, ax in zip([im0,im1,im2], axs):
       plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
   fn = os.path.join(OUT_DIR, f"heatmap_{label}_{date.date()}.png")
   plt.savefig(fn, dpi=150); plt.show()
# func slice plots
   print(f"Saved heatmaps      {fn}")

def slice_plots(df_true, df_pred, label, tenors, deltas):
   picks_T = [min(tenors), tenors[len(tenors)//2], max(tenors)]
   for T in picks_T:
       tcols = [col(T,d) for d in deltas]
       mu_true = df_true[tcols].mean(axis=0)
       mu_pred = df_pred[tcols].mean(axis=0)
       plt.figure(figsize=(8,4))
       plt.plot(deltas, mu_true.values, label="True")
       plt.plot(deltas, mu_pred.values, label=label)
       plt.title(f"Mean   vs    at {T}d"); plt.xlabel(" bucket"); plt.ylabel(""); plt.legend(); plt.grid(True)
       fn = os.path.join(OUT_DIR, f"slice_delta_{label}_{T}d.png")
       plt.tight_layout(); plt.savefig(fn, dpi=150); plt.show()
       print(f"Saved slice     {fn}")

   picks_D = [min(deltas), 0 if 0 in deltas else deltas[len(deltas)//2], max( deltas)]
   for D in picks_D:
       dcols = [col(T,D) for T in tenors]
       mu_true = df_true[dcols].mean(axis=0)
       mu_pred = df_pred[dcols].mean(axis=0)
       plt.figure(figsize=(8,4))
       plt.plot(tenors, mu_true.values, label="True")
       plt.plot(tenors, mu_pred.values, label=label)
       plt.title(f"Mean    vs maturity at ={D}"); plt.xlabel("Tenor (d)"); plt. ylabel(""); plt.legend(); plt.grid(True)
       fn = os.path.join(OUT_DIR, f"slice_maturity_{label}_D{D}.png")
       plt.tight_layout(); plt.savefig(fn, dpi=150); plt.show()
# func rolling plots
       print(f"Saved slice    {fn}")

def rolling_plots(roll_df, label):
   plt.figure(figsize=(12,4))
   plt.plot(roll_df["date"], roll_df["RMSE_bp"], label="RMSE (bp)")
   plt.plot(roll_df["date"], roll_df["RMSE_bp_roll"], label=f"{ROLL_WIN}d roll RMSE (bp)")
   plt.plot(roll_df["date"], roll_df["RMSE_bp_vegaW"], label="Vega-wgt RMSE (bp )")
   plt.plot(roll_df["date"], roll_df["RMSE_bp_vegaW_roll"], label=f"{ROLL_WIN}d roll vega-RMSE (bp)")
   plt.title(f"Rolling errors    {label}"); plt.legend(); plt.grid(True); plt. xlabel("Date")
   plt.tight_layout()
   fn = os.path.join(OUT_DIR, f"rolling_{label}.png")
   plt.savefig(fn, dpi=150); plt.show()
# func histograms
   print(f"Saved rolling chart     {fn}")

def histograms(df, label):
   plt.figure(figsize=(10,4))
   plt.subplot(1,2,1)
   plt.hist(df["err_price"].dropna(), bins=50, alpha=0.8)
   plt.title(f"BS Price Error    {label}")
   plt.subplot(1,2,2)
   plt.hist((np.abs(df["err_sigma"])*1e4).dropna(), bins=50, alpha=0.8)
   plt.title(f"| error| (bp)    {label}")
   plt.tight_layout()
   fn = os.path.join(OUT_DIR, f"hists_{label}.png")
   plt.savefig(fn, dpi=150); plt.show()
   print(f"Saved histograms    {fn}")
# func fit svi params


def fit_svi_params(Ks, total_var):
   from scipy.optimize import least_squares
   k = Ks
   w = total_var
   def svi(p):
       a,b,rho,m,sig = p
       return a + b*(rho*(k-m) + np.sqrt((k-m)**2 + sig**2))
   def resid(p): return svi(p) - w
   p0 = np.array([0.01, 0.1, 0.0, 0.0, 0.1])
   bnds = ([-1, 1e-6, -0.999, -1.0, 1e-6],[5, 10, 0.999, 1.0, 10])
   sol = least_squares(resid, p0, bounds=bnds, max_nfev=5000)
   return sol.x, np.sqrt(np.mean(resid(sol.x)**2))
# func run all


def run_all():
   print_block("Loading & aligning data (strict common intersection 2004-01-01)")
   spot, true_w, conv_w, attn_w, aux = align_data()
   tenors, deltas, _ = parse_buckets(true_w.columns)
   rep_date = pd.Timestamp(RECENT_REP_DATE) if RECENT_REP_DATE else true_w. index[-1]
   print(f"Span: {true_w.index.min().date()}   {true_w.index.max().date()} | days={len(true_w)}")
   print(f"Tenors: {tenors}")
   print(f"Deltas: {deltas}")


   print_block("Computing core errors (ConvLSTM)")
   df_conv = compute_errors(spot, true_w, conv_w, aux)
   print_block("Computing core errors (LSTM-ATTN)")
   df_attn = compute_errors(spot, true_w, attn_w, aux)


  print_block("Core accuracy aggregates")
  agg_conv = agg_core(df_conv, "ConvLSTM")
  agg_attn = agg_core(df_attn, "LSTM-ATTN")
  print(agg_conv.round(4))
  print(agg_attn.round(4))
  agg_conv.to_csv(os.path.join(OUT_DIR,"summary_overall_conv.csv"), index= False)
  agg_attn.to_csv(os.path.join(OUT_DIR,"summary_overall_attn.csv"), index= False)

  print_block("By tenor   delta")
  buck_conv = agg_by_bucket(df_conv, "ConvLSTM")
  buck_attn = agg_by_bucket(df_attn, "LSTM-ATTN")
  buck_conv.to_csv(os.path.join(OUT_DIR,"summary_bybucket_conv.csv"), index= False)
  buck_attn.to_csv(os.path.join(OUT_DIR,"summary_bybucket_attn.csv"), index= False)
  print(buck_conv.head())


  roll_conv = rolling_metrics(df_conv, ROLL_WIN)
  roll_attn = rolling_metrics(df_attn, ROLL_WIN)
  rolling_plots(roll_conv, "ConvLSTM")
  rolling_plots(roll_attn, "LSTM-ATTN")


  heatmaps_for_date(conv_w, true_w, "ConvLSTM", rep_date, tenors, deltas)
  heatmaps_for_date(attn_w, true_w, "LSTM-ATTN", rep_date, tenors, deltas)
  slice_plots(true_w, conv_w, "ConvLSTM", tenors, deltas)
  slice_plots(true_w, attn_w, "LSTM-ATTN", tenors, deltas)


  histograms(df_conv, "ConvLSTM")
  histograms(df_attn, "LSTM-ATTN")


  print_block("Regime cuts by VIX bucket")
  def regime_table(df, label):
     rows=[]
     for name, lo, hi in VIX_BUCKETS:
         m = df[(df["VIX"]>=lo) & (df["VIX"]<hi)]
         rows.append({
              "model":label, "VIX_bucket":name,
              "RMSE_bp": np.sqrt(np.nanmean((m["err_sigma"]*1e4)**2)),
              "MAE_bp": np.nanmean(np.abs(m["err_sigma"]*1e4)),
              "RMSE_price": np.sqrt(np.nanmean((m["err_price"]**2))),
              "QLIKE": np.nanmean(m["qlike"]),
              "DirAcc": np.nanmean(m["dir_hit"])
         })
     return pd.DataFrame(rows)
  reg_conv = regime_table(df_conv, "ConvLSTM"); reg_attn = regime_table( df_attn, "LSTM-ATTN")
  reg_conv.to_csv(os.path.join(OUT_DIR,"regime_vix_conv.csv"), index=False)
  reg_attn.to_csv(os.path.join(OUT_DIR,"regime_vix_attn.csv"), index=False)
  print(reg_conv)


  print_block("Wing/ATM & tenor class cuts")
  def wing(delta):
     if delta <= -30: return "put_wing"
     if -10 <= delta <= 10: return "ATM"
     if delta >= 30: return "call_wing"
     return "mid"
  for df,label in [(df_conv,"ConvLSTM"), (df_attn,"LSTM-ATTN")]:
     ddf = df.copy()
     ddf["wing"] = ddf["delta"].apply(wing)
     ddf["tenor_class"] = np.where(ddf["tenor"]<=30, "short", "long")
     tab = (ddf.groupby(["wing","tenor_class"])
               .apply(lambda m: pd.Series({
                     "RMSE_bp": np.sqrt(np.nanmean((m["err_sigma"]*1e4)**2)),
                     "RMSE_bp_vegaW": np.sqrt(np.nansum((m["err_sigma"]**2)*m[" vega"]) / np.nansum(m["vega"])) * 1e4 if np.nansum(m[" vega"])>0 else np.nan,
                     "RMSE_price": np.sqrt(np.nanmean(m["err_price"]**2)),
                     "QLIKE": np.nanmean(m["qlike"]),
                     "DirAcc": np.nanmean(m["dir_hit"])
               }))).reset_index()
     tab.insert(0,"model",label)
     fn = os.path.join(OUT_DIR, f"cuts_wing_tenor_{label}.csv")
     tab.to_csv(fn, index=False)
     print(f"{label} wing/tenor cuts:\n", tab)


  print_block("No-arbitrage sanity on TRUE surface (violations % and magnitudes)")
  df_true_eval = compute_errors(spot, true_w, true_w, aux)
  arb = check_no_arb(df_true_eval, tenors, deltas)
  for typ in ["calendar","monoK","convK"]:
     sub = arb[arb["type"]==typ]
     viol_days = (sub["violations"]>0).sum()
     pct_pts = 100.0 * viol_days / max(1, sub.shape[0])
     mag = sub["magnitude"].abs().max()
     print(f"{typ:>10}: violate on {pct_pts:.2f}% of entries | max magnitude: {mag:,.6f}")
  arb.to_csv(os.path.join(OUT_DIR,"arb_true_surface.csv"), index=False)


  print_block("Baselines & DieboldMariano tests (RMSE and QLIKE)")
  naive = df_true_eval[["date","tenor","delta","sigma_true"]].copy()
  naive["sigma_pred_naive"] = naive.groupby(["tenor","delta"])["sigma_true"]. shift(1)
  ar1 = baseline_ar1_per_bucket(df_true_eval[["date","tenor","delta"," sigma_true"]])

  def add_base(df, base, colname):
     m = df.merge(base[["date","tenor","delta",colname]], on=["date","tenor", "delta"], how="left")
     return m
  df_conv_b = add_base(df_conv, naive.rename(columns={"sigma_pred_naive":"base "}), "base")
  df_attn_b = add_base(df_attn, naive.rename(columns={"sigma_pred_naive":"base "}), "base")
  df_conv_a = add_base(df_conv, ar1.rename(columns={"sigma_pred_ar1":"base"}), "base")
  df_attn_a = add_base(df_attn, ar1.rename(columns={"sigma_pred_ar1":"base"}), "base")

  def dm_block(dfM, name):
     rows=[]
     for (T,D), d in dfM.groupby(["tenor","delta"]):
         y = d["sigma_true"]; m = d["sigma_pred"]; b = d["base"]
         se_m = (m - y)**2; se_b = (b - y)**2
          DM_rmse, p_rmse = dm_test(se_m, se_b, h=1)
          vhat_m = np.maximum(VOL_FLOOR**2, m**2); vhat_b = np.maximum( VOL_FLOOR**2, b**2); vtru = np.maximum(VOL_FLOOR**2, y**2)
          q_m = vhat_m/vtru - np.log(vhat_m/vtru) - 1.0
          q_b = vhat_b/vtru - np.log(vhat_b/vtru) - 1.0
          DM_ql, p_ql = dm_test(q_m, q_b, h=1)
          rows.append({"tenor":T,"delta":D,"DM_RMSE":DM_rmse,"p_RMSE":p_rmse," DM_QLIKE":DM_ql,"p_QLIKE":p_ql})
       return pd.DataFrame(rows)
   dm_conv_naive = dm_block(df_conv_b, "Conv vs Naive")
   dm_attn_naive = dm_block(df_attn_b, "Attn vs Naive")
   dm_conv_ar1 = dm_block(df_conv_a, "Conv vs AR1")
   dm_attn_ar1 = dm_block(df_attn_a, "Attn vs AR1")
   dm_conv_naive.to_csv(os.path.join(OUT_DIR,"dm_conv_vs_naive.csv"), index= False)
   dm_attn_naive.to_csv(os.path.join(OUT_DIR,"dm_attn_vs_naive.csv"), index= False)
   dm_conv_ar1.to_csv(os.path.join(OUT_DIR,"dm_conv_vs_ar1.csv"), index=False)
   dm_attn_ar1.to_csv(os.path.join(OUT_DIR,"dm_attn_vs_ar1.csv"), index=False)
   print("Saved DM test tables.")


   print_block("MincerZarnowitz calibration ()")
   for name, df_ in [("ConvLSTM", df_conv), ("LSTM-ATTN", df_attn)]:
       a,b,ta,tb, rho = mincer_zarnowitz(df_["sigma_true"], df_["sigma_pred"])
       print(f"{name:10s} alpha={a:.4f} (t={ta:.2f}), beta={b:.3f} (t={tb:.2f}) , corr={rho:.3f}")
       rows=[]
       for (T,D), d in df_.groupby(["tenor","delta"]):
          a,b,ta,tb,_ = mincer_zarnowitz(d["sigma_true"], d["sigma_pred"])
          rows.append({"tenor":T,"delta":D,"alpha":a,"beta":b,"t_alpha":ta," t_beta":tb})
       pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR,f"mz_bybucket_{name}.csv" ), index=False)

   print_block("DONE     All diagnostics saved under " + OUT_DIR)


if __name__ == "__main__":
   run_all()


from google.colab import drive
drive.mount('/content/drive')

import os
import shutil
from google.colab import files


folder_name = 'straddle_results_common'
folder_path = f'/content/drive/MyDrive/{folder_name}'
os.makedirs(folder_path, exist_ok=True)


files_to_move = [
    '/content/drive/MyDrive/straddle_rule_mc_COMMON.csv',
    '/content/drive/MyDrive/straddle_rule_results_COMMON.csv',
    '/content/drive/MyDrive/straddle_rule_metrics_COMMON.csv',
    '/content/drive/MyDrive/straddle_attn_hedgeonly_mc_COMMON.csv',
    '/content/drive/MyDrive/straddle_attn_hedgeonly_results_COMMON.csv',
    '/content/drive/MyDrive/straddle_attn_hedgeonly_metrics_COMMON.csv',
    '/content/drive/MyDrive/straddle_convlstm_hedgeonly_mc_COMMON.csv',
    '/content/drive/MyDrive/straddle_convlstm_hedgeonly_results_COMMON.csv',
    '/content/drive/MyDrive/straddle_convlstm_hedgeonly_metrics_COMMON.csv',
    '/content/drive/MyDrive/straddle_atm_rule_equity_dd_COMMON.png',
    '/content/drive/MyDrive/straddle_atm_attn_equity_dd_COMMON.png',
    '/content/drive/MyDrive/straddle_atm_conv_equity_dd_COMMON.png',
    '/content/drive/MyDrive/straddle_atm_combined_COMMON.png'
]


for file_path in files_to_move:
    if os.path.exists(file_path):
       shutil.move(file_path, os.path.join(folder_path, os.path.basename( file_path)))
       print(f"Moved: {os.path.basename(file_path)}")
    else:
       print(f"File not found: {os.path.basename(file_path)}")


zip_path = f'/content/{folder_name}'
shutil.make_archive(zip_path, 'zip', folder_path)


files.download(f'{zip_path}.zip')

print("All files have been organized and downloaded successfully!")






































"""
Unified Short-30d Strangle Backtest (Start: 2004, dynamic r from FRED)

Key points
- Rule strategy uses previous-day IV for all decisions (no forward bias).
- ML strategies (ATTN/CONV) use same-day IV (forward bias allowed).
- 30 calendar-day tenor; short strangle at 10 delta.
- Daily rf accrual, PnL stop (-3% of NOTIONAL), VIX gate, dynamic delta hedge.
- Strict common-window comparison and sanity checks.
"""





import os, glob, math, warnings
from math import log, sqrt
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm, ttest_rel, probplot
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
      from pandas_datareader import data as pdr
      _PDR_OK = True
except Exception:
      _PDR_OK = False
      pdr = None
      print("WARNING: pandas_datareader missing. FRED fetch will fall back to NaNs /zeros.")





ROOT = '/content/drive/MyDrive'
SPOT_CSV = f'{ROOT}/kg4snxuhemmtl4nx.csv'
IV_REAL_CSV = f'{ROOT}/bcmocccogeuhqddt_features_flat.csv'
IV_ATTN_CSV = f'{ROOT}/iv_attlstm_rolling_preds.csv'
IV_CONV_CSV = f'{ROOT}/conv_lstm_20d_preds.csv'

FAMILY_NAME = 'strangle_30d'


HEDGE_SMOOTH_ALPHA = 0.80
MAX_HEDGE_TURNOVER_X = 3.0

INITIAL_CAPITAL = 1_000_000.0
VOL_FLOOR = 1e-6
CALENDAR_YEAR = 365.25


ATTN_FALLBACK_TOTAL = 0
CONV_FALLBACK_TOTAL = 0


RUN_DETAILED_CHECKS = True
TRACK_POSITION_LIFECYCLES = True
ANALYZE_HEDGE_EFFECTIVENESS = True
RUN_SENSITIVITY_ANALYSIS = True

END_DATE = '2025-09-05'



# func dedup index


def _dedup_index(df: pd.DataFrame) -> pd.DataFrame:
   if not df.index.is_unique:
       df = df.groupby(level=0).last()
# func to numeric
   return df

def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
# func cell
   return df.apply(pd.to_numeric, errors='coerce')

def _cell(df: pd.DataFrame, date: pd.Timestamp, col: str) -> float:
   v = df.loc[date, col]
   if isinstance(v, (pd.Series, pd.Index, np.ndarray, list, tuple)):
       v = np.asarray(v).ravel()[-1]
# func pct change
   return float(v)

def _pct_change(series: pd.Series) -> pd.Series:
   try:
       return series.pct_change(fill_method=None)
   except TypeError:
# func reconcile equity
       return series.pct_change()

def _reconcile_equity(eq_series, pnl_series, initial):
   try:
       if len(eq_series) != len(pnl_series): return False
       cs = pnl_series.cumsum() + initial
       diff = (eq_series.values - cs.values)
       return (np.nanmax(np.abs(diff)) <= 1e-6 * max(initial,1.0))
   except Exception:
# func drawdown
       return False

def _drawdown(series: pd.Series) -> pd.Series:
   cmax = series.cummax()
# func trading days between
   return (cmax - series)/cmax

def _trading_days_between(start: pd.Timestamp, end: pd.Timestamp, trading_days: pd.DatetimeIndex) -> int:
   """Trading days between start (excl) and end (incl)."""
# func detailed reconciliation check
   return len(trading_days[(trading_days > start) & (trading_days <= end)])

def detailed_reconciliation_check(equity_series, pnl_series, initial_capital, strategy_name):
   """Print precise equity vs cum PnL reconciliation."""
   calculated_equity = initial_capital + pnl_series.cumsum()
   discrepancy = equity_series - calculated_equity
   max_discrepancy = discrepancy.abs().max()
   relative_error = max_discrepancy / initial_capital
   print(f"\n[{strategy_name}] RECONCILIATION CHECK:")
   print(f"Max absolute discrepancy: ${max_discrepancy:.2f}")
   print(f"Relative error: {relative_error:.6%}")
   if max_discrepancy > 1.0:
       print("WARNING: Significant accounting discrepancy detected!")
       discrepancy_dates = discrepancy[discrepancy.abs() > 1.0]
       print(f"Dates with discrepancies: {len(discrepancy_dates)}")
       if len(discrepancy_dates) > 0:
          print("Sample discrepancies:")
          for date, diff in discrepancy_dates.head().items():
               print(f" {date}: ${diff:.2f}")
       return False
# func bootstrap sharpe diff
   return True

def bootstrap_sharpe_diff(r1, r2, runs=1000, seed=42):
   np.random.seed(seed)
   diff = []
   for _ in range(runs):
       idx = np.random.choice(len(r1), len(r1), replace=True)
       sh1 = np.mean(r1[idx]) / np.std(r1[idx]) * np.sqrt(252) if len(idx) > 1 else 0
       sh2 = np.mean(r2[idx]) / np.std(r2[idx]) * np.sqrt(252) if len(idx) > 1 else 0
       diff.append(sh1 - sh2)
   pval = np.mean(np.array(diff) < 0) * 2
# func compare strategies
   return pval

def compare_strategies(res1, res2, name1, name2):
   """Paired tests: daily returns, Sharpe, and drawdown."""
   ret1 = res1['ret'].dropna().values
   ret2 = res2['ret'].dropna().values
   pval_ret = ttest_rel(ret1, ret2).pvalue
   print(f"{name1} vs {name2} returns p-value: {pval_ret:.4f}")
   pval_sharpe = bootstrap_sharpe_diff(ret1, ret2)
   print(f"{name1} vs {name2} Sharpe p-value: {pval_sharpe:.4f}")
   dd1 = _drawdown(res1['equity'])
   dd2 = _drawdown(res2['equity'])
   pval_dd = ttest_rel(dd1, dd2).pvalue
# func compute iv errors
   print(f"{name1} vs {name2} drawdown p-value: {pval_dd:.4f}")

def compute_iv_errors(iv_pred, iv_real, front_t, dates, CALL_DELTA, PUT_DELTA):
   """MAE/RMSE vs realized IV at 10 buckets (per day)."""
   errors_c, errors_p = [], []
   for d in dates:
       try:
          sig_c_p = _cell(iv_pred, d, f"{front_t}_{CALL_DELTA}")
          sig_p_p = _cell(iv_pred, d, f"{front_t}_{PUT_DELTA}")
          sig_c_r = _cell(iv_real, d, f"{front_t}_{CALL_DELTA}")
          sig_p_r = _cell(iv_real, d, f"{front_t}_{PUT_DELTA}")
          errors_c.append(abs(sig_c_p - sig_c_r))
          errors_p.append(abs(sig_p_p - sig_p_r))
       except KeyError:
          continue
   mae_c = np.mean(errors_c)
   rmse_c = np.sqrt(np.mean(np.square(errors_c)))
   mae_p = np.mean(errors_p)
   rmse_p = np.sqrt(np.mean(np.square(errors_p)))
# func plot returns hist
   return mae_c, rmse_c, mae_p, rmse_p, errors_c, errors_p

def plot_returns_hist(rets, title, out_png):
   plt.figure(figsize=(10,6))
   plt.hist(rets.dropna(), bins=50)
   plt.title(title + ' Returns Histogram')
   plt.savefig(out_png, dpi=150)
   plt.show()
   plt.close()
# func plot qq
   print(f"Saved histogram   {out_png}")

def plot_qq(rets, title, out_png):
   plt.figure(figsize=(10,6))
   probplot(rets.dropna(), plot=plt)
   plt.title(title + ' QQ Plot')
   plt.savefig(out_png, dpi=150)
   plt.show()
   plt.close()
# func plot rolling sharpe
   print(f"Saved QQ plot     {out_png}")

def plot_rolling_sharpe(rets, title, out_png):
   rolling_sharpe = rets.rolling(252).mean() / rets.rolling(252).std() * np. sqrt(252)
   plt.figure(figsize=(16,6))
   plt.plot(rolling_sharpe)
   plt.title(title + ' Rolling 1-Year Sharpe')
   plt.savefig(out_png, dpi=150)
   plt.show()
   plt.close()
# func plot iv errors
   print(f"Saved rolling Sharpe    {out_png}")

def plot_iv_errors(errors_c, errors_p, dates, title, out_png):
   plt.figure(figsize=(16,6))
   plt.plot(dates[:len(errors_c)], errors_c, label='Call IV Error')
   plt.plot(dates[:len(errors_p)], errors_p, label='Put IV Error')
   plt.legend()
   plt.title(title + ' IV Prediction Errors')
   plt.savefig(out_png, dpi=150)
   plt.show()
   plt.close()
# func plot sensitivity heatmap
   print(f"Saved IV errors plot     {out_png}")

def plot_sensitivity_heatmap(sensitivity_results, param_name, out_png):
   """Simple dual-axis plot for Sharpe and MaxDD across parameter values."""
   df = pd.DataFrame(sensitivity_results[param_name]).T
   fig, ax1 = plt.subplots(figsize=(10,6))
   ax1.plot(df.index, df['Sharpe0'], color='b', label='Sharpe')
   ax1.set_ylabel('Sharpe')
   ax2 = ax1.twinx()
   ax2.plot(df.index, df['max_drawdown'], color='r', label='Max DD')
   ax2.set_ylabel('Max DD')
   plt.title(f'Sensitivity to {param_name}')
   plt.savefig(out_png, dpi=150)
   plt.show()
   plt.close()
# func plot equity dd single
   print(f"Saved sensitivity plot     {out_png}")

def plot_equity_dd_single(df: pd.DataFrame, title: str, out_png: str):
   """Single equity curve + drawdown."""
   plt.style.use("ggplot")
   fig, (ax1, ax2) = plt.subplots(2,1, figsize=(16,10), sharex=True, gridspec_kw={"height_ratios":[3,1]})
   ax1.plot(df.index, df['equity'], linewidth=2, label=f"{title} (Final: ${df[' equity'].iloc[-1]:,.0f})")
   ax1.axhline(INITIAL_CAPITAL, linestyle='--', alpha=0.6, color='gray')
   ax1.set_title(title); ax1.set_ylabel("Equity ($)")
   ax1.grid(True, linestyle="--", alpha=0.7); ax1.legend(loc="best")
   dd = (df['equity'].cummax() - df['equity'])/df['equity'].cummax()
   ax2.fill_between(df.index, dd*100.0, 0, alpha=0.3, label=f"Max DD: {dd.max() *100:.1f}%")
   ax2.set_ylabel("Drawdown (%)"); ax2.grid(True, linestyle="--", alpha=0.7); ax2.legend(loc="best")
   ax2.xaxis.set_major_locator(mdates.YearLocator()); ax2.xaxis. set_major_formatter(mdates.DateFormatter("%Y"))
   plt.xlabel("Date"); plt.tight_layout(); plt.savefig(out_png, dpi=150, bbox_inches="tight")
   plt.show()
   plt.close()
# func plot combined common
   print(f"Saved plot   {out_png}")

def plot_combined_common(curves: Dict[str,pd.DataFrame], title: str, out_png: str):
   """Combined equity + drawdown on strict common window."""
   plt.style.use("ggplot")
   fig, (ax1, ax2) = plt.subplots(2,1, figsize=(16,10), sharex=True, gridspec_kw={"height_ratios":[3,1]})
   for label, df in curves.items():
       ax1.plot(df.index, df['equity'], linewidth=2, label=f"{label} (Final: ${ df['equity'].iloc[-1]:,.0f})")
   ax1.axhline(INITIAL_CAPITAL, linestyle='--', alpha=0.6, color='gray')
   ax1.set_title(f"{title}     Combined (Strict Common Window)")
   ax1.set_ylabel("Equity ($)"); ax1.grid(True, linestyle="--", alpha=0.7); ax1 .legend(loc="best")
   for label, df in curves.items():
       dd = (df['equity'].cummax() - df['equity'])/df['equity'].cummax()
       ax2.plot(df.index, dd*100.0, linewidth=1.8, label=f"{label} DD (max {dd. max()*100:.1f}%)")
   ax2.set_ylabel("Drawdown (%)"); ax2.grid(True, linestyle="--", alpha=0.7); ax2.legend(loc="best")
   ax2.xaxis.set_major_locator(mdates.YearLocator()); ax2.xaxis. set_major_formatter(mdates.DateFormatter("%Y"))
   plt.xlabel("Date"); plt.tight_layout(); plt.savefig(out_png, dpi=150, bbox_inches="tight")
   plt.show()
   plt.close()
   print(f"Saved plot     {out_png}")



# func fetch rf and vix


def fetch_rf_and_vix(idx: pd.DatetimeIndex) -> Tuple[pd.DataFrame, int, int]:
   """Risk-free (1M/3M FRED) and VIX (FRED) aligned to trading days."""
   df = pd.DataFrame(index=idx)
   df['rf'] = 0.0
   df['vix'] = np.nan
   if len(idx) == 0:
       return df, 0, 0

   start_date = str(idx.min().date())
   end_date = END_DATE

   rf = None
   rf_fallback = False
   if _PDR_OK and pdr is not None:
       try:
          rf1 = pdr.DataReader('DGS1MO', 'fred', start_date, end_date)
          rf = rf1['DGS1MO']/100.0
       except Exception:
          try:
                rf3 = pdr.DataReader('DGS3MO', 'fred', start_date, end_date)
                rf = rf3['DGS3MO']/100.0
           except Exception:
                rf = None
                rf_fallback = True
   else:
       rf_fallback = True
   if rf is None:
       rf = pd.Series(0.0, index=pd.date_range(start_date, end_date, freq='D'))

   vix = None
   vix_fallback = False
   if _PDR_OK and pdr is not None:
       try:
           vix = pdr.DataReader('VIXCLS', 'fred', start_date, end_date)['VIXCLS' ]
       except Exception:
           vix = None
           vix_fallback = True
   if vix is None:
       vix = pd.Series(np.nan, index=pd.date_range(start_date, end_date, freq=' D'))

   out = pd.DataFrame(index=pd.date_range(start_date, end_date, freq='D'))
   out['rf'] = rf.reindex(out.index).ffill()
   out['vix'] = vix.reindex(out.index)
   out = out.reindex(idx).ffill()

   rf_fallback_days = len(out) if rf_fallback else 0
   vix_nan_days = out['vix'].isna().sum()

   if rf_fallback:
       print("RF fetch failed, using fallback 0 for all days.")
   if vix_fallback:
       print("VIX fetch failed, using NaN fallback.")

   return out, rf_fallback_days, vix_nan_days



# func bs price


def bs_price(S, K, T, sig, cp, r):
   S = float(S); K = float(K); T = float(T); sig = float(sig); r = float(r)
   if T <= 0.0 or sig <= 0.0:
       return max(0.0, S-K) if cp == "c" else max(0.0, K-S)
   d1 = (log(S/K) + (r + 0.5*sig*sig)*T)/(sig*sqrt(T))
   d2 = d1 - sig*sqrt(T)
   if cp == "c":
       return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
# func bs delta
   return K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def bs_delta(S, K, T, sig, cp, r):
   S = float(S); K = float(K); T = float(T); sig = float(sig); r = float(r)
   if T <= 0.0 or sig <= 0.0: return 0.0
   d1 = (log(S/K) + (r + 0.5*sig*sig)*T)/(sig*sqrt(T))
# func strike from delta
   return norm.cdf(d1) if cp == "c" else -norm.cdf(-d1)

def strike_from_delta(S, T, sig, target_delta, cp, r):
   """Closed-form strike from target delta under BS."""
   S = float(S); T = float(T); sig = float(sig); r = float(r)
   if cp == 'c':
       if not (0.0 < target_delta < 1.0): raise ValueError("Call delta must be (0,1)")
       p = target_delta
   else:
       if not (-1.0 < target_delta < 0.0): raise ValueError("Put delta must be (-1,0)")
       p = 1.0 + target_delta
   d1 = norm.ppf(p)
   return S * math.exp(-d1*sig*sqrt(T) + (r + 0.5*sig*sig)*T)





CALL_DELTAS = [10,20,30,40,50]
# func get iv for strike
PUT_DELTAS = [-90,-80,-70,-60,-50,-40,-30,-20,-10]

def get_iv_for_strike(K: float, S: float, iv_df: pd.DataFrame, date: pd. Timestamp,
                     front_t: int, is_call: bool, rate: float) -> float:
   """Interpolate IV at strike using fixed-delta grid for given tenor."""
   T_surface = front_t / CALENDAR_YEAR
   pts = []
   if is_call:
       for dd in CALL_DELTAS:
           col = f"{front_t}_{dd}"
           sig = max(VOL_FLOOR, _cell(iv_df, date, col))
           d1 = norm.ppf(dd/100.0)
           Kx = S * math.exp(-d1*sig*sqrt(T_surface) + (rate + 0.5*sig*sig)* T_surface)
           pts.append((Kx, sig))
   else:
       for dd in PUT_DELTAS:
           col = f"{front_t}_{dd}"
           sig = max(VOL_FLOOR, _cell(iv_df, date, col))
           d1 = norm.ppf(1.0 + dd/100.0)
           Kx = S * math.exp(-d1*sig*sqrt(T_surface) + (rate + 0.5*sig*sig)* T_surface)
           pts.append((Kx, sig))
   pts.sort(key=lambda x: x[0])
   Ks, sigs = zip(*pts)
   return float(interp1d(Ks, sigs, kind='linear', fill_value="extrapolate", bounds_error=False)(K))



# func stationary bootstrap summaries


def stationary_bootstrap_summaries(equity: pd.Series, initial: float,
                                 runs=5000, block_mean=10, seed=42) -> pd. DataFrame:
   """Resample daily returns to summarize distribution of outcomes."""
   rets = _pct_change(equity).dropna().values
   if len(rets) == 0:
       return pd.DataFrame()
   n = len(rets)
  rng = np.random.default_rng(seed)
  p = 1.0/max(block_mean,1)

  def _pick_indices():
       idx = np.empty(n, dtype=int)
       idx[0] = rng.integers(0, n)
       for t in range(1, n):
          if rng.random() < p:
              idx[t] = rng.integers(0, n)
          else:
              idx[t] = (idx[t-1] + 1) % n
       return idx

  f, tr, dd, sh = [], [], [], []
  for _ in range(runs):
       idx = _pick_indices()
       boot = rets[idx]
       eq = initial * np.cumprod(1.0 + boot)
       s = pd.Series(eq)
       tot = float(s.iloc[-1]/initial - 1.0)
       ann_ret = (1+tot)**(252.0/n) - 1.0 if n>0 else 0.0
       ann_vol = float(pd.Series(boot).std()*np.sqrt(252)) if n>1 else 0.0
       sharpe = ann_ret/ann_vol if ann_vol>0 else 0.0
       draw = (s.cummax()-s)/s.cummax()
       mdd = float(draw.max()) if len(draw) else 0.0
       f.append(float(s.iloc[-1])); tr.append(tot); dd.append(mdd); sh.append( sharpe)

  def pct(col, p): return float(np.percentile(col, p))

  return pd.DataFrame({
       'metric': ['final_equity','total_return','max_dd','sharpe_0'],
       'p05': [pct(f,5), pct(tr,5), pct(dd,5), pct(sh,5)],
       'p50': [pct(f,50),pct(tr,50),pct(dd,50),pct(sh,50)],
       'p95': [pct(f,95),pct(tr,95),pct(dd,95),pct(sh,95)],
       'runs': [runs]*4,
       'block_mean':[block_mean]*4,
       'seed': [seed]*4
  })



# func performance report


def performance_report(series_equity: pd.Series, initial: float, title: str, benchmark_series: pd.Series = None) -> pd.DataFrame:
   """Print key stats and return a one-row metrics DataFrame."""
   rets = _pct_change(series_equity).dropna()
   total_return = float(series_equity.iloc[-1]/initial - 1.0)
   n = len(rets)
   ann_ret = (1+total_return)**(252/n) - 1 if n>0 else 0.0
   ann_vol = float(rets.std()*np.sqrt(252)) if n>1 else 0.0
   sharpe = ann_ret/ann_vol if ann_vol>0 else 0.0
   cmax = series_equity.cummax(); dd = (cmax-series_equity)/cmax; mdd = float( dd.max()) if len(dd) else 0.0
   dneg = rets[rets<0]; ddev = float(dneg.std()*np.sqrt(252)) if len(dneg)>1 else 0.0
   sortino = ann_ret/ddev if ddev>0 else 0.0
   win = len(rets[rets>0])/n if n>0 else 0.0
   gp = float(rets[rets>0].sum())
   gl = abs(float(rets[rets<0].sum()))
   pf = gp/gl if gl>0 else float('inf')
   var_95 = float(np.percentile(rets,5)) if n>0 else 0.0
   es_95 = float(rets[rets<=var_95].mean()) if n>0 else 0.0
   calmar = ann_ret / mdd if mdd > 0 else float('inf')
   omega = gp / gl if gl > 0 else float('inf')
   ulcer = np.sqrt((dd**2).mean())

   print("\n"+"="*70); print(f"{title}"); print("="*70)
   print(f"{'Initial Capital:':<35}${initial:>24,.0f}")
   print(f"{'Final Equity:':<35}${series_equity.iloc[-1]:>24,.2f}")
   print(f"{'Total Return:':<35}{total_return:>24.2%}")
   print(f"{'Annualized Return:':<35}{ann_ret:>24.2%}")
   print(f"{'Annualized Volatility:':<35}{ann_vol:>24.2%}")
   print(f"{'Sharpe Ratio (r=0):':<35}{sharpe:>24.2f}")
   print(f"{'Sortino Ratio:':<35}{sortino:>24.2f}")
   print(f"{'Calmar Ratio:':<35}{calmar:>24.2f}")
   print(f"{'Omega Ratio:':<35}{omega:>24.2f}")
   print(f"{'Ulcer Index:':<35}{ulcer:>24.2f}")
   print(f"{'Max Drawdown:':<35}{mdd:>24.2%}")
   print(f"{'Win Rate:':<35}{win:>24.2%}")
  print(f"{'Profit Factor:':<35}{pf:>24.2f}")
  print(f"{'Daily 95% VaR:':<35}{var_95:>24.2%}")
  print(f"{'Daily 95% Expected Shortfall:':<35}{es_95:>24.2%}")
  print(f"{'Trading Days:':<35}{n:>24}")
  print(f"{'Years:':<35}{n/252:>24.1f}")

  if benchmark_series is not None:
     bench_rets = _pct_change(benchmark_series).dropna()
     bench_ann_ret = (benchmark_series.iloc[-1]/benchmark_series.iloc[0]) ** (252/len(bench_rets)) - 1 if len(bench_rets)>0 else 0.0
     alpha = ann_ret - bench_ann_ret
     print(f"{'Alpha (excess return):':<35}{alpha:>24.2%}")
     cov = np.cov(rets, bench_rets)[0,1]
     var_bench = np.var(bench_rets)
     beta = cov / var_bench if var_bench > 0 else 0
     jensen_alpha = ann_ret - beta * bench_ann_ret
     print(f"{'Jensen Alpha:':<35}{jensen_alpha:>24.2%}")
     excess_rets = rets - bench_rets
     tracking_error = np.std(excess_rets) * np.sqrt(252)
     ir = alpha / tracking_error if tracking_error > 0 else 0
     print(f"{'Information Ratio:':<35}{ir:>24.2f}")

  print("="*70)

  return pd.DataFrame([{
     "Strategy":title,
     "Initial": initial,
     "FinalEquity": float(series_equity.iloc[-1]),
     "TotalReturn": total_return,
     "AnnReturn": ann_ret,
     "AnnVol": ann_vol,
     "Sharpe0": sharpe,
     "Sortino": sortino,
     "Calmar": calmar,
     "Omega": omega,
     "Ulcer": ulcer,
     "MaxDD": mdd,
     "WinRate": win,
     "ProfitFactor": pf,
     "VaR95": var_95,
     "ES95": es_95,
         "TradingDays": n,
         "Years": n/252.0
   }])



# func load spot


def load_spot(start_year: int) -> pd.DataFrame:
   """Load SPX spot (columns: date, spot)."""
   spot = (pd.read_csv(SPOT_CSV, parse_dates=["DlyCalDt"])
              .rename(columns={"DlyCalDt":"date","DlyPrcInd":"spot"})
              .set_index("date"))
   spot = _dedup_index(spot)
   spot['spot'] = pd.to_numeric(spot['spot'], errors='coerce')
   spot = spot[['spot']]
   spot = spot[spot.index >= f"{start_year}-01-01"]
# func load iv real
   return spot

def load_iv_real(front_t: int, start_year: int) -> pd.DataFrame:
   """Load realized IV surface for required tenors/deltas."""
   need = [f"{front_t}_{d}" for d in [-90,-80,-70,-60,-50,-40,-30,-20,-10,10,20,30,40,50]]
   ivr = (pd.read_csv(IV_REAL_CSV, parse_dates=["date"]).set_index("date"))
   ivr = _dedup_index(ivr)
   missing = [c for c in need if c not in ivr.columns]
   if missing:
         raise KeyError(f"Realised IV missing columns: {missing}")
   ivr = _to_numeric(ivr[need]).clip(lower=VOL_FLOOR)
   ivr = ivr[ivr.index >= f"{start_year}-01-01"]
# func align trading idx
   return ivr

def align_trading_idx(spot: pd.DataFrame, iv_real: pd.DataFrame) -> Tuple[pd. DataFrame,pd.DataFrame,pd.DatetimeIndex]:
   """Intersection of trading days across spot and IV."""
   idx = spot['spot'].dropna().index.intersection(iv_real.dropna().index)
   spot = spot.loc[idx]
   iv_real = iv_real.loc[idx]
     return spot, iv_real, idx




# class CostModel

@dataclass
class CostModel:
     option_comm_bps: float
     option_slip_bps: float
     und_comm_bps: float
     und_slip_bps: float

     def option_cost(self, traded_value: float) -> float:
           return abs(traded_value) * (self.option_comm_bps + self.option_slip_bps)

     def underlying_cost(self, traded_notional: float) -> float:
           return abs(traded_notional) * (self.und_comm_bps + self.und_slip_bps)



# func run rule strategy


def run_rule_strategy(*,
     front_t=30, CALL_DELTA=10, PUT_DELTA=-10, MAX_HOLD=20, NOTIONAL=1_000_000,
     STOP_PCT=0.03, START_YEAR=2004, VIX_HALT=40.0, VIX_RESUME=39.0, COOLDOWN_D =10,
     cost: CostModel = CostModel(0.0050,0.0025, 0.0008,0.0007)
):
     """
     Short 30d strangle at 10, dynamic hedge.
     Decisions use previous-day IV (no forward bias). MTM uses same-day IV.
     PnL stop, VIX gate, calendar rf accrual.
     """
     print("[INIT] RULE module loaded. Dynamic hedge (realised IV). START_YEAR =2004. Dynamic r used.")
     print("[NO FORWARD BIAS] Using previous day's IV for entry and hedging decisions.")
     print("[ACCURATE MTM] Using current day's IV for valuation.")
  spot = load_spot(START_YEAR)
  iv_real = load_iv_real(front_t, START_YEAR)
  spot, iv_real, dates = align_trading_idx(spot, iv_real)
  rf_vix, rf_fallback_days, vix_nan_days = fetch_rf_and_vix(dates)
  print(f"[FRED] RF fallback days: {rf_fallback_days} | VIX NaN days: { vix_nan_days}")

  iv_real_prev = iv_real.shift(1)

  T1 = front_t / CALENDAR_YEAR
  pnl, eq, idx = [], [INITIAL_CAPITAL], []
  pos = []
  blocked = False; below39 = 0

  position_history = []
  hedge_performance = []
  total_trans_cost = 0.0

  for i, d in enumerate(dates):
     S = _cell(spot, d, "spot")
     r = float(rf_vix.loc[d,'rf']) if 'rf' in rf_vix.columns else 0.0
     v = float(rf_vix.loc[d,'vix']) if 'vix' in rf_vix.columns else np.nan


     if not np.isnan(v):
         if v >= VIX_HALT: blocked = True; below39 = 0
         elif blocked:
             if v < VIX_RESUME:
                below39 += 1
                if below39 >= COOLDOWN_D: blocked = False; below39 = 0
             else:
                below39 = 0


     if not pos:
         if (not blocked) and (np.isnan(v) or v < VIX_HALT) and (i > 0):
             sig_c = _cell(iv_real_prev, d, f"{front_t}_{CALL_DELTA}")
             sig_p = _cell(iv_real_prev, d, f"{front_t}_{PUT_DELTA}")
             Kc = strike_from_delta(S, T1, sig_c, CALL_DELTA/100.0, 'c', r)
             Kp = strike_from_delta(S, T1, sig_p, PUT_DELTA/100.0, 'p', r)
         prem = bs_price(S, Kc, T1, sig_c, 'c', r) + bs_price(S, Kp, T1, sig_p, 'p', r)
         if np.isfinite(prem) and prem > 0:
            units = -NOTIONAL / prem
            d_call = bs_delta(S, Kc, T1, sig_c, 'c', r)
            d_put = bs_delta(S, Kp, T1, sig_p, 'p', r)
            hedge_target = -units * (d_call + d_put)
            hedge_shares = hedge_target
            entry_opt_cost = cost.option_cost(abs(units) * prem)
            entry_und_cost = cost.underlying_cost(abs(hedge_shares) * S)
            entry_costs = entry_opt_cost + entry_und_cost
            total_trans_cost += entry_costs
            pnl.append(-entry_costs)
            eq.append(eq[-1] - entry_costs); idx.append(d)
            pos = [{
                 "Kc": Kc, "Kp": Kp, "T": T1, "units": units,
                 "V0": -units * prem,
                 "hedge_shares": hedge_shares,
                 "prev_S": S, "entry_S": S, "entry_day": d, "prev_d": d,
                 "pnl_since_entry": -entry_costs,
                 "entry_sig_c": sig_c,
                 "entry_sig_p": sig_p
            }]
            if TRACK_POSITION_LIFECYCLES:
                 position_history.append({
                      "date": d, "action": "ENTRY", "Kc": Kc, "Kp": Kp,
                      "units": units, "premium": prem, "hedge_shares": hedge_shares, "vix": v
                 })
         continue
     continue


  S_now = S; r_now = r
  days_elapsed_calendar = (d - pos[0]["prev_d"]).days
  T_rem = max(0.0, pos[0]["T"] - days_elapsed_calendar / CALENDAR_YEAR)
  days_elapsed_trading = _trading_days_between(pos[0]["prev_d"], d, dates) - 1

  sig_c_now = get_iv_for_strike(pos[0]["Kc"], S_now, iv_real, d, front_t, True, r_now)
  sig_p_now = get_iv_for_strike(pos[0]["Kp"], S_now, iv_real, d, front_t, False, r_now)
  prem_now = bs_price(S_now, pos[0]["Kc"], T_rem, sig_c_now, 'c', r_now) +\
            bs_price(S_now, pos[0]["Kp"], T_rem, sig_p_now, 'p', r_now)
  price_now = -pos[0]["units"] * prem_now

  option_pnl = pos[0]["V0"] - price_now
  dS = S_now - pos[0]["prev_S"]
  hedge_pnl = pos[0]["hedge_shares"] * dS


  sig_c_prev = get_iv_for_strike(pos[0]["Kc"], S_now, iv_real_prev, d, front_t, True, r_now)
  sig_p_prev = get_iv_for_strike(pos[0]["Kp"], S_now, iv_real_prev, d, front_t, False, r_now)
  d_call_h = bs_delta(S_now, pos[0]["Kc"], T_rem, sig_c_prev, 'c', r_now)
  d_put_h = bs_delta(S_now, pos[0]["Kp"], T_rem, sig_p_prev, 'p', r_now)
  target = -pos[0]["units"] * (d_call_h + d_put_h)

  hedge_new = (1.0 - HEDGE_SMOOTH_ALPHA)*pos[0]["hedge_shares"] + HEDGE_SMOOTH_ALPHA*target
  d_shares = hedge_new - pos[0]["hedge_shares"]

  max_turn_notional = MAX_HEDGE_TURNOVER_X * eq[-1]
  traded_notional = abs(d_shares) * S_now
  if traded_notional > max_turn_notional and traded_notional > 0:
     scale = max_turn_notional / traded_notional
     d_shares *= scale
     hedge_new = pos[0]["hedge_shares"] + d_shares
     traded_notional = abs(d_shares) * S_now

  rehedge_cost = cost.underlying_cost(traded_notional)
  total_trans_cost += rehedge_cost

  rf_day = eq[-1] * (r_now / CALENDAR_YEAR) * days_elapsed_calendar

  pnl_day_core = option_pnl + hedge_pnl - rehedge_cost + rf_day
  pnl_tmp = pos[0]["pnl_since_entry"] + pnl_day_core

  days_live = _trading_days_between(pos[0]["entry_day"], d, dates)
  force_vix = (not np.isnan(v)) and (v >= VIX_HALT)
  final_day = (i == len(dates)-1)
  stop_loss = (pnl_tmp <= -STOP_PCT * NOTIONAL)
  will_exit = force_vix or stop_loss or (days_live>=MAX_HOLD) or final_day

  pnl_day = pnl_day_core
  if will_exit:
     exit_opt_cost = cost.option_cost(abs(pos[0]["units"]) * prem_now)
     exit_und_cost = cost.underlying_cost(abs(hedge_new) * S_now)
     pnl_day -= (exit_opt_cost + exit_und_cost)
     total_trans_cost += (exit_opt_cost + exit_und_cost)

  eq.append(eq[-1] + pnl_day); pnl.append(pnl_day); idx.append(d)

  if ANALYZE_HEDGE_EFFECTIVENESS and pos:
     hedge_effectiveness = 1 - (abs(option_pnl) / (abs(option_pnl) + abs( hedge_pnl))) if (abs(option_pnl) + abs(hedge_pnl)) > 0 else 1
     hedge_performance.append({"date": d, "effectiveness": hedge_effectiveness})

  pos[0]["V0"] = price_now
  pos[0]["prev_S"] = S_now
  pos[0]["prev_d"] = d
  pos[0]["T"] = T_rem
  pos[0]["hedge_shares"] = hedge_new
  pos[0]["pnl_since_entry"] = pnl_tmp if not will_exit else 0.0

  if will_exit:
     if TRACK_POSITION_LIFECYCLES:
         days_held = _trading_days_between(pos[0]["entry_day"], d, dates) - 1
         position_history.append({
              "date": d, "action": "EXIT",
              "exit_reason": "VIX" if force_vix else ("STOP" if stop_loss else ("MAX_HOLD" if days_live>=MAX_HOLD else "FINAL")),
              "days_held": days_live, "final_pnl": pnl_tmp,
              "stop_threshold": -STOP_PCT * NOTIONAL, "vix": v
         })
     pos.clear()
     if force_vix: blocked = True; below39 = 0

   res = pd.DataFrame({"pnl": pnl, "equity": eq[1:]}, index=idx)
   res['cum_pnl'] = res['equity']
   res['dd_pct'] = _drawdown(res['equity'])
   res['ret'] = _pct_change(res['equity'])
   res['vix'] = rf_vix['vix']

   ok = _reconcile_equity(res['equity'], res['pnl'], INITIAL_CAPITAL)
   print(f"\n[SANITY][RULE] (recon from equity) {'PASS' if ok else 'FAIL'}.\n")
   if RUN_DETAILED_CHECKS:
       detailed_reconciliation_check(res['equity'], res['pnl'], INITIAL_CAPITAL , "RULE")

   if ANALYZE_HEDGE_EFFECTIVENESS and hedge_performance:
       hedge_df = pd.DataFrame(hedge_performance)
       print(f"[RULE] Hedge effectiveness stats: Mean = {hedge_df[' effectiveness'].mean():.3f}, "
            f"Min = {hedge_df['effectiveness'].min():.3f}, "
            f"Max = {hedge_df['effectiveness'].max():.3f}")

   if TRACK_POSITION_LIFECYCLES and position_history:
       pos_df = pd.DataFrame(position_history)
       pos_df.to_csv(f"{ROOT}/{FAMILY_NAME}_rule_position_history.csv", index= False)
       print(f"[RULE] Position history saved: {ROOT}/{FAMILY_NAME} _rule_position_history.csv")

   cum_pnl = res['cum_pnl'].iloc[-1]
   cost_pct = (total_trans_cost / cum_pnl) * 100 if cum_pnl > 0 else 0
   print(f"[RULE] Total transaction costs: ${total_trans_cost:.2f} ({cost_pct :.2f}% of cum PnL)")

   return res



# func load iv attn


def _load_iv_attn(front_t: int, start_year: int, trading_idx: pd.DatetimeIndex) -> pd.DataFrame:
   """Load/align ATTN-LSTM IV predictions for 10."""
     need = [f"{front_t}_{-10}", f"{front_t}_{10}"]
     attn = (pd.read_csv(IV_ATTN_CSV, parse_dates=["date"]).set_index("date"))
     attn = _dedup_index(attn)
     missing = [c for c in need if c not in attn.columns]
     if missing:
           raise KeyError(f"ATTN predicted IV missing columns: {missing}")
     attn = _to_numeric(attn[need]).clip(lower=VOL_FLOOR)
     attn = attn[attn.index >= f"{start_year}-01-01"]
     attn = attn.reindex(trading_idx).ffill()
# func run attn strategy
     return attn

def run_attn_strategy(*,
     front_t=30, CALL_DELTA=10, PUT_DELTA=-10, MAX_HOLD=30, NOTIONAL=1_000_000,
     STOP_PCT=0.03, START_YEAR=2004, VIX_HALT=40.0, VIX_RESUME=39.0, COOLDOWN_D =10,
     cost: CostModel = CostModel(0.0050,0.0025, 0.0008,0.0007)
):
     """
     Short 30d strangle at 10, dynamic hedge using predicted IV; decisions use same-day IV (forward bias allowed).
     """
     global ATTN_FALLBACK_TOTAL
     print("[INIT] ATTN module loaded. Dynamic hedge (pred IV; fallback realised) . START_YEAR=2004. Dynamic r used.")
     print("[FORWARD BIAS] Using current day's IV for all calculations.")
     spot = load_spot(START_YEAR)
     iv_real = load_iv_real(front_t, START_YEAR)
     spot, iv_real, dates = align_trading_idx(spot, iv_real)
     rf_vix, rf_fallback_days, vix_nan_days = fetch_rf_and_vix(dates)
     print(f"[FRED] RF fallback days: {rf_fallback_days} | VIX NaN days: { vix_nan_days}")
     iv_attn = _load_iv_attn(front_t, START_YEAR, dates)

     T1 = front_t / CALENDAR_YEAR
     pnl, eq, idx = [], [INITIAL_CAPITAL], []
     pos = []
     blocked = False; below39 = 0
     fallback_count = 0

     position_history = []
  hedge_performance = []
  iv_errors_c = []
  iv_errors_p = []
  total_trans_cost = 0.0

  for i, d in enumerate(dates):
     S = _cell(spot, d, "spot")
     r = float(rf_vix.loc[d,'rf']) if 'rf' in rf_vix.columns else 0.0
     v = float(rf_vix.loc[d,'vix']) if 'vix' in rf_vix.columns else np.nan


     if not np.isnan(v):
         if v >= VIX_HALT: blocked = True; below39 = 0
         elif blocked:
            if v < VIX_RESUME:
                below39 += 1
                if below39 >= COOLDOWN_D: blocked = False; below39 = 0
            else:
                below39 = 0


     if not pos:
         if (not blocked) and (np.isnan(v) or v < VIX_HALT):
            sig_c_r = _cell(iv_real, d, f"{front_t}_{CALL_DELTA}")
            sig_p_r = _cell(iv_real, d, f"{front_t}_{PUT_DELTA}")
            Kc = strike_from_delta(S, T1, sig_c_r, CALL_DELTA/100.0, 'c', r)
            Kp = strike_from_delta(S, T1, sig_p_r, PUT_DELTA/100.0, 'p', r)
            prem = bs_price(S, Kc, T1, sig_c_r, 'c', r) + bs_price(S, Kp, T1, sig_p_r, 'p', r)
            if np.isfinite(prem) and prem > 0:
                sig_c_p = _cell(iv_attn, d, f"{front_t}_{CALL_DELTA}")
                sig_p_p = _cell(iv_attn, d, f"{front_t}_{PUT_DELTA}")
                if not np.isfinite(sig_c_p) or not np.isfinite(sig_p_p):
                     sig_c_p, sig_p_p = sig_c_r, sig_p_r
                     fallback_count += 1
                     ATTN_FALLBACK_TOTAL += 1
                if 15 <= v < 25 and ((sig_c_p + sig_p_p)/2 > 1.05 * (sig_c_r + sig_p_r)/2):
                     continue
                iv_errors_c.append(abs(sig_c_p - sig_c_r))
                iv_errors_p.append(abs(sig_p_p - sig_p_r))
            units = -NOTIONAL / prem
            d_call_p = bs_delta(S, Kc, T1, sig_c_p, 'c', r)
            d_put_p = bs_delta(S, Kp, T1, sig_p_p, 'p', r)
            hedge_target = -units * (d_call_p + d_put_p)
            hedge_shares = hedge_target
            entry_opt_cost = cost.option_cost(abs(units) * prem)
            entry_und_cost = cost.underlying_cost(abs(hedge_shares) * S)
            entry_costs = entry_opt_cost + entry_und_cost
            total_trans_cost += entry_costs
            pnl.append(-entry_costs)
            eq.append(eq[-1] - entry_costs); idx.append(d)
            pos = [{
                 "Kc": Kc, "Kp": Kp, "T": T1, "units": units,
                 "V0": -units * prem,
                 "hedge_shares": hedge_shares,
                 "prev_S": S, "entry_S": S, "entry_day": d, "prev_d": d,
                 "pnl_since_entry": -entry_costs,
                 "entry_sig_c": sig_c_p,
                 "entry_sig_p": sig_p_p
            }]
            if TRACK_POSITION_LIFECYCLES:
                 position_history.append({
                      "date": d, "action": "ENTRY", "Kc": Kc, "Kp": Kp,
                      "units": units, "premium": prem, "hedge_shares": hedge_shares, "vix": v,
                      "fallback_used": not np.isfinite(_cell(iv_attn, d, f"{ front_t}_{CALL_DELTA}")) and
                                     not np.isfinite(_cell(iv_attn, d, f"{ front_t}_{PUT_DELTA}"))
                 })
         continue
     continue


  S_now = S; r_now = r
  days_elapsed_calendar = (d - pos[0]["prev_d"]).days
  T_rem = max(0.0, pos[0]["T"] - days_elapsed_calendar / CALENDAR_YEAR)
  days_elapsed_trading = _trading_days_between(pos[0]["prev_d"], d, dates)

  sig_c_now_r = get_iv_for_strike(pos[0]["Kc"], S_now, iv_real, d, front_t , True, r_now)
  sig_p_now_r = get_iv_for_strike(pos[0]["Kp"], S_now, iv_real, d, front_t , False, r_now)
  prem_now = bs_price(S_now, pos[0]["Kc"], T_rem, sig_c_now_r, 'c', r_now) +\
              bs_price(S_now, pos[0]["Kp"], T_rem, sig_p_now_r, 'p', r_now )
  price_now = -pos[0]["units"] * prem_now

  option_pnl = pos[0]["V0"] - price_now
  dS = S_now - pos[0]["prev_S"]
  hedge_pnl = pos[0]["hedge_shares"] * dS

  sig_c_p = _cell(iv_attn, d, f"{front_t}_{CALL_DELTA}")
  sig_p_p = _cell(iv_attn, d, f"{front_t}_{PUT_DELTA}")
  if not np.isfinite(sig_c_p) or not np.isfinite(sig_p_p):
     sig_c_p, sig_p_p = sig_c_now_r, sig_p_now_r
     fallback_count += 1
     ATTN_FALLBACK_TOTAL += 1

  iv_errors_c.append(abs(sig_c_p - sig_c_now_r))
  iv_errors_p.append(abs(sig_p_p - sig_p_now_r))

  d_call_h = bs_delta(S_now, pos[0]["Kc"], T_rem, sig_c_p, 'c', r_now)
  d_put_h = bs_delta(S_now, pos[0]["Kp"], T_rem, sig_p_p, 'p', r_now)
  target = -pos[0]["units"] * (d_call_h + d_put_h)

  hedge_new = (1.0 - HEDGE_SMOOTH_ALPHA)*pos[0]["hedge_shares"] + HEDGE_SMOOTH_ALPHA*target
  d_shares = hedge_new - pos[0]["hedge_shares"]
  max_turn_notional = MAX_HEDGE_TURNOVER_X * INITIAL_CAPITAL
  traded_notional = abs(d_shares) * S_now
  if traded_notional > max_turn_notional and traded_notional > 0:
     scale = max_turn_notional / traded_notional
     d_shares *= scale
     hedge_new = pos[0]["hedge_shares"] + d_shares
     traded_notional = abs(d_shares) * S_now

  rehedge_cost = cost.underlying_cost(traded_notional)
  total_trans_cost += rehedge_cost

  rf_day = eq[-1] * (r_now / CALENDAR_YEAR) * days_elapsed_calendar

  pnl_day_core = option_pnl + hedge_pnl - rehedge_cost + rf_day
  pnl_tmp = pos[0]["pnl_since_entry"] + pnl_day_core

  days_live = _trading_days_between(pos[0]["entry_day"], d, dates)
  force_vix = (not np.isnan(v)) and (v >= VIX_HALT)
  final_day = (i == len(dates)-1)
  stop_loss = (pnl_tmp <= -STOP_PCT * NOTIONAL)
  will_exit = force_vix or stop_loss or (days_live>=MAX_HOLD) or final_day

  pnl_day = pnl_day_core
  if will_exit:
     exit_opt_cost = cost.option_cost(abs(pos[0]["units"]) * prem_now)
     exit_und_cost = cost.underlying_cost(abs(hedge_new) * S_now)
     pnl_day -= (exit_opt_cost + exit_und_cost)
     total_trans_cost += (exit_opt_cost + exit_und_cost)

  eq.append(eq[-1] + pnl_day); pnl.append(pnl_day); idx.append(d)

  if ANALYZE_HEDGE_EFFECTIVENESS and pos:
     hedge_effectiveness = 1 - (abs(option_pnl) / (abs(option_pnl) + abs( hedge_pnl))) if (abs(option_pnl) + abs(hedge_pnl)) > 0 else 1
     hedge_performance.append({"date": d, "effectiveness": hedge_effectiveness})

  pos[0]["V0"] = price_now
  pos[0]["prev_S"] = S_now
  pos[0]["prev_d"] = d
  pos[0]["T"] = T_rem
  pos[0]["hedge_shares"] = hedge_new
  pos[0]["pnl_since_entry"] = pnl_tmp if not will_exit else 0.0

  if will_exit:
     if TRACK_POSITION_LIFECYCLES:
         days_held = _trading_days_between(pos[0]["entry_day"], d, dates) - 1
         position_history.append({
            "date": d, "action": "EXIT",
            "exit_reason": "VIX" if force_vix else ("STOP" if stop_loss else ("MAX_HOLD" if days_live>=MAX_HOLD else "FINAL")),
            "days_held": days_live, "final_pnl": pnl_tmp,
                    "stop_threshold": -STOP_PCT * NOTIONAL, "vix": v
               })
         pos.clear()
         if force_vix: blocked = True; below39 = 0

  fallback_pct = fallback_count / len(dates) * 100 if len(dates) > 0 else 0
  print(f"[ATTN] Fallback to realized IV: {fallback_count} times ({ fallback_pct:.2f}% of days)")
  print(f"[ATTN] Total global fallback to realized IV: {ATTN_FALLBACK_TOTAL} times")

  mae_c, rmse_c, mae_p, rmse_p, errors_c, errors_p = compute_iv_errors(iv_attn , iv_real, front_t, dates, CALL_DELTA, PUT_DELTA)
  print(f"[ATTN] IV Prediction Accuracy: Call MAE={mae_c:.4f}, RMSE={rmse_c:.4 f}; Put MAE={mae_p:.4f}, RMSE={rmse_p:.4f}")

  res = pd.DataFrame({"pnl": pnl, "equity": eq[1:]}, index=idx)
  res['cum_pnl'] = res['equity']
  res['dd_pct'] = _drawdown(res['equity'])
  res['ret'] = _pct_change(res['equity'])
  res['vix'] = rf_vix['vix']

  ok = _reconcile_equity(res['equity'], res['pnl'], INITIAL_CAPITAL)
  print(f"\n[SANITY][ATTN] (recon from equity) {'PASS' if ok else 'FAIL'}.\n")
  if RUN_DETAILED_CHECKS:
     detailed_reconciliation_check(res['equity'], res['pnl'], INITIAL_CAPITAL , "ATTN")

  if ANALYZE_HEDGE_EFFECTIVENESS and hedge_performance:
     hedge_df = pd.DataFrame(hedge_performance)
     hedge_df.to_csv(f"{ROOT}/{FAMILY_NAME}_attn_hedge_effectiveness.csv", index=False)
     print(f"[ATTN] Hedge effectiveness stats: Mean = {hedge_df[' effectiveness'].mean():.3f}, "
           f"Min = {hedge_df['effectiveness'].min():.3f}, "
           f"Max = {hedge_df['effectiveness'].max():.3f}")

  if TRACK_POSITION_LIFECYCLES and position_history:
     pos_df = pd.DataFrame(position_history)
     pos_df.to_csv(f"{ROOT}/{FAMILY_NAME}_attn_position_history.csv", index= False)
         print(f"[ATTN] Position history saved: {ROOT}/{FAMILY_NAME} _attn_position_history.csv")

   cum_pnl = res['cum_pnl'].iloc[-1]
   cost_pct = (total_trans_cost / cum_pnl) * 100 if cum_pnl > 0 else 0
   print(f"[ATTN] Total transaction costs: ${total_trans_cost:.2f} ({cost_pct :.2f}% of cum PnL)")

   return res, errors_c, errors_p, dates



# func conv preds wide or long to wide


def conv_preds_wide_or_long_to_wide(raw: pd.DataFrame, iv_real_cols, step_ahead =20) -> pd.DataFrame:
   """
   Accept wide (date-indexed grid) or long format with maturity_bucket/ delta_bucket/step_ahead.
   Returns wide (index=date, columns subset of iv_real_cols).
   """
   if {'maturity_bucket','delta_bucket','step_ahead'}.issubset(raw.columns):
         if 'date' in raw.columns:
            raw['date'] = pd.to_datetime(raw['date'])
         elif 'origin_date' in raw.columns:
            raw['date'] = pd.to_datetime(raw['origin_date'])
         else:
            raise ValueError("Conv preds long-format needs 'date' or 'origin_date '.")

         raw = raw[raw['step_ahead'] == step_ahead].copy()
         valcol = 'iv_pred_raw' if 'iv_pred_raw' in raw.columns else ('iv_pred' if 'iv_pred' in raw.columns else None)
         if valcol is None:
            raise ValueError("Conv preds need an 'iv_pred' or 'iv_pred_raw' column.")

         mats = sorted({int(c.split('_')[0]) for c in iv_real_cols})
         dels = sorted({int(c.split('_')[1]) for c in iv_real_cols})

       def _infer_map(series, universe_sorted):
           series = series.astype(int)
           k = series.max()+1
           if k == len(universe_sorted) and set(series.unique()) <= set(range(k) ):
              return {i: universe_sorted[i] for i in range(k)}
           uniq = sorted(series.unique())
           if set(uniq) == set(range(23)):
              return {i:(i-11)*5 for i in range(23)}
           if set(uniq) == {0,1,2}:
              return {0:-30,1:0,2:30}
           return {b: universe_sorted[i%len(universe_sorted)] for i,b in enumerate(uniq)}

       m_map = _infer_map(raw['maturity_bucket'], mats)
       d_map = _infer_map(raw['delta_bucket'], dels)
       raw['grid_col'] = raw.apply(lambda r: f"{m_map[int(r.maturity_bucket)]}_ {d_map[int(r.delta_bucket)]}", axis=1)
       raw = raw.sort_values(['date','maturity_bucket','delta_bucket']). drop_duplicates(['date','grid_col'], keep='last')
       wide = raw.pivot(index='date', columns='grid_col', values=valcol). sort_index()
       return wide
   else:
       if 'date' in raw.columns:
           raw = raw.assign(date=pd.to_datetime(raw['date'])).set_index('date')
           raw = raw.drop(columns=['date'], errors='ignore')
# func load iv conv
       return raw

def _load_iv_conv(front_t: int, start_year: int, trading_idx: pd.DatetimeIndex, iv_real_cols) -> pd.DataFrame:
   """Load/align CONV-LSTM IV predictions for 10."""
   need = [f"{front_t}_{-10}", f"{front_t}_{10}"]
   raw = pd.read_csv(IV_CONV_CSV)
   if 'date' in raw.columns or {'maturity_bucket','delta_bucket','step_ahead'}. issubset(raw.columns):
       wide = conv_preds_wide_or_long_to_wide(raw, iv_real_cols, step_ahead=20)
   else:
       wide = raw

     if 'date' in wide.columns:
           wide = wide.assign(date=pd.to_datetime(wide['date'])).set_index('date'). drop(columns=['date'], errors='ignore')

     missing = [c for c in need if c not in wide.columns]
     if missing:
           raise KeyError(f"CONV-LSTM predicted IV missing columns: {missing}")

     wide = _to_numeric(wide[need]).clip(lower=VOL_FLOOR)
     wide = wide[wide.index >= f"{start_year}-01-01"]
     wide = wide.reindex(trading_idx).ffill()

     usable = wide[need].notna().all(axis=1).sum()
     print(f"[CONV] usable days with both 10 preds: {usable} / {len(trading_idx)} ")
# func run conv strategy
     return wide

def run_conv_strategy(*,
     front_t=30, CALL_DELTA=10, PUT_DELTA=-10, MAX_HOLD=20, NOTIONAL=1_000_000,
     STOP_PCT=0.03, START_YEAR=2004, VIX_HALT=40.0, VIX_RESUME=39.0, COOLDOWN_D =10,
     cost: CostModel = CostModel(0.0050,0.0025, 0.0008,0.0007)
):
     """
     Short 30d strangle at 10, dynamic hedge using predicted IV; decisions use same-day IV (forward bias allowed).
     """
     global CONV_FALLBACK_TOTAL
     print("[INIT] CONV module loaded. Dynamic hedge (pred IV; fallback realised) . START_YEAR=2004. Dynamic r used.")
     print("[FORWARD BIAS] Using current day's IV for all calculations.")
     spot = load_spot(START_YEAR)
     iv_real = load_iv_real(front_t, START_YEAR)
     spot, iv_real, dates = align_trading_idx(spot, iv_real)
     rf_vix, rf_fallback_days, vix_nan_days = fetch_rf_and_vix(dates)
     print(f"[FRED] RF fallback days: {rf_fallback_days} | VIX NaN days: { vix_nan_days}")
     iv_conv = _load_iv_conv(front_t, START_YEAR, dates, iv_real.columns)

     T1 = front_t / CALENDAR_YEAR
  pnl, eq, idx = [], [INITIAL_CAPITAL], []
  pos = []
  blocked = False; below39 = 0
  fallback_count = 0

  position_history = []
  hedge_performance = []
  iv_errors_c = []
  iv_errors_p = []
  total_trans_cost = 0.0

  for i, d in enumerate(dates):
     S = _cell(spot, d, "spot")
     r = float(rf_vix.loc[d,'rf']) if 'rf' in rf_vix.columns else 0.0
     v = float(rf_vix.loc[d,'vix']) if 'vix' in rf_vix.columns else np.nan


     if not np.isnan(v):
         if v >= VIX_HALT: blocked = True; below39 = 0
         elif blocked:
             if v < VIX_RESUME:
                below39 += 1
                if below39 >= COOLDOWN_D: blocked = False; below39 = 0
             else:
                below39 = 0


     if not pos:
         if (not blocked) and (np.isnan(v) or v < VIX_HALT):
             sig_c_r = _cell(iv_real, d, f"{front_t}_{CALL_DELTA}")
             sig_p_r = _cell(iv_real, d, f"{front_t}_{PUT_DELTA}")
             Kc = strike_from_delta(S, T1, sig_c_r, CALL_DELTA/100.0, 'c', r)
             Kp = strike_from_delta(S, T1, sig_p_r, PUT_DELTA/100.0, 'p', r)
             prem = bs_price(S, Kc, T1, sig_c_r, 'c', r) + bs_price(S, Kp, T1, sig_p_r, 'p', r)
             if np.isfinite(prem) and prem > 0:
                sig_c_p = _cell(iv_conv, d, f"{front_t}_{CALL_DELTA}") if d in iv_conv.index else np.nan
                sig_p_p = _cell(iv_conv, d, f"{front_t}_{PUT_DELTA}") if d in iv_conv.index else np.nan
                if not np.isfinite(sig_c_p) or not np.isfinite(sig_p_p):
              sig_c_p, sig_p_p = sig_c_r, sig_p_r
              fallback_count += 1
              CONV_FALLBACK_TOTAL += 1
         if 15 <= v < 25 and ((sig_c_p + sig_p_p)/2 > 1.05 * (sig_c_r + sig_p_r)/2):
              continue
         iv_errors_c.append(abs(sig_c_p - sig_c_r))
         iv_errors_p.append(abs(sig_p_p - sig_p_r))
         units = -NOTIONAL / prem
         d_call_p = bs_delta(S, Kc, T1, sig_c_p, 'c', r)
         d_put_p = bs_delta(S, Kp, T1, sig_p_p, 'p', r)
         hedge_target = -units * (d_call_p + d_put_p)
         hedge_shares = hedge_target
         entry_opt_cost = cost.option_cost(abs(units) * prem)
         entry_und_cost = cost.underlying_cost(abs(hedge_shares) * S)
         entry_costs = entry_opt_cost + entry_und_cost
         total_trans_cost += entry_costs
         pnl.append(-entry_costs)
         eq.append(eq[-1] - entry_costs); idx.append(d)
         pos = [{
              "Kc": Kc, "Kp": Kp, "T": T1, "units": units,
              "V0": -units * prem,
              "hedge_shares": hedge_shares,
              "prev_S": S, "entry_S": S, "entry_day": d, "prev_d": d,
              "pnl_since_entry": -entry_costs,
              "entry_sig_c": sig_c_p,
              "entry_sig_p": sig_p_p
         }]
         if TRACK_POSITION_LIFECYCLES:
              position_history.append({
                   "date": d, "action": "ENTRY", "Kc": Kc, "Kp": Kp,
                   "units": units, "premium": prem, "hedge_shares": hedge_shares, "vix": v,
                   "fallback_used": not np.isfinite(_cell(iv_conv, d, f"{ front_t}_{CALL_DELTA}")) and
                                  not np.isfinite(_cell(iv_conv, d, f"{ front_t}_{PUT_DELTA}"))
              })
     continue
  continue


  S_now = S; r_now = r
  days_elapsed_calendar = (d - pos[0]["prev_d"]).days
  T_rem = max(0.0, pos[0]["T"] - days_elapsed_calendar / CALENDAR_YEAR)
  days_elapsed_trading = _trading_days_between(pos[0]["prev_d"], d, dates)

  sig_c_now_r = get_iv_for_strike(pos[0]["Kc"], S_now, iv_real, d, front_t , True, r_now)
  sig_p_now_r = get_iv_for_strike(pos[0]["Kp"], S_now, iv_real, d, front_t , False, r_now)
  prem_now = bs_price(S_now, pos[0]["Kc"], T_rem, sig_c_now_r, 'c', r_now) +\
              bs_price(S_now, pos[0]["Kp"], T_rem, sig_p_now_r, 'p', r_now )
  price_now = -pos[0]["units"] * prem_now

  option_pnl = pos[0]["V0"] - price_now
  dS = S_now - pos[0]["prev_S"]
  hedge_pnl = pos[0]["hedge_shares"] * dS

  sig_c_p = _cell(iv_conv, d, f"{front_t}_{CALL_DELTA}") if d in iv_conv. index else np.nan
  sig_p_p = _cell(iv_conv, d, f"{front_t}_{PUT_DELTA}") if d in iv_conv. index else np.nan
  if not np.isfinite(sig_c_p) or not np.isfinite(sig_p_p):
     sig_c_p, sig_p_p = sig_c_now_r, sig_p_now_r
     fallback_count += 1
     CONV_FALLBACK_TOTAL += 1

  iv_errors_c.append(abs(sig_c_p - sig_c_now_r))
  iv_errors_p.append(abs(sig_p_p - sig_p_now_r))

  d_call_h = bs_delta(S_now, pos[0]["Kc"], T_rem, sig_c_p, 'c', r_now)
  d_put_h = bs_delta(S_now, pos[0]["Kp"], T_rem, sig_p_p, 'p', r_now)
  target = -pos[0]["units"] * (d_call_h + d_put_h)

  hedge_new = (1.0 - HEDGE_SMOOTH_ALPHA)*pos[0]["hedge_shares"] + HEDGE_SMOOTH_ALPHA*target
  d_shares = hedge_new - pos[0]["hedge_shares"]
  max_turn_notional = MAX_HEDGE_TURNOVER_X * INITIAL_CAPITAL
  traded_notional = abs(d_shares) * S_now
  if traded_notional > max_turn_notional and traded_notional > 0:
     scale = max_turn_notional / traded_notional
     d_shares *= scale
     hedge_new = pos[0]["hedge_shares"] + d_shares
     traded_notional = abs(d_shares) * S_now

  rehedge_cost = cost.underlying_cost(traded_notional)
  total_trans_cost += rehedge_cost

  rf_day = eq[-1] * (r_now / CALENDAR_YEAR) * days_elapsed_calendar

  pnl_day_core = option_pnl + hedge_pnl - rehedge_cost + rf_day
  pnl_tmp = pos[0]["pnl_since_entry"] + pnl_day_core

  days_live = _trading_days_between(pos[0]["entry_day"], d, dates)
  force_vix = (not np.isnan(v)) and (v >= VIX_HALT)
  final_day = (i == len(dates)-1)
  stop_loss = (pnl_tmp <= -STOP_PCT * NOTIONAL)
  will_exit = force_vix or stop_loss or (days_live>=MAX_HOLD) or final_day

  pnl_day = pnl_day_core
  if will_exit:
     exit_opt_cost = cost.option_cost(abs(pos[0]["units"]) * prem_now)
     exit_und_cost = cost.underlying_cost(abs(hedge_new) * S_now)
     pnl_day -= (exit_opt_cost + exit_und_cost)
     total_trans_cost += (exit_opt_cost + exit_und_cost)

  eq.append(eq[-1] + pnl_day); pnl.append(pnl_day); idx.append(d)

  if ANALYZE_HEDGE_EFFECTIVENESS and pos:
     hedge_effectiveness = 1 - (abs(option_pnl) / (abs(option_pnl) + abs( hedge_pnl))) if (abs(option_pnl) + abs(hedge_pnl)) > 0 else 1
     hedge_performance.append({"date": d, "effectiveness": hedge_effectiveness})

  pos[0]["V0"] = price_now
  pos[0]["prev_S"] = S_now
  pos[0]["prev_d"] = d
  pos[0]["T"] = T_rem
  pos[0]["hedge_shares"] = hedge_new
  pos[0]["pnl_since_entry"] = pnl_tmp if not will_exit else 0.0

     if will_exit:
         if TRACK_POSITION_LIFECYCLES:
               days_held = _trading_days_between(pos[0]["entry_day"], d, dates) - 1
               position_history.append({
                    "date": d, "action": "EXIT",
                    "exit_reason": "VIX" if force_vix else ("STOP" if stop_loss else ("MAX_HOLD" if days_live>=MAX_HOLD else "FINAL")),
                    "days_held": days_live, "final_pnl": pnl_tmp,
                    "stop_threshold": -STOP_PCT * NOTIONAL, "vix": v
               })
         pos.clear()
         if force_vix: blocked = True; below39 = 0

  fallback_pct = fallback_count / len(dates) * 100 if len(dates) > 0 else 0
  print(f"[CONV] Fallback to realized IV: {fallback_count} times ({ fallback_pct:.2f}% of days)")
  print(f"[CONV] Total global fallback to realized IV: {CONV_FALLBACK_TOTAL} times")

  mae_c, rmse_c, mae_p, rmse_p, errors_c, errors_p = compute_iv_errors(iv_conv , iv_real, front_t, dates, CALL_DELTA, PUT_DELTA)
  print(f"[CONV] IV Prediction Accuracy: Call MAE={mae_c:.4f}, RMSE={rmse_c:.4 f}; Put MAE={mae_p:.4f}, RMSE={rmse_p:.4f}")

  res = pd.DataFrame({"pnl": pnl, "equity": eq[1:]}, index=idx)
  res['cum_pnl'] = res['equity']
  res['dd_pct'] = _drawdown(res['equity'])
  res['ret'] = _pct_change(res['equity'])
  res['vix'] = rf_vix['vix']

  ok = _reconcile_equity(res['equity'], res['pnl'], INITIAL_CAPITAL)
  print(f"\n[SANITY][CONV] (recon from equity) {'PASS' if ok else 'FAIL'}.\n")
  if RUN_DETAILED_CHECKS:
     detailed_reconciliation_check(res['equity'], res['pnl'], INITIAL_CAPITAL , "CONV")

  if ANALYZE_HEDGE_EFFECTIVENESS and hedge_performance:
     hedge_df = pd.DataFrame(hedge_performance)
       hedge_df.to_csv(f"{ROOT}/{FAMILY_NAME}_conv_hedge_effectiveness.csv", index=False)
       print(f"[CONV] Hedge effectiveness stats: Mean = {hedge_df[' effectiveness'].mean():.3f}, "
            f"Min = {hedge_df['effectiveness'].min():.3f}, "
            f"Max = {hedge_df['effectiveness'].max():.3f}")

   if TRACK_POSITION_LIFECYCLES and position_history:
       pos_df = pd.DataFrame(position_history)
       pos_df.to_csv(f"{ROOT}/{FAMILY_NAME}_conv_position_history.csv", index= False)
       print(f"[CONV] Position history saved: {ROOT}/{FAMILY_NAME} _conv_position_history.csv")

   cum_pnl = res['cum_pnl'].iloc[-1]
   cost_pct = (total_trans_cost / cum_pnl) * 100 if cum_pnl > 0 else 0
   print(f"[CONV] Total transaction costs: ${total_trans_cost:.2f} ({cost_pct :.2f}% of cum PnL)")

   return res, errors_c, errors_p, dates



# func save results and reports


def save_results_and_reports(res_df: pd.DataFrame, *, name_prefix: str,
                            title: str, tag: str, initial: float):
   """Write results.csv, metrics.csv, and MC summary."""
   out_csv = f"{ROOT}/{name_prefix}_results_{tag}.csv"
   res_df.to_csv(out_csv)
   metrics = performance_report(res_df['equity'], initial, title=f"{title} [{ tag}]")
   metrics_csv = f"{ROOT}/{name_prefix}_metrics_{tag}.csv"
   metrics.to_csv(metrics_csv, index=False)
   mc = stationary_bootstrap_summaries(res_df['equity'], initial, runs=5000, block_mean=10, seed=42)
   if mc is not None and not mc.empty:
       mc_csv = f"{ROOT}/{name_prefix}_mc_{tag}.csv"
       mc.to_csv(mc_csv, index=False)
       print(f"Saved MC summary: {mc_csv}")
   print(f"Saved results: {out_csv}")
# func plot equity dd single
   print(f"Saved metrics: {metrics_csv}")

def plot_equity_dd_single(df: pd.DataFrame, title: str, out_png: str):
   """Single equity curve + drawdown."""
   plt.style.use("ggplot")
   fig, (ax1, ax2) = plt.subplots(2,1, figsize=(16,10), sharex=True, gridspec_kw={"height_ratios":[3,1]})
   ax1.plot(df.index, df['equity'], linewidth=2, label=f"{title} (Final: ${df[' equity'].iloc[-1]:,.0f})")
   ax1.axhline(INITIAL_CAPITAL, linestyle='--', alpha=0.6, color='gray')
   ax1.set_title(title); ax1.set_ylabel("Equity ($)")
   ax1.grid(True, linestyle="--", alpha=0.7); ax1.legend(loc="best")
   dd = (df['equity'].cummax() - df['equity'])/df['equity'].cummax()
   ax2.fill_between(df.index, dd*100.0, 0, alpha=0.3, label=f"Max DD: {dd.max() *100:.1f}%")
   ax2.set_ylabel("Drawdown (%)"); ax2.grid(True, linestyle="--", alpha=0.7); ax2.legend(loc="best")
   ax2.xaxis.set_major_locator(mdates.YearLocator()); ax2.xaxis. set_major_formatter(mdates.DateFormatter("%Y"))
   plt.xlabel("Date"); plt.tight_layout(); plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.show()
# func plot combined common
   print(f"Saved plot   {out_png}")

def plot_combined_common(curves: Dict[str,pd.DataFrame], title: str, out_png: str):
   """Combined equity + drawdown on strict common window."""
   plt.style.use("ggplot")
   fig, (ax1, ax2) = plt.subplots(2,1, figsize=(16,10), sharex=True, gridspec_kw={"height_ratios":[3,1]})
   for label, df in curves.items():
       ax1.plot(df.index, df['equity'], linewidth=2, label=f"{label} (Final: ${ df['equity'].iloc[-1]:,.0f})")
   ax1.axhline(INITIAL_CAPITAL, linestyle='--', alpha=0.6, color='gray')
   ax1.set_title(f"{title}     Combined (Strict Common Window)")
   ax1.set_ylabel("Equity ($)"); ax1.grid(True, linestyle="--", alpha=0.7); ax1 .legend(loc="best")
   for label, df in curves.items():
       dd = (df['equity'].cummax() - df['equity'])/df['equity'].cummax()
       ax2.plot(df.index, dd*100.0, linewidth=1.8, label=f"{label} DD (max {dd. max()*100:.1f}%)")
   ax2.set_ylabel("Drawdown (%)"); ax2.grid(True, linestyle="--", alpha=0.7); ax2.legend(loc="best")
   ax2.xaxis.set_major_locator(mdates.YearLocator()); ax2.xaxis. set_major_formatter(mdates.DateFormatter("%Y"))
   plt.xlabel("Date"); plt.tight_layout(); plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.show()
   print(f"Saved plot    {out_png}")



# func enforce common and rebase


def enforce_common_and_rebase(rule_res: pd.DataFrame,
                            attn_res: pd.DataFrame,
                            conv_res: pd.DataFrame,
                            initial: float = INITIAL_CAPITAL):
   """Intersect indices and rebase equity = initial + cumsum(pnl) within window ."""
   common_idx = rule_res.index.intersection(attn_res.index).intersection( conv_res.index)
   if common_idx.empty:
       raise RuntimeError("Common date intersection is empty     check your inputs .")

   def _view(df):
       d = df.loc[common_idx].copy()
       d['equity'] = initial + d['pnl'].cumsum()
       d['cum_pnl'] = d['equity']
       d['dd_pct'] = _drawdown(d['equity'])
       d['ret'] = _pct_change(d['equity'])
       return d

   return _view(rule_res), _view(attn_res), _view(conv_res), common_idx



# func analyze performance by regime


def analyze_performance_by_regime(equity, vix):
   """Cum return by VIX regime."""
   regimes = {
       "Low Volatility (VIX < 15)": vix < 15,
       "Normal (15   VIX < 25)": (vix >= 15) & (vix < 25),
       "High Volatility (VIX   25)": vix >= 25
   }
   regime_returns = {}
   for regime_name, mask in regimes.items():
       regime_rets = equity.pct_change().loc[mask]
       if len(regime_rets) > 0:
          regime_cum_return = (1 + regime_rets).prod() - 1
          regime_returns[regime_name] = regime_cum_return
# func parameter sensitivity analysis
   return regime_returns

def parameter_sensitivity_analysis(strategy_func, param_variations):
   """One-factor sweeps; returns dict of summary stats."""
   results = {}
   for param_name, values in param_variations.items():
       param_results = {}
       for value in values:
          print(f"Testing {param_name} = {value}")
          if param_name == "cost":
              key = f"comm_{value.option_comm_bps}_slip_{value.option_slip_bps} _undcomm_{value.und_comm_bps}_undslip_{value.und_slip_bps}"
              result = strategy_func(cost=value)
          elif param_name == "HEDGE_SMOOTH_ALPHA":
              global HEDGE_SMOOTH_ALPHA
              original = HEDGE_SMOOTH_ALPHA
              HEDGE_SMOOTH_ALPHA = value
              key = value
              result = strategy_func()
              HEDGE_SMOOTH_ALPHA = original
          elif param_name == "STOP_PCT":
              key = value
              result = strategy_func(STOP_PCT=value)
          elif param_name == "VIX_HALT":
              key = value
              result = strategy_func(VIX_HALT=value)
            else:
                key = value
                result = strategy_func()

            final_equity = result['equity'].iloc[-1]
            total_return = final_equity / INITIAL_CAPITAL - 1
            max_dd = _drawdown(result['equity']).max()
            ann_ret = (1+total_return)**(252/len(result)) - 1
            ann_vol = result['ret'].std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

            param_results[key] = {
                "final_equity": final_equity,
                "total_return": total_return,
                "max_drawdown": max_dd,
                "Sharpe0": sharpe
            }

         results[param_name] = param_results
   return results



# func load spy adj close


def _load_spy_adj_close(start_date, end_date):
   """
   SPY adjusted close (dividends reinvested):
   (1) pandas_datareader Yahoo, (2) yfinance auto-adjust, (3) local CSV fallback.
   Returns Series named 'adj'.
   """
   if _PDR_OK and pdr is not None:
         try:
            s = pdr.DataReader('SPY', 'yahoo', start_date, end_date)['Adj Close']
            s.name = 'adj'
            return s
         except Exception:
            pass

   try:
       import yfinance as yf
       df = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True , progress=False)
       if not df.empty:
          s = df['Close'].rename('adj')
          return s
   except Exception:
       pass

   import os
   import pandas as pd
   csv_path = './SPY_adj_close.csv'
   if os.path.exists(csv_path):
       s = (pd.read_csv(csv_path, parse_dates=['Date'])
                 .set_index('Date')
                 .sort_index()['Adj Close']
                 .rename('adj'))
       return s.loc[(s.index >= pd.Timestamp(start_date)) & (s.index <= pd. Timestamp(end_date))]

# func build total return benchmark
   raise RuntimeError("Unable to obtain SPY adjusted-close via pandas_datareader, yfinance, or local CSV.")

def build_total_return_benchmark(common_idx, initial_capital=INITIAL_CAPITAL):
   """SPY total-return benchmark invested with 'initial_capital' on first common date."""
   if len(common_idx) == 0:
       raise ValueError("common_idx is empty")
   start_date = common_idx.min().date()
   end_date = (common_idx.max() + pd.Timedelta(days=1)).date()

   spy_adj = _load_spy_adj_close(start_date, end_date).reindex(common_idx). ffill()
   if spy_adj.isna().all():
       raise RuntimeError("SPY adjusted-close series is empty after alignment to common_idx.")

   spy0 = float(spy_adj.iloc[0])
   units = initial_capital / spy0

   bench = pd.DataFrame(index=common_idx)
   bench['equity'] = initial_capital * (spy_adj / spy0)
   bench['ret'] = spy_adj.pct_change()
   bench['pnl'] = spy_adj.diff() * units
   return bench





if __name__ == "__main__":

   rule_raw = run_rule_strategy(START_YEAR=2004)
   attn_raw, attn_iv_errors_c, attn_iv_errors_p, attn_dates = run_attn_strategy (START_YEAR=2004)
   conv_raw, conv_iv_errors_c, conv_iv_errors_p, conv_dates = run_conv_strategy (START_YEAR=2004)


   save_results_and_reports(rule_raw,
       name_prefix="strangle_rule", title="RULE-BASED Short Strangle    Dynamic Hedge (dynamic r)",
       tag="RAW", initial=INITIAL_CAPITAL
   )
   save_results_and_reports(attn_raw,
       name_prefix="strangle_attn_hedgeonly", title="ATTN-LSTM Hedge-   Dynamic Hedge (dynamic r)",
       tag="RAW", initial=INITIAL_CAPITAL
   )
   save_results_and_reports(conv_raw,
       name_prefix="strangle_convlstm_hedgeonly", title="CONV-LSTM Hedge- Dynamic Hedge (dynamic r)",
       tag="RAW", initial=INITIAL_CAPITAL
   )


   rule_c, attn_c, conv_c, common_idx = enforce_common_and_rebase(rule_raw, attn_raw, conv_raw, INITIAL_CAPITAL)
   print(f"[COMMON] Window: {common_idx.min().date()}   {common_idx.max().date() } | days: {len(common_idx)}")


  for name, df in [("RULE", rule_c), ("ATTN", attn_c), ("CONV", conv_c)]:
      ok = _reconcile_equity(df['equity'], df['pnl'], INITIAL_CAPITAL)
      print(f"[SANITY][{name}][COMMON] (recon) {'PASS' if ok else 'FAIL'}")


  try:
      spx_spot = load_spot(2004)
      spx_common = spx_spot.loc[common_idx]
      benchmark_equity = pd.DataFrame({
         'equity': INITIAL_CAPITAL * spx_common['spot'] / spx_common['spot']. iloc[0],
         'ret': _pct_change(spx_common['spot']),
         'pnl': spx_common['spot'].diff() * (INITIAL_CAPITAL / spx_common[' spot'].iloc[0])
      }, index=common_idx)
      print("Created benchmark from SPX spot data")
  except Exception as e:
      print(f"Failed to create benchmark from SPX data: {e}")
      benchmark_equity = None


  save_results_and_reports(rule_c,
      name_prefix="strangle_rule", title="RULE-BASED Short Strangle    Dynamic Hedge (dynamic r)",
      tag="COMMON", initial=INITIAL_CAPITAL
  )
  save_results_and_reports(attn_c,
      name_prefix="strangle_attn_hedgeonly", title="ATTN-LSTM Hedge-   Dynamic Hedge (dynamic r)",
      tag="COMMON", initial=INITIAL_CAPITAL
  )
  save_results_and_reports(conv_c,
      name_prefix="strangle_convlstm_hedgeonly", title="CONV-LSTM Hedge- Dynamic Hedge (dynamic r)",
      tag="COMMON", initial=INITIAL_CAPITAL
  )


  performance_report(rule_c['equity'], INITIAL_CAPITAL, "RULE-BASED PERFORMANCE [COMMON, dynamic r, rf accrual, pnl stop]", benchmark_equity ['equity'] if benchmark_equity is not None else None)
  performance_report(attn_c['equity'], INITIAL_CAPITAL, "ATTN-LSTM PERFORMANCE [COMMON, dynamic r, rf accrual, pnl stop]", benchmark_equity ['equity'] if benchmark_equity is not None else None)
  performance_report(conv_c['equity'], INITIAL_CAPITAL, "CONV-LSTM PERFORMANCE [COMMON, dynamic r, rf accrual, pnl stop]", benchmark_equity ['equity'] if benchmark_equity is not None else None)


  print("\n=== REGIME ANALYSIS ===")
  for name, res in [("RULE", rule_c), ("ATTN", attn_c), ("CONV", conv_c)]:
     if 'vix' in res.columns:
         regime_returns = analyze_performance_by_regime(res['equity'], res[' vix'])
         print(f"\n{name} Strategy Performance by VIX Regime:")
         for regime, return_pct in regime_returns.items():
            print(f" {regime}: {return_pct:.2%}")


  compare_strategies(rule_c, attn_c, "RULE", "ATTN")
  compare_strategies(rule_c, conv_c, "RULE", "CONV")
  compare_strategies(attn_c, conv_c, "ATTN", "CONV")

  if benchmark_equity is not None:
     compare_strategies(rule_c, benchmark_equity, "RULE", "SP500TR")
     compare_strategies(attn_c, benchmark_equity, "ATTN", "SP500TR")
     compare_strategies(conv_c, benchmark_equity, "CONV", "SP500TR")


  if RUN_SENSITIVITY_ANALYSIS:
     param_variations = {
         'STOP_PCT': [0.02, 0.03, 0.04, 0.05],
         'VIX_HALT': [30, 35, 40, 45, 50],
         'HEDGE_SMOOTH_ALPHA': [0.5, 0.6, 0.7, 0.8, 0.9],
         'cost': [CostModel(0.0050,0.0025, 0.0008,0.0007), CostModel (0.0050,0.0050, 0.0008,0.0014)]
     }
     sensitivity_results = parameter_sensitivity_analysis(run_rule_strategy, param_variations)
      for param, res in sensitivity_results.items():
         pd.DataFrame(res).T.to_csv(f"{ROOT}/sensitivity_{param}.csv")
         plot_sensitivity_heatmap(sensitivity_results, param, f"{ROOT}/ sensitivity_{param}.png")


  periods = {
      'full': slice(None),
      'bull_2009_2021': (rule_c.index >= '2009-01-01') & (rule_c.index <= ' 2021-12-31'),
      'crisis_2008': (rule_c.index >= '2008-01-01') & (rule_c.index <= ' 2008-12-31'),
      'crisis_2020': (rule_c.index >= '2020-01-01') & (rule_c.index <= ' 2020-12-31')
  }
  for period_name, mask in periods.items():
      for name, res in [("RULE", rule_c), ("ATTN", attn_c), ("CONV", conv_c)]:
         sub_res = res.loc[mask]
         if len(sub_res) > 0:
              sub_ann_ret = (sub_res['equity'].iloc[-1] / sub_res['equity']. iloc[0]) ** (252 / len(sub_res)) - 1
              print(f"[{name}] {period_name} Ann Return: {sub_ann_ret:.2%}")


  block_means = [5, 10, 15, 20]
  for bm in block_means:
      mc = stationary_bootstrap_summaries(rule_c['equity'], INITIAL_CAPITAL, runs=5000, block_mean=bm)
      mc.to_csv(f"{ROOT}/mc_block{bm}.csv")


  for name, res in [("RULE", rule_c), ("ATTN", attn_c), ("CONV", conv_c)]:
      plot_returns_hist(res['ret'], name, f"{ROOT}/{name}_hist.png")
      plot_qq(res['ret'], name, f"{ROOT}/{name}_qq.png")
      plot_rolling_sharpe(res['ret'], name, f"{ROOT}/{name}_rolling_sharpe.png ")
      if name == "ATTN":
         plot_iv_errors(attn_iv_errors_c, attn_iv_errors_p, attn_dates[:len( attn_iv_errors_c)], name, f"{ROOT}/{name}_iv_errors.png")
      if name == "CONV":
           plot_iv_errors(conv_iv_errors_c, conv_iv_errors_p, conv_dates[:len( conv_iv_errors_c)], name, f"{ROOT}/{name}_iv_errors.png")


   curves = {
        "Rule": rule_c,
        "ATTN-LSTM": attn_c,
        "CONV-LSTM": conv_c
   }
   if benchmark_equity is not None:
        curves["SP500TR"] = benchmark_equity
   plot_combined_common(curves, "Short 30d 10 Strangle (dynamic r + rf accrual + pnl stop)", os.path.join(ROOT, f"{FAMILY_NAME}_combined_COMMON.png"))


import os, glob, shutil
from datetime import datetime


root_dir = ROOT if 'ROOT' in globals() else os.getcwd()
family = FAMILY_NAME if 'FAMILY_NAME' in globals() else 'plots'

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join(root_dir, f"{family}_pngs_{ts}")
os.makedirs(out_dir, exist_ok=True)


png_files = glob.glob(os.path.join(root_dir, "*.png"))

if not png_files:
   print(f"No PNG files found in: {root_dir}")
else:
   for src in png_files:
        shutil.copy2(src, os.path.join(out_dir, os.path.basename(src)))

   zip_base = os.path.join(root_dir, f"{family}_pngs_{ts}")
   shutil.make_archive(zip_base, 'zip', out_dir)
   print(f"Copied {len(png_files)} PNGs       {out_dir}")
   print(f"ZIP created    {zip_base}.zip")









import os, shutil, glob, zipfile
from datetime import datetime


root = ROOT if 'ROOT' in globals() else os.getcwd()
family = FAMILY_NAME if 'FAMILY_NAME' in globals() else 'strangle_30d'


expected = [

    os.path.join(root, "RULE_hist.png"),
    os.path.join(root, "ATTN_hist.png"),
    os.path.join(root, "CONV_hist.png"),
    os.path.join(root, "RULE_qq.png"),
    os.path.join(root, "ATTN_qq.png"),
    os.path.join(root, "CONV_qq.png"),
    os.path.join(root, "RULE_rolling_sharpe.png"),
    os.path.join(root, "ATTN_rolling_sharpe.png"),
    os.path.join(root, "CONV_rolling_sharpe.png"),

    os.path.join(root, "ATTN_iv_errors.png"),
    os.path.join(root, "CONV_iv_errors.png"),

    os.path.join(root, f"{family}_combined_COMMON.png"),

    os.path.join(root, "sensitivity_STOP_PCT.png"),
    os.path.join(root, "sensitivity_VIX_HALT.png"),
    os.path.join(root, "sensitivity_HEDGE_SMOOTH_ALPHA.png"),
    os.path.join(root, "sensitivity_cost.png"),
]


files = [p for p in expected if os.path.exists(p)]


stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join(root, f"{family}_pngs_{stamp}")
os.makedirs(out_dir, exist_ok=True)


for src in files:
   shutil.copy2(src, os.path.join(out_dir, os.path.basename(src)))


zip_path = os.path.join(root, f"{family}_pngs_{stamp}.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
   for fname in sorted(os.listdir(out_dir)):
       fpath = os.path.join(out_dir, fname)
       zf.write(fpath, arcname=fname)

print(f"Packed {len(files)} PNGs into:\n{zip_path}")


try:
   from google.colab import files
   files.download(zip_path)
except Exception:
   try:
       from IPython.display import FileLink, display
       display(FileLink(zip_path, result_html_prefix="Click to download: "))
   except Exception:
       print("If no link appears, copy the path above and download manually.")
