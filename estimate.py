import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

def get_data(path):
  dataset = pd.read_csv(path)
  return dataset

def scaler(X_train, X_val):
  X_train_scaled = X_train.copy()
  X_val_scaled = X_val.copy()

  scaler = MinMaxScaler()
  scaler.fit(X_train_scaled)
  x_train_scaled_1 = scaler.transform(X_train_scaled)
  x_val_scaled_1 = scaler.transform(X_val_scaled)

  return x_train_scaled_1, x_val_scaled_1

def rf(X_train, X_val, Y_train, Y_val):
  rc_best = RandomForestClassifier(n_estimators = 150,  max_features = 0.6, min_samples_leaf = 1, max_depth = 20, random_state = 0 )
  rc_best.fit(X_train, Y_train)
  rc_tr_pred = rc_best.predict(X_train)
  rc_val_pred = rc_best.predict(X_val)

  print("Precision Score : ", precision_score(Y_val, rc_val_pred, pos_label='positive', average='weighted'))

dataset = get_data("./train.csv")
X = dataset.drop('price_range', axis=1)
y = dataset['price_range']
X_train, X_val, Y_train, Y_val = train_test_split(X, y)
X_train_scaled, X_val_scaled = scaler(X_train, X_val)
rf(X_train_scaled, X_val_scaled, Y_train, Y_val)