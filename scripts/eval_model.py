import pickle # Serialiser des objets (y comporis des modeles)
import json 

import pandas as pd
from sklearn.metrics import f1_score, precision_score

from config import Config

X_test = pd.read_csv(str(Config.FEATURES_PATH / "test_features.csv"))
y_test = pd.read_csv(str(Config.FEATURES_PATH / "test_labels.csv"))
y_test = y_test.to_numpy().ravel()

# Restaurer le modèle
model = pickle.load(open(str(Config.MODELS_PATH / "model.pk"), mode='rb'))

y_pred = model.predict(X_test)
test_score = f1_score(y_true=y_test, y_pred=y_pred)
test_precision = precision_score(y_true=y_test, y_pred=y_pred)

with open(str(Config.METRICS_FILE_PATH), mode='w') as f:
    json.dump(dict(f1_score=test_score, precision=test_precision), f)