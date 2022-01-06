# Considerer train.csv et test.csv
# Extraire les caracteriques
# Enregistrer

import pandas as pd

import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from config import Config

nltk.download('stopwords')

Config.FEATURES_PATH.mkdir(parents=True, exist_ok=True)

df_train = pd.read_csv(str(Config.DATASET_PATH / "train.csv"))
df_test = pd.read_csv(str(Config.DATASET_PATH / "test.csv"))

# Racine : journalisation -> jour
def stem(row):
    tokens_tmp = row['tweet'].split()
    stemmed_tokens_tmp = [st.stem(t) for t in tokens_tmp]
    # stemmed_tokens_tmp = []
    # for t in tokens_tmp:
    #     stemmed_tokens_tmp.append(st.stem(t))
    return ' '.join(stemmed_tokens_tmp)

#
st = PorterStemmer()

# Appliquer le stemm sur df_train et df_test
df_train['tweet'] = df_train.apply(lambda row: stem(row), axis=1)
df_test['tweet'] = df_test.apply(lambda row: stem(row), axis=1)

# Vectorizer
tfid_vectorizer = TfidfVectorizer(
                        lowercase=True, stop_words=Config.STOP_WORDS, 
                        max_features=Config.FEATURES_SIZE)
tmp_df = pd.concat([df_train, df_test])
tfid_vectorizer.fit(tmp_df['tweet'])

train_features = tfid_vectorizer.transform(df_train['tweet']).todense()
test_features = tfid_vectorizer.transform(df_test['tweet']).todense()

train_features = pd.DataFrame(train_features)
test_features = pd.DataFrame(test_features)

# Enregistrement des features pour train et test
train_features.to_csv(str(Config.FEATURES_PATH / "train_features.csv"), index=None)
test_features.to_csv(str(Config.FEATURES_PATH / "test_features.csv"), index=None)

# Enregistrement des labels pour train et test
df_train.label.to_csv(str(Config.FEATURES_PATH / "train_labels.csv"), index=None)
df_test.label.to_csv(str(Config.FEATURES_PATH / "test_labels.csv"), index=None)
