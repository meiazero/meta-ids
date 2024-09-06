# Importações organizadas no topo, removendo duplicatas e não utilizadas
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

# Carregar dataset (explicação: o dataset deve ser carregado corretamente antes do uso)
# TODO: Verifique se o caminho do arquivo './dataset/wustl_iiot_2021.csv' está correto.
df = pd.read_csv('./dataset/wustl-iiot-2021.csv')

# Label encoding da coluna 'Traffic'
labelencoder = LabelEncoder()
df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])

# Separação dos dados em features e target
X = df.drop(['Traffic'], axis=1).values
y = df.iloc[:, -1].values

# Divisão dos dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0, stratify=y)

# Aplicação de oversampling com SMOTE para balancear o dataset
smote = SMOTE(n_jobs=-1, sampling_strategy={2: 1000})
X_train, y_train = smote.fit_resample(X_train, y_train)

# Função para o RandomForest
def random_forest_classifier(X_train, y_train, X_test, y_test):
    """
    Treina e avalia um modelo RandomForestClassifier.

    Parâmetros:
    - X_train: Dados de treino.
    - y_train: Labels de treino.
    - X_test: Dados de teste.
    - y_test: Labels de teste.

    Retorna:
    - accuracy: Acurácia do modelo.
    """
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X_train, y_train)
    rf_score = rf.score(X_test, y_test)
    return rf_score

# Função para o DecisionTree
def decision_tree_classifier(X_train, y_train, X_test, y_test):
    """
    Treina e avalia um modelo DecisionTreeClassifier.

    Parâmetros:
    - X_train: Dados de treino.
    - y_train: Labels de treino.
    - X_test: Dados de teste.
    - y_test: Labels de teste.

    Retorna:
    - accuracy: Acurácia do modelo.
    """
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    dt_score = dt.score(X_test, y_test)
    return dt_score

# Função para o AdaBoost
def adaboost_classifier(X_train, y_train, X_test, y_test):
    """
    Treina e avalia um modelo AdaBoostClassifier.

    Parâmetros:
    - X_train: Dados de treino.
    - y_train: Labels de treino.
    - X_test: Dados de teste.
    - y_test: Labels de teste.

    Retorna:
    - accuracy: Acurácia do modelo.
    """
    ab = AdaBoostClassifier(random_state=0)
    ab.fit(X_train, y_train)
    ab_score = ab.score(X_test, y_test)
    return ab_score

# Função para o MLPClassifier
def mlp_classifier(X_train, y_train, X_test, y_test):
    """
    Treina e avalia um modelo MLPClassifier.

    Parâmetros:
    - X_train: Dados de treino.
    - y_train: Labels de treino.
    - X_test: Dados de teste.
    - y_test: Labels de teste.

    Retorna:
    - accuracy: Acurácia do modelo.
    """
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(39, 30, 2), random_state=1)
    mlp.fit(X_train, y_train)
    mlp_score = mlp.score(X_test, y_test)
    return mlp_score

# Função para o XGBoost
def xgboost_classifier(X_train, y_train, X_test, y_test):
    """
    Treina e avalia um modelo XGBoostClassifier.

    Parâmetros:
    - X_train: Dados de treino.
    - y_train: Labels de treino.
    - X_test: Dados de teste.
    - y_test: Labels de teste.

    Retorna:
    - accuracy: Acurácia do modelo.
    """
    xg = XGBClassifier(learning_rate=0.653, n_estimators=65, max_depth=50)
    xg.fit(X_train, y_train)
    xg_score = xg.score(X_test, y_test)
    return xg_score

# Execução dos modelos e exibição dos resultados
rf_score = random_forest_classifier(X_train, y_train, X_test, y_test)
dt_score = decision_tree_classifier(X_train, y_train, X_test, y_test)
ab_score = adaboost_classifier(X_train, y_train, X_test, y_test)
mlp_score = mlp_classifier(X_train, y_train, X_test, y_test)
xg_score = xgboost_classifier(X_train, y_train, X_test, y_test)

print(f"Random Forest Accuracy: {rf_score}")
print(f"Decision Tree Accuracy: {dt_score}")
print(f"AdaBoost Accuracy: {ab_score}")
print(f"MLP Accuracy: {mlp_score}")
print(f"XGBoost Accuracy: {xg_score}")
