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

# Carregar dataset (verificar o caminho do arquivo)
df = pd.read_csv('./dataset/wustl-iiot-2021.csv')

# Label encoding da coluna 'Traffic'
labelencoder = LabelEncoder()
df['Traffic'] = labelencoder.fit_transform(df['Traffic'])

# Separação dos dados em features e target
X = df.drop(columns=['Traffic']).values
y = df['Traffic'].values

# Divisão dos dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0, stratify=y)

# Aplicação de oversampling com SMOTE para balancear o dataset
smote = SMOTE(n_jobs=-1, sampling_strategy={2: 1000})
X_train, y_train = smote.fit_resample(X_train, y_train)

# Função genérica para treinar e avaliar modelos
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Treina e avalia um modelo de classificação.

    Parâmetros:
    - model: O modelo de classificação a ser treinado.
    - X_train: Dados de treino.
    - y_train: Labels de treino.
    - X_test: Dados de teste.
    - y_test: Labels de teste.

    Retorna:
    - accuracy: Acurácia do modelo.
    """
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return score

# Dicionário de modelos
models = {
    "Random Forest": RandomForestClassifier(random_state=0),
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "AdaBoost": AdaBoostClassifier(random_state=0),
    "MLP": MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(39, 30, 2), random_state=1),
    "XGBoost": XGBClassifier(learning_rate=0.653, n_estimators=65, max_depth=50)
}

# Avaliação dos modelos
for name, model in models.items():
    accuracy = evaluate_model(model, X_train, y_train, X_test, y_test)
    print(f"{name} Accuracy: {accuracy:.4f}")
