# Importações organizadas no topo e remoção de duplicatas
import dgl.nn as dglnn
from dgl import from_networkx
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import dgl.function as fn
import networkx as nx
import pandas as pd
import socket
import struct
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import category_encoders as ce
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Carregar o dataset (verificar o caminho do arquivo e o dataset em uso)
data = pd.read_csv('./datasets/iotid20.csv')

# Processamento do dataset e remoção de colunas desnecessárias
data['Src_IP'] = data.Src_IP.apply(lambda x: socket.inet_ntoa(struct.pack('>I', random.randint(0xac100001, 0xac1f0001))))
data['Src_IP'] = data['Src_IP'] + ':' + data['Src_Port'].apply(str)
data['Dst_IP'] = data['Dst_IP'] + ':' + data['Dst_Port'].apply(str)
data.drop(columns=['Src_Port', 'Dst_Port', 'Timestamp', 'Flow_ID', 'Cat', 'Sub_Cat', 'Flow_Byts/s', 'Flow_Pkts/s'], inplace=True)

# Label encoding
le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])

# Separação das features e target
label = data['Label']
data.drop(columns=['Label'], inplace=True)
scaler = StandardScaler()
data = pd.concat([data, label], axis=1)

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=123, stratify=label)

# Encoder para variáveis categóricas
encoder = ce.TargetEncoder(cols=['Fwd_PSH_Flags', 'Bwd_PSH_Flags', 'Protocol'])
encoder.fit(X_train, y_train)
X_train = encoder.transform(X_train)

# Normalização das colunas
cols_to_norm = list(set(X_train.columns) - {'Label'})
X_train[cols_to_norm] = scaler.fit_transform(X_train[cols_to_norm])

# Criação do grafo com NetworkX
X_train['h'] = X_train[cols_to_norm].values.tolist()
G = nx.from_pandas_edgelist(X_train, "Src_IP", "Dst_IP", ['h', 'Label'], create_using=nx.MultiGraph())
G = G.to_directed()
G = from_networkx(G, edge_attrs=['h', 'Label'])

# Definição das features dos nós e arestas
G.ndata['h'] = th.ones(G.num_nodes(), G.edata['h'].shape[1])
G.edata['train_mask'] = th.ones(len(G.edata['h']), dtype=th.bool)

# Função para calcular a acurácia
def compute_accuracy(pred, labels):
    """Calcula a acurácia com base nas predições e rótulos."""
    return (pred.argmax(1) == labels).float().mean().item()

# Definição da camada SAGE
class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super(SAGELayer, self).__init__()
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation

    def message_func(self, edges):
        """Função de mensagem para o gráfico."""
        return {'m': self.W_msg(th.cat([edges.src['h'], edges.data['h']], 2))}

    def forward(self, g_dgl, nfeats, efeats):
        """Propagação das informações no grafo."""
        with g_dgl.local_scope():
            g_dgl.ndata['h'] = nfeats
            g_dgl.edata['h'] = efeats
            g_dgl.update_all(self.message_func, fn.mean('m', 'h_neigh'))
            g_dgl.ndata['h'] = F.relu(self.W_apply(th.cat([g_dgl.ndata['h'], g_dgl.ndata['h_neigh']], 2)))
            return g_dgl.ndata['h']

# Definição do modelo SAGE
class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList([
            SAGELayer(ndim_in, edim, 128, activation),
            SAGELayer(128, edim, ndim_out, activation)
        ])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        """Propaga os dados pelas camadas do modelo."""
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats.sum(1)

# Definição do preditor MLP
class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super(MLPPredictor, self).__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        """Aplica predições às arestas."""
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(th.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h):
        """Realiza as predições para as arestas."""
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

# Definição do modelo completo
class Model(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super(Model, self).__init__()
        self.gnn = SAGE(ndim_in, ndim_out, edim, activation, dropout)
        self.pred = MLPPredictor(ndim_out, 2)

    def forward(self, g, nfeats, efeats):
        """Executa a predição com base no grafo e suas features."""
        h = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)

# Definição da função de perda com pesos balanceados
class_weights = class_weight.compute_sample_weight('balanced', np.unique(G.edata['Label'].cpu().numpy()))
class_weights = th.FloatTensor(class_weights)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Features de nós e arestas
node_features = G.ndata['h']
edge_features = G.edata['h']
edge_label = G.edata['Label']
train_mask = G.edata['train_mask']

# Inicialização do modelo e otimizador
model = Model(G.ndata['h'].shape[2], 128, G.ndata['h'].shape[2], F.relu, 0.2)
opt = th.optim.Adam(model.parameters())

# Treinamento do modelo
for epoch in range(1, 50):
    pred = model(G, node_features, edge_features)
    loss = criterion(pred[train_mask], edge_label[train_mask])
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 10 == 0:
        print('Training acc:', compute_accuracy(pred[train_mask], edge_label[train_mask]))

# Teste do modelo (código adicional para preparação e execução do teste)
X_test = encoder.transform(X_test)
X_test[cols_to_norm] = scaler.transform(X_test[cols_to_norm])
X_test['h'] = X_test[cols_to_norm].values.tolist()

# Geração do grafo de teste
G_test = nx.from_pandas_edgelist(X_test, "Src_IP", "Dst_IP", ['h', 'Label'], create_using=nx.MultiGraph())
G_test = G_test.to_directed()
G_test = from_networkx(G_test, edge_attrs=['h', 'Label'])
actual = G_test.edata.pop('Label')
G_test.ndata['feature'] = th.ones(G_test.num_nodes(), G.ndata['h'].shape[2])
G_test.ndata['feature'] = th.reshape(G_test.ndata['feature'], (G_test.ndata['feature'].shape[0], 1, G_test.ndata['feature'].shape[1]))
G_test.edata['h'] = th.reshape(G_test.edata['h'], (G_test.edata['h'].shape[0], 1, G_test.edata['h'].shape[1]))

# Avaliação e predição
node_features_test = G_test.ndata['feature']
edge_features_test = G_test.edata['h']
test_pred = model(G_test, node_features_test, edge_features_test)
test_pred = test_pred.argmax(1).cpu().detach().numpy()

# Exibição dos resultados de métricas
actual = ["Normal" if i == 0 else "Attack" for i in actual]
test_pred = ["Normal" if i == 0 else "Attack" for i in test_pred]
cnf_matrix = confusion_matrix(actual, test_pred)
precision, recall, fscore, _ = precision_recall_fscore_support(actual, test_pred, average='weighted')

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {fscore}')
