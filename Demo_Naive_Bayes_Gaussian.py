#--esse codigo tem como objetico demonstrar como podemos fazer previsão para determinar o risco de um acidente veicular de um motorista
#--usando um algoritimo de bayes

import pandas as pd
from sklearn.naive_bayes import GaussianNB

#para mostrar melhor as colunas
pd.set_option('display.max_columns', 12)

base_risco_acidente = pd.read_csv('C:\\TCC\\MeuTCC\\risco_de_acidente_dataset.csv')

#Mostrando base de dados
print(base_risco_acidente)

#separa os atributos da clase
X_risco_acidente = base_risco_acidente.iloc[:, 0:4].values

y_risco_acidente = base_risco_acidente.iloc[:, 4].values

#transformo os atributos categoricos em numericos
from sklearn.preprocessing import LabelEncoder
label_encoder_historico = LabelEncoder()
label_encoder_tdirigindo = LabelEncoder()
label_encoder_vistoria = LabelEncoder()
label_encoder_kmedio = LabelEncoder()

#atribuindo as transformações na matriz
X_risco_acidente[:, 0] = label_encoder_historico.fit_transform(X_risco_acidente[:, 0])
X_risco_acidente[:, 1] = label_encoder_tdirigindo.fit_transform(X_risco_acidente[:, 1])
X_risco_acidente[:, 2] = label_encoder_vistoria.fit_transform(X_risco_acidente[:, 2])
X_risco_acidente[:, 3] = label_encoder_kmedio.fit_transform(X_risco_acidente[:, 3])

#mostrando a matriz com os dados mudados para numeros
print(X_risco_acidente)

#criando um arquivo tipo pkl para salvar todas as constantes
import pickle
with open('risco_acidente.pkl', 'wb') as f:
    pickle.dump([X_risco_acidente, y_risco_acidente], f)

#criando a tabela de probabilidades com distribuição Gaussiana, ou seja, o modelo treinado
naive_risco_acidente = GaussianNB()
(naive_risco_acidente.fit(X_risco_acidente, y_risco_acidente))


#entre parentesses colocarei o valor dos atributos numericos de tal forma que permita o facil entendimento do uso deles como parametros na função
#agora faremos a previsão usando dois motoristas de teste com esses atributos respectivamente
#Primeiro teste: historico desconhecido(1), tdirigindo muito(1), vistoria adequada (0), kmedio 40km_100km(0)
#Segundo teste: historico ruim(2), tdirigido pouco(0), vistoria nenhuma(1), kmedio acima_100(0)

#define 'previsao' com os parametros a cima
previsao = naive_risco_acidente.predict([[1,1,0,0],[2,0,1,0]])

#diz a previsão de risco neste caso é respectivamente, 'moderado', e 'alto'
print(previsao)