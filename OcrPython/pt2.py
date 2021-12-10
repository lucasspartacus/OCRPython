# -*- coding: utf-8 -*-

# Trabalho Prático - Parte 2
# Disciplina: Processamento de Imagens
# Professor: Alexei Manso Correa Machado
# Grupo:
# Igor Marques Reis
# Lucas Spartacus Vieira Carvalho
# Rafael Mourão Cerqueira Figueiredo

import pickle, os, seaborn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from datetime import datetime
from tkinter import messagebox

# Variáveis globais
digitos_preditos_svm = None
digitos_preditos_mlp = []

# Metodo para formatacao dos tempos de execucao:
def formata_tempo(tempo_execucao):
    tempo_formatado = ""
    aux = ["h", "min"]
    cont = 0
    for i, c in enumerate(tempo_execucao):
        if c == ':':
            tempo_formatado += aux[cont]
            cont += 1
        else : tempo_formatado += tempo_execucao[i]
    tempo_formatado += 's'

    return tempo_formatado

# Metodo para treinar a SVM:
def treina_svm(proj_digitos_treino, rotulos_treino, proj_digitos_teste, rotulos_teste):
    x_treino = tf.keras.utils.normalize(proj_digitos_treino) # normaliza projecoes
    y_treino = rotulos_treino
    
    # Seta e treina svm
    svm = SVC(C = 100)
    t_inicio = datetime.now()   # armazena tempo no inicio do treinamento
    svm.fit(x_treino, y_treino) # treina svm 
    t_fim = datetime.now()      # armazena tempo ao final do treinamento

    # Salva svm treinada
    filename = "svm_treinada.dat"
    pickle.dump(svm, open(filename, "wb"))

    # Testa svm treinada
    digitos_preditos = svm.predict( tf.keras.utils.normalize(proj_digitos_teste) )

    # Calcular matriz de confusao:
    matriz_confusao = metrics.confusion_matrix(rotulos_teste, digitos_preditos)
    seaborn.heatmap(matriz_confusao, annot=True, annot_kws={"size":16},  fmt='g')
    plt.savefig("matriz_confusao_svm.png")

    tempo_formatado = formata_tempo( str(t_fim-t_inicio) )

    # Mede acuracia da svm
    acuracia = metrics.accuracy_score(rotulos_teste, digitos_preditos)
    # print("acc ", acuracia)
    # print("Tempo para treinar a SVM = ", tempo_formatado )

    # Abre pop-up com tempo de treinamento da SVM
    messagebox.showinfo(title="Tempo decorrido", message="Tempo para treinar a SVM = {}".format(tempo_formatado))

# Método para rodar a SVM na imagem de entrada:
def roda_svm(proj_digitos_treino, rotulos_treino, proj_digitos_teste, rotulos_teste, proj_digitos_imagem):
    # Se a svm ainda nao foi treinada, entao a treina
    if os.path.isfile("svm_treinada.dat") == False:
        treina_svm(proj_digitos_treino, rotulos_treino, proj_digitos_teste, rotulos_teste)

    # Carrega svm treinada
    svm_treinada = pickle.load( open("svm_treinada.dat", "rb"))

    # Rodar svm 
    t_inicio = datetime.now()
    digitos_preditos = svm_treinada.predict( tf.keras.utils.normalize(proj_digitos_imagem) )
    t_fim = datetime.now()

    tempo_formatado = formata_tempo( str(t_fim-t_inicio) )
    # Armazena resultado obtido
    global digitos_preditos_svm 
    digitos_preditos_svm = digitos_preditos

    # Abre pop-up com tempo p/ rodar a SVM
    messagebox.showinfo(title="Tempo decorrido", message="Tempo para rodar a SVM = {}".format(tempo_formatado))

# Método para treinar o MLP:
def treina_mlp(proj_digitos_treino, rotulos_treino, proj_digitos_teste, rotulos_teste, epocas):
    x_treino = tf.keras.utils.normalize(proj_digitos_treino) # normaliza projecoes
    y_treino = rotulos_treino
    formato_dados = x_treino[0].shape[0]
    print("formato_dados =", formato_dados)

    # Declara o mlp e suas camadas
    mlp = tf.keras.models.Sequential()
    mlp.add(tf.keras.layers.Input(formato_dados))    # camada de input
    mlp.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))    # primeira hidden layer, com 128 neuronios e relu como funcao de ativacao 
    mlp.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))    # segunda hidden layer
    mlp.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))  # ultima camada, com 10 neuronios (numero de classificacoes possiveis)

    # Parametros de treino do mlp
    mlp.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

    # Treina mlp
    t_inicio = datetime.now()                       # armazena tempo no inicio do treinamento
    mlp.fit(x_treino, y_treino, epochs = epocas)    # treina mlp
    t_fim = datetime.now()                          # armazena tempo no inicio do treinamento

    # Salva mlp treinado
    mlp.save("mlp_treinado")

    # Testar mlp treinado
    digitos_preditos = []
    result = mlp.predict( tf.keras.utils.normalize(proj_digitos_teste) )
    for digit in result: digitos_preditos.append( np.argmax(digit) ) # interpreta resultados do MLP

    # Calcular matriz de confusao:
    matriz_confusao = metrics.confusion_matrix(rotulos_teste, digitos_preditos)
    seaborn.heatmap(matriz_confusao, annot=True, annot_kws={"size":16},  fmt='g')
    plt.savefig("matriz_confusao_mlp.png")

    # Mede loss e acuracia do mlp
    loss, acuracia = mlp.evaluate(tf.keras.utils.normalize(proj_digitos_teste), rotulos_teste)

    tempo_formatado = formata_tempo( str(t_fim-t_inicio) )

    # print("Tempo para treinar o MLP = ", tempo_formatado )

    # Abre pop-up com tempo de treinamento do MLP
    messagebox.showinfo(title="Tempo decorrido", message="Tempo para treinar o MLP = {}".format(tempo_formatado))

# Método para rodar o MLP na imagem de entrada:
def roda_mlp(proj_digitos_treino, rotulos_treino, proj_digitos_teste, rotulos_teste, proj_digitos_imagem, epocas):
    # Se o mlp ainda nao foi treinado, entao o treina
    if os.path.isdir("mlp_treinada") == False:
        treina_mlp(proj_digitos_treino, rotulos_treino, proj_digitos_teste, rotulos_teste, epocas)

    # Carrega mlp treinado
    mlp_treinado = tf.keras.models.load_model("mlp_treinada")

    # Roda o mlp
    t_inicio = datetime.now()
    digitos_preditos = mlp_treinado.predict( tf.keras.utils.normalize(proj_digitos_imagem) )
    t_fim = datetime.now()

    tempo_formatado = formata_tempo( str(t_fim-t_inicio) )

    # Guarda os digitos preditos
    global digitos_preditos_mlp
    for digit in digitos_preditos: digitos_preditos_mlp.append( np.argmax(digit) )

    # Abre pop-up com tempo de treinamento
    messagebox.showinfo(title="Tempo decorrido", message="Tempo para rodar o MLP = {}".format(tempo_formatado))


def main():
    print('Este programa é um módulo de outro. Sua execução "por si só" não está definida.')

if __name__ == '__main__':
    main()