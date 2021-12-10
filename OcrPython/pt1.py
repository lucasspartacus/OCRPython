# -*- coding: utf-8 -*-

# Trabalho Prático - Parte 1
# Disciplina: Processamento de Imagens
# Professor: Alexei Manso Correa Machado
# Grupo:
# Igor Marques Reis
# Lucas Spartacus Vieira Carvalho
# Rafael Mourão Cerqueira Figueiredo

import pt2 # parte 2 do trabalho (classificar os digitos)
import os, cv2 #opencv
import numpy as np
import tkinter as tk
from matplotlib import pyplot as plt
from tkinter import filedialog as fd
from tkinter.simpledialog import askinteger
from PIL import ImageTk, Image  # pip install Pillow
from mnist import MNIST

# Variáveis globais
imgPath = None
img = None
imagem_limiarizada = None

janela = tk.Tk()
lblImg = tk.Label(image = "")
largura = 850
altura = 650

rotacionou_imagem = False   # flag que indica se a imagem foi rotacionada
imagem_rotacionada = None   # imagem que pode vir a ser rotacionada
projecoes_treino_mnist = [] # projecoes mnist - base de treino
lbl_mnist_treino = []       # rotulos mnist - base de treino
projecoes_teste_mnist = []  # projecoes mnist - base de teste
lbl_mnist_teste = []        # rotulos mnist - base de treino
projecoes_digitos = []      # projecoes dos digitos da imagem importada
digitos_np = []             # numpy array com os digitos recortados da imagem

# Metodo que permite exibir a imagem na interface grafica
def toPNG(img, nomeImagem):
    cv2.imwrite(nomeImagem, img)

# Redimensiona imagem para que ela caiba na interface grafica
def redimensiona_imagem(imagem, width=504, height=504): 
    if imagem.width > width:
        if imagem.height > height:
            imagem = imagem.resize((width, height))
        else: imagem = imagem.resize((width, imagem.height))
    else:
        if imagem.height > height:
            imagem = imagem.resize((imagem.width, height))

    return imagem

# Rotaciona a imagem em "x" graus
def rotaciona_imagem():
    global rotacionou_imagem, imagem_rotacionada

    graus = askinteger("Graus", "Em quantos graus (°) a imagem deve ser rotacionada?")
    im_aux = imagem_limiarizada
    centro_img = tuple(np.array(im_aux.shape[1::-1]) / 2)
    matriz_rotacao = cv2.getRotationMatrix2D(centro_img, graus, 1.0)
    imagem_rotacionada = cv2.warpAffine(im_aux, matriz_rotacao, im_aux.shape[1::-1], flags=cv2.INTER_LINEAR)

    plota_imagem(imagem_rotacionada)

    rotacionou_imagem = True

# Plota a imagem na interface grafica
def plota_imagem(imagem_npArray):
    toPNG(imagem_npArray, "tmp.png")

    imagem_tk = Image.open( os.path.join(os.getcwd(), "tmp.png") )

    if imagem_tk.width > 504 or imagem_tk.height > 504:
        imagem_tk = redimensiona_imagem(imagem_tk)
    imagem_tk = ImageTk.PhotoImage(imagem_tk)
    
    global lblImg
    lblImg.config(image = "")
    lblImg = tk.Label(image = imagem_tk)
    lblImg.image = imagem_tk
    lblImg.place(x = 30, y = 70)

# Mostra os digitos na interface grafica
def mostra_digitos(titulos_digitos_np):
    num_digitos = len(digitos_np)

    numLinhas = num_digitos/2
    numColunas = num_digitos/2

    # Oculta a escala e os eixos dos graficos
    plt.rcParams["figure.figsize"] = [5.0, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.rc('axes.spines', top=False, bottom=False, left=False, right=False)
    plt.rc('axes', facecolor=(1,1,1,0), edgecolor=(1,1,1,0) )
    plt.rc( ('xtick', 'ytick'), color=(1,1,1,0) )

    for idx, digito in enumerate(digitos_np, 1):
        fig = plt.subplot(numLinhas, numColunas, idx)
        fig.title.set_text( str(titulos_digitos_np[ idx-1 ]) )
        plt.imshow(digito, cmap="gray")
    plt.show()

# Utiliza Otsu para limiarizar a imagem
# inv -> flag que indica se o objeto sera branco ou preto (e o fundo, preto ou branco)
def tira_limiar(inv):
    global imagem_limiarizada, rotacionou_imagem
    
    # le a imagem corrente
    if rotacionou_imagem: img_limiar = imagem_rotacionada.copy()
    else: img_limiar = cv2.imread(imgPath) 
    
    # converte-a para NC
    img_limiar_NC = cv2.cvtColor(img_limiar, cv2.COLOR_BGR2GRAY)

    # filtro gaussiano
    img_filtroGauss = cv2.GaussianBlur(img_limiar_NC.copy(), (5, 5), 0) # kernel 5

    if inv: limiar, im = cv2.threshold(img_filtroGauss.copy(), 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    else: limiar, im = cv2.threshold(img_filtroGauss.copy(), 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

    imagem_limiarizada = im.copy()
    
    plota_imagem(im)

# Recorta cada digito da imagem, os exibe e tira suas projecoes
def acha_contorno():
    global img, imagem_limiarizada, digitos_np
    
    contornos, _ = cv2.findContours(imagem_limiarizada.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img2 = imagem_limiarizada.copy()
    digitos = []
    for c in contornos:
        x,y,w,h = cv2.boundingRect(c)
        
        # Faz retangulo delimitando cada digito
        cv2.rectangle(img2, (x,y), (x+w, y+h), color=(255, 255, 255), thickness=2)
        
        # Guarda posicao do digito na imagem
        digito = imagem_limiarizada[y:y+h, x:x+w]
        
        # Redimensiona o digito p/ (18, 18)
        digito_redimensionado = cv2.resize(digito, (18,18))
        
        # Adiciona 5px de margem para adequar o digito ao padrao mnist
        digito_com_margem = np.pad(digito_redimensionado, ((5,5),(5,5)), "constant", constant_values=0)
        
        digitos.append(digito_com_margem)
    

    # converte para array numpy
    digitos_np = np.array(digitos)
    num_digitos = len(digitos_np)

    # tira projecoes dos digitos
    global projecoes_digitos
    projecoes_digitos = []
    for i in range(num_digitos):
        digitos_np[i] = 255 - digitos_np[i] # faz fundo branco e digito preto
        projecoes_digitos.append(np.concatenate((projHorizontal(digitos_np[i].copy()), projVertical(digitos_np[i].copy()))))

    titulos = []
    for digito in digitos_np: titulos.append("")
    mostra_digitos( titulos )

# Obtem path da imagem e a exibe na interface
def carregar_imagem():
    global imgPath, rotacionou_imagem
    rotacionou_imagem = False
    imgPath = fd.askopenfilename()
    print(imgPath)

    global img
    img = cv2.imread(imgPath)

    plota_imagem(img)

# Exclui o arquivo temp ao fechar o programa
def on_closing():
    if os.path.isfile("tmp.png"):
        os.remove("tmp.png")
    janela.destroy()

# Seta parametros da interface grafica
def inicializa_janela():

    # Parametros da interface grafica:
    janela.geometry(str(largura) + "x" + str(altura))
    janela.columnconfigure(0, weight=1)
    janela.rowconfigure(0, weight=1)
    janela.title("OCR")
    
    janela.protocol("WM_DELETE_WINDOW", on_closing) # Acoes ao fechar o programa

    # Botoes - declaracao
    btnCarregarImg = tk.Button(janela, text = "Carregar imagem", bg = "#7E7E7E", fg = "#ffffff", font = ("Calibri", 10), command = lambda : carregar_imagem())
    btnLimiar = tk.Button(janela, text = "Limiariza imagem", bg = "#7E7E7E", fg = "#ffffff", font = ("Calibri", 10), command = lambda : tira_limiar(False))
    btnLimiarInvertido = tk.Button(janela, text = "Limiariza imagem - invertido ", bg = "#7E7E7E", fg = "#ffffff", font = ("Calibri", 10), command = lambda : tira_limiar(True))
    btnRecortaDigitos = tk.Button(janela, text = "Recortar dígitos", bg = "#7E7E7E", fg = "#ffffff", font = ("Calibri", 10), command = lambda : acha_contorno())
    btnRotImagem = tk.Button(janela, text = "Rotacionar imagem", bg = "#7E7E7E", fg = "#ffffff", font = ("Calibri", 10), command = lambda : rotaciona_imagem())
    btnRodarSVM = tk.Button(janela, text = "Rodar SVM", bg = "#7E7E7E", fg = "#ffffff", font = ("Calibri", 10), command = lambda : roda_svm())
    btnRodarMLP = tk.Button(janela, text = "Rodar MLP", bg = "#7E7E7E", fg = "#ffffff", font = ("Calibri", 10), command = lambda : roda_mlp())

    # Botoes - posicao
    btnCarregarImg.place(x = 15, y = 20)
    btnLimiar.place(x = 130, y = 20)
    btnLimiarInvertido.place(x = 250, y = 20)
    btnRecortaDigitos.place(x = 430, y = 20)
    btnRotImagem.place(x = 540, y = 20)
    btnRodarSVM.place(x = 670, y = 20)
    btnRodarMLP.place(x = 755, y = 20)

# Retorna um vetor que representa a projecao horizontal da imagem
def projHorizontal(img):
  
    return np.sum(img, axis = 1) 

# Retorna um vetor que representa a projecao vertical da imagem.
def projVertical(img):
  
    return np.sum(img, axis = 0)

# Invoca o metodo que roda a SVM na parte 2
def roda_svm():
    # rodar svm:
    pt2.roda_svm( np.array(projecoes_treino_mnist), np.array(lbl_mnist_treino), np.array(projecoes_teste_mnist), np.array(lbl_mnist_teste), np.array(projecoes_digitos) )

    # imprimir resultados obtidos na interface:
    mostra_digitos( pt2.digitos_preditos_svm )

# Invoca o metodo que roda a MLP na parte 2
def roda_mlp():
    # rodar mlp:
    epocas = askinteger("Epochs", "Número de épocas (epochs) para o MLP")
    pt2.roda_mlp( np.array(projecoes_treino_mnist), np.array(lbl_mnist_treino), np.array(projecoes_teste_mnist), np.array(lbl_mnist_teste), np.array(projecoes_digitos), epocas )

    # imprimir resultados obtidos na interface:
    mostra_digitos( pt2.digitos_preditos_mlp )


def main():
    global projecoes_treino_mnist, projecoes_teste_mnist, lbl_mnist_treino, lbl_mnist_teste

    inicializa_janela()

    
    mndata = MNIST('samples')

    mnist_treino, lbl_mnist_treino = mndata.load_training()
    mnist_teste, lbl_mnist_teste = mndata.load_testing()

    projecoes_treino_mnist = []
    projecoes_teste_mnist = []

    print("Preparando a base de dados...")
    
    # Converte imagens para numpy.array (dim. 28x28), aplica um filtro de media e tira suas projecoes
    for i in range(len(mnist_treino)):
        mnist_treino[i] = np.array(mnist_treino[i], dtype="uint8").reshape((28,28))
        mnist_treino[i] = 255 - mnist_treino[i] # faz fundo branco e digito preto
        mnist_treino[i] = cv2.GaussianBlur(mnist_treino[i], (3,3), 0) # filtro gaussiano, kernel 3
        projecoes_treino_mnist.append(np.concatenate((projHorizontal(mnist_treino[i].copy()), projVertical(mnist_treino[i].copy()))))


    for i in range(len(mnist_teste)):
        mnist_teste[i] = np.array(mnist_teste[i], dtype="uint8").reshape((28,28))
        mnist_teste[i] = 255 - mnist_teste[i] # faz fundo branco e digito preto
        mnist_teste[i] = cv2.GaussianBlur(mnist_teste[i], (3,3), 0) # filtro gaussiano, kernel 3
        projecoes_teste_mnist.append(np.concatenate((projHorizontal(mnist_teste[i].copy()), projVertical(mnist_teste[i].copy()))))
    


    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    janela.mainloop()

if __name__ == "__main__":
    main()