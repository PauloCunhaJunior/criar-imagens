"""
Reconhecimento de imagens com CNN + TensorFlow usando MNIST.

Este exemplo faz 3 coisas:
1. Treina uma rede convolucional para reconhecer numeros de 0 a 9.
2. Cria imagens novas com numeros desenhados por codigo.
3. Envia essas imagens criadas para o modelo avaliar.

Funciona bem no Google Colab. No VS Code pode falhar se o Python local
for muito novo para o TensorFlow, como Python 3.14.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras import datasets, layers, models


IMAGE_SIZE = 28
OUTPUT_DIR = Path("imagens_criadas")


def treinar_modelo():
    """Carrega o MNIST, treina a CNN e devolve modelo + dados de teste."""
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    model = models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_images,
        train_labels,
        epochs=5,
        batch_size=64,
        validation_data=(test_images, test_labels),
    )

    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Perda nos dados de teste: {test_loss:.4f}")
    print(f"Precisao nos dados de teste: {test_accuracy:.4f}")

    return model, test_images, test_labels


def carregar_fonte(tamanho):
    """Tenta usar uma fonte comum no Colab/Windows e cai para a fonte padrao."""
    fontes = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "arialbd.ttf",
        "arial.ttf",
    ]

    for fonte in fontes:
        try:
            return ImageFont.truetype(fonte, tamanho)
        except OSError:
            pass

    return ImageFont.load_default()


def criar_imagem_digito(digito, caminho, tamanho_fonte=24, deslocamento=(0, -2)):
    """
    Cria uma imagem parecida com o MNIST: 28x28, fundo preto e digito branco.

    O parametro deslocamento ajuda a testar se o modelo reconhece o numero
    mesmo quando ele nao esta perfeitamente centralizado.
    """
    imagem = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0)
    desenho = ImageDraw.Draw(imagem)
    fonte = carregar_fonte(tamanho_fonte)
    texto = str(digito)

    caixa = desenho.textbbox((0, 0), texto, font=fonte)
    largura = caixa[2] - caixa[0]
    altura = caixa[3] - caixa[1]
    x = (IMAGE_SIZE - largura) // 2 + deslocamento[0]
    y = (IMAGE_SIZE - altura) // 2 + deslocamento[1]

    desenho.text((x, y), texto, fill=255, font=fonte)
    imagem.save(caminho)
    return caminho


def preparar_imagem_para_modelo(caminho):
    """
    Abre uma imagem externa e transforma no formato esperado pela CNN:
    (1, 28, 28, 1), valores entre 0 e 1.

    Se a imagem tiver fundo branco e numero preto, a funcao inverte as cores,
    pois o MNIST usa fundo escuro e numero claro.
    """
    imagem = Image.open(caminho).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    array = np.array(imagem).astype("float32") / 255.0

    if array.mean() > 0.5:
        array = 1.0 - array

    return array.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)


def avaliar_imagem(model, caminho):
    """Mostra a imagem criada e imprime a previsao da rede."""
    imagem_modelo = preparar_imagem_para_modelo(caminho)
    probabilidades = model.predict(imagem_modelo, verbose=0)[0]
    previsao = int(np.argmax(probabilidades))
    confianca = float(probabilidades[previsao])

    plt.figure(figsize=(2, 2))
    plt.imshow(imagem_modelo.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray")
    plt.title(f"Previsto: {previsao} ({confianca:.1%})")
    plt.axis("off")
    plt.show()

    print(f"Arquivo: {caminho}")
    print(f"Numero previsto: {previsao}")
    print(f"Confianca: {confianca:.1%}")
    print("-" * 30)

    return previsao, confianca


def testar_com_imagens_do_mnist(model, test_images, test_labels, quantidade=10):
    """Testa com imagens que ja existem no conjunto MNIST."""
    imagens = test_images[:quantidade]
    rotulos = test_labels[:quantidade]
    previsoes = model.predict(imagens, verbose=0)
    rotulos_previstos = np.argmax(previsoes, axis=1)

    for i, imagem in enumerate(imagens):
        plt.figure(figsize=(1.5, 1.5))
        plt.imshow(imagem.reshape(28, 28), cmap="gray")
        plt.title(f"Real: {rotulos[i]} | Previsto: {rotulos_previstos[i]}")
        plt.axis("off")
        plt.show()


def criar_e_avaliar_novas_imagens(model):
    """Cria uma imagem para cada digito e avalia todas com o modelo."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    for digito in range(10):
        caminho = OUTPUT_DIR / f"digito_{digito}.png"
        criar_imagem_digito(digito, caminho)
        avaliar_imagem(model, caminho)


if __name__ == "__main__":
    modelo, imagens_teste, rotulos_teste = treinar_modelo()

    print("\nTeste com imagens reais do MNIST:")
    testar_com_imagens_do_mnist(modelo, imagens_teste, rotulos_teste)

    print("\nTeste com imagens criadas por codigo:")
    criar_e_avaliar_novas_imagens(modelo)
