# Criar imagens e avaliar com uma CNN no TensorFlow

Este projeto usa o conjunto MNIST, que tem imagens 28x28 de numeros escritos a mao.
A CNN aprende a reconhecer numeros de 0 a 9 e depois avaliamos imagens novas criadas por codigo.

## Melhor caminho: Google Colab

Como o TensorFlow pode nao funcionar ainda no Python 3.14, o jeito mais simples e usar o Google Colab.

1. Abra <https://colab.research.google.com/>
2. Crie um notebook novo.
3. Copie o conteudo do arquivo `imagem.py` para uma celula.
4. Execute a celula.

O codigo vai:

- treinar a CNN com MNIST;
- mostrar a precisao nos dados de teste;
- criar arquivos PNG em `imagens_criadas/`;
- pedir para a rede prever qual numero aparece em cada imagem criada.

## Ideia principal

O modelo foi treinado com imagens assim:

- tamanho: `28x28` pixels;
- cor: preto e branco;
- fundo: escuro;
- numero: claro;
- formato que entra na rede: `(1, 28, 28, 1)`.

Por isso, antes de avaliar qualquer imagem nova, precisamos converter a imagem para esse mesmo padrao.

## Trecho mais importante

```python
imagem = Image.open(caminho).convert("L").resize((28, 28))
array = np.array(imagem).astype("float32") / 255.0

if array.mean() > 0.5:
    array = 1.0 - array

array = array.reshape(1, 28, 28, 1)
predicao = model.predict(array)
numero_previsto = np.argmax(predicao)
```

Esse trecho abre a imagem, deixa em escala de cinza, redimensiona para 28x28,
normaliza os pixels e ajusta para o formato esperado pela CNN.

## Como testar com sua propria imagem

No Colab, depois de treinar o modelo, voce pode enviar uma imagem do seu computador:

```python
from google.colab import files

uploaded = files.upload()

for nome_arquivo in uploaded.keys():
    avaliar_imagem(modelo, nome_arquivo)
```

Desenhe um numero simples, de preferencia com bastante contraste. Se usar fundo branco
e numero preto, o codigo tenta inverter automaticamente para ficar parecido com o MNIST.

## Observacao importante

As imagens criadas por fonte de computador nao sao exatamente iguais aos numeros escritos
a mao do MNIST. Entao o modelo pode errar algumas. Isso tambem e parte boa do estudo:
mostra que uma rede neural costuma funcionar melhor quando os dados novos parecem com os
dados usados no treinamento.
