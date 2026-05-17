# Criar Imagens e Avaliar com uma CNN no TensorFlow

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PauloCunhaJunior/criar-imagens/blob/main/imagem.ipynb)

Este projeto utiliza uma Rede Neural Convolucional, conhecida como CNN, para reconhecer imagens de números de 0 a 9 usando o conjunto de dados MNIST.

O MNIST é formado por imagens pequenas, no tamanho 28x28 pixels, contendo números escritos à mão. A CNN é treinada com essas imagens e, depois do treinamento, o projeto também cria novas imagens por código para testar se o modelo consegue reconhecer os números corretamente.

## Objetivo do projeto

O objetivo deste projeto é demonstrar, de forma simples, como uma CNN pode ser usada para reconhecer imagens de dígitos numéricos.

O algoritmo realiza três etapas principais:

1. Treina uma CNN usando o conjunto de dados MNIST.
2. Avalia o desempenho do modelo com imagens reais de teste.
3. Cria novas imagens de números por código e verifica se o modelo consegue reconhecê-las.

## Melhor forma de executar

A forma recomendada de executar este projeto é pelo Google Colab, pois ele já possui um ambiente mais preparado para trabalhar com bibliotecas de Inteligência Artificial, como o TensorFlow.

Para abrir o projeto no Colab, clique no botão abaixo:

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PauloCunhaJunior/criar-imagens/blob/main/imagem.ipynb)

## Por que usar o Google Colab?

O TensorFlow pode apresentar problemas em versões muito novas do Python instaladas no computador, como o Python 3.14.

Por isso, o Google Colab é uma opção mais simples, pois permite executar o código diretamente pelo navegador, sem a necessidade de instalar todas as bibliotecas manualmente no computador.

## Tecnologias utilizadas

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pillow
- Google Colab
- GitHub

## Como o algoritmo funciona

Primeiro, o algoritmo carrega o conjunto de dados MNIST, que contém imagens de números escritos à mão.

Depois, as imagens são normalizadas, ou seja, seus valores são ajustados para ficarem entre 0 e 1. Isso facilita o processamento pela rede neural.

Em seguida, uma CNN é criada e treinada para identificar padrões nas imagens, como formas, curvas e traços dos números.

Após o treinamento, o modelo é avaliado com imagens de teste e também com imagens novas criadas pelo próprio código.

## Estrutura principal do modelo

O modelo utiliza camadas convolucionais para identificar características visuais das imagens, camadas de redução para simplificar as informações e camadas densas para realizar a classificação final do número.

A saída final do modelo possui 10 possibilidades, representando os números de 0 a 9.

## Ideia principal das imagens

O modelo foi treinado com imagens no seguinte padrão:

- tamanho: `28x28` pixels;
- cor: preto e branco;
- fundo: escuro;
- número: claro;
- formato de entrada na rede: `(1, 28, 28, 1)`.

Por isso, antes de avaliar qualquer imagem nova, ela precisa ser convertida para esse mesmo padrão.

## Trecho importante do código

```python
imagem = Image.open(caminho).convert("L").resize((28, 28))
array = np.array(imagem).astype("float32") / 255.0

if array.mean() > 0.5:
    array = 1.0 - array

array = array.reshape(1, 28, 28, 1)
predicao = model.predict(array)
numero_previsto = np.argmax(predicao)