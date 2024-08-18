# Detecção de Spam com LSTM em Python

Este projeto tem como objetivo construir um modelo de detecção de spam utilizando redes Long Short-Term Memory (LSTM). O projeto é implementado em Python usando Jupyter Notebook, com foco em pré-processamento de texto e aprendizado profundo.

## Visão Geral do Projeto

O objetivo deste projeto é classificar mensagens de texto como spam ou não spam usando um modelo de rede neural construído com TensorFlow e Keras. O projeto envolve:
- Pré-processamento e exploração de dados
- Tokenização de texto e preenchimento de sequências
- Construção e treinamento de um modelo baseado em LSTM
- Avaliação do desempenho do modelo

## Dependências

Para rodar este projeto, você precisará das seguintes bibliotecas:

```python
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Flatten, Dense, Dropout, LSTM, SpatialDropout1D, Bidirectional
```

### Instalação
Você pode instalar as bibliotecas necessárias usando pip:

```bash
pip install tensorflow numpy pandas seaborn matplotlib wordcloud scikit-learn
```

## Uso
Clone o repositório:

```bash
git clone https://github.com/seunomeusuario/deteccao-de-spam-lstm.git
cd deteccao-de-spam-lstm
```

Abra o Jupyter Notebook:

```bash

jupyter notebook deteccao_de_spam.ipynb
```

Execute as células do notebook para pré-processar os dados, construir o modelo e treiná-lo no conjunto de dados.

## Arquitetura do Modelo
A arquitetura do modelo LSTM consiste em:

- Camada de Embedding para representação vetorial de palavras
- Camadas LSTM com dropout para aprendizado de sequência
- Camada Flatten para converter a matriz 2D em um vetor 1D
- Camada Dense com ativação sigmoid para classificação binária
- Treinamento
- O modelo é treinado utilizando early stopping para evitar overfitting. O número de épocas é definido como 30, com paciência de 2 para o early stopping.

## Resultados
O desempenho do modelo é avaliado com base em precisão, perda e outras métricas relevantes. Ferramentas de visualização como Seaborn e Matplotlib são usadas para plotar os resultados.

## Licença
Este projeto é licenciado sob a Licença MIT - veja o arquivo LICENSE para mais detalhes.

