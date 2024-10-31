# Análise de Fraude em Transações

Este projeto visa desenvolver uma aplicação de Machine Learning para identificar fraudes em transações financeiras. A aplicação utiliza a biblioteca Streamlit para criar uma interface interativa, permitindo aos usuários explorar e visualizar os dados, realizar pré-processamento, treinar um modelo e avaliar seu desempenho.

## Índice

- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Descrição do Dataset](#descrição-do-dataset)
- [Funcionalidades](#funcionalidades)
- [Como Usar](#como-usar)
- [Estrutura do Código](#estrutura-do-código)
- [Contribuições](#contribuições)
- [Licença](#licença)

## Tecnologias Utilizadas

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn

## Descrição do Dataset

O dataset utilizado neste projeto é o "Fraud.csv", que contém informações sobre transações financeiras. As colunas principais incluem:

- `isFraud`: Indica se a transação é fraudulenta (1) ou não (0).
- Outras colunas numéricas e categóricas que descrevem características das transações.

## Funcionalidades

A aplicação oferece as seguintes funcionalidades:

1. **Introdução**: Descrição do projeto e sua finalidade.
2. **Análise Exploratória**: Visualização dos dados e análise estatística.
   - Resumo estatístico
   - Contagem de valores únicos
   - Verificação de dados nulos
   - Matriz de correlação
   - Gráfico de pizza para variáveis categóricas
3. **Pré-processamento**: Preparação dos dados para o modelo.
   - Codificação de variáveis categóricas
   - Normalização dos dados
   - Sobreamostragem usando SMOTE
   - Divisão dos dados em conjuntos de treino e teste
4. **Treinamento do Modelo**: Treinamento de um modelo de Regressão Logística.
5. **Avaliação do Modelo**: Avaliação do desempenho do modelo treinado.
   - Acurácia
   - Relatório de classificação
   - Matriz de confusão
   - Histogramas das variáveis numéricas
   - Comparação entre transações normais e fraudulentas

## Como Usar

Acesse a aplicação diretamente no seguinte link: [Deploy da Aplicação](<LINK_DO_DEPLOY>).

## Estrutura do Código

- **`strem.py`**: O arquivo principal que contém toda a lógica da aplicação.
- **`data/Fraud.csv`**: O dataset utilizado para a análise.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir um issue ou enviar um pull request.
