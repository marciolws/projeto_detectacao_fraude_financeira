import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Configuração do título da aplicação
st.title("Análise de Fraude em Transações")

# Carregar os dados automaticamente
@st.cache_data
def load_data():
    """Carrega os dados a partir de um arquivo CSV."""
    return pd.read_csv("arquivo_reduzido.csv")


# Carregar os dados
df = load_data()

# Barra lateral para informações do autor
st.sidebar.header("Informações do Autor")
st.sidebar.write("Autor: Marcio Silva")
# Substitua pelo seu LinkedIn
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/marcio-d-silva/)")

# Barra lateral para navegação
st.sidebar.header("Navegação")
options = st.sidebar.radio("Escolha uma seção:",
                           ["Introdução", "Análise Exploratória", "Pré-processamento",
                            "Treinamento do Modelo", "Avaliação do Modelo"])

# Introdução
if options == "Introdução":
    st.write(
        "Esta aplicação analisa transações para identificar fraudes utilizando Machine Learning. "
        "O modelo é treinado para prever a probabilidade de uma transação ser fraudulenta com base em dados históricos.")

# Análise Exploratória
elif options == "Análise Exploratória":
    st.header("2. Análise Exploratória de Dados")
    st.write("Dados Carregados:")
    st.dataframe(df.head())

    if st.checkbox("Mostrar Resumo Estatístico"):
        st.write(df.describe())

    if st.checkbox("Mostrar Contagem de Valores Únicos"):
        st.write(df.nunique())

    if st.checkbox("Verificar Dados Nulos"):
        st.write(df.isnull().sum())

    if st.checkbox("Mostrar Matriz de Correlação"):
        """Visualiza a correlação entre variáveis numéricas."""
        numeric_df = df.select_dtypes(include=[np.number])
        plt.figure(figsize=(10, 5))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="Blues")
        plt.title("Matriz de Correlação")
        st.pyplot(plt)

    if st.checkbox("Mostrar Gráfico de Pizza das Variáveis Categóricas"):
        """Cria um gráfico de pizza para a variável categórica 'type'."""
        col = 'type'
        category_counts = df[col].value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(category_counts, labels=category_counts.index,
                autopct='%1.1f%%', startangle=140)
        plt.title(f'Gráfico de Pizza para a Variável {col}')
        plt.axis('equal')
        st.pyplot(plt)

# Pré-processamento
elif options == "Pré-processamento":
    st.header("3. Pré-processamento de Dados")

    # Codificação de variáveis categóricas
    if st.checkbox("Codificar Variáveis Categóricas"):
        """Codifica variáveis categóricas usando Label Encoding."""
        encoder = LabelEncoder()
        for column in df.select_dtypes('object').columns:
            df[column] = encoder.fit_transform(df[column])
        st.write("Variáveis Categóricas Codificadas")

    # Normalização dos dados
    if st.checkbox("Normalizar Dados"):
        """Normaliza os dados numéricos usando Min-Max Scaler."""
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        scaler = MinMaxScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        st.write("Dados Normalizados")

    # Divisão de dados
    X = df.drop(['isFraud'], axis=1)
    y = df['isFraud']

    # Oversampling
    if st.checkbox("Sobreamostrar Dados"):
        """Aplica SMOTE para lidar com o desbalanceamento de classes."""
        smote = SMOTE(random_state=0)
        X, y = smote.fit_resample(X, y)
        st.write("Dados Sobreamostrados")

    # Dividir o conjunto de dados
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=0)

    # Armazenar conjuntos de dados no estado da sessão
    st.session_state.x_train = x_train
    st.session_state.x_test = x_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

# Treinamento do Modelo
elif options == "Treinamento do Modelo":
    st.header("4. Treinamento do Modelo")

    if 'x_train' in st.session_state and 'y_train' in st.session_state:
        """Treina o modelo de regressão logística com os dados de treino."""
        model = LogisticRegression(max_iter=1000)
        model.fit(st.session_state.x_train, st.session_state.y_train)

        # Fazer previsões
        y_pred = model.predict(st.session_state.x_test)
        st.session_state.y_pred = y_pred  # Armazenando previsões no estado da sessão
    else:
        st.write("Por favor, realize o pré-processamento primeiro.")

# Avaliação do Modelo
elif options == "Avaliação do Modelo":
    st.header("5. Avaliação do Modelo")

    if 'y_pred' in st.session_state:
        """Avalia o desempenho do modelo utilizando métricas de classificação."""
        y_pred = st.session_state.y_pred
        accuracy = accuracy_score(st.session_state.y_test, y_pred)
        st.write(f"Acurácia: {accuracy:.2f}")

        st.write("Relatório de Classificação:")
        st.text(classification_report(st.session_state.y_test, y_pred))

        st.header("6. Matriz de Confusão")
        confusion = confusion_matrix(st.session_state.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Previsões')
        plt.ylabel('Valores Reais')
        plt.title('Matriz de Confusão')
        st.pyplot(plt)

        if st.checkbox("Mostrar Histogramas das Variáveis Numéricas"):
            """Exibe histogramas das variáveis numéricas, comparando transações normais e fraudulentas."""
            num_cols = df.select_dtypes(include=['float64', 'int64']).columns
            plt.figure(figsize=(12, 28*4))
            gs = gridspec.GridSpec(28, 1)
            for i, cn in enumerate(df[num_cols]):
                ax = plt.subplot(gs[i])
                sns.distplot(df[cn][df.isFraud == 0], bins=50, color='blue', label='Normal')
                sns.distplot(df[cn][df.isFraud == 1], bins=50, color='red', label='Fraude')
                ax.set_xlabel('')
                ax.set_title('Histograma de Recursos: ' + str(cn))
                ax.legend()
            st.pyplot(plt)

        if st.checkbox("Comparar Transações Normais e Fraudulentas"):
            """Compara a distribuição de transações normais e fraudulentas ao longo do tempo."""
            f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 4))
            bins = 50
            ax1.hist(df.step[df.isFraud == 0], bins=bins, color='blue')
            ax1.set_title('Transações Normais')
            ax2.hist(df.step[df.isFraud == 1], bins=bins, color='red')
            ax2.set_title('Transações Fraudulentas')
            plt.xlabel('Tempo (30 dias)')
            plt.ylabel('Número de Transações')
            st.pyplot(plt)
    else:
        st.write("Por favor, treine o modelo primeiro para ver a avaliação.")
