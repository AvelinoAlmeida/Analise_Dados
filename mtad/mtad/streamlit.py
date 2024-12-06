import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt

# Carregar o modelo
model = XGBClassifier()
model.load_model("xgb_model.json")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("important_features.pkl")

# Configurar o Streamlit
st.set_page_config(page_title="Sistema de Previsão de Diabetes", layout="wide")

# Configurar a barra lateral para navegação
st.sidebar.title("Menu de Navegação")
menu = st.sidebar.radio("Ir para", ["Início", "Estatísticas", "Previsão"])

# Carregar o dataset
df = pd.read_csv("diabetes.csv")

# Página inicial (Início)
if menu == "Início":
    st.markdown("<h2>🩺 Bem-vindo ao Sistema de Previsão de Diabetes</h2>", unsafe_allow_html=True)
    st.image("diabetes_image.jpg")
    st.markdown("<p style='font-size:18px;'>Este sistema foi desenvolvido para ajudar a prever o risco de diabetes com base em dados clínicos e comportamentais.</p>", unsafe_allow_html=True)

# Página de análises
elif menu == "Estatísticas":
    st.title("📊 Estatísticas e Análises dos Dados")
    
    # ---- Gráfico de Distribuição de Diabetes por Faixa Etária ----
    st.subheader("Distribuição de Diabetes por Faixa Etária")

    # faixas etárias
    age_labels = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49',
                  '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+']
    df['Age_Group'] = df['Age'].map(lambda x: age_labels[int(x) - 1])

    # filtros interativos
    category_filter = st.multiselect(
        "Selecione as categorias de diabetes:",
        options=["Sem Diabetes", "Pré-Diabetes", "Diabetes"],
        default=["Sem Diabetes", "Pré-Diabetes", "Diabetes"]
    )
    category_map = {"Sem Diabetes": 0, "Pré-Diabetes": 1, "Diabetes": 2}
    selected_categories = [category_map[cat] for cat in category_filter]

    filtered_df = df[df["Diabetes_012"].isin(selected_categories)]

    # Contagem de sem diabetes, pré-diabetes e diabetes por faixa etária
    age_diabetes_counts = filtered_df.groupby(['Age_Group', 'Diabetes_012']).size().unstack(fill_value=0)

    for col in [0, 1, 2]:
        if col not in age_diabetes_counts.columns:
            age_diabetes_counts[col] = 0

    age_diabetes_counts = age_diabetes_counts.reset_index().melt(
        id_vars="Age_Group",
        value_vars=[0, 1, 2],
        var_name="Categoria",
        value_name="Frequência"
    )

    # mapear os valores da coluna 'Categoria' para nomes mais legíveis
    age_diabetes_counts['Categoria'] = age_diabetes_counts['Categoria'].map({0: 'Sem Diabetes', 1: 'Pré-Diabetes', 2: 'Diabetes'})

    # gráfico interativo com Plotly
    fig = px.bar(
        age_diabetes_counts,
        x="Age_Group",
        y="Frequência",
        color="Categoria",
        barmode="group",
        labels={"Age_Group": "Faixa Etária", "Frequência": "Frequência", "Categoria": "Categoria de Diabetes"},
        title="Distribuição de Diabetes por Faixa Etária"
    )

    st.plotly_chart(fig, use_container_width=True)


    # ---- Gráfico de Correlação com Filtros ----
    st.subheader("Correlação das Variáveis com Diabetes")
    
    # Calcular correlações com a variável 'Diabetes_012'
    correlations = df.drop('Diabetes_012', axis=1).corrwith(df['Diabetes_012'])
    correlations_df = correlations.reset_index()
    correlations_df.columns = ['Variável', 'Correlação']
    correlations_df = correlations_df.sort_values(by='Correlação', ascending=False)

    # Filtro de Tipo de Correlação
    filter_sign = st.radio(
        "Selecione o tipo de correlação",
        options=["Todas", "Positivas", "Negativas"],
        index=0,
        horizontal=True
    )

    # Aplicar filtros às correlações
    filtered_correlations = correlations_df
    if filter_sign == "Positivas":
        filtered_correlations = filtered_correlations[filtered_correlations['Correlação'] > 0]
    elif filter_sign == "Negativas":
        filtered_correlations = filtered_correlations[filtered_correlations['Correlação'] < 0]

    # Criar gráfico interativo com Plotly
    fig_corr = px.bar(
        filtered_correlations,
        x='Variável',
        y='Correlação',
        color='Correlação',
        color_continuous_scale=px.colors.diverging.Tealrose,
        title="Correlação das Variáveis com Diabetes_012",
        labels={"Variável": "Variável", "Correlação": "Valor da Correlação"}
    )

    st.plotly_chart(fig_corr, use_container_width=True)


# Página de previsão
elif menu == "Previsão":
    st.title("⚕️ Previsão de Diabetes")
    st.write("Preencha as informações abaixo para prever o risco de diabetes:")

    with st.expander("ℹ️ O que significam as variáveis?"):
        st.markdown("""
        - **HighBP**: Indica se a pessoa tem pressão alta (0 = Não, 1 = Sim).
        - **HighChol**: Indica se a pessoa tem colesterol alto (0 = Não, 1 = Sim).
        - **BMI**: Índice de Massa Corporal (12 = baixo peso, 25 = peso normal, >40 = obesidade).
        - **Smoker**: Indica se a pessoa fuma (0 = Não, 1 = Sim).
        - **Stroke**: Indica se a pessoa já teve um acidente vascular cerebral (0 = Não, 1 = Sim).
        - **HeartDiseaseorAttack**: Indica se a pessoa já teve um ataque cardíaco ou sofre de doença cardíaca (0 = Não, 1 = Sim).
        - **PhysActivity**: Indica se a pessoa pratica atividade física regularmente (0 = Não, 1 = Sim).
        - **HvyAlcoholConsump**: Indica se a pessoa consome álcool em excesso (0 = Não, 1 = Sim).
        - **GenHlth**: Autoavaliação da saúde geral (1 = Excelente, 5 = Pobre).
        - **MentHlth**: Dias nos últimos 30 em que a saúde mental foi ruim (1 = Nenhum dia, 5 = Todos os dias).
        - **PhysHlth**: Dias nos últimos 30 em que a saúde física foi ruim (1 = Nenhum dia, 5 = Todos os dias)
        - **DiffWalk**: Indica se a pessoa tem dificuldade para caminhar (0 = Não, 1 = Sim).
        - **Age**: Intervalo de idades (1 = 18-24 anos, 2 = 25-29, 3 = 30-34, 4 = 35-39, 5 = 40-44, 6 = 45-49, 7 = 50-54, 8 = 55-59, 9 = 60-64 anos, 10 = 65-69, 11 = 70-74, 12 = 75-79, 13 = 80+ anos).
        - **Income**: Intervalo de rendimento anual (1 = Menos de 10.000 $, ..., 5 = Mais de 75.000 $).
        """)

    # Features contínuas e categóricas
    continuous_features = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income', ]
    binary_features = ['HighBP', 'HighChol', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'DiffWalk', 'Smoker', 'HvyAlcoholConsump']

    important_features = ['HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'HvyAlcoholConsump', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Age', 'Income']
    
    unwanted_features = ['Education']

    # Valores padrão para os sliders
    default_values = {
        "HighBP": 0,
        "HighChol": 0,
        "BMI": 30.0,
        "Smoker": 0,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": 0,
        "HvyAlcoholConsump": 0,
        "GenHlth": 3,
        "MentHlth": 0,
        "PhysHlth": 0,
        "DiffWalk": 0,
        "Age": 6,
        "Income": 1,
        "Education": 1,
    }

    # Criar sliders para variáveis contínuas
    input_data = {}
    for feature in continuous_features:
        if feature == "Age":
            input_data[feature] = st.slider(feature, 1, 13, default_values[feature])
        elif feature == "BMI":
            input_data[feature] = st.slider(feature, 12.0, 98.0, default_values[feature])
        elif feature == "Income":
            input_data[feature] = st.slider(feature, 1, 5, default_values[feature])
        elif feature in ["GenHlth", "MentHlth", "PhysHlth"]:
            input_data[feature] = st.slider(feature, 1, 5, default_values[feature])
        elif feature in unwanted_features:
            input_data[feature] = default_values[feature]

    # Criar sliders para variáveis categóricas
    for feature in binary_features:
        input_data[feature] = st.slider(feature, 0, 1, default_values[feature])

    # Converter input_data para um dataframe
    input_df = pd.DataFrame([input_data], columns=feature_columns + unwanted_features)

    # print valores das features contínuas
    input_df[continuous_features].head()

    # normalizar as features contínuas
    input_df[continuous_features] = scaler.transform(input_df[continuous_features])

    # drop das features não necessárias para a previsão
    input_df = input_df.drop(columns=unwanted_features)

    # Garantir que as colunas estejam na mesma ordem do treino
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Botão para fazer a previsão
    if st.button("Fazer Previsão"):
        # Fazer a previsão
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # Exibir resultado amigável
        st.write(f"**Risco Previsto:** {'Sem Diabetes' if prediction[0] == 0 else 'Diabetes'}")
        st.write(f"**Probabilidade de Sem Diabetes (0):** {np.round(prediction_proba[0][0], 2)}")
        st.write(f"**Probabilidade de Diabetes (1):** {np.round(prediction_proba[0][1], 2)}")

        # Gráfico de probabilidades
        prob_df = pd.DataFrame(
            {
                "Label": ["Sem Diabetes (0)", "Diabetes (1)"],
                "Probabilidade": prediction_proba[0]
            }
        ).set_index("Label")
        st.bar_chart(prob_df)