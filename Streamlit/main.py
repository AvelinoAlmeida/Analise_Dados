# Importar bibliotecas

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import timedelta

# Função para carregar dados

@st.cache_data
def carregar_dados(empresas):
    """
    Carrega dados históricos de ações listadas na Euronext Lisbon.
    
    Args:
        empresas (list): Lista de símbolos das empresas.
    
    Returns:
        pd.DataFrame: Preços de fecho das ações.
    """
    # Junta os símbolos num único texto para consulta

    texto_tickers = " ".join(empresas)

    # Obtém os dados históricos das ações

    dados_acao = yf.Tickers(texto_tickers)
    cotacoes_acao = dados_acao.history(period="1d", start="2020-01-01", end="2024-10-30")

    # Mantém apenas os preços de fecho
    cotacoes_acao = cotacoes_acao["Close"]
    return cotacoes_acao

# Lista de ações para análise
acoes = ["GALP.LS", "JMT.LS", "EDP.LS", "BCP.LS", "SEM.LS"]
dados = carregar_dados(acoes)

# Interface do Streamlit
st.write("""
# Preço de Ações
Este gráfico apresenta a evolução do preço das ações selecionadas.
""")

# Sidebar com filtros
st.sidebar.header("Filtros")

# Filtro de ações
acoes_selecionadas = st.sidebar.multiselect("Escolha as ações para análise", dados.columns)
if acoes_selecionadas:
    dados = dados[acoes_selecionadas]
    # Renomear a coluna para facilitar manipulação em caso de uma única ação selecionada
    if len(acoes_selecionadas) == 1:
        acao_unica = acoes_selecionadas[0]
        dados = dados.rename(columns={acao_unica: "Close"})

# Filtro de datas
data_inicial = dados.index.min().to_pydatetime()
data_final = dados.index.max().to_pydatetime()
intervalo_data = st.sidebar.slider(
    "Selecionar o período",
    min_value=data_inicial,
    max_value=data_final,
    value=(data_inicial, data_final),
    step=timedelta(days=1),
)

# Filtrar os dados pelas datas selecionadas
dados = dados.loc[intervalo_data[0]:intervalo_data[1]]

# Criar o gráfico de evolução dos preços
st.line_chart(dados)

# Calcular a performance da carteira
texto_performance_ativos = ""

# Garantir que todas as ações estão consideradas
if len(acoes_selecionadas) == 0:
    acoes_selecionadas = list(dados.columns)
elif len(acoes_selecionadas) == 1:
    dados = dados.rename(columns={"Close": acao_unica})

# Definir um investimento inicial por ativo
carteira = [200 for _ in acoes_selecionadas]
total_inicial_carteira = sum(carteira)

# Calcular performance por ativo e atualizar a carteira
for i, acao in enumerate(acoes_selecionadas):
    # Performance do ativo (percentual de valorização)
    performance_ativo = dados[acao].iloc[-1] / dados[acao].iloc[0] - 1
    performance_ativo = float(performance_ativo)

    # Atualizar o valor da carteira com base na performance
    carteira[i] = carteira[i] * (1 + performance_ativo)

    # Adicionar informações sobre a performance do ativo ao texto
    if performance_ativo > 0:
        texto_performance_ativos += f"  \n{acao}: :green[{performance_ativo:.1%}]"
    elif performance_ativo < 0:
        texto_performance_ativos += f"  \n{acao}: :red[{performance_ativo:.1%}]"
    else:
        texto_performance_ativos += f"  \n{acao}: {performance_ativo:.1%}"

# Calcular a performance total da carteira
total_final_carteira = sum(carteira)
performance_carteira = total_final_carteira / total_inicial_carteira - 1

# Formatar a performance da carteira com cores
if performance_carteira > 0:
    texto_performance_carteira = f" Total da minha carteira \n :green[{performance_carteira:.1%}]"
elif performance_carteira < 0:
    texto_performance_carteira = f" Total da minha carteira \n :red[{performance_carteira:.1%}]"
else:
    texto_performance_carteira = f" Total da minha carteira \n  {performance_carteira:.1%}"

# Exibir o desempenho dos ativos
st.write(f"""
## Desempenho por Ativo:

{texto_performance_ativos}
""")

# Exibir o desempenho da carteira
st.write(f"""
## Avaliação da Carteira:

{texto_performance_carteira}
""")


