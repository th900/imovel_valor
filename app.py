import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Configuração da página do Streamlit
st.set_page_config(page_title="Preditor de Preços de Imóveis", page_icon="🏠", layout="centered")

st.title("🏠 Precificação Inteligente de Imóveis")
st.markdown("Insira as características do imóvel para calcular a estimativa de preço de venda via Machine Learning (CatBoost).")

# 2. Carregar o modelo e as colunas salvas no Colab
@st.cache_resource
def load_model_artifacts():
    model = joblib.load('modelo_imoveis_catboost.pkl')
    columns = joblib.load('colunas_modelo.pkl')
    return model, columns

try:
    model, model_columns = load_model_artifacts()
except Exception as e:
    st.error("Erro ao carregar os arquivos do modelo. Certifique-se de que 'modelo_imoveis_catboost.pkl' e 'colunas_modelo.pkl' estão na mesma pasta.")
    st.stop()

# 3. Criando a Interface de Usuário (Inputs) baseada nos insights da EDA
st.sidebar.header("📍 Localização")
# Listamos alguns bairros principais do dataset (você pode adicionar mais se quiser)
bairro_escolhido = st.sidebar.selectbox("Selecione o Bairro:", ["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst", "NridgHt", "OldTown", "BrkSide", "Sawyer", "NWAmes", "SawyerW", "IDOTRR", "MeadowV", "Edwards", "Timber", "Gilbert", "StoneBr", "ClearCr", "NPkVW", "Blmngtn", "BrDale", "SWISU", "Blueste"])

st.subheader("📐 Dimensões e Qualidade")
col1, col2 = st.columns(2)
with col1:
    gr_liv_area = st.number_input("Área Construída Acima do Solo (sqft):", min_value=300, max_value=6000, value=1500)
    total_bsmt_sf = st.number_input("Área do Porão (sqft):", min_value=0, max_value=6000, value=1000)
with col2:
    overall_qual = st.slider("Qualidade Geral do Material (1 a 10):", min_value=1, max_value=10, value=6)
    ano_construcao = st.number_input("Ano de Construção:", min_value=1800, max_value=2026, value=2000)
    ano_venda = st.number_input("Ano da Venda:", min_value=2006, max_value=2026, value=2010)

st.subheader("✨ Diferenciais e Acessórios")
col3, col4, col5 = st.columns(3)
with col3:
    tem_piscina = st.checkbox("Possui Piscina?")
with col4:
    tem_lareira = st.checkbox("Possui Lareira?")
with col5:
    tem_porao = st.checkbox("Possui Porão?")

# 4. Processamento dos dados inseridos para o formato que o modelo espera
if st.button("💰 Calcular Preço Estimado", type="primary"):
    
    # Criamos um dicionário com os valores padrão zerados para TODAS as colunas que o modelo espera
    input_data = {col: 0 for col in model_columns}
    
    # Preenchemos as variáveis numéricas diretas
    input_data['GrLivArea'] = gr_liv_area
    input_data['TotalBsmtSF'] = total_bsmt_sf
    input_data['OverallQual'] = overall_qual
    input_data['YearBuilt'] = ano_construcao
    input_data['YrSold'] = ano_venda
    
    # Engenharia de Features idêntica ao Colab
    input_data['TotalSF'] = gr_liv_area + total_bsmt_sf
    input_data['IdadeImovel'] = max(ano_venda - ano_construcao, 0)
    input_data['HasPool'] = 1 if tem_piscina else 0
    input_data['HasFireplace'] = 1 if tem_lareira else 0
    input_data['HasBasement'] = 1 if tem_porao else 0
    
    # Ativando a coluna correspondente ao One-Hot Encoding do Bairro (ex: Neighborhood_NoRidge = 1)
    coluna_bairro = f"Neighborhood_{bairro_escolhido}"
    if coluna_bairro in input_data:
        input_data[coluna_bairro] = 1
        
    # Transforma em DataFrame com a ordem exata de colunas do modelo
    df_input = pd.DataFrame([input_data])[model_columns]
    
    # Executa a previsão
    previsao = model.predict(df_input)[0]
    
    # Exibe o resultado na tela de forma destacada
    st.success(f"### Preço Estimado de Venda: **${previsao:,.2f}**")