import streamlit as st
import pandas as pd
import joblib

# 1. Configuração da página do Streamlit
st.set_page_config(page_title="Preditor de Preços de Imóveis", page_icon="🏠", layout="centered")

st.title("🏠 Precificação Inteligente de Imóveis")
st.markdown("Insira as características do imóvel para calcular a estimativa de preço de venda via Machine Learning (CatBoost).")

# 2. Carregar o modelo, as colunas e os valores padrão de treino
@st.cache_resource
def load_model_artifacts():
    model = joblib.load('modelo_imoveis_catboost.pkl')
    columns = joblib.load('colunas_modelo.pkl')
    defaults = joblib.load('valores_padrao.pkl') # Carrega as medianas/modas de treino
    return model, columns, defaults

try:
    model, model_columns, valores_padrao = load_model_artifacts()
except Exception as e:
    st.error("Erro ao carregar os artefatos do modelo. Verifique se os arquivos .pkl estão no repositório.")
    st.stop()

# 3. Interface de Usuário (Inputs principais baseados na EDA)
st.sidebar.header("📍 Localização")
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

# 4. Processamento dos dados
if st.button("💰 Calcular Preço Estimado", type="primary"):
    
    #  Cria a base de inputs usando as MEDIANAS de treino como padrão, e não ZERO!
    # Cria um DataFrame vazio mapeando as colunas originais (antes do get_dummies)
    df_base_original = pd.DataFrame([valores_padrao])
    
    # Substituí os valores padrão do dicionário pelos valores que o usuário digitou na tela
    df_base_original['GrLivArea'] = gr_liv_area
    df_base_original['TotalBsmtSF'] = total_bsmt_sf
    df_base_original['OverallQual'] = overall_qual
    df_base_original['YearBuilt'] = ano_construcao
    df_base_original['YrSold'] = ano_venda
    df_base_original['Neighborhood'] = bairro_escolhido
    
    # Recriando a engenharia de features 
    df_base_original['TotalSF'] = gr_liv_area + total_bsmt_sf
    df_base_original['IdadeImovel'] = max(ano_venda - ano_construcao, 0)
    df_base_original['HasPool'] = 1 if tem_piscina else 0
    df_base_original['HasFireplace'] = 1 if tem_lareira else 0
    df_base_original['HasBasement'] = 1 if tem_porao else 0
    
    # Aplicando o One-Hot Encoding na linha de input
    df_encoded = pd.get_dummies(df_base_original)
    
    # Criando o dicionário final com todas as colunas esperadas pelo modelo, inicializadas em 0
    input_final = {col: 0 for col in model_columns}
    
    # Preenche o dicionário com os valores numéricos e os dummies gerados pelo get_dummies do input
    for col in model_columns:
        if col in df_encoded.columns:
            input_final[col] = df_encoded[col].values[0]
        elif col in df_base_original.columns:
            # Caso a coluna numérica não tenha sofrido encoding (ex: LotArea), pega o valor padrão
            input_final[col] = df_base_original[col].values[0]
            
    # Transforma no DataFrame final ordenado exatamente como o modelo espera
    df_input_modelo = pd.DataFrame([input_final])[model_columns]
    
    # Executa a previsão com dados realistas e protegidos
    previsao = model.predict(df_input_modelo)[0]
    
    st.success(f"### Preço Estimado de Venda: **${previsao:,.2f}**")