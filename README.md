# 🏠 Precificação Inteligente de Imóveis com Machine Learning

[![Streamlit App](https://static.streamlit.io/badge_svg.svg)](https://imovelvalor-3jmdrw5gjn6xfnrup5pyfo.streamlit.app/)

Este projeto apresenta o desenvolvimento e o deploy de um modelo preditivo de alta performance para estimar o preço de venda de imóveis. O objetivo principal é transformar dados brutos em uma ferramenta de suporte à decisão de negócios, otimizando margens imobiliárias e reduzindo o tempo de avaliação manual de propriedades.

A aplicação final foi implementada em produção e pode ser testada interativamente através do link acima.


## 🏢 1. O Problema de Negócio

No mercado imobiliário, a precificação incorreta de imóveis gera gargalos operacionais e financeiros:
* **Subprecificação:** Reduz a margem de lucro e a comissão da empresa, além de frustrar o proprietário.
* **Superprecificação:** Deixa o imóvel "encalhado" no mercado, gerando custos de anúncio sem conversão.
* **Lentidão:** Processos manuais de avaliação demoram dias para ponderar dezenas de variáveis estruturais e regionais.

**Solução:** Um algoritmo de regressão avançado capaz de prever o valor justo de mercado (`SalePrice`) em segundos, servindo como um balizador de preços automatizado para a operação.

---

## 🔍 2. Principais Insights da Análise Exploratória (EDA)

* **Relação Área vs. Preço:** Existe uma tendência linear clara de valorização proporcional à área construída acima do solo (`GrLivArea`).
* **Teto de Valorização por Localização:** Bairros como `NoRidge`, `NridgHt` e `StoneBr` dominam o topo de preços. Insights multivariados mostraram que imóveis grandes em bairros desvalorizados sofrem forte penalização de preço, provando que a localização atua como um moderador crítico do preço por m².
* **Acessórios de Margem:** Adicionais como **Piscina, Lareira e Porão** estão correlacionados com saltos expressivos no preço médio dos imóveis.

---

## 🛠️ 3. Engenharia de Features e Rigor Técnico

Para garantir a robustez estatística e simular um ambiente de produção real, o projeto adotou práticas rigorosas recomendadas pelo mercado:

1. **Tratamento de Outliers Cirúrgico:** Em vez de remover dados cegamente com base em análises bidimensionais isoladas, seguiu-se a recomendação do autor original do dataset (Dean De Cock) para remover estritamente **2 pontos de vendas atípicas** (casas com mais de 4000 sqft vendidas a preços irrisórios por razões externas/familiares). Outliers multivariados explicados pelas demais features foram preservados para manter o sinal real do modelo.
2. **Prevenção de Data Leakage (Vazamento de Dados):** O conjunto de dados foi dividido em Treino e Validação (Holdout) **antes** de qualquer etapa de ajuste fino ou otimização. O ajuste de hiperparâmetros foi realizado exclusivamente na base de treino.
3. **Engenharia de Recursos:** Criação de variáveis agregadas como `TotalSF` (Área Total Computada), `IdadeImovel` no momento da venda e flags lógicas para a presença de acessórios.

---

## 🤖 4. O "Campeonato" de Modelos e Otimização

Avaliamos múltiplos algoritmos utilizando **Validação Cruzada (K-Fold = 5)** com base na métrica **MAE (Erro Médio Absoluto)** para garantir estabilidade:

* KNN Regressor -> MAE: \$27,944.34
* Regressão Linear -> MAE: \$18,400.80
* Random Forest -> MAE: \$16,799.93
* Gradient Boosting -> MAE: \$15,430.78
* **CatBoost Regressor -> MAE: \$13,909.13 (Vencedor)**

### Ajuste Fino de Hiperparâmetros
O CatBoost foi submetido a um **GridSearchCV** para refinar seus pesos, encontrando a combinação ideal: `{'depth': 6, 'iterations': 1000, 'learning_rate': 0.03}`.

---

## 📈 5. Avaliação Final (Dados Inéditos e Isolados)

Após a remoção dos ruídos (outliers) e fixação dos hiperparâmetros, o modelo definitivo foi testado na base de **Validação (Holdout)** — dados completamente inéditos que o algoritmo nunca viu em nenhuma etapa anterior:

* **Erro Médio Absoluto (MAE Real):** \$14,097.30
* **Coeficiente de Determinação ($R^2$ Real):** 0.9297

> **Conclusão:** O modelo demonstra altíssima capacidade de generalização e estabilidade, sendo capaz de **explicar 92.97% da variação dos preços** de novos imóveis no mercado.

---

## 🚀 6. Pipeline de Produção e Deploy Robustos

A interface de usuário foi desenvolvida em **Streamlit** (Python) e implantada no Streamlit Cloud. 

### Mitigação de Dados Fora de Distribuição (Out-of-Distribution)
Para evitar o erro comum de enviar inputs zerados para o modelo em produção (o que destruiria a confiabilidade das previsões em colunas numéricas ocultas ao formulário, como `LotArea` ou `GarageCars`), o pipeline foi blindado:
* Foi exportado um artefato com as **medianas e modas estatísticas** do treino (`valores_padrao.pkl`).
* O aplicativo em produção consome esses dados padrão para preencher automaticamente as variáveis não solicitadas no formulário, garantindo que o input do usuário permaneça sempre dentro da distribuição conhecida pelo CatBoost.

### Como rodar localmente:
1. Clone o repositório.
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute o app: `streamlit run app.py`
