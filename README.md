# 🏠 Precificação Inteligente de Imóveis com Machine Learning

[![Streamlit App](https://static.streamlit.io/badge_svg.svg)](INSIRA_AQUI_O_LINK_DO_SEU_STREAMLIT)

Este projeto apresenta o desenvolvimento e o deploy de um modelo preditivo robusto para estimar o preço de venda de imóveis. O objetivo principal é transformar dados brutos em uma ferramenta de suporte à decisão de negócios, otimizando margens imobiliárias e reduzindo o tempo de avaliação manual de propriedades.

A aplicação final foi implementada em produção e pode ser testada interativamente através do link acima.
 
<<<<<<< HEAD
# 🏠 Precificação Inteligente de Imóveis com Machine Learning

[![Streamlit App](https://static.streamlit.io/badge_svg.svg)](INSIRA_AQUI_O_LINK_DO_SEU_STREAMLIT)

Este projeto apresenta o desenvolvimento e o deploy de um modelo preditivo robusto para estimar o preço de venda de imóveis. O objetivo principal é transformar dados brutos em uma ferramenta de suporte à decisão de negócios, otimizando margens imobiliárias e reduzindo o tempo de avaliação manual de propriedades.

A aplicação final foi implementada em produção e pode ser testada interativamente através do link acima.

---
=======

>>>>>>> 1549e102f366ed7780e4b78506ddf158fb44696c

## 🏢 1. O Problema de Negócio

No mercado imobiliário competitivo, a avaliação manual de imóveis por corretores gera gargalos operacionais e financeiros:
* **Subprecificação:** Reduz a margem de comissão da empresa e frustra o proprietário.
* **Superprecificação:** Deixa o imóvel "encalhado", gerando custos de anúncio sem conversão.
* **Lentidão:** Processos manuais demoram dias para ponderar dezenas de variáveis estruturais e regionais.

**Solução:** Um algoritmo de regressão capaz de prever o valor justo de mercado (`SalePrice`) em segundos, servindo como um balizador de preços para a operação.

---

## 🔍 2. Principais Insights da Análise Exploratória (EDA)

* **Relação Área vs. Preço:** Existe uma tendência linear clara de valorização proporcional à área construída acima do solo (`GrLivArea`).
* **Teto de Valorização por Localização:** Bairros como `NoRidge`, `NridgHt` e `StoneBr` dominam o topo de preços. Insights multivariados mostraram que imóveis grandes em bairros desvalorizados sofrem forte penalização de preço, provando que a localização atua como um moderador crítico do preço por m².
* **Acessórios de Margem:** Adicionais como **Piscina, Lareira e Porão** estão correlacionados com saltos expressivos no preço médio dos imóveis.

---

## 🛠️ 3. Engenharia de Features e Rigor Técnico

Para garantir a robustez estatística e simular um ambiente de produção real, o projeto adotou práticas rigorosas:

1. **Tratamento de Outliers Cirúrgico:** Em vez de remover dados cegamente, seguiu-se a recomendação do autor original do dataset (Dean De Cock) para remover estritamente **2 pontos atípicos** (casas com mais de 4000 sqft vendidas a preços irrisórios por razões externas/familiares). Outliers multivariados explicados pelas demais features foram preservados para não eliminar sinal real de mercado.
2. **Prevenção de Data Leakage (Vazamento de Dados):** O conjunto de dados foi dividido em Treino e Validação (Holdout) **antes** de qualquer ajuste ou otimização. O ajuste fino foi realizado exclusivamente na base de treino.
3. **Engenharia de Recursos:** Criação de variáveis agregadas como `TotalSF` (Área Total Computada), `IdadeImovel` no momento da venda e flags lógicas para a presença de acessórios.

---

## 🤖 4. O "Campeonato" de Modelos e Ajuste Fino

Avaliamos múltiplos algoritmos utilizando **Validação Cruzada (K-Fold = 5)** com base na métrica **MAE (Erro Médio Absoluto)** para garantir estabilidade:

* Regressão Linear -> MAE: \$19,798.85
* KNN Regressor -> MAE: \$27,996.68
* Random Forest -> MAE: \$17,512.30
* Gradient Boosting -> MAE: \$16,325.62
* **CatBoost Regressor -> MAE: \$14,583.98 (Vencedor)**

### Otimização de Hiperparâmetros
O CatBoost foi submetido a um **GridSearchCV**, encontrando a combinação ideal (`depth: 6`, `iterations: 1000`, `learning_rate: 0.03`).

---

## 📈 5. Avaliação Final (Dados Inéditos)

O modelo final foi testado na base de **Validação (Holdout)** — dados que o algoritmo nunca viu em nenhuma etapa de treino ou ajuste:

* **MAE Real:** \$15,504.87
* **Coeficiente de Determinação ($R^2$):** 0.9094

> **Conclusão:** O modelo final demonstra alta capacidade de generalização, sendo capaz de **explicar 90.94% da variação dos preços** de imóveis completamente novos.

---

## 🚀 6. Deploy e Produção

A interface de usuário foi desenvolvida em **Streamlit** (Python puro) e implantada no Streamlit Cloud. 

Para mitigar a fragilidade do pipeline em produção, o script mapeia os inputs do usuário diretamente para a estrutura de colunas original exportada do modelo (`colunas_modelo.pkl`), prevenindo bugs silenciosos causados por One-Hot Encoding ou categorias ausentes no momento do input.

### Como rodar localmente:
1. Clone o repositório.
2. Instale as dependências: `pip install -r requisitos.txt`
<<<<<<< HEAD
3. Execute o app: `streamlit run app.py`
=======
3. Execute o app: `streamlit run app.py`
>>>>>>> 1549e102f366ed7780e4b78506ddf158fb44696c
