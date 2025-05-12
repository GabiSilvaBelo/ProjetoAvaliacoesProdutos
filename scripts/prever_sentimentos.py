import pandas as pd
import joblib
import os

# Caminhos
CAMINHO_ARQUIVO = 'data/olist_order_reviews_dataset.csv'
CAMINHO_MODELO = 'models/modelo_sentimento.joblib'
CAMINHO_SAIDA = 'data/olist_order_reviews_com_sentimento.csv'

# Verificações
if not os.path.exists(CAMINHO_ARQUIVO):
    raise FileNotFoundError(f"Arquivo CSV não encontrado: {CAMINHO_ARQUIVO}")

if not os.path.exists(CAMINHO_MODELO):
    raise FileNotFoundError(f"Modelo de sentimento não encontrado: {CAMINHO_MODELO}")

# Carregar os dados
df = pd.read_csv(CAMINHO_ARQUIVO)

# Verifica se a coluna de mensagem existe
if 'review_comment_message' not in df.columns:
    raise ValueError("A coluna 'review_comment_message' não foi encontrada no arquivo CSV.")

# Carregar modelo treinado
modelo = joblib.load(CAMINHO_MODELO)

# Aplicar o modelo apenas nas linhas com texto não nulo
df['sentimento'] = None
mascaras_validas = df['review_comment_message'].notnull()

# Previsão
df.loc[mascaras_validas, 'sentimento'] = modelo.predict(df.loc[mascaras_validas, 'review_comment_message'])

# Converter valores para inteiros (0 = negativo, 1 = positivo)
df['sentimento'] = df['sentimento'].astype('Int64')  # aceita valores nulos com pandas >=1.0

# Salvar novo CSV com a coluna de sentimento
df.to_csv(CAMINHO_SAIDA, index=False)

print(f"✅ Arquivo com sentimentos salvo em: {CAMINHO_SAIDA}")
