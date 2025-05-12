import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Criar a pasta 'figures' caso não exista
os.makedirs('figures', exist_ok=True)

# Carregar os dados
df = pd.read_csv('data/olist_order_reviews_dataset.csv')

# Exibir primeiras linhas
print("👀 Primeiras linhas do dataset:")
print(df.head())

# Informações básicas
print("\n📊 Informações gerais:")
print(df.info())

# Estatísticas descritivas
print("\n📈 Estatísticas descritivas:")
print(df.describe())

# Verificar valores nulos
print("\n❓ Valores nulos por coluna:")
print(df.isnull().sum())

# Visualizações
sns.set(style="whitegrid")

# 1. Histograma das notas de avaliação
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='review_score', palette='viridis')
plt.title("Distribuição das Notas de Avaliação")
plt.xlabel("Nota")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.savefig('figures/histograma_notas.png')
plt.close()

# 2. Presença de comentário vs. nota média
df['tem_comentario'] = df['review_comment_message'].notnull()

plt.figure(figsize=(6, 4))
sns.barplot(data=df, x='tem_comentario', y='review_score', palette='coolwarm')
plt.title("Nota média: com e sem comentário")
plt.xlabel("Tem Comentário?")
plt.ylabel("Nota média")
plt.xticks([0, 1], ['Não', 'Sim'])
plt.tight_layout()
plt.savefig('figures/media_notas_comentario.png')
plt.close()

# 3. Quantidade de avaliações por data
df['review_creation_date'] = pd.to_datetime(df['review_creation_date'])
avaliacoes_por_data = df.groupby('review_creation_date').size()

plt.figure(figsize=(10, 5))
avaliacoes_por_data.plot()
plt.title("Número de avaliações ao longo do tempo")
plt.xlabel("Data")
plt.ylabel("Quantidade de Avaliações")
plt.tight_layout()
plt.savefig('figures/avaliacoes_por_data.png')
plt.close()
