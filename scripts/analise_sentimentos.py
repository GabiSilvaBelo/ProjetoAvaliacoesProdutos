# scripts/analise_sentimentos.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Estilo dos gráficos
sns.set(style="whitegrid", palette="pastel")

# Lê os dados com sentimentos
df = pd.read_csv('data/olist_order_reviews_com_sentimento.csv')

# Remove registros sem sentimento (comentários vazios)
df = df.dropna(subset=['sentimento'])

# Converte sentimento para inteiro (estava como float por causa do NaN)
df['sentimento'] = df['sentimento'].astype(int)

# Converte a data para datetime
df['review_creation_date'] = pd.to_datetime(df['review_creation_date'])

# Agrupa por mês
df['mes'] = df['review_creation_date'].dt.to_period('M')

# Conta sentimentos por mês
df_agrupado = df.groupby(['mes', 'sentimento']).size().unstack().fillna(0)
df_agrupado.columns = ['Negativo', 'Positivo']
df_agrupado.index = df_agrupado.index.to_timestamp()

# Plot: Volume de sentimentos por mês
plt.figure(figsize=(12, 6))
df_agrupado.plot(kind='bar', stacked=True, color=['salmon', 'skyblue'])
plt.title('Volume de avaliações positivas e negativas por mês')
plt.xlabel('Mês')
plt.ylabel('Quantidade de avaliações')
plt.tight_layout()
plt.xticks(rotation=45)
plt.savefig('figures/sentimentos_mensal.png')
plt.show()
