import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import nltk
import re
from nltk.corpus import stopwords
import os

# Baixar stopwords
nltk.download('stopwords')
stopwords_pt = set(stopwords.words('portuguese'))

# Criar diretório figures se não existir
os.makedirs('figures', exist_ok=True)

# Carregar o dataset
df = pd.read_csv('data/olist_order_reviews_dataset.csv')

# Remover nulos
df = df[df['review_comment_message'].notnull()]

# Função de limpeza
def limpar_texto(texto):
    texto = texto.lower()  # minúsculas
    texto = re.sub(r'[^\w\s]', '', texto)  # remove pontuação
    texto = re.sub(r'\d+', '', texto)  # remove números
    palavras = texto.split()
    palavras = [p for p in palavras if p not in stopwords_pt]  # remove stopwords
    return ' '.join(palavras)

# Aplicar limpeza
df['comentario_limpo'] = df['review_comment_message'].apply(limpar_texto)

# Juntar tudo em um só texto
texto_unido = ' '.join(df['comentario_limpo'])

# Gerar nuvem de palavras
nuvem = WordCloud(width=800, height=400, background_color='white').generate(texto_unido)

# Mostrar e salvar a nuvem
plt.figure(figsize=(10, 5))
plt.imshow(nuvem, interpolation='bilinear')
plt.axis('off')
plt.title("Nuvem de Palavras dos Comentários")
plt.tight_layout()
plt.savefig('figures/nuvem_palavras.png')
plt.show()

# Análise de sentimentos
df['polaridade'] = df['comentario_limpo'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Exibir estatísticas de sentimento
print("📊 Estatísticas da polaridade (sentimento):")
print(df['polaridade'].describe())

# Histograma da polaridade
plt.figure(figsize=(8, 5))
df['polaridade'].hist(bins=30, color='skyblue')
plt.title("Distribuição da Polaridade dos Comentários")
plt.xlabel("Polaridade (Sentimento)")
plt.ylabel("Quantidade de Comentários")
plt.tight_layout()
plt.savefig('figures/polaridade_sentimentos.png')
plt.show()
