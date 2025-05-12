import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import os

# Criar pasta models se não existir
os.makedirs('models', exist_ok=True)

# Carregar os dados
df = pd.read_csv('data/olist_order_reviews_dataset.csv')

# Criar coluna de sentimento
df = df[df['review_comment_message'].notnull()]  # remove sem texto
df['sentimento'] = df['review_score'].apply(lambda x: 1 if x >= 4 else 0)

# Separar variáveis
X = df['review_comment_message']
y = df['sentimento']

# Separar em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Criar pipeline de TF-IDF + Modelo
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=None)),
    ('modelo', LogisticRegression(max_iter=1000))
])

# Treinar o modelo
pipeline.fit(X_train, y_train)

# Avaliar
y_pred = pipeline.predict(X_test)

print("📊 Relatório de Classificação:\n")
print(classification_report(y_test, y_pred))

print("🧩 Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# Salvar modelo
joblib.dump(pipeline, 'models/modelo_sentimento.joblib')
print("\n✅ Modelo salvo em: models/modelo_sentimento.joblib")
