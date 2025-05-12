import joblib

# Caminho para o modelo salvo
modelo_path = 'models/modelo_sentimento.joblib'

# Carregar o modelo treinado
modelo = joblib.load(modelo_path)

print("🤖 Modelo carregado com sucesso!")

while True:
    texto = input("\nDigite uma avaliação (ou 'sair' para encerrar): ")
    if texto.lower() == 'sair':
        print("Encerrando...")
        break

    # Fazer a predição
    predicao = modelo.predict([texto])[0]
    sentimento = 'Positivo 😊' if predicao == 1 else 'Negativo 😠'

    print(f"🔍 Sentimento previsto: {sentimento}")
