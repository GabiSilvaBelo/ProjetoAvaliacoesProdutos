# 🛍️ Análise de Sentimentos de Avaliações de Produtos

Este projeto tem como objetivo analisar os sentimentos presentes nas avaliações dos clientes de uma loja online utilizando **Processamento de Linguagem Natural (PLN)** e **Machine Learning**.

---

## 📌 Objetivos

- Limpar e pré-processar os dados textuais das avaliações.
- Treinar um modelo de classificação para detectar sentimentos (positivo ou negativo).
- Salvar o modelo para uso futuro.
- Adicionar os sentimentos classificados ao conjunto de dados original.

---

## 🧠 Tecnologias e Bibliotecas

- Python 3.10+
- pandas
- scikit-learn
- joblib
- re (expressões regulares)
- matplotlib (em etapas futuras)

---

## ⚙️ Como Executar o Projeto

1. **Clone o repositório:**

```bash
git clone https://github.com/seu-usuario/seu-repo.git
cd ProjetoAvaliacoesProdutos


2. **Crie e ative um ambiente virtual**

python3 -m venv venv
source venv/bin/activate  # macOS/Linux

3. **Instale as dependências**

pip install -r requirements.txt

4. **Execute os scripts principais**

Para treinar o modelo:
python3 scripts/modelo_sentimento.py

Para aplicar o modelo nos dados:
python3 scripts/aplica_modelo.py

📊 Resultados

O modelo apresenta uma acurácia de 90% na classificação de sentimentos com base nas avaliações. Isso demonstra sua boa capacidade de generalização e identificação de padrões positivos e negativos.

✍️ Autora

Gabriela Belo da Silva
Cientista de Dados em formação | Estagiária em tecnologia | Criadora do canal Ampulhetta

📌 Observação

Este projeto faz parte da trilha de estudos em Ciência de Dados, abordando as etapas de Análise Exploratória, NLP, e Machine Learning.
