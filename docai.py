import streamlit as st
import pdfplumber
import openai
import numpy as np
import os

# Configuração da API do OpenAI
openai.api_key = os.getenv('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')  # Use variáveis de ambiente para a chave API

# Função para extrair texto de arquivos PDF
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Erro ao ler o arquivo {file_path}: {e}")
    return text

# Função para gerar embeddings usando OpenAI
def generate_embeddings(texts):
    embeddings = []
    try:
        for text in texts:
            embedding = openai.Embedding.create(input=text, model="text-embedding-ada-002")['data'][0]['embedding']
            embeddings.append(embedding)
    except Exception as e:
        st.error(f"Erro ao gerar embeddings: {e}")
    return embeddings

# Função para buscar respostas usando o modelo de linguagem
def get_answer(question, texts, embeddings):
    try:
        question_embedding = openai.Embedding.create(input=question, model="text-embedding-ada-002")['data'][0]['embedding']
        similarities = [np.dot(question_embedding, embedding) for embedding in embeddings]
        most_similar_index = np.argmax(similarities)
        context = texts[most_similar_index]
        prompt = f"Com base no seguinte trecho da norma técnica, responda à pergunta:\n\n{context}\n\nPergunta: {question}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Erro ao buscar resposta: {e}")
        return ""

# Interface do Streamlit
st.title("🔍 Consulta de Normas Técnicas")

# Diretório onde os PDFs estão armazenados
pdf_directory = 'normas'

# Verifica se o diretório existe
if not os.path.exists(pdf_directory):
    st.error(f"O diretório '{pdf_directory}' não existe.")
else:
    # Lista todos os arquivos PDF no diretório
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

    if not pdf_files:
        st.warning(f"Não há arquivos PDF no diretório '{pdf_directory}'.")
    else:
        # Extrai texto de todos os PDFs
        texts = [extract_text_from_pdf(os.path.join(pdf_directory, file)) for file in pdf_files]
        embeddings = generate_embeddings(texts)

        question = st.text_input("❓ Faça sua pergunta sobre as normas técnicas:")

        if question:
            with st.spinner("Buscando resposta..."):
                answer = get_answer(question, texts, embeddings)
                st.success("Resposta:")
                st.write(answer)
