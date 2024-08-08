import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import json
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence


OPENAI_API_KEY = os.environ['SUA CHAVE DA OPENAI']

llm = ChatOpenAI(model="gpt-3.5-turbo")

def agentePesquisa(query, llm):
    ferramentas = load_tools(["ddg-search", "wikipedia"], llm=llm)
    prompt = hub.pull("hwchase17/react")
    agente = create_react_agent(llm, ferramentas, prompt)
    executor_agente = AgentExecutor(agent=agente, tools=ferramentas, prompt=prompt)
    contextoWeb = executor_agente.invoke({"input": query})
    return contextoWeb['output']

def carregarDados(url):
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("postcontentwrap", "pagetitleloading background-imaged loading-dark"))),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    return retriever

def obterDocsRelevantes(query, url):
    retriever = carregarDados(url)
    documentos_relevantes = retriever.invoke(query)
    return documentos_relevantes

def agenteSupervisor(query, llm, contextoWeb, documentos_relevantes):
    prompt_template = """
    Você é um gerente de uma agência de viagens. Sua resposta final deverá ser um roteiro de viagem completo e detalhado. 
    Utilize o contexto de eventos e preços de passagens, o input do usuário e também os documentos relevantes para elaborar o roteiro.
    Contexto: {contextoWeb}
    Documento relevante: {documentos_relevantes}
    Usuário: {query}
    Assistente:
    """

    prompt = PromptTemplate(
        input_variables=['contextoWeb', 'documentos_relevantes', 'query'],
        template=prompt_template
    )

    sequence = RunnableSequence(prompt | llm)
    resposta = sequence.invoke({"contextoWeb": contextoWeb, "documentos_relevantes": documentos_relevantes, "query": query})
    return resposta

def obterResposta(query, llm, url):
    contextoWeb = agentePesquisa(query, llm)
    documentos_relevantes = obterDocsRelevantes(query, url)
    resposta = agenteSupervisor(query, llm, contextoWeb, documentos_relevantes)
    return resposta

import streamlit as st

# Visual no  Streamlit
st.title("Assistente de Agência de Viagens")

query = st.text_area("Digite sua consulta de viagem:")
nome_pais = st.text_input("Digite o nome do país:").lower()

if st.button("Obter Plano de Viagem"):
    if query and nome_pais:
        url_pais = f"https://www.dicasdeviagem.com/{nome_pais}/"
        resposta = obterResposta(query, llm, url_pais)
        st.subheader("Plano de Viagem")
        st.write(resposta.content)
    else:
        st.error("Por favor, digite uma consulta de viagem e o nome do país.")