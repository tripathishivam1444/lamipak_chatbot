from langchain.prompts import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores.chroma import  Chroma 
import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
import time 
from langchain_community.vectorstores import Qdrant
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA

from langchain_openai import OpenAIEmbeddings
import qdrant_client
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Qdrant
from langchain_community.llms import OpenAI

load_dotenv()


import os 



#------------------------------------------------------------------------
"""LOad All the Pdf Data """

def pdf_to_text(pdf_files):
    all_pdf_text_list = []
    source_list = []
    for file in pdf_files:
        pdf = PdfReader(file)
        for page_number in range(len(pdf.pages)):
            page = pdf.pages[page_number]
            text = page.extract_text()
            all_pdf_text_list.append(text)
            source_list.append(file.name + "_page_" + str(page_number))
            
    text_splitter = RecursiveCharacterTextSplitter( chunk_size = 1000, chunk_overlap = 20, length_function = len,)
    
    # vector_db = Chroma.from_texts(all_pdf_text_list, OpenAIEmbeddings())
    pdf_docs = text_splitter.create_documents(all_pdf_text_list, metadatas = [{"sounrce" : s} for s in source_list])
    
    st.write( "pdf_docs -------> ",pdf_docs)
    
    url="https://f881097e-33f9-43d6-a53f-c9d330c43384.europe-west3-0.gcp.cloud.qdrant.io:6333"
    api_key="iYrvlwR9ksNus0h7O7-IUSyxwgJTCcGoZjgmb5pO0JkNkbRgspMQJg"
   
       
    qdrant = Qdrant.from_documents(pdf_docs,
                                   OpenAIEmbeddings(), 
                                   url=url,
                                   prefer_grpc=True,
                                   api_key=api_key,
                                   collection_name="Lamipak_chatbot")
    
    # pdf_chroma_db = Chroma.from_documents(pdf_docs, embedding=embeddings, persist_directory= "Vector_DB/")
    # pdf_chroma_db.persist()
    return qdrant
    

# #-----------------------------------------------------------------------------------------------------
# """It Loads all the url  Data"""

def web_data_loader(urls):
    try:
        loader = SeleniumURLLoader(urls=urls)
        web_text = loader.load()
    except:
        st.write("SeleniumURLLoader Error Occurred ")
        print("\n\n ----------->> SeleniumURLLoader Error Occurred <<------------ \n\n ")
        try:
            web_data = WebBaseLoader(urls)
            web_text = web_data.load()
        except:
            st.write("WebBaseLoader Error Occurred ")
            print("\n\n --------->> WebBaseLoader Error Occurred <<--------- \n\n ")
            web_data = UnstructuredURLLoader(urls)
            web_text = web_data.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20, length_function=len)

    web_docs = text_splitter.split_documents(web_text)
    st.write("web_docs -------------> ", web_docs)

    # Assuming doc_content is the actual content of the document
    embeddings = OpenAIEmbeddings()

    # Creating a dummy class with page_content and metadata attributes
    class Document:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata 

    # Iterate over each document and create Chroma database
    
    st.write(f"\n\n total ---> {len(web_docs)} documents are these \n\n ")
    st.write("\n\n Document Saving in DB ......\n\n " )
    for i, doc_content in enumerate(web_docs):
        # Creating a Document instance for each document
        document = Document(doc_content.page_content, doc_content.metadata)
        print(document)

        # Creating Chroma database from a single document
        #web_chroma_db = Chroma.from_documents([document], embedding=embeddings, persist_directory="Vector_DB/")
        
        
        url="https://2443ceb5-760b-48d0-a96c-9c9a65b2b8f1.us-east4-0.gcp.cloud.qdrant.io:6333"
        api_key="HX0t_FWV5dP4HH9PtI755AnehSgRmIQyLaqoQ8sCxZKJw33eUV5hWQ"
   
   
        qdrant = Qdrant.from_documents([document],
                               OpenAIEmbeddings(),
                               url=url,
                               prefer_grpc=True,
                               api_key=api_key,
                               collection_name="Lamipak_chatbot")
        
        st.write( f"{i}/{len(web_docs)}" )
        time.sleep(25)
    return qdrant





url="https://2443ceb5-760b-48d0-a96c-9c9a65b2b8f1.us-east4-0.gcp.cloud.qdrant.io:6333"
api_key="HX0t_FWV5dP4HH9PtI755AnehSgRmIQyLaqoQ8sCxZKJw33eUV5hWQ"
    

client = qdrant_client.QdrantClient(url=url, prefer_grpc=True, api_key=api_key)
    
vectorstore = Qdrant(client=client,
                     collection_name= 'Lamipak_chatbot',
                     embeddings= OpenAIEmbeddings())



def genrating_answer_from_db(query, vectorstore= vectorstore):
    # chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo'),
    #                                           retriever=vectorstore.as_retriever(),
    #                                           )
    

    qa = RetrievalQA.from_chain_type(llm=OpenAI(),
                                chain_type="stuff",
                                retriever=vectorstore.as_retriever() )

    answer = qa.run(query)
    
    return answer







# print("\n\n qa ----->>  ", qa)
# print(f"\n\n\n ------> {qa.run("what milk production in india?")}")
    


    
