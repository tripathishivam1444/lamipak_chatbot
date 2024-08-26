import os 
import time
import qdrant_client
import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import OpenAI
from langchain_qdrant import Qdrant
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import WebBaseLoader
# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader


import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Access the environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
QDERANT_URL = os.getenv('QDERANT_URL')
QDRANT_API_API = os.getenv('QDRANT_API_API')

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
            
    text_splitter = RecursiveCharacterTextSplitter( chunk_size = 1000, chunk_overlap = 100, length_function = len,)
    
    # vector_db = Chroma.from_texts(all_pdf_text_list, OpenAIEmbeddings())
    pdf_docs = text_splitter.create_documents(all_pdf_text_list, metadatas = [{"sounrce" : s} for s in source_list])
    
    st.write( "pdf_docs -------> ",pdf_docs)
    

   
       
    qdrant = Qdrant.from_documents(pdf_docs,
                                   OpenAIEmbeddings(), 
                                   url=QDERANT_URL,
                                   prefer_grpc=True,
                                   api_key=QDRANT_API_API,
                                   collection_name="Lamipak_chatbot")
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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)

    web_docs = text_splitter.split_documents(web_text)
    st.write("web_docs -------------> ", web_docs)

    class Document:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata 

    st.write(f"\n\n total ---> {len(web_docs)} documents are these \n\n ")
    st.write("\n\n Document Saving in DB ......\n\n " )

    for i, doc_content in enumerate(web_docs):
        document = Document(doc_content.page_content, doc_content.metadata)
        print(document)
        
        qdrant = Qdrant.from_documents([document],
                               OpenAIEmbeddings(),
                               url=QDERANT_URL,
                               prefer_grpc=True,
                               api_key=QDRANT_API_API,
                               collection_name="Lamipak_chatbot")
        
        st.write( f"{i+1}/{len(web_docs)}" )
        time.sleep(25)
    return qdrant


client = qdrant_client.QdrantClient(url=QDERANT_URL, prefer_grpc=True, api_key=QDRANT_API_API)
    
vectorstore = Qdrant(client=client,
                     collection_name= 'Lamipak_chatbot',
                     embeddings= OpenAIEmbeddings(),
                     )


def genrating_answer_from_db(query, vectorstore= vectorstore):
    qa = RetrievalQA.from_chain_type(llm=OpenAI(),
                                chain_type="stuff",
                                retriever=vectorstore.as_retriever() )

    answer = qa.run(query)
    
    return answer
