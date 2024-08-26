import re
import streamlit as st
from streamlit_chat import message
from utils import pdf_to_text, web_data_loader, genrating_answer_from_db

st.markdown("<h1 style='color: maroon;'>TRIPATHI UTKARSH</h1>", unsafe_allow_html=True)

if "user_input" not in st.session_state:
    st.session_state['user_input'] = []
    
if "AI_response" not in st.session_state:
    st.session_state['AI_response'] = []

st.sidebar.header("Upload Data")


pdf_files = st.sidebar.file_uploader("Upload PDFs", type=['pdf'], accept_multiple_files=True)
url_container = st.sidebar.text_area("Paste your link(s)")

upload_button = st.sidebar.button("Upload Data")

if upload_button:
    visited_urls = set()
    all_urls = []


    url_pattern = re.compile(r'https?://\S+')
    url_list = url_pattern.findall(url_container)


    if pdf_files or url_list:
        if pdf_files:
            st.sidebar.write("Extracting text from PDF files...")
            all_pdf_docs = pdf_to_text(pdf_files)
            st.sidebar.success("PDF data saved into Chroma Database", icon="✅")
        
        if url_list:
            st.sidebar.write("Extracting text from web data...")
            all_web_docs = web_data_loader(url_list)
            st.sidebar.success("Web data saved into Chroma Database", icon="✅")
    else:
        st.sidebar.write("Please add at least a URL or PDF")


input_query = st.chat_input("Input your query")

if input_query:
    answer = genrating_answer_from_db(input_query)
    st.session_state.user_input.append(input_query)
    st.session_state.AI_response.append(answer)


message_history = st.empty()

if message_history:
    for i in range(len(st.session_state['user_input'])):

        message(st.session_state['user_input'][i], avatar_style="avataaars", is_user=True, key=f"{i}_user")
        message(st.session_state['AI_response'][i], key=f"{i}_ai", avatar_style="bottts")
