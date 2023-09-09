import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question, conversation_chain, pdf_title):
    response = conversation_chain({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    st.write(f"Response for PDF '{pdf_title}':")
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    return response['chat_history'][-1].content  # Return the last response content

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation1" not in st.session_state:
        st.session_state.conversation1 = None
    if "conversation2" not in st.session_state:
        st.session_state.conversation2 = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")

    # Add two file upload buttons for PDFs
    uploaded_pdf1 = st.file_uploader("Upload the first PDF:", type=["pdf"])
    uploaded_pdf2 = st.file_uploader("Upload the second PDF:", type=["pdf"])

    if "conversation1" not in st.session_state or "conversation2" not in st.session_state:
        st.warning("Please upload PDFs to start the conversation.")

    if uploaded_pdf1 and uploaded_pdf2:
        with st.spinner("Processing"):
            # Get text from the first PDF
            raw_text1 = get_pdf_text([uploaded_pdf1])

            # Get text from the second PDF
            raw_text2 = get_pdf_text([uploaded_pdf2])

            # Split text into chunks
            text_chunks1 = get_text_chunks(raw_text1)
            text_chunks2 = get_text_chunks(raw_text2)

            # Create vector stores for both PDFs
            vectorstore1 = get_vectorstore(text_chunks1)
            vectorstore2 = get_vectorstore(text_chunks2)

            # Create conversation chains for both PDFs
            st.session_state.conversation1 = get_conversation_chain(vectorstore1)
            st.session_state.conversation2 = get_conversation_chain(vectorstore2)

    # Example usage:
    user_question = st.text_input("Ask a question about the PDFs:")
    
    if user_question:
        if st.session_state.conversation1 and st.session_state.conversation2:
            response1 = handle_userinput(user_question, st.session_state.conversation1, "PDF 1")
            response2 = handle_userinput(user_question, st.session_state.conversation2, "PDF 2")

            # Construct a new question using the original question and two answers
            new_question = f"{user_question}. Processor 1 data:{response1} Processor 2 data: {response2}"

            # Get an answer to the new question
            difference_answer = handle_userinput(new_question, st.session_state.conversation1, "Difference")

            # Display the answer to the new question
            st.write("Answer to the new question:")
            st.write(difference_answer)

if __name__ == '__main__':
    main()