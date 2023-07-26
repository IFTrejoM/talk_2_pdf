import streamlit as st
# from dotenv import load_dotenv # conda install -c "conda-forge/label/cf201901" python-dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from InstructorEmbedding import INSTRUCTOR

def obtener_texto_pdf(docs_pdf):
    """Función que extrae el texto del (los) PDF"""
    
    texto = ""
    
    for pdf in docs_pdf:
        lector_pdf = PdfReader(pdf)
        
        for page in lector_pdf.pages:
            texto += page.extract_text()
    
    return texto 

def fraccionar_texto(texto):
    """"Función que fracciona el texto"""
    
    divisor_de_texto = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    fracciones_de_texto = divisor_de_texto.split_text(texto)
    
    return fracciones_de_texto

def crear_vectorstore(fracciones_de_texto):
    """Función que crea almacén de vectores con FAISS"""
    
    vectores_de_palabras = OpenAIEmbeddings()
    # vectores_de_palabras = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl') # Alternativa gratuita
    # vectores_de_palabras = INSTRUCTOR('hkunlp/instructor-xl')
    
    vectorstore = FAISS.from_texts(texts=fracciones_de_texto, embedding=vectores_de_palabras)
    
    return vectorstore

def crear_cadena_de_conversación(vectorstore):
#    llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", 
                        model_kwargs={"temperature":0.5,
                                      "max_length":1024} # 512
                        ) # Alternativa gratuita

    memoria = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
        )
    cadena_de_conversación = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memoria
        )
    return cadena_de_conversación

def procesar_pregunta(pregunta_de_usuario):
    """Procesa la pregunta del usuario y devuelve respuesta"""
    
    respuesta = st.session_state.conversation({'question': pregunta_de_usuario})
    
    st.session_state.chat_history = respuesta['chat_history']
    
    for i, mensaje in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace(
                    "{{MSG}}",
                    mensaje.content
                    ),
                unsafe_allow_html=True
                )
        else:
            st.write(
                bot_template.replace(
                    "{{MSG}}",
                    mensaje.content
                    ),
                unsafe_allow_html=True
                )

def main():
    
    # load_dotenv()
    
    ## ENTORNO PRINCIPAL ##
    
    # Configuración de página:
    st.set_page_config(
        page_title="Talk to PDF",
        page_icon=":books:"
        )
    st.write(css, unsafe_allow_html=True)
    
    # Crea atributos "conversation" y "chat_history" en la sesión, si estos aún no existen:
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    # Título de la página:
    st.header("Chatea con tus PDF :books:")
    
    # Preguntas del usuario:
    pregunta_de_usuario = st.text_input("Haz una pregunta sobre tus PDF:")
    
    if pregunta_de_usuario:
        procesar_pregunta(pregunta_de_usuario)
    
    # Barra lateral:
    with st.sidebar:
        st.subheader("Documentos")
        
        # Espacio de carga de documentos:
        docs_pdf = st.file_uploader(
            "Carga tus PDF aquí", 
            accept_multiple_files=True
            )
        # Acciones cuando el usuario presiona el botón "Procesar":
        if st.button("Procesar"): 
            
            # Agrega un "spinner":
            with st.spinner("Un momento..."):
                
                # Extrae el texto del (los) PDF:
                texto_bruto = obtener_texto_pdf(docs_pdf)
                
                # Fracciona el (los) PDF:
                fracciones_de_texto = fraccionar_texto(texto_bruto)
                # st.write(fracciones_de_texto)
                
                # Crea el almacén de vectores:
                vectorstore = crear_vectorstore(fracciones_de_texto)
                
                # Crea cadena de conversación:
                st.session_state.conversation = crear_cadena_de_conversación(vectorstore)
                
            # Añade un mensaje de éxito si el archivo PDF fue procesado correctamente:
            st.success("¡PDF procesado correctamente!")

if __name__ == "__main__":
    main()

# Implemented from:
# https://github.com/alejandro-ao/ask-multiple-pdfs
# Thank you a lot!