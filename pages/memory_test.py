import streamlit as st, ollama, json, os, tempfile, re, chromadb, uuid
from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI
from streamlit_pdf_viewer import pdf_viewer

st.set_page_config(
page_title="Monocle Memory Chat",
page_icon="📋",
layout="centered",
initial_sidebar_state="expanded"
)

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
  )

# Define functions to be used in streamlit app.

def local_css(file):
  with open(file) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style.css")

def create_chat_markdown():
  chat_history = ""
  for message in st.session_state.messages:
    role = message['role'].upper()
    content = message["content"]
    chat_history += f"**{role}**: {content}\n\n"
  return chat_history

def export_chat_json():
  if not os.path.exists("chat_history/"):
    os.makedirs("chat_history/")
  chat_data = st.session_state.messages
  if len(chat_data) == 0:
    return
  special_chars = r"[^a-zA-Z0-9\s]"
  title_content = chat_data[0]["content"]
  first_message = re.sub(special_chars,"", title_content) 
  chat_file = f"{first_message}"
  chat_file = os.path.join("chat_history/", f"{chat_file}.json")
  chat_history = st.session_state.messages
  with open(chat_file,"w") as f:
    json.dump(chat_history,f, indent = 2)
  st.success(f"Chat saved as {chat_file}")

def load_chat_history(selected_chat):
    with open(os.path.join("chat_history/", selected_chat), "r") as f:
        chat_history = json.load(f)
        st.session_state.messages = chat_history

def delete_chat_history(selected_chat):
  os.remove(os.path.join("chat_history/", selected_chat))
  
def new_chat():
    st.session_state.messages = []
    st.session_state.vector_store = []

def load_pdf(uploaded_file):
  with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
      temp_file.write(uploaded_file.getbuffer())
      temp_file_path = temp_file.name
  loader = PyMuPDFLoader(temp_file_path)
  chunks = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000, chunk_overlap = 100)
  doc_to_embed = text_splitter.split_documents(chunks)
  return doc_to_embed

def load_spread(uploaded_file):
  with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
      temp_file.write(uploaded_file.getbuffer())
      temp_file_path = temp_file.name
  loader = CSVLoader(temp_file_path)
  doc_to_embed = loader.load()
  return doc_to_embed

def embed_document(doc_to_embed):
  hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': False},
    show_progress = True
  )
  ids = [str(uuid.uuid4()) for _ in doc_to_embed]
  collection_name = "rag_chat"
  vector_store = Chroma.from_documents(
    documents = doc_to_embed, embedding = hf_embeddings, collection_name= collection_name, ids = ids
  )
  return vector_store

# Instantiate LLM and establish parameters through prompt engineering.
def q_a(question, vector_store):
  llm = ChatOllama(model = st.session_state["ollama_model"])
  question_prompt = PromptTemplate(
    input_variables = ["question"], 
    template = """You are a helpful AI assistant and when replying, you need to generate three different
    versions of the question and retrieve the relevant documents from the vector database. This will allow
    you to gather multiple perspectives on the question and aid in your ability to overcome the limitations
    of your inherent training."""
  )
  # Instantiate retriever to submit questions to LLM based on embedded file.
  retriever = MultiQueryRetriever.from_llm(
    vector_store.as_retriever(),
    llm = llm,
    prompt = question_prompt
  )
  # Define template for LLM to follow.
  chat_template = """
  Answer the question using the following context: {context}

  Question: {question}
  Do not make up an answer, only use the information provided from {context}.
  """
  chat_prompt = ChatPromptTemplate.from_template(chat_template)
  # Create chain to pass question through to LLM.
  chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | chat_prompt
    | llm
    | StrOutputParser()
  )
  reply = chain.invoke(question)
  return reply

if "authentication_status" not in st.session_state:
  st.switch_page("llm_login.py")

if "messages" not in st.session_state:
  st.session_state["messages"] = []

def main():

  st.header("💬 Monocle Memory Chat 🗨️")

  available_models = [model["name"] for model in ollama.list()["models"]]
  with st.sidebar:
    selected_model = st.selectbox("Select a model", available_models, key = "selected_model")
    if "ollama_model" not in st.session_state:
      st.session_state["ollama_model"] = "llama3.1"
    else:
      st.session_state["ollama_model"] = selected_model

  with st.sidebar:
    st.button("Start a new chat", on_click = new_chat)

  uploaded_file = st.file_uploader("Upload a file to view.", type = ('pdf', 'csv', 'xlsx', 'xls'))
  
  vector_store = []
  if uploaded_file and uploaded_file.name.endswith(("pdf")):
    documents = load_pdf(uploaded_file)
    vector_store = embed_document(documents)
  if uploaded_file and uploaded_file.name.endswith(("csv", "xlsx", "xls")):
    documents = load_spread(uploaded_file)
    vector_store = embed_document(documents)

  st.session_state.vector_store = vector_store
  
  for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

  if prompt:= st.chat_input("What would you like to ask?"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar = "🧑‍💻"):
      st.markdown(prompt)

    with st.chat_message("assistant", avatar = "🧙‍♂️"):
      with st.spinner("Reading..."):
        if vector_store is not None:
          answer = q_a(prompt, st.session_state.vector_store)
          st.markdown(answer)
        else:
          st.warning("Upload a file")
      st.session_state.messages.append({"role": "assistant", "content": answer})

  with st.sidebar:
    chat_history_markdown = create_chat_markdown()
    chat_data = st.session_state.messages
    if len(chat_data) == 0:
      return
    first_message = chat_data[0]["content"]
    chat_file = f"{first_message}"
    st.download_button(
      label = "Download chat as Markdown",
      data = chat_history_markdown,
      file_name = f"{chat_file}.md",
      mime = "text/markdown"
    )

    st.button("Save chat to history", on_click = export_chat_json)

  # st.write(st.session_state.vector_store)

if __name__ == "__main__":
  main()