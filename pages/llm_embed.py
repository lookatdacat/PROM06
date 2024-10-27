import streamlit as st, chromadb, tempfile
from time import sleep
from langchain.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

st.set_page_config(
  page_title="File Upload and Embedding",
  page_icon="📃",
  layout="centered",
  initial_sidebar_state="expanded"
)

# Define function for writing files from memory buffer to disk and then loading through loader of choice.
def load_pdf(file):
  with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
      temp_file.write(file.getbuffer())
      temp_file_path = temp_file.name
  loader = PyMuPDFLoader(temp_file_path)
  documents = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000, chunk_overlap = 100)
  documents = text_splitter.split_documents(documents)
  file_content = "\n".join([doc.page_content for doc in documents])
  return file_content

# Define HuggingFace embeddings as a variable
hf_embeddings = HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-mpnet-base-v2",
  model_kwargs={'device': 'cuda'},
  encode_kwargs={'normalize_embeddings': False},
  show_progress = True
)
ollama_embeddings = OllamaEmbeddings(model = "llama3.1", num_gpu = 1)

# Define function for viewing available collections.
def view_collections():
  client = chromadb.PersistentClient(path="./chromadb_data")
  all_collections = client.list_collections()
  collection_names = [collection.name for collection in all_collections]
  if not collection_names:
      st.write("No collections found.")
  else:
      st.write("Available collections:")
      for name in collection_names:
          st.write(f"- {name}")

def delete_collection(all_collections, collection_name):
  if all_collections:
    for collection in collection_name:
      client = chromadb.PersistentClient(path="./chromadb_data")
      client.delete_collection(collection)
      st.success(f"{collection} successfully deleted")
      sleep(1)
  else:
    st.info("No collections to delete.")

def main():

  if "authentication_status" not in st.session_state:
    st.switch_page("llm_login.py")

  st.header("📃 Document Management 📃", divider="orange", anchor=False)

  uploaded_files = st.file_uploader("Upload your files", accept_multiple_files=True)
  collection_name = st.text_input("Enter name of document collection", placeholder="data_embeddings")
  
  if uploaded_files and st.button("Upload and Embed"):
    chroma_client = chromadb.PersistentClient(path="chromadb_data")
    collection = chroma_client.get_or_create_collection(name=collection_name)

    for uploaded_file in uploaded_files:
      file_content = load_pdf(uploaded_file)
      file_name = uploaded_file.name
  # Use get_or_create_collection to handle the existence check and creation in one step
      collection.add(
        documents=[file_content],
        metadatas=[{"file_name": file_name}],
        ids=[file_name]
      )
      st.success("Files have been uploaded and embedded successfully.")

  if st.button("View Available Collections"):
    view_collections()

  client = chromadb.PersistentClient(path="./chromadb_data")
  all_collections = client.list_collections()
  collection_names = [collection.name for collection in all_collections]
  collection_name = st.multiselect("Select the collections you want to delete", collection_names)

  delete_collections = st.button("🗑️ **Delete Collections**")

  if delete_collections:
    delete_collection(all_collections, collection_name)

if __name__ == "__main__":
  main()