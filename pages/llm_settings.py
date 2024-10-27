import streamlit as st
import ollama
from time import sleep

st.set_page_config(
    page_title="Model management",
    page_icon="⚙️",
    layout="centered",
    initial_sidebar_state="expanded",
)

def main():

  if "authentication_status" not in st.session_state:
    st.switch_page("llm_login.py")

  st.header("⚙️ Model Management ⚙️", divider="orange", anchor=False)

  available_models = [model["name"] for model in ollama.list()["models"]]
  st.selectbox("View available models", available_models, key = "selected_model")

  # Download models
  st.subheader("Download Models", anchor=False)

  model_name = st.text_input(
    "Enter the name of the model you want to download. Do not close this window until the download is complete", placeholder="llama3.2"
    )

  if st.button(f"⬇️ **Download** :green[{model_name}]"):
    if model_name:
      ollama.pull(model_name)
      st.success(f"Downloaded model: {model_name}")
      st.toast('The model has been downloaded!', icon='😁')
      sleep(2)
      st.rerun()
    else:
      st.warning("Please enter a model name.", icon="⚠️")

  url_ollama_models = "https://ollama.com/library"
  st.markdown("[Available models can be found at Ollama.com/library.](%s)" % url_ollama_models)

  st.divider()

  # Create models
  st.subheader("Create model", anchor=False)

  modelfile=st.text_area(
        "Enter the modelfile",
        height=100,
        placeholder="FROM llama3.1\nSYSTEM You are Bowser from Super Mario Bros.",
      )

  model_name = st.text_input(
    "Enter the name of the model you want to create", placeholder="Bowser"
    )

  def new_model():
    if model_name and modelfile:
      ollama.create(model=model_name, modelfile=modelfile)
      st.success(f"Model created!")
      st.toast('The model has been created!', icon='😁')

  st.button("🧬 **Create Model**", on_click=new_model)
  url_ollama_parameters = "https://github.com/ollama/ollama/blob/main/docs/modelfile.md"
  st.markdown("[More details on available parameters can be found here.](%s)" % url_ollama_parameters)

  st.divider()

  # Delete models

  st.subheader("Delete Models", anchor=False)

  models_info = ollama.list()
  available_models = [m["name"] for m in models_info["models"]]
  selected_models = st.multiselect("Select the models you want to delete", available_models)

  def delete_models():
    if available_models:
      for model in selected_models:
        ollama.delete(model)
        st.success(f"{model} successfully deleted")
    else:
      st.info("No models to delete, pull a new one")

  st.button("🗑️ **Delete**", on_click=delete_models)

if __name__ == "__main__":
  main()