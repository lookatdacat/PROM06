import streamlit as st, ollama, json, os, re
from openai import OpenAI

st.set_page_config(
  page_title="Monocle Chat",
  page_icon="🍻",
  layout="centered",
  initial_sidebar_state="expanded"
  )

client = OpenAI(
  base_url="http://localhost:11434/v1",
  api_key="monocle",
)

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
  st.rerun()
  
def new_chat():
    st.session_state.messages = []

def main():

  st.header("💬 Monocle Chat 🗨️", divider="orange", anchor=False)
  
  if "authentication_status" not in st.session_state:
    st.switch_page("llm_login.py")

  # Instantiate Session State key for "messages". This is used to store the chat history between runs of the app.
  if "messages" not in st.session_state:
    st.session_state.messages = []

  if "embedded_chat" not in st.session_state:
    st.session_state.embedded_chat = []

  # Instantiate Session State key for "ollama_model". The two different Session States statements allow the model to be updated part way through a conversation without restarting the app.
  available_models = [model["name"] for model in ollama.list()["models"]]
  with st.sidebar:
    selected_model = st.selectbox("Select a model", available_models, key = "selected_model")

  if "ollama_model" not in st.session_state:
    st.session_state["ollama_model"] = "llama3.1"
  else:
    st.session_state["ollama_model"] = selected_model

  with st.sidebar:
    st.button("Start a new chat", on_click = new_chat)

  chat_files = [f for f in os.listdir("chat_history/") if f.endswith(".json")]
  with st.sidebar:
    selected_chat = st.selectbox("Chat history", chat_files)
    if st.button("Load chat"):
      load_chat_history(selected_chat)
    if st.button("Delete chat"):
      delete_chat_history(selected_chat)
  
  for message in st.session_state.messages:
    with st.chat_message(
      message["role"], 
      avatar = "🧙‍♂️" if message["role"] == "assistant" else "🧑‍💻"
    ):
      st.markdown(message["content"])

  if prompt := st.chat_input("What would you like to ask?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar = "🧑‍💻"):
      st.markdown(prompt)

    with st.chat_message("assistant", avatar = "🧙‍♂️"):
      with st.spinner("Contemplating..."):
        stream = client.chat.completions.create(
          model=st.session_state["ollama_model"],
          messages=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
          ],
          stream=True,
        )
    response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})

  # st.write(st.session_state.messages)

  # Display options for saving chat messages if required.

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

if __name__ == "__main__":
  main()