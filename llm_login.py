import streamlit as st, streamlit_authenticator as stauth, time

conn = st.connection("postgresql", type = "sql")

def user_authentication(user_name, password):
    user_query = conn.query("SELECT * FROM users;", ttl =0)
    for row in user_query.itertuples():
      if user_name == row.user_name and password == row.password:
        return True

def main():

  st.title("Monocle Login")

  if "authentication_status" in st.session_state:
    st.switch_page("pages/llm_chat.py")

  with st.form(key="login_form"):
    user_name = st.text_input("Username", key = "user_name")
    password = st.text_input("Password", type = "password", key = "password")
    submit_button = st.form_submit_button(label="Login")

    if submit_button:
      if user_authentication(user_name, password):
        st.session_state['authentication_status'] = True
        st.success("Login successful!")
        st.switch_page("pages/llm_chat.py")
      else:
        st.session_state['authentication_status'] = False
        st.success("Login failed! Invalid username or password")

if __name__ == "__main__":
  main()