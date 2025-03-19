import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler #this library is used to display the thoughts and actions of an agent in an interactive Streamlit app

## Arxiv and wikipedia Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

search=DuckDuckGoSearchRun(name="Search")


st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assisstant","content":"Hi,I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content']) #this will display the messages in the chat

if prompt:=st.chat_input(placeholder="What is machine learning?"): #chat_input is used to take input from the user
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt) #this means that the user's message will be displayed in the chat

    llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True) #streaming=True is used to stream the response
    tools=[search,arxiv,wiki]

    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)

    #ZERO_SHOT_REACT_DESCRIPTION is used to react to the user's query and provide a description of the query. i.e, gives response based only on current query
    #CHAT_SHOT_REACT_DESCRIPTION is used to react to the user's query and provide a description of the query. i.e, gives response based on current query and previous queries

    with st.chat_message("assistant"): 
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False) #this will display the thoughts and actions of an agent in an interactive Streamlit app
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb]) 
        st.session_state.messages.append({'role':'assistant',"content":response}) #agent response is stored in assistant role messages along with the content
        st.write(response)

