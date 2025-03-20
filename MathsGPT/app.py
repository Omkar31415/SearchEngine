import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

## Set upi the Stramlit app
st.set_page_config(page_title="Text To Math Problem Solver And Data Search Assistant",page_icon="🧮")
st.title("Text To Math Problem Solver Uing Google Gemma 2")

groq_api_key=st.sidebar.text_input(label="Groq API Key",type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()

llm=ChatGroq(model="gemma2-9b-it",groq_api_key=groq_api_key)

## Initializing the tools
wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Internet to find information on topics mentioned in the question."
)

## Initializa the Math tool
math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tools for answering math related questions. Only input mathematical expression need to be provided"
)

prompt="""
You are an agent specialized in solving mathematical questions. 
Approach the problem systematically by:
1. Identifying the relevant variables and quantities
2. Setting up the appropriate equations or relationships
3. Solving step-by-step
4. Verifying the solution makes sense in context

Provide a clear, detailed explanation with each step labeled for the question below:
Question: {question}
Answer: 
"""

prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

## Combine all the tools into chain
chain=LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool=Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions that require step-by-step thinking."
)

## initialize the agents
assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I'm a Math chatbot who can solve your math problems and answer questions that require reasoning or research!"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Create a container for the input area
input_container = st.container()

# Use a form to ensure the input area reappears after submission
with input_container:
    with st.form(key="question_form"):
        question = st.text_area(
            "Enter your question:",
            "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?",
            key=f"question_input_{len(st.session_state.messages)}"  # Unique key based on message count
        )
        submit_button = st.form_submit_button("Find Answer")

# Process the form submission
if submit_button and question:
    with st.spinner("Generating response..."):
        # Add user message to chat
        st.session_state.messages.append({"role":"user","content":question})
        st.chat_message("user").write(question)
        
        # Get response from agent
        try:
            # Fix the parameter in assistant_agent.run() to pass only the question string
            response = assistant_agent.run(question, callbacks=[StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)])
            
            # Add assistant response to chat
            st.session_state.messages.append({'role':'assistant', "content":response})
            st.chat_message("assistant").write(response)
            st.success(response)
            # Force a rerun to update the UI with a fresh form
            st.rerun()
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({'role':'assistant', "content":error_msg})
            st.rerun()