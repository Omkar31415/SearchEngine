from langchain_text_splitters import RecursiveCharacterTextSplitter
import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader #unstructured url loader is used to load the website data

## streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YouTube or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YouTube or Website")
st.subheader('Summarize URL')

## Get the Groq API Key and url(YT or website)to be summarized
with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value="",type="password")

if not groq_api_key:
    st.info("Please add the groq api key")
    st.stop()

generic_url=st.text_input("URL",label_visibility="collapsed")

## Gemma Model USsing Groq API
llm =ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

prompt_template="""
Provide a concise summary of the following content in 300 words or less. Your summary should contain the following elements,Capture the main ideas and key points, Maintain the original tone and intent, Include any critical data, statistics, or findings, Highlight the most important conclusions or takeaways, Avoid unnecessary details or tangential information, Format your response as a coherent paragraph that would be useful for someone who hasn't seen the original content.

Content:{text}
"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can may be a YouTube video URL or website URL")
    else:
        try:
            with st.spinner("Please wait while summary being generated..."):
                ## loading the website or yt video data
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs=loader.load()
                
                final_documents=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=100).split_documents(docs)
                
                ## Chain For Summarization
                chain=load_summarize_chain(llm,chain_type="refine",prompt=prompt)
                output_summary=chain.run(final_documents)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")
                    