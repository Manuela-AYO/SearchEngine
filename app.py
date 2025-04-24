from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.summarize import load_summarize_chain
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import validators
import os

enabled_tracing = True

os.environ["USER_AGENT"]="ManuelaGenAI/1.0"
os.environ.pop("SSL_CERT_FILE", None)

model = ChatGroq(model_name="Llama3-8b-8192", api_key=st.secrets["GROQ_API_KEY"])
map_prompt = ChatPromptTemplate.from_messages([
    ("system", "Make a concise summary of {text}")
])
combine_prompt = ChatPromptTemplate.from_messages([
    ("system", "Using each of the summary provided, make an overall summary {text}")
])

chain = load_summarize_chain(llm=model, chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=combine_prompt)


st.title("Youtube video and website summarizer")

if enabled_tracing:
    with st.container():
        site_url = st.text_input(label="Youtube or website link", placeholder="Enter a valid Youtube or website link")
        if st.button(label="Valider", type="primary"):
            site_url = site_url.strip()
            url_validator = validators.url(site_url)
            if url_validator:
                try:
                    docs = []
                    if "youtube.com" in site_url:
                        docs = YoutubeLoader.from_youtube_url(site_url).load()
                    else:
                        docs = WebBaseLoader(site_url).load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    documents = text_splitter.split_documents(docs)
                    # st.text("Your awesome summary is being prepared...")
                    summary = chain.run(documents)
                    st.text("Your summary")
                    st.text(summary)
                except Exception as e:
                    st.error("An error occured ", e)
            else: 
                    st.error("This is not a valid url")
