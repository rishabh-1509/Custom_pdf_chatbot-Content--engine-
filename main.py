import streamlit as st
import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from pdf2image import convert_from_path
from transformers import AutoTokenizer, TextStreamer, pipeline

# Setting the device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

st.title("Content - Engine ")

# File uploader
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    # Loading the PDF documents
    loader = PyPDFLoader(uploaded_files)
    docs = loader.load()
    
    st.write(f"Loaded {len(docs)} documents")

    # Splitting the text from the PDF into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, schunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    
    st.write(f"Split into {len(texts)} text chunks")

    # Performing the embedding on the smaller text chunks
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE})

    # Creating a Chroma VectorStore
    db = Chroma.from_documents(texts, embeddings, persist_directory="db")
    st.write("Created Chroma VectorStore")

    # Initializing the local LLM
    model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
    model_basename = "model"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    model = AutoGPTQForCausalLM.from_quantized(
        model_name_or_path=model_name_or_path,
        revision="gptq-4bit-128g-actorder_True",
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        inject_fused_attention=False,
        device="cuda:0",
        quantize_config=None
    )
    st.write("Initialized local LLM")

    # Initializing the prompt format
    DEFAULT_SYSTEM_PROMPT = """
    You are a helpful, respectful and honest assistant. Always answer as helpfully
    as possible, while being safe. Your answers should not include any harmful,
    unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your
    responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain
    why instead of answering something not correct. If you don't know the answer to a
    question, please don't share false information.
    """.strip()

    def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        return f"""
    [INST] <<SYS>>
    {system_prompt}
    <</SYS>>

    {prompt} [/INST]
    """.strip()

    # Initializing the pipeline
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        streamer=streamer,
        temperature=0.1,
        max_new_tokens=1024,
        temperature=0,
        top_p=0.05,
        repetition_penalty=1.15,
    )

    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})
    SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."

    template = generate_prompt(
        """
    {context}

    Question: {question}
    """,
        system_prompt=SYSTEM_PROMPT,
    )

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Initializing the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"K": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    st.write("Initialized RetrievalQA chain")

    question = st.text_input("Ask a question about the PDF content")
    if question:
        result = qa_chain.run(question)
        st.write("Answer:")
        st.write(result["answer"])

        st.write("Source documents:")
        for doc in result["source_documents"]:
            st.write(doc.page_content)
