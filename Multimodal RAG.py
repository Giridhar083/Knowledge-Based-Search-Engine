import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.utils.constants import PartitionStrategy

st.set_page_config(page_title="Multimodal RAG", layout="wide")
pdfs_directory = r'F:\RAG\Multimodal RAG\pdfs'
if not os.path.exists(pdfs_directory):
    os.makedirs(pdfs_directory)

@st.cache_resource
def load_models():
    # Make sure you pulled these models in terminal:
    # ollama pull nomic-embed-text
    # ollama pull moondream
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = InMemoryVectorStore(embeddings)
    model = OllamaLLM(model="moondream")
    return vector_store, model
vector_store, model = load_models()

def upload_pdf(file):
    file_path = os.path.join(pdfs_directory, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def extract_text_from_image(image_path):
    # This asks the Vision model to describe the image
    model_with_image_context = model.bind(images=[image_path])
    response = model_with_image_context.invoke("Describe this image in detail.")
    return response

def load_pdf(file_path):
    for f in os.listdir(pdfs_directory):
        if f.endswith((".png", ".jpg", ".jpeg")):
            os.remove(os.path.join(pdfs_directory, f))

    st.info("Step 1/3: Analyzing PDF Layout...")
    elements = partition_pdf(
        file_path,
        strategy=PartitionStrategy.HI_RES,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir=pdfs_directory
    )
    text_elements = [element.text for element in elements if element.category not in ["Image", "Table"]]
    image_files = [f for f in os.listdir(pdfs_directory) if f.endswith((".png", ".jpg", ".jpeg"))]

    if image_files:
        st.info(f"Step 2/3: Analyzing {len(image_files)} images with AI...")
        progress_bar = st.progress(0)

        for i, file in enumerate(image_files):
            progress_bar.progress((i + 1) / len(image_files))

            image_path = os.path.join(pdfs_directory, file)
            if os.path.getsize(image_path) > 4000:
                image_description = extract_text_from_image(image_path)
                text_elements.append(f"[Image Description: {image_description}]")

    return "\n\n".join(text_elements)

def index_docs(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_text(text)
    vector_store.add_texts(chunks)
    return True

st.title("📄 Multimodal RAG Chatbot")
if "processed" not in st.session_state:
    st.session_state.processed = False

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    if not st.session_state.processed:
        saved_path = upload_pdf(uploaded_file)

        with st.spinner("Processing document... please wait..."):
            raw_text = load_pdf(saved_path)
            index_docs(raw_text)

        st.session_state.processed = True
        st.success("Step 3/3: Finished! You can now chat below.")
    question = st.chat_input("Ask a question about the PDF...")

    if question:
        st.chat_message("user").write(question)
        related_documents = vector_store.similarity_search(question)
        context = "\n\n".join([doc.page_content for doc in related_documents])
        template = """
        Answer the question based ONLY on the following context.
        Context: {context}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chain.invoke({"question": question, "context": context})
                st.write(answer)