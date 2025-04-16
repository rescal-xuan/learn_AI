import warnings
warnings.filterwarnings('ignore')
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import io
import pdfplumber

# Streamlit UI
st.title("Document Q&A with Ollama and Langchain")
llm = Ollama(model="deepseek-r1:1.5b")
# File Upload
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

@st.cache_resource  # Cache the embedding and vector store
def load_and_process_data(uploaded_file):
    """Loads, splits, and embeds the PDF document."""

    if uploaded_file is not None:
        try:
            # Read the PDF content from the uploaded file as bytes
            pdf_content = uploaded_file.read()

            # Use io.BytesIO to create a file-like object from the bytes
            pdf_file = io.BytesIO(pdf_content)


            # 1. Load the PDF using pdfplumber directly
            with pdfplumber.open(pdf_file) as pdf:
                pages = []
                for page in pdf.pages:
                    pages.append(page.extract_text())

            # Create a document-like structure for Langchain
            documents = [f"Page {i+1}: {page}" for i, page in enumerate(pages)]

            # 2. Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.create_documents(documents)  # Use create_documents instead

            # 3. Create embeddings using HuggingFace
            embeddings = HuggingFaceEmbeddings()  # Or another suitable model

            # 4. Create a vector store (FAISS)
            db = FAISS.from_documents(texts, embeddings)
            return db

        except Exception as e:
            st.error(f"An error occurred while processing the PDF: {e}")
            st.error(f"Detailed error: {type(e).__name__} - {e}") # Show the type and the specific error message
            return None
    else:
        return None



db = load_and_process_data(uploaded_file)
prompt="""
1. 仅使用下面的上下文。
2. 如果不确定，请说“我不知道”。
3. 答案保持在 4 句话以内。
上下文:{context}
问题:{question}
答案:
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

# User Query
query = st.text_input("Enter your question about the document:")

if query:
    if db is not None:
        # 5. Set up the Ollama language model
        llm = Ollama(model="deepseek-r1:1.5b")  # Ensure the model is available

        # 6. Create a RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # Or "refine", "map_reduce", "map_rerank" (experiment with different chain types)
            retriever=db.as_retriever(search_kwargs={"k": 4}),  # Adjust 'k' as needed
            return_source_documents=True  # Option to return source documents
        )

        # 7. Run the query
        with st.spinner("Generating answer..."):
            result = qa_chain({"query": query})

        # 8. Display the results
        st.subheader("Answer:")
        st.write(result["result"]) # Access the answer from the result dictionary

        # (Optional) Display source documents
        if "source_documents" in result:
            st.subheader("Source Documents:")
            for doc in result["source_documents"]:
                st.write(doc.page_content)
                st.write("---")
    else:
        st.warning("Please upload a document first.")