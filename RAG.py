from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests
import io
import os
import tempfile
from urllib.parse import quote
from dotenv import load_dotenv

# Configuration
BUCKET_NAME = "files"
ACCOUNT_ID = "a647af0b197a174668773e5c87e93c8b"

class RAGSystem:
    def __init__(self):
        load_dotenv()
        self.retriever = None
        self.rag_chain = None
        self.R2_CUSTOM_DOMAIN = "christophhein.me"
        self.R2_API_TOKEN = os.getenv("R2_API_TOKEN")
        os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")

    # Initialize immediately with error handling
    try:
        self._initialize()
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        raise

    def _download_from_r2(self, file_key):
        url = f"https://{self.R2_CUSTOM_DOMAIN}/{quote(file_key)}"
        headers = {
            "Authorization": f"Bearer {self.R2_API_TOKEN}",
            "Content-Type": "application/octet-stream"
        }

        print(f"\nAttempting download from: {url}")
        if not self.R2_API_TOKEN:
            raise ValueError("R2_API_TOKEN is not set in environment variables")

        try:
            response = requests.get(url, headers=headers, timeout=10)
            print(f"HTTP Status: {response.status_code}")

            suffix = os.path.splitext(file_key)[1]
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                tmp_file.write(response.content)
                return tmp_file.name
        
        except requests.exceptions.RequestException as e:
            print(f"Download failed. Details: {e}")
            raise

    def _load_documents(self):
        documents = []
        file_config = [
            ("SEAdv_Report.pdf", PyPDFLoader),
            ("Fake.docx", Docx2txtLoader)
        ]

        for filename, loader_cls in file_config:
            try:
                print(f"\n📂 Processing {filename}...")
                file_path = self._download_from_r2(filename)
                loader = loader_cls(file_path)
                documents.extend(loader.load())
                print(f"✅ Successfully loaded {filename}")
                os.unlink(file_path)
            except Exception as e:
                print(f"❌ Failed to load {filename}: {str(e)}")
                if 'file_path' in locals():
                    try:
                        os.unlink(file_path)
                    except:
                        pass

        if not documents:
            raise ValueError("No documents loaded - check configuration")
        return documents

    def _initialize(self):
        docs_list = self._load_documents()
        doc_splits = self._split_documents(docs_list)
        self.retriever = self._create_retriever(doc_splits)
        self.rag_chain = self._create_rag_chain()

    def _split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return text_splitter.split_documents(documents)

    def _create_retriever(self, documents):
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = SKLearnVectorStore.from_documents(documents=documents, embedding=embedding)
        return vectorstore.as_retriever(k=4)

    def _create_rag_chain(self):
        prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks.
Use the following documents to answer the question.
If you don't know the answer, just say that you don't know.
Use a few sentences maximum and keep the answer concise but informative:

Question: {question}
Documents: {documents}
Answer:
""",
            input_variables=["question", "documents"],
        )

        llm = ChatMistralAI(
            model="mistral-small",
            temperature=0,
        )

        return prompt | llm | StrOutputParser()

    def query(self, question):
        try:
            documents = self.retriever.invoke(question)
            doc_texts = "\n\n".join([doc.page_content for doc in documents])
            return self.rag_chain.invoke({
                "question": question,
                "documents": doc_texts
            })
        except Exception as e:
            print(f"Error during query: {str(e)}")
            return "I encountered an error processing your request. Please try again."

