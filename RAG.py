from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests
import io
import os
import tempfile
from urllib.parse import quote
from dotenv import load_dotenv

# Configuration - UPDATE THESE!
BUCKET_NAME = "files"  # Just the name, no .r2.dev
ACCOUNT_ID = "a647af0b197a174668773e5c87e93c8b"
                   # If files are in subfolder (or "" if in root)


class RAGSystem:
    def __init__(self):
        load_dotenv()
        
        os.environ['USER_AGENT'] = 'MyRAGApplication/1.0'
        self.retriever = None
        self.rag_chain = None

        self.R2_CUSTOM_DOMAIN = "christophhein.me"  # Your connected domain
        
       
    

    def _download_from_r2(self, file_key):
        """Download file via your custom domain"""
        url = f"https://{self.R2_CUSTOM_DOMAIN}/{quote(file_key)}"
        headers = {
            "Authorization": f"Bearer {self.R2_API_TOKEN}",
            "Content-Type": "application/octet-stream"
        }
        
        print(f"\nAttempting download from: {url}")
        if not self.R2_API_TOKEN:
            raise ValueError("R2_API_TOKEN is not set in environment variables")
        
        print(f"🔑 Using R2 token starting with: {self.R2_API_TOKEN[:6]}...")
       
        
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            print(f"HTTP Status: {response.status_code}")

                # Create temporary file
            suffix = os.path.splitext(file_key)[1]
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                tmp_file.write(response.content)
                return tmp_file.name
            
            if response.status_code == 404:
                raise ValueError(
                    f"File not found at: {url}\n"
                    "Verify:\n"
                    "1. File exists in R2 bucket\n"
                    "2. Custom domain is properly connected\n"
                    "3. No typos in filename"
                )
            
            response.raise_for_status()
            return io.BytesIO(response.content)
            
        except requests.exceptions.RequestException as e:
            print(f"Download failed. Details:")
            if hasattr(e, 'response'):
                print(f"Response: {e.response.text}")
            raise

    def _load_documents(self):
        """Load documents with proper file handling"""
        documents = []
        file_config = [
            ("SEAdv_Report.pdf", PyPDFLoader),
            ("Fake.docx", Docx2txtLoader)
        ]
        
        for filename, loader_cls in file_config:
            try:
                print(f"\n📂 Processing {filename}...")
                file_path = self._download_from_r2(filename)
                
                # Use the file path instead of BytesIO
                loader = loader_cls(file_path)
                documents.extend(loader.load())
                print(f"✅ Successfully loaded {filename}")
                
                # Clean up temporary file
                os.unlink(file_path)
            except Exception as e:
                print(f"❌ Failed to load {filename}: {str(e)}")
                if 'file_path' in locals():
                    try:
                        os.unlink(file_path)
                    except:
                        pass
                raise
        
        if not documents:
            raise ValueError("No documents loaded - check configuration")
        return documents
        
    def _initialize(self):
        """Initialize all components"""
        # Load and split documents
        docs_list = self._load_documents()
        if not docs_list:
            raise ValueError("No documents were loaded - please check your input files")
        
        doc_splits = self._split_documents(docs_list)
        
        # Create vector store and retriever
        self.retriever = self._create_retriever(doc_splits)
        
        # Create RAG chain
        self.rag_chain = self._create_rag_chain()
    
    def _split_documents(self, documents):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=350,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return text_splitter.split_documents(documents)
    
    def _create_retriever(self, documents):
        """Create vector store and retriever"""
        vectorstore = SKLearnVectorStore.from_documents(
            documents=documents,
            embedding=OllamaEmbeddings(model="llama3.1:8b"),
        )
        return vectorstore.as_retriever(k=4)
    
    def _create_rag_chain(self):
        """Create the RAG processing chain"""
        prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks.
            Use the following documents to answer the question.
            If you don't know the answer, just say that you don't know.
            Use a few sentences maximum and keep the answer concise but don't leave out important information:
            Question: {question}
            Documents: {documents}
            Answer:
            """,
            input_variables=["question", "documents"],
        )
        
        llm = ChatOllama(
            model="llama3.1:8b",
            temperature=0,
        )
        
        return prompt | llm | StrOutputParser()
    
    def query(self, question):
        """Query the RAG system"""
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
        

#    def test_download(filename):
        url = f"https://christophhein.me/{filename}"
        headers = {"Authorization": f"Bearer {R2_API_TOKEN}"}
        
        print(f"\nAttempting to download: {filename}")
        print(f"Full URL: {url}")
        print(f"Token starts with: {R2_API_TOKEN[:6]}...")
        
        try:
            response = requests.get(url, headers=headers)
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                print("✓ Success! File exists and is accessible")
                return True
            else:
                print(f"Response: {response.text}")
        except Exception as e:
            print(f"Error: {str(e)}")
        return False
            
#    if __name__ == "__main__":
        files_to_test = ["SEAdv_Report.pdf", "Fake.docx"]
        for file in files_to_test:
            if not test_download(file):
                print("\nTroubleshooting Steps:")
                print("1. Verify bucket name and account ID")
                print("2. Check file exists in R2 dashboard")
                print("3. Regenerate API token with 'Object Read' permissions")
                print("4. Ensure no typos in filenames (case-sensitive)")
                break