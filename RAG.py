from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
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
        
        os.environ['USER_AGENT'] = 'MyRAGApplication/1.0'
        self.retriever = None
        self.rag_chain = None
        self.embedding = None  # Will hold our cached embedder

        self.R2_CUSTOM_DOMAIN = "christophhein.me"
        self.R2_API_TOKEN = os.getenv("R2_API_TOKEN")
        os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")

        # Initialize embedding cache
        self._initialize_embedding_cache()
    
    def _initialize_embedding_cache(self):
        """Initialize the embedding cache system"""
        # Create cache directory if it doesn't exist
        os.makedirs("./embedding_cache", exist_ok=True)
        
        # Set up the cache store and embedder
        store = LocalFileStore("./embedding_cache")
        base_embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.embedding = CacheBackedEmbeddings.from_bytes_store(
            base_embedding,
            store,
            namespace="miniLM"  # Unique namespace for this model
        )

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
            #("Masterthesis.pdf", PyPDFLoader),
            ("Fake.pdf", pyPDFLoader),
            #("Anthropocentrism.pdf", PyPDFLoader),
            #("SEAdv_Report.pdf", PyPDFLoader),
            #("Airbalanced_bite.pdf", PyPDFLoader),
            #("Transport_Interviews.pdf", PyPDFLoader),
            #("UCD_Ubicomp.pdf", PyPDFLoader)
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
        """Create vector store and retriever using cached embeddings"""
        vectorstore = SKLearnVectorStore.from_documents(
            documents=documents,
            embedding=self.embedding  # Using the cached embedder
        )
        return vectorstore.as_retriever(k=4)
    
    def _create_rag_chain(self):
        """Create the RAG processing chain"""
        prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks.
            Use the following documents to answer the question.
            If you don't know the answer, just say that you don't know.
            Use a few sentences maximum and keep the answer concise but don't leave out important information:

            Abstract from the document:
            This study investigates the integration of LLMs into Audience Participation
                            Games to mediate collaborative storytelling on Twitch, addressing the lack of
                            frameworks for equitable multi-user prompting. A mixed-methods approach
                            is used to evaluate the LLM as a game agent and record the user experience.
                            This approach combines qualitative interviews, game expereince questionnaire
                            surveys, and exact match analysis. This process is supposed to provide in
                            sights that contribute to a better understanding of the user experience when
                            LLMs/GPTs are integrated into gaming contexts. The findings indicate that,
                            while the used LLM (Qwq) effectively merged numerous inputs into a cohesive
                            narrative, it exhibited an "early-input bias," preferring initial contributions
                            and thereby compromising inclusivity in later turns. This resulted in a decrease
                            of more than 50 percent in the EM score by the fifth turn. Participants
                            reported moderate engagement and low perceived competence, suggesting deficiencies
                            in user experience design concerning visibility and feedback. Positively
                            there was next to no tension and frustrations recorded. Mutual influence while
                            prompting the LLM to create the narrative received moderate ratings, indi
                            cating a moderate impact. Additionally, the LLM's adherence to more than
                            human perspectives was noted as inconsistent with anthropocentric framing
                            appearing in a few occasions. However, humor increased enjoyment and was
                            partially successful combined with critiques of human ecological impact. Over
                            all, the study is able to contribute to human-AI collaboration in multi-user
                            contexts, but the LLM instructed as a game agent is not able to fulfill the
                            task sufficiently all the time. Due to limitations such as small sample sizes
                            and reliance on prompts to facilitate the agent, the thesis explores more the
                            feasibility of creating narrations with users remotely providing multiple inputs
                            in a prompt.

            Question: {question}
            Documents: {documents}
            Answer:
            """,
            input_variables=["question", "documents"],
        )
        
        llm = ChatMistralAI(
            model="mistral-medium",
            temperature=0.1
        )
        
        return prompt | llm | StrOutputParser()
    
    def query(self, question):
        """Query the RAG system"""
        try:
            documents = self.retriever.invoke(question)
            doc_texts = "\n\n".join([doc.page_content for doc in documents])
            
            # Debug logging
            print("\n=== RETRIEVED DOCUMENTS ===")
            for i, doc in enumerate(documents, 1):
                print(f"\nDocument {i}:")
                print(f"Source: {doc.metadata.get('source', 'unknown')}")
                print(f"Content: {doc.page_content[:200]}...")  # First 200 chars
            
            print("\n=== FULL PROMPT SENT TO LLM ===")
            print(f"Question: {question}")
            print(f"Documents: {doc_texts[:500]}...")  # First 500 chars of combined docs
            
            return self.rag_chain.invoke({
                "question": question,
                "documents": doc_texts,
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
