from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests
import io
import os

class RAGSystem:
    def __init__(self):
        os.environ['USER_AGENT'] = 'MyRAGApplication/1.0'
        self.retriever = None
        self.rag_chain = None
        self._initialize()
    
    def _load_documents(self):
        """Load documents from Cloudflare R2 with error handling"""
        documents = []
        
        # File type configuration - REPLACE WITH YOUR ACTUAL FILES
        file_types = {
            'pdf': {
                'loader': PyPDFLoader,
                'files': ["SEAdv_Report.pdf", "User-Centered Design Approaches of a Color-Based Communication System for Emotional Support of Employees as a Ubiquitous Computing Solution.pdf"]
            },
            'docx': {
                'loader': Docx2txtLoader,
                'files': ["Fake Projectreport.docx"]
            },
            'txt': {
                'loader': TextLoader,
                'files': ["Plans.txt"]
            }
        }

        # Load each file type
        for file_type, config in file_types.items():
            loader_cls = config['loader']
            for filename in config['files']:
                try:
                    # Download from R2
                    file_stream = self._download_from_r2(filename)
                    
                    # Initialize appropriate loader
                    if file_type == 'pdf':
                        loader = PyPDFLoader(file_stream)
                    elif file_type == 'docx':
                        loader = Docx2txtLoader(file_stream)
                    else:  # txt
                        loader = TextLoader(file_stream)
                    
                    documents.extend(loader.load())
                    print(f"Successfully loaded {filename}")
                    
                except ImportError as e:
                    if file_type == 'docx':
                        print("docx2txt not installed - skipping Word documents")
                    else:
                        print(f"Import error for {filename}: {str(e)}")
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
        
        print(f"\nTotal loaded documents: {len(documents)}")
        return documents

    def _download_from_r2(self, file_key):
        """Download file from Cloudflare R2"""
        url = f"https://a647af0b197a174668773e5c87e93c8b.eu.r2.cloudflarestorage.com/{file_key}"
        headers = {"Authorization": f"Bearer {os.getenv('R2_API_TOKEN')}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise HTTP errors
        return io.BytesIO(response.content)
        
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