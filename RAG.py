from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os


class RAGSystem:
    def __init__(self, documents_path='C:/Users/chris/Documents/Papers'):
        os.environ['USER_AGENT'] = 'MyRAGApplication/1.0'
        self.documents_path = documents_path
        self.retriever = None
        self.rag_chain = None
        self._initialize()
        
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
    
    def _load_documents(self):
        """Load documents with error handling"""
        documents = []
        
        # PDFs
        try:
            pdf_loader = DirectoryLoader(
                path=self.documents_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            documents.extend(pdf_loader.load())
        except Exception as e:
            print(f"Error loading PDFs: {str(e)}")
        
        # Word docs (if docx2txt available)
        try:
            docx_loader = DirectoryLoader(
                path=self.documents_path,
                glob="**/*.docx",
                loader_cls=Docx2txtLoader,
                show_progress=True
            )
            documents.extend(docx_loader.load())
        except ImportError:
            print("docx2txt not installed - skipping Word documents")
        except Exception as e:
            print(f"Error loading Word documents: {str(e)}")
        
        # Text files
        try:
            txt_loader = DirectoryLoader(
                path=self.documents_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={'autodetect_encoding': True},
                show_progress=True
            )
            documents.extend(txt_loader.load())
        except Exception as e:
            print(f"Error loading text files: {str(e)}")
            
        print(f"\nLoaded {len(documents)} documents")
        return documents
    
    def _split_documents(self, documents):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=350,
            chunk_overlap=150
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
        documents = self.retriever.invoke(question)
        doc_texts = "\n".join([doc.page_content for doc in documents])
        return self.rag_chain.invoke({
            "question": question,
            "documents": doc_texts
        })