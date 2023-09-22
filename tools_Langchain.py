#Ray Serve
import os
import yaml
import numpy
from langchain.retrievers import AzureCognitiveSearchRetriever
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI,AzureChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import GraphCypherQAChain
from langchain.chains import RetrievalQA
from langchain.graphs import Neo4jGraph
from langchain.llms import OpenAI, AzureOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader



import uuid
import pinecone
import inspect
from langchain.vectorstores.azuresearch import AzureSearch
# ----------------------------------------------------------------------------------------------------------------------
import tools_time_profiler
import tools_Azure_Search
# ----------------------------------------------------------------------------------------------------------------------
class Assistant(object):
    def __init__(self, filename_config_chat_model,filename_config_emb_model,filename_config_vectorstore, vectorstore_index_name,chain_type='QA'):
        self.TP = tools_time_profiler.Time_Profiler()
        self.embeddings = self.init_emb_model(filename_config_emb_model)
        self.chain = self.init_chain(filename_config_chat_model, chain_type=chain_type)
        self.azure_search = tools_Azure_Search.Client_Search(filename_config_vectorstore,index_name=vectorstore_index_name,filename_config_emb_model=filename_config_emb_model)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_chain(self,filename_config_chat_model,filename_config_neo4j=None,chain_type='QA'):
        with open(filename_config_chat_model, 'r') as config_file:
            config = yaml.safe_load(config_file)
            if 'openai' in config:
                openai_api_key = config['openai']['key']
                if chain_type == 'QA':
                    model = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
                else:
                    model = OpenAI(temperature=0, openai_api_key=openai_api_key)

            elif 'azure' in config:
                os.environ["OPENAI_API_TYPE"] = "azure"
                os.environ["OPENAI_API_TYPE"] = "azure"
                os.environ["OPENAI_API_VERSION"] = "2023-05-15"
                os.environ["OPENAI_API_BASE"] = config['azure']['openai_api_base']
                os.environ["OPENAI_API_KEY"] = config['azure']['openai_api_key']
                if chain_type == 'QA':
                    model = AzureChatOpenAI(deployment_name=config['azure']['deployment_name'], openai_api_version=os.environ["OPENAI_API_VERSION"],openai_api_key=os.environ["OPENAI_API_KEY"],openai_api_base=os.environ["OPENAI_API_BASE"])
                else:
                    model = AzureOpenAI(deployment_name=config['azure']['deployment_name'], openai_api_version=os.environ["OPENAI_API_VERSION"],openai_api_key=os.environ["OPENAI_API_KEY"],openai_api_base=os.environ["OPENAI_API_BASE"])

        self.LLM = model

        chain = None
        if chain_type == 'QA':
            chain = load_qa_chain(model)
            #chain = RetrievalQA.from_chain_type(llm=self.LLM,retriever=vectordb.as_retriever(), chain_type="stuff")
        elif chain_type == 'Summary':
            chain = load_summarize_chain(model)
        elif chain_type=='Neo4j':
            with open(filename_config_neo4j, 'r') as config_file:
                config = yaml.safe_load(config_file)
                url = f"bolt://{config['database']['host']}:{config['database']['port']}"
            self.graph = Neo4jGraph(url=url,username=config['database']['user'],password=config['database']['password'])
            self.chain = GraphCypherQAChain.from_llm(model,graph=self.graph,verbose=True,return_intermediate_steps=True)

        return chain
# ----------------------------------------------------------------------------------------------------------------------
    def init_emb_model(self, filename_config_emb_model):
        with open(filename_config_emb_model, 'r') as config_file:
            config = yaml.safe_load(config_file)
            if 'openai' in config:
                openai_api_key = config['openai']['key']
                model = OpenAIEmbeddings(openai_api_key=openai_api_key)

            elif 'azure' in config:
                os.environ["OPENAI_API_TYPE"] = "azure"
                os.environ["OPENAI_API_VERSION"] = "2023-05-15"
                os.environ["OPENAI_API_BASE"] = config['azure']['openai_api_base']
                os.environ["OPENAI_API_KEY"] = config['azure']['openai_api_key']
                model = OpenAIEmbeddings(deployment=config['azure']['deployment_name'])

        return model
# ----------------------------------------------------------------------------------------------------------------------
    def pdf_to_texts(self, filename_pdf):
        loader = PyPDFLoader(filename_pdf)
        pages = loader.load_and_split()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(pages)
        texts = [t for t in set([doc.page_content for doc in docs])]
        return texts
# ----------------------------------------------------------------------------------------------------------------------
    def file_to_texts(self,filename_in):
        with open(filename_in, 'r') as f:
            text_document = f.read()
        texts = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=1000, chunk_overlap=50).create_documents([text_document])
        texts = [text.page_content for text in texts]
        return texts
# ----------------------------------------------------------------------------------------------------------------------
    def add_document(self,filename_in):

        loader = TextLoader(filename_in, encoding="cp1252")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)



        self.azure_search.upload_document()

        return
# ----------------------------------------------------------------------------------------------------------------------
    def embed_texts(self,texts):
        res = self.embeddings.embed_documents([t for t in texts])
        res = numpy.array(res)
        return res
# ----------------------------------------------------------------------------------------------------------------------
    def do_docsearch(self, query):



        docs = vector_store.similarity_search(query=query,k=3,search_type="hybrid")

        return docs
# ----------------------------------------------------------------------------------------------------------------------
    def run_chain(self, query, text_key=None, do_debug=False):

        docs = self.do_docsearch(query,text_key=None)

        if do_debug:
            texts = [t for t in set([doc.page_content for doc in docs])]
            for t in texts:
                print(t)
                print('------------------')

        response = self.chain.run(question=query,input_documents=docs)
        return response
# ----------------------------------------------------------------------------------------------------------------------