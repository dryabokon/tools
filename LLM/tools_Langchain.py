import warnings
warnings.filterwarnings("ignore")
import json
import uuid
import inspect
import os
import yaml
# ----------------------------------------------------------------------------------------------------------------------
from langchain.schema.document import Document
# from langchain.retrievers import AzureCognitiveSearchRetriever
# from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI,AzureChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from langchain.llms import OpenAI, AzureOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Pinecone
from langchain.schema import HumanMessage
# ----------------------------------------------------------------------------------------------------------------------
import tools_Azure_NLP
import tools_time_profiler
import tools_Azure_Search
import pinecone
# ----------------------------------------------------------------------------------------------------------------------
class Assistant(object):
    def __init__(self, filename_config_chat_model,filename_config_emb_model,filename_config_vectorstore, filename_config_NLP=None,vectorstore_index_name=None,filename_config_neo4j=None,search_mode_hybrid=True,chain_type='QA'):
        self.TP = tools_time_profiler.Time_Profiler()
        self.embeddings = self.init_emb_model(filename_config_emb_model)
        self.chain = self.init_chain(filename_config_chat_model, filename_config_neo4j=filename_config_neo4j,chain_type=chain_type)
        self.azure_search = tools_Azure_Search.Client_Search(filename_config_vectorstore,index_name=vectorstore_index_name,filename_config_emb_model=filename_config_emb_model)
        self.init_vectorstore_pinecone(filename_config_vectorstore)
        self.search_mode_hybrid=search_mode_hybrid
        self.init_cache()
        self.NER = tools_Azure_NLP.Client_NLP(filename_config_NLP)
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
        self.chain_type = chain_type
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
            chain = GraphCypherQAChain.from_llm(model,graph=self.graph,verbose=True,return_intermediate_steps=True)

        return chain
# ----------------------------------------------------------------------------------------------------------------------
    def init_vectorstore_pinecone(self,filename_config,text_key=None):
        if filename_config is None:
            return
        with open(filename_config, 'r') as config_file:
            config = yaml.safe_load(config_file)
            if 'pinecone' in config:
                self.vectorstore_desc = 'pinecone'
                self.pinecone_api_key = config['pinecone']['api_key']
                self.pinecone_index_name = config['pinecone']['index_name']
                pinecone.init(api_key=self.pinecone_api_key, environment='gcp-starter')
                self.vectorstore_pinecone = Pinecone(pinecone.Index(self.pinecone_index_name), self.embeddings, text_key)
            else:
                self.vectorstore_desc = 'azure'

        return
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
        texts = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=500, chunk_overlap=100).create_documents([text_document])
        texts = [text.page_content for text in texts]
        return texts
# ----------------------------------------------------------------------------------------------------------------------
    def add_document_pinecone(self, filename_in, text_key=None):
        print(filename_in)
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        if text_key is None:
            text_key = filename_in.split('/')[-1].split('.')[0]

        texts = self.pdf_to_texts(filename_in) if filename_in.split('/')[-1].split('.')[-1].find('pdf') >= 0 else self.file_to_texts(filename_in)

        self.vectorstore_pinecone.add_texts(texts)
        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def add_document_azure(self, filename_in, azure_search_index_name):
        print(filename_in)
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        texts = self.pdf_to_texts(filename_in) if filename_in.split('/')[-1].split('.')[-1].find('pdf') >= 0 else self.file_to_texts(filename_in)
        docs =[{'uuid':uuid.uuid4().hex,'text':t} for t in texts]
        docs_e = self.azure_search.tokenize_documents(docs, field_source='text', field_embedding='token')

        if azure_search_index_name not in self.azure_search.get_indices():
            fields = self.azure_search.create_fields(docs_e, field_embedding='token')
            search_index = self.azure_search.create_search_index(azure_search_index_name, fields)
            self.azure_search.search_index_client.create_index(search_index)

        self.azure_search.search_client = self.azure_search.get_search_client(azure_search_index_name)
        self.azure_search.upload_documents(docs)
        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def texts_to_docs(self,texts):
        docs = [Document(page_content=t, metadata={}) for t in texts]
        return docs
# ----------------------------------------------------------------------------------------------------------------------
    def do_docsearch_azure(self, query,azure_search_index_name,select='text',limit=4):
        self.azure_search.search_client = self.azure_search.get_search_client(azure_search_index_name)
        texts = self.azure_search.search_texts(query,select=select)
        docs = self.texts_to_docs(texts)
        return docs
# ----------------------------------------------------------------------------------------------------------------------
    def do_docsearch_azure_hybrid(self, query, azure_search_index_name, search_field='token', select='text',limit=4):
        self.azure_search.search_client = self.azure_search.get_search_client(azure_search_index_name)
        texts = self.azure_search.search_texts_hybrid(query,field=search_field,select=select,limit=limit)
        docs = self.texts_to_docs(texts)
        return docs
# ----------------------------------------------------------------------------------------------------------------------
    def do_docsearch_pinecone(self, query, text_key=None,limit=4):
        docsearch = Pinecone(Pinecone.get_pinecone_index(self.pinecone_index_name), self.embeddings, text_key=text_key)
        docs = docsearch.similarity_search(query=query,k=limit)#docs = docsearch.as_retriever(search_type="mmr").get_relevant_documents(query)
        return docs
# ----------------------------------------------------------------------------------------------------------------------
    def pretify_string(self,text,N=120):
        lines = []
        line = ""
        for word in text.split():
            if len(line + word) + 1 <= N:
                if line:
                    line += " "
                line += word
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)

        result = '\n'.join(lines)

        return result
# ----------------------------------------------------------------------------------------------------------------------
    def run_chain(self, query, azure_search_index_name=None, text_key=None,search_field='token',select='text',limit=4):

        if self.vectorstore_desc == 'pinecone':
            docs = self.do_docsearch_pinecone(query,text_key=text_key,limit=limit)
        else:
            if self.search_mode_hybrid:
                docs = self.do_docsearch_azure_hybrid(query, azure_search_index_name,search_field=search_field,select=select,limit=limit)
            else:
                docs = self.do_docsearch_azure(query, azure_search_index_name,select=select)

        texts = [d.page_content for d in docs]

        try:
            response = self.chain.run(question=query,input_documents=docs)
        except:
            response = ''

        return response,texts
# ----------------------------------------------------------------------------------------------------------------------
    def init_search_index(self,azure_search_index_name,search_field,text_key):
        self.azure_search_index_name =azure_search_index_name
        self.search_field=search_field
        self.text_key = text_key
        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_cache(self):
        self.filename_cache = './data/output/' + 'cache.json'
        self.dct_cache = {}

        if os.path.isfile(self.filename_cache):
            with open(self.filename_cache, 'r') as f:
                self.dct_cache = json.load(f)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def Q(self,query,context_free=False,texts=None,q_post_proc=None,html_allowed=False):

        if (texts is None or len(texts) ==0) and (query in self.dct_cache.keys()):
            responce = self.dct_cache[query]
        else:
            if (texts is not None and len(texts) > 0):
                context_free = True
                #query+= '\nUse data below.\n'
                query+= '.'.join(texts)
            if context_free:
                try:
                    if self.chain_type=='QA':
                        responce = self.LLM([HumanMessage(content=query)]).content
                    else:
                        responce = self.LLM(query)
                except:
                    responce = ''
            else:
                responce, texts= self.run_chain(query, azure_search_index_name=self.azure_search_index_name, search_field=self.search_field,select=self.text_key)

            self.dct_cache[query] = responce
            with open(self.filename_cache, "w") as f:
                f.write(json.dumps(self.dct_cache, indent=4))

        if q_post_proc is not None:
            responce = self.Q(f'{q_post_proc} Q:{query} A:{responce}', context_free=True)

        if responce.find('Unfortunately') == 0 or responce.find('I\'m sorry') == 0 or responce.find('N/A') == 0:
            #responce = "⚠️" + responce
            responce = '&#9888 ' + responce

        if html_allowed:
            # responce = f'<span style="color: #086A6A;background-color:#EEF4F4">{query}</span>'
            responce = f'<span style="color: #000000;background-color:#D5E5E5">{responce}</span>'

        return responce
# ----------------------------------------------------------------------------------------------------------------------

