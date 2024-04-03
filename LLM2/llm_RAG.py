import warnings
warnings.filterwarnings("ignore")
import json
import uuid
import inspect
import os
from halo import Halo
# ----------------------------------------------------------------------------------------------------------------------
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
# ----------------------------------------------------------------------------------------------------------------------
from LLM2 import llm_interaction
import tools_time_profiler
import tools_Azure_Search
# ----------------------------------------------------------------------------------------------------------------------
class RAG(object):
    def __init__(self, chain,filename_config_vectorstore,vectorstore_index_name,filename_config_emb_model,do_debug=False,do_spinner=False):
        self.TP = tools_time_profiler.Time_Profiler()
        self.chain = chain
        self.azure_search = tools_Azure_Search.Client_Search(filename_config_vectorstore,index_name=vectorstore_index_name,filename_config_emb_model=filename_config_emb_model)
        self.init_search_index(vectorstore_index_name, search_field='token')
        self.search_mode_hybrid = True
        print('RAG search mode: ' + ('hybrid' if self.search_mode_hybrid else 'semantic'))
        self.init_cache()
        self.do_debug = do_debug
        self.do_spinner = do_spinner
        return
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
        with open(filename_in, 'r',encoding='utf8', errors ='ignore') as f:
            text_document = f.read()
        texts = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=500, chunk_overlap=100).create_documents([text_document])
        texts = [text.page_content for text in texts]
        return texts
# ----------------------------------------------------------------------------------------------------------------------
    def add_document_azure(self, filename_in, azure_search_index_name,tag=None,do_tokenize=True):
        print(filename_in)
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        texts = self.pdf_to_texts(filename_in) if filename_in.split('/')[-1].split('.')[-1].find('pdf') >= 0 else self.file_to_texts(filename_in)

        if tag is None:
            docs =[{'uuid':uuid.uuid4().hex,'text':t} for t in texts]
        else:
            docs =[{'uuid':uuid.uuid4().hex,'tag':tag,'text':t} for t in texts]

        field_embedding = None
        if do_tokenize:
            field_embedding = 'token'
            docs = self.azure_search.tokenize_documents(docs, field_source='text', field_embedding=field_embedding)
            #docs = self.azure_search.tokenize_documents_SentenceTransformerEmbeddings(docs, field_source='text', field_embedding=field_embedding) #not integrated to search yet in the logic of the class

            search_index = self.azure_search.create_search_index(docs,field_embedding=field_embedding, index_name=azure_search_index_name)
            if azure_search_index_name not in self.azure_search.get_indices():
                self.azure_search.search_index_client.create_index(search_index,field_embedding=field_embedding, index_name=azure_search_index_name)
        else:
            if azure_search_index_name not in self.azure_search.get_indices():
                self.azure_search.create_search_index(docs, field_embedding=field_embedding, index_name=azure_search_index_name)

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
        texts = self.azure_search.search_semantic(query,select=select,limit=limit)
        docs = self.texts_to_docs(texts)
        return docs
# ----------------------------------------------------------------------------------------------------------------------
    def do_docsearch_azure_hybrid(self, query, azure_search_index_name, search_field='token', select='text',limit=4):
        self.azure_search.search_client = self.azure_search.get_search_client(azure_search_index_name)
        #texts = self.azure_search.search_hybrid(query,field=search_field,select=select,limit=limit)
        texts = self.azure_search.search_vector(query,field=search_field,select=select,limit=limit)
        docs = self.texts_to_docs(texts)
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
    def init_search_index(self,azure_search_index_name,search_field):
        self.azure_search_index_name =azure_search_index_name
        self.search_field=search_field
        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_cache(self):
        self.filename_cache = './data/output/' + 'cache.json'
        self.dct_cache = {}

        # if os.path.isfile(self.filename_cache):
        #     with open(self.filename_cache, 'r') as f:
        #         self.dct_cache = json.load(f)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def run_query(self, query:str, search_field='token',limit=6):

        if self.do_spinner:
            self.TP.tic('RAG', verbose=False,reset=True)
            spinner = Halo(spinner={'interval': 100, 'frames': ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']})
            spinner.start()

        if self.search_mode_hybrid:
            docs = self.do_docsearch_azure_hybrid(query, self.azure_search_index_name,search_field=search_field,select=self.select,limit=limit)
        else:
            docs = self.do_docsearch_azure(query, self.azure_search_index_name,select=self.select,limit=limit)

        if self.do_spinner:
            spinner.stop()
            spinner.succeed(self.TP.print_duration('RAG', verbose=False))

        texts = [d.page_content[:1024] for d in docs]
        #response = self.chain.run(question=query, input_documents=docs)
        query = query + '\n\n'.join(texts)
        response = self.chain.run(question=query,input_documents=[])
        # try:
        #     response = self.chain.run(question=query,input_documents=docs)
        # except:
        #     response = ''

        if self.do_debug:
            llm_interaction.display_debug_info(texts)

        mode = 'w' if not os.path.isfile(self.filename_cache) else 'a'
        with open(self.filename_cache, mode) as f:
            dct_texts = dict(zip(range(len(texts)),texts))
            json.dump(dct_texts, f, indent=4)
            json.dump({'query':query,'response':response}, f, indent=4)

        return response,texts
# ----------------------------------------------------------------------------------------------------------------------