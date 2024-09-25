#https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/sample_vector_search.py
#https://github.com/Azure-Samples/azure-search-python-samples/blob/main/Quickstart/v11/azure-search-quickstart.ipynb
#https://github.com/Azure/azure-search-vector-samples/blob/main/demo-python/code/azure-search-vector-python-sample.ipynb
#----------------------------------------------------------------------------------------------------------------------
import pandas as pd
import yaml
import uuid
from tqdm import tqdm
from LLM2 import llm_interaction
#----------------------------------------------------------------------------------------------------------------------
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex,SearchField,SearchFieldDataType,SimpleField,SearchableField,VectorSearch
from azure.search.documents.indexes.models import HnswAlgorithmConfiguration , VectorSearchAlgorithmKind, HnswParameters, VectorSearchAlgorithmMetric,ExhaustiveKnnAlgorithmConfiguration, ExhaustiveKnnParameters, VectorSearchProfile, SemanticConfiguration, SemanticField, SemanticPrioritizedFields, SemanticSearch
from azure.search.documents.models import VectorizedQuery, VectorQuery
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

from langchain.schema.document import Document
#----------------------------------------------------------------------------------------------------------------------
class Client_Search(object):
    def __init__(self, filename_config,index_name=None,filename_config_emb_model=None):
        if filename_config is None:
            return
        with open(filename_config, 'r') as config_file:
            self.config_search = yaml.safe_load(config_file)
            if not 'azure' in self.config_search.keys():
                return

            self.search_index_client = SearchIndexClient(self.config_search['azure']['azure_search_endpoint'], AzureKeyCredential(self.config_search['azure']['azure_search_key']))

        self.index_name = index_name if index_name is not None else (self.config_search['azure']['index_name'] if 'index_name' in self.config_search['azure'].keys() else None)
        self.search_client = self.get_search_client(self.index_name)
        self.model_embedding = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.model_embedding.cuda()

        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_search_client(self,index_name):
        if index_name is None:
            return None
        return SearchClient(self.config_search['azure']['azure_search_endpoint'], index_name,AzureKeyCredential(self.config_search['azure']['azure_search_key']))
# ----------------------------------------------------------------------------------------------------------------------
    def get_embedding_chroma(self,text):
        self.chroma_db = Chroma.from_documents([Document(page_content=text, metadata={})], SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))
        embedding = self.chroma_db._collection.get(include=['embeddings'])['embeddings']
        return embedding[0]
# ----------------------------------------------------------------------------------------------------------------------
    def get_embedding_huggingface(self, text):
        embedding = self.model_embedding.encode(text)
        embedding = embedding.tolist()
        return embedding
# ----------------------------------------------------------------------------------------------------------------------
    def get_embedding(self,text):
        return self.get_embedding_huggingface(text)
# ----------------------------------------------------------------------------------------------------------------------
    def tokenize_documents(self, dct_records, field_source, field_embedding):
        for d in tqdm(dct_records, total=len(dct_records), desc='Tokenizing'):
            d[field_embedding] = self.get_embedding(d[field_source])

        return dct_records
# ----------------------------------------------------------------------------------------------------------------------
    def create_search_index(self,docs,field_embedding, index_name):
        df = pd.DataFrame(docs[:1])
        fields = []
        vector_search_dimensions = 0
        if field_embedding is not None and field_embedding in df.columns:
            lst = docs[0][field_embedding]
            vector_search_dimensions = len(lst[0]) if len(lst)==1 else len(lst)
        for r in range(df.shape[1]):
            name = df.columns[r]
            if r == 0:
                field = SimpleField(name=name, type=SearchFieldDataType.String, key=True)
            elif name == field_embedding:
                field = SearchField(name=name, type=SearchFieldDataType.Collection(SearchFieldDataType.Single),searchable=True, vector_search_dimensions=vector_search_dimensions,vector_search_profile_name="myHnswProfile")
            else:
                field = SearchableField(name=name, type=SearchFieldDataType.String)
            fields.append(field)

        vector_search = None
        if vector_search_dimensions>0:
            vector_search = VectorSearch(
                algorithms=[ExhaustiveKnnAlgorithmConfiguration(name="myHnsw",kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,parameters=ExhaustiveKnnParameters(metric=VectorSearchAlgorithmMetric.COSINE))],
                profiles=[VectorSearchProfile(name="myHnswProfile",algorithm_configuration_name="myHnsw")]
            )

        index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
        result = self.search_index_client.create_or_update_index(index)
        return result
# ----------------------------------------------------------------------------------------------------------------------
    def get_indices(self):
        res = [x for x in self.search_index_client.list_index_names()]
        return res
# ----------------------------------------------------------------------------------------------------------------------
    def get_document(self,key):
        result = self.search_client.get_document(key=key)
        return result
# ----------------------------------------------------------------------------------------------------------------------
    def get_document_count(self):
        result = self.search_client.get_document_count()
        return result
# ----------------------------------------------------------------------------------------------------------------------
    def upload_documents(self,docs):
        if not isinstance(docs,list):
            docs = [docs]
        result = self.search_client.upload_documents(documents=docs)
        return result[0].succeeded
# ----------------------------------------------------------------------------------------------------------------------
    def delete_document(self,dict_doc):
        #delete_document(dict_doc={"hotelId": "1000"})
        self.search_client.delete_documents(documents=[dict_doc])
        return
# ----------------------------------------------------------------------------------------------------------------------
    def search_semantic(self, query,select=None,as_df=True,limit=5):
        search_res = self.search_client.search(search_text=query,select=select,top=limit)
        result = llm_interaction.from_list_of_dict([r for r in search_res],select=select,as_df=as_df)
        return result
# ----------------------------------------------------------------------------------------------------------------------
    def search_hybrid(self, query,field='token',select=None,as_df=True,limit=5):
        vector = self.get_embedding(query)
        search_client = self.get_search_client(self.index_name)
        vector_query = VectorizedQuery(fields=field, exhaustive=True,vector=vector)
        search_res = search_client.search(search_text=query,vector_queries=[vector_query],top=limit)
        list_of_dict = [r for r in search_res]
        result = llm_interaction.from_list_of_dict(list_of_dict, select=select, as_df=as_df)
        result['score'] = [l['@search.score'] for l in list_of_dict]
        return result
# ----------------------------------------------------------------------------------------------------------------------
    def search_vector(self,query,field='token',select=None,as_df=True,search_filter_dict=None,limit=4):
        vector = self.get_embedding(query)
        search_client = self.get_search_client(self.index_name)
        vector_query = VectorizedQuery(fields=field, exhaustive=True,vector=vector)
        search_res = search_client.search(search_text=None,vector_queries=[vector_query],top=limit)
        list_of_dict = [r for r in search_res]
        result = llm_interaction.from_list_of_dict(list_of_dict, select=select, as_df=as_df)
        result['score'] = [l['@search.score'] for l in list_of_dict]

        return result
# ----------------------------------------------------------------------------------------------------------------------
    def add_book(self, filename_in, azure_search_index_name, do_tokenize=True):

        texts = llm_interaction.pdf_to_texts(filename_in) if filename_in.split('/')[-1].split('.')[-1].find('pdf') >= 0 else llm_interaction.file_to_texts(filename_in)
        docs =[{'uuid':uuid.uuid4().hex,'text':t} for t in texts]

        field_embedding = None
        if do_tokenize:
            field_embedding = 'token'
            docs = self.tokenize_documents(docs, field_source='text', field_embedding=field_embedding)

            search_index = self.create_search_index(docs,field_embedding=field_embedding, index_name=azure_search_index_name)
            if azure_search_index_name not in self.get_indices():
                self.search_index_client.create_index(search_index,field_embedding=field_embedding, index_name=azure_search_index_name)
        else:
            if azure_search_index_name not in self.get_indices():
                self.create_search_index(docs, field_embedding=field_embedding, index_name=azure_search_index_name)

        self.search_client = self.get_search_client(azure_search_index_name)
        self.upload_documents(docs)
        return
# ----------------------------------------------------------------------------------------------------------------------