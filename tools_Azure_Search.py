#https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/sample_vector_search.py
#https://github.com/Azure-Samples/azure-search-python-samples/blob/main/Quickstart/v11/azure-search-quickstart.ipynb
#https://github.com/Azure/azure-search-vector-samples/blob/main/demo-python/code/azure-search-vector-python-sample.ipynb
#----------------------------------------------------------------------------------------------------------------------
import openai
import pandas as pd
import yaml
import uuid
from tqdm import tqdm
#----------------------------------------------------------------------------------------------------------------------
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex,SearchField,SearchFieldDataType,SimpleField,SearchableField,VectorSearch
from azure.search.documents.indexes.models import HnswAlgorithmConfiguration , VectorSearchAlgorithmKind, HnswParameters, VectorSearchAlgorithmMetric,ExhaustiveKnnAlgorithmConfiguration, ExhaustiveKnnParameters, VectorSearchProfile, SemanticConfiguration, SemanticField, SemanticPrioritizedFields, SemanticSearch
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document
from azure.search.documents.models import VectorizedQuery, VectorQuery
from sentence_transformers import SentenceTransformer
#----------------------------------------------------------------------------------------------------------------------
#from azure.ai.textanalytics import TextAnalyticsClient
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

        # if filename_config_emb_model is not None:
        #     with open(filename_config_emb_model, 'r') as config_file:
        #         self.config_emb = yaml.safe_load(config_file)
        #         openai.api_type = "azure"
        #         openai.api_version = self.config_emb['azure']['openai_api_version']
        #         openai.api_base = self.config_emb['azure']['openai_api_base']
        #         openai.api_key = self.config_emb['azure']['openai_api_key']
        #         self.model_deployment_name = self.config_emb['azure']['deployment_name']

        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_search_client(self,index_name):
        if index_name is None:
            return None
        return SearchClient(self.config_search['azure']['azure_search_endpoint'], index_name,AzureKeyCredential(self.config_search['azure']['azure_search_key']))
#----------------------------------------------------------------------------------------------------------------------
    # def get_NER_client(self):
    #     return TextAnalyticsClient(endpoint=self.config_search['azure']['azure_search_endpoint'], credential=AzureKeyCredential(self.config_search['azure']['azure_search_key']))
# ----------------------------------------------------------------------------------------------------------------------
    def tokenize_documents(self, dct_records, field_source, field_embedding):

        for d in tqdm([d for d in dct_records],total=len(dct_records),desc='Tokenizing'):
            d[field_embedding]=self.get_embedding_huggingface(d[field_source])

        return dct_records
# ----------------------------------------------------------------------------------------------------------------------
    def tokenize_documents_SentenceTransformerEmbeddings(self, dct_records, field_source, field_embedding):

        texts = [d[field_source] for d in dct_records]
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        docs = [Document(page_content=t, metadata={}) for t in texts]
        db = Chroma.from_documents(docs, embedding_function)
        embeddings = db._collection.get(include=['embeddings'])['embeddings']

        print(len(dct_records))
        for i,d in enumerate(dct_records):
            print('.')
            d[field_embedding] = embeddings[i]

        return dct_records
# ----------------------------------------------------------------------------------------------------------------------
    def hash_documents(self, dct_records, field_source, field_hash):
        for d in dct_records:
            d[field_hash]=hash(d[field_source])
        return
# ----------------------------------------------------------------------------------------------------------------------
    def add_uuid_to_documents(self, dct_records):
        for d in dct_records:
            d['uuid']=uuid.uuid4().hex
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

        # fields2 = [
        #     SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True,facetable=True),
        #     SearchableField(name="title", type=SearchFieldDataType.String),
        #     SearchableField(name="content", type=SearchFieldDataType.String),
        #     SearchableField(name="category", type=SearchFieldDataType.String,filterable=True),
        #     SearchField(name="titleVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
        #     SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
        # ]

        vector_search = None
        if vector_search_dimensions>0:
            vector_search = VectorSearch(
                algorithms=[ExhaustiveKnnAlgorithmConfiguration(name="myHnsw",kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,parameters=ExhaustiveKnnParameters(metric=VectorSearchAlgorithmMetric.COSINE))],
                profiles=[VectorSearchProfile(name="myHnswProfile",algorithm_configuration_name="myHnsw")]
            )

            # vector_search2 = VectorSearch(
            #     algorithms=[HnswAlgorithmConfiguration(name="myHnsw",kind=VectorSearchAlgorithmKind.HNSW,parameters=HnswParameters(m=4,ef_construction=400,ef_search=500,metric=VectorSearchAlgorithmMetric.COSINE)),
            #                 ExhaustiveKnnAlgorithmConfiguration(name="myExhaustiveKnn",kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,parameters=ExhaustiveKnnParameters(metric=VectorSearchAlgorithmMetric.COSINE))],
            #     profiles=[
            #         VectorSearchProfile(name="myHnswProfile",algorithm_configuration_name="myHnsw",),
            #         VectorSearchProfile(name="myExhaustiveKnnProfile",algorithm_configuration_name="myExhaustiveKnn",)
            #     ]
            # )

        index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
        result = self.search_index_client.create_or_update_index(index)
        return result
# ----------------------------------------------------------------------------------------------------------------------
    def init_vector_store(self):
        #
        # self.vector_store = AzureSearch(
        #     azure_search_endpoint=self.config_search['azure']['azure_search_endpoint'],
        #     azure_search_key=self.config_search['azure']['openai_api_key'],
        #     index_name=self.index_name,
        #     embedding_function=self.embedding.embed_query,
        #     fields=fields)
        return
# ----------------------------------------------------------------------------------------------------------------------
#     def get_embedding_chroma(self,text):
#         #embedding = self.embedding.create(input=text, deployment_id=self.model_deployment_name)["data"][0]["embedding"]
#         #embedding = self.embedding.create(input=text,model=self.model_deployment_name).data[0].embedding
#         #embedding = self.embedding.embed_query(text)
#         self.chroma_db = Chroma.from_documents([Document(page_content=text, metadata={})], SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))
#         embedding = self.chroma_db._collection.get(include=['embeddings'])['embeddings']
#         return embedding[0]
# ----------------------------------------------------------------------------------------------------------------------
    def get_embedding_huggingface(self, text):
        embedding = self.model_embedding.encode(text)
        embedding = embedding.tolist()
        return embedding
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
    def from_list_of_dict(self,list_of_dict,select,as_df):
        if as_df:
            df = pd.DataFrame(list_of_dict)
            df = df.iloc[:,[c.find('@')<0 for c in df.columns]]
            if select is not None and df.shape[0]>0:
                df = df[select]
            result = df
        else:
            if isinstance(select, list):
                result = [';'.join([x + ':' + str(r[x]) for x in select]) for r in list_of_dict]
            else:
                if select is not None:
                    result = [r[select] for r in list_of_dict]
                else:
                    result = '\n'.join([str(r) for r in list_of_dict])
        return result
# ----------------------------------------------------------------------------------------------------------------------
    def search_semantic(self, query,select=None,as_df=True,limit=5):
        search_res = self.search_client.search(search_text=query,select=select,top=limit)
        result = self.from_list_of_dict([r for r in search_res],select=select,as_df=as_df)
        return result
# ----------------------------------------------------------------------------------------------------------------------
    def search_hybrid(self, query,field='token',select=None,as_df=True,limit=5):
        vector = self.get_embedding_huggingface(query)
        search_client = self.get_search_client(self.index_name)
        vector_query = VectorizedQuery(fields=field, exhaustive=True,vector=vector)
        search_res = search_client.search(search_text=query,vector_queries=[vector_query],top=limit)
        list_of_dict = [r for r in search_res]
        result = self.from_list_of_dict(list_of_dict, select=select, as_df=as_df)
        return result
# ----------------------------------------------------------------------------------------------------------------------
    def search_vector(self,query,field='token',select=None,as_df=True,limit=4):
        vector = self.get_embedding_huggingface(query)
        search_client = self.get_search_client(self.index_name)
        vector_query = VectorizedQuery(fields=field, exhaustive=True,vector=vector)
        search_res = search_client.search(search_text=None,vector_queries=[vector_query],top=limit)
        list_of_dict = [r for r in search_res]
        result = self.from_list_of_dict(list_of_dict, select=select, as_df=as_df)
        return result
# ----------------------------------------------------------------------------------------------------------------------
