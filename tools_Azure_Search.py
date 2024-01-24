#https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/sample_vector_search.py
#https://github.com/Azure-Samples/azure-search-python-samples/blob/main/Quickstart/v11/azure-search-quickstart.ipynb
#https://github.com/Azure/azure-search-vector-samples/blob/main/demo-python/code/azure-search-vector-python-sample.ipynb
#----------------------------------------------------------------------------------------------------------------------
import openai
import pandas as pd
import yaml
import uuid
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
from azure.search.documents.models import VectorizedQuery
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

        if filename_config_emb_model is not None:
            with open(filename_config_emb_model, 'r') as config_file:
                self.config_emb = yaml.safe_load(config_file)
                openai.api_type = "azure"
                openai.api_version = self.config_emb['azure']['openai_api_version']
                openai.api_base = self.config_emb['azure']['openai_api_base']
                openai.api_key = self.config_emb['azure']['openai_api_key']


                # self.embedding = AzureOpenAIEmbeddings(
                #     openai_api_version=str(self.config_emb['azure']['openai_api_version']),
                #     openai_api_key=self.config_emb['azure']['openai_api_key'],
                #     azure_endpoint=self.config_emb['azure']['openai_api_base'],
                #     azure_deployment = self.config_emb['azure']['deployment_name']
                # )

                # self.embedding = AzureOpenAIEmbeddings(
                #     api_key=self.config_emb['azure']['openai_api_key'],
                #     api_version=str(self.config_emb['azure']['openai_api_version']),
                #     azure_endpoint=self.config_emb['azure']['openai_api_base']
                # )

                # self.embedding = AzureOpenAI(
                #     api_key=self.config_emb['azure']['openai_api_key'],
                #     api_version=self.config_emb['azure']['openai_api_version'],
                #     azure_endpoint=self.config_emb['azure']['openai_api_base']
                # ).embeddings

                self.model_deployment_name = self.config_emb['azure']['deployment_name']

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
    def create_fields(self, docs, field_embedding):
        df = pd.DataFrame(docs[:1])
        fields = []
        vector_search_dimensions = len(self.get_embedding("Text"))
        for r in range(df.shape[1]):
            name = df.columns[r]
            if r==0:
                field = SimpleField(name=name, type=SearchFieldDataType.String, key=True)
            elif name==field_embedding:
                field = SearchField(name=name, type=SearchFieldDataType.Collection(SearchFieldDataType.Single),searchable=True, vector_search_dimensions=vector_search_dimensions,vector_search_configuration="default")
            else:
                field = SearchableField(name=name, type=SearchFieldDataType.String)
            fields.append(field)


        return fields
# ----------------------------------------------------------------------------------------------------------------------
    def tokenize_documents(self, dct_records, field_source, field_embedding):
        print(len(dct_records))
        for d in dct_records:
            print('.')
            d[field_embedding]=self.get_embedding_chroma(d[field_source])

        return dct_records
# ----------------------------------------------------------------------------------------------------------------------
    def tokenize_documents_chroma(self, dct_records, field_source, field_embedding):

        texts = [d[field_source] for d in dct_records]
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        docs = [Document(page_content=t, metadata={}) for t in texts]
        db = Chroma.from_documents(docs, embedding_function)
        embeddings = db._collection.get(include=['embeddings'])['embeddings']

        for i,d in enumerate(dct_records):
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

        vector_search_dimensions = len(docs[0][field_embedding])
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
    def get_embedding_chroma(self,text):
        #embedding = self.embedding.create(input=text, deployment_id=self.model_deployment_name)["data"][0]["embedding"]
        #embedding = self.embedding.create(input=text,model=self.model_deployment_name).data[0].embedding
        #embedding = self.embedding.embed_query(text)

        self.chroma_db = Chroma.from_documents([Document(page_content=text, metadata={})], SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))
        embedding = self.chroma_db._collection.get(include=['embeddings'])['embeddings']
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
    def upload_documents(self,docs):
        if not isinstance(docs,list):
            docs = [docs]
        result = self.search_client.upload_documents(documents=docs)
        return result[0].succeeded
# ----------------------------------------------------------------------------------------------------------------------
    def delete_document(self,dict_doc):
        self.search_client.delete_documents(documents=[dict_doc])
        return
# ----------------------------------------------------------------------------------------------------------------------
    def search_document(self, query,select=None):
        results = self.search_client.search(search_text=query,select=select)
        df = pd.DataFrame([r for r  in results])
        df = df.iloc[:,[c.find('@')<0 for c in df.columns]]
        if not select is None and df.shape[0]>0:
            df = df[select]

        return df
# ----------------------------------------------------------------------------------------------------------------------
    def search_document_hybrid(self, query,field_embedding,select=None):
        vector = Vector(value=self.get_embedding(query), k=3, fields=field_embedding)
        results = self.search_client.search(search_text=query,vectors=[vector],select=select)
        df = pd.DataFrame([r for r in results])
        df = df.iloc[:, [c.find('@') < 0 for c in df.columns]]
        if not select is None:
            df = df[select]
        return df
# ----------------------------------------------------------------------------------------------------------------------
    def search_texts(self,query,select=None):
        results = self.search_client.search(search_text=query,select=select)
        list_of_dict = [r for r in results]
        if isinstance(select,list):
            texts = [';'.join([x+':'+str(r[x]) for x in select]) for r in list_of_dict]
        else:
            texts = [r[select] for r in list_of_dict]
        return texts
# ----------------------------------------------------------------------------------------------------------------------
    def search_texts_hybrid(self,query,field,select,limit=4):

        #vector = Vector(value=self.get_embedding_chroma(query), fields=field)
        #results = self.search_client.search(search_text=query, vectors=[vector], select=select,top=limit)
        #results = self.search_client.search(search_text=None, vector_queries=[vector], select=select)

        vector = self.get_embedding_chroma(query)[-1]
        vector_query = VectorizedQuery(vector=vector, k_nearest_neighbors=3,fields=field)
        results = self.search_client.search(search_text=None, vector_queries=[vector_query], select=select, top=limit)
        list_of_dict = [r for r in results]
        texts = [r[select] for r in list_of_dict]
        return texts
# ----------------------------------------------------------------------------------------------------------------------