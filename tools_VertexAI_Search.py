import sys
from PIL import Image as PImage
import datetime
import os
import uuid
from tqdm import tqdm
import pandas as pd
import yaml
import tools_DF
#----------------------------------------------------------------------------------------------------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
#----------------------------------------------------------------------------------------------------------------------
from google.cloud import storage
from google.oauth2 import service_account
from google.cloud import bigquery
from google.auth.transport.requests import Request

from langchain.schema.document import Document
from langchain_google_community import BigQueryVectorSearch
from langchain.vectorstores.utils import DistanceStrategy
from langchain_google_vertexai import VertexAIEmbeddings
#----------------------------------------------------------------------------------------------------------------------
import tools_time_profiler
#----------------------------------------------------------------------------------------------------------------------
class VertexAI_Search(object):
    def __init__(self,filename_config,service_account_file=None,table_name=None):
        with open(filename_config, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_file
        self.credentials = service_account.Credentials.from_service_account_file(service_account_file)

        self.bigqwery_client = bigquery.Client(project=self.config['GCP']['PROJECT_ID'], location=self.config['GCP']['REGION'])
        #self.model_emb_langchain = VertexAIEmbeddings(model_name="textembedding-gecko@latest", project=self.config['GCP']['PROJECT_ID'])
        self.model_emb_langchain = VertexAIEmbeddings(model_name="text-embedding-004", project=self.config['GCP']['PROJECT_ID'])
        self.BQ_dataset = 'my_vector_store'
        self.table_name = table_name
        self.storage_client = storage.Client(project=self.config['GCP']['PROJECT_ID'],credentials=self.credentials)
        self.bucket_name = self.config['GCP']['BUCKET']
        self.bucket = self.storage_client.bucket(self.bucket_name,user_project=self.config['GCP']['PROJECT_ID'])

        self.TP = tools_time_profiler.Time_Profiler()
        print('VertexAI_Search initialized')
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_identity_token(self, target_audience):
        credentials = service_account.IDTokenCredentials.from_service_account_file(os.environ['GOOGLE_APPLICATION_CREDENTIALS'],target_audience=target_audience)
        credentials.refresh(Request())
        return credentials.token
# ----------------------------------------------------------------------------------------------------------------------
    def switch_bucket(self,name):
        self.bucket_name = name
        self.storage_client.bucket(name,user_project=self.config['GCP']['PROJECT_ID'])
# ----------------------------------------------------------------------------------------------------------------------
    def get_embedding(self,text):
        res = self.model_emb_langchain.embed_query(text)
        return res
# ----------------------------------------------------------------------------------------------------------------------
    def upload_blob(self,bucket_name, source_file_name, destination_blob_name):
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        command = 'gsutil cp ' + source_file_name +'gs://' +bucket_name + '/' + destination_blob_name
        # print(command)
        # os.system(command)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def generate_signed_url(self,gs_url,duration_sec=60):
        bucket_name = '/'.join(gs_url.split('gs://')[1].split('/')[:-1])
        destination_blob_name = gs_url.split('gs://')[1].split('/')[-1]

        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        url = blob.generate_signed_url(expiration=datetime.timedelta(seconds=duration_sec),method='GET')
        return url
# ----------------------------------------------------------------------------------------------------------------------
    def create_bucket_if_not_exists(self,bucket_name):
        bucket = self.storage_client.bucket(bucket_name)
        if not bucket.exists():
            bucket.create()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def cleanup_bucket(self,bucket_name):
        bucket = self.storage_client.bucket(bucket_name)
        if bucket.exists():
            blobs = bucket.list_blobs()
            for blob in blobs:
                blob.delete()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def pdf_to_texts(self, filename_pdf, chunk_size=2000):

        loader = PyPDFLoader(filename_pdf)
        pages = loader.load_and_split()
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
        docs = text_splitter.split_documents(pages)
        texts = [t for t in set([doc.page_content for doc in docs])]
        return texts

    # ----------------------------------------------------------------------------------------------------------------------
    def pdf_to_texts_and_images(self, filename_pdf, chunk_size=2000):
        from pdf2image import convert_from_path
        images = convert_from_path(filename_pdf)
        loader = PyPDFLoader(filename_pdf)
        pages = loader.load_and_split()
        texts = [page.page_content for page in pages]

        images = [img.resize((640, int(640 * img.size[1] / img.size[0]))) for img in images]
        images = [PImage.merge("RGB", (img.split())) for img in images]

        return texts, images
# ----------------------------------------------------------------------------------------------------------------------
    def file_to_texts(self, filename_in, chunk_size=2000):
        with open(filename_in, 'r', encoding='utf8', errors='ignore') as f:
            text_document = f.read()
        texts = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=chunk_size,
                                               chunk_overlap=100).create_documents([text_document])
        texts = [text.page_content for text in texts]
        return texts

    # ----------------------------------------------------------------------------------------------------------------------
    def add_book(self,table_name,bucket_name,filename_in,chunk_size=2000,add_images=True,remove_table=False,clean_bucket=False):
        self.TP.tic('add_book')
        self.switch_bucket(bucket_name)
        self.create_bucket_if_not_exists(bucket_name)
        if clean_bucket:
            self.cleanup_bucket(bucket_name)

        is_pdf = (filename_in.split('/')[-1].split('.')[-1].find('pdf') >= 0)

        if is_pdf and add_images:
            texts,images = self.pdf_to_texts_and_images(filename_in,chunk_size=chunk_size)
            metadata_list_of_dict = []
            for i,img in enumerate(images):
                uuid_filename = str(uuid.uuid4()) + '.png'
                img.save(uuid_filename)

                self.upload_blob(self.bucket_name, uuid_filename, uuid_filename)
                os.remove(uuid_filename)
                metadata_list_of_dict.append({'filename':filename_in.split('/')[-1],'url':'gs://'+self.bucket_name+ '/' +uuid_filename})
        else:
            texts = self.pdf_to_texts(filename_in,chunk_size=chunk_size) if is_pdf else self.file_to_texts(filename_in,chunk_size=chunk_size)
            metadata_list_of_dict =[{'filename':filename_in.split('/')[-1]}]*len(texts)

        self.bigqwery_client.create_dataset(dataset=self.BQ_dataset, exists_ok=True)

        table_id = f"{self.config['GCP']['PROJECT_ID']}.{self.BQ_dataset}.{table_name}"
        if remove_table:
            self.bigqwery_client.delete_table(table_id, not_found_ok=True)
        self.bigqwery_client.create_table(table_id, exists_ok=True)

        store = BigQueryVectorSearch(
            project_id=self.config['GCP']['PROJECT_ID'],
            dataset_name=self.BQ_dataset,
            table_name=table_name,
            location=self.config['GCP']['REGION'],
            embedding=self.model_emb_langchain,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        )

        store.add_texts(texts,metadatas=metadata_list_of_dict)
        print(filename_in + ': %d' % len(texts) + ' chunks of %d added to BigQuery table ' % chunk_size + table_name)
        self.TP.print_duration('add_book')
        return
# ----------------------------------------------------------------------------------------------------------------------
    def add_tabular_data(self,table_name,df,col_text,remove_table=False):

        metadata_list_of_dict = [df.iloc[i].to_dict() for i in range(df.shape[0])]
        texts = [(' '.join([str(df[c].iloc[i]) for c in col_text])) for i in range(df.shape[0])]

        self.bigqwery_client.create_dataset(dataset=self.BQ_dataset, exists_ok=True)

        table_id = f"{self.config['GCP']['PROJECT_ID']}.{self.BQ_dataset}.{table_name}"
        if remove_table:
            self.bigqwery_client.delete_table(table_id, not_found_ok=True)
        self.bigqwery_client.create_table(table_id, exists_ok=True)

        store = BigQueryVectorSearch(
            project_id=self.config['GCP']['PROJECT_ID'],
            dataset_name=self.BQ_dataset,
            table_name=table_name,
            location=self.config['GCP']['REGION'],
            embedding=self.model_emb_langchain,
            distance_strategy=DistanceStrategy.COSINE,
        )

        store.add_texts(texts, metadatas=metadata_list_of_dict)
        print('%d' % len(texts) + ' records added to BigQuery table '+ table_name)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def add_images_and_texts(self,table_name,bucket_name,folder_images,image_filenames,texts,distance_strategy='cosine',remove_table=False,clean_bucket=False):

        if distance_strategy is None:d_strat = DistanceStrategy.COSINE
        elif distance_strategy == 'cosine':d_strat = DistanceStrategy.COSINE
        elif distance_strategy == 'euclidean':d_strat = DistanceStrategy.EUCLIDEAN_DISTANCE
        elif distance_strategy == 'max_inner_product':d_strat = DistanceStrategy.MAX_INNER_PRODUCT
        elif distance_strategy == 'dot_product':d_strat = DistanceStrategy.DOT_PRODUCT
        elif distance_strategy == 'jaccard':d_strat = DistanceStrategy.JACCARD
        else:d_strat = DistanceStrategy.COSINE

        self.switch_bucket(bucket_name)
        self.create_bucket_if_not_exists(bucket_name)
        if clean_bucket:
            self.cleanup_bucket(bucket_name)

        metadata_list_of_dict = []

        for filename in tqdm(image_filenames,total=len(image_filenames),desc='Uploading scenes'):
            #self.upload_blob(self.bucket_name, folder_images+filename, filename)
            #os.system(f'gcloud cp {folder_images}{filename} gs://{self.bucket_name}')
            metadata_list_of_dict.append({'filename': filename,'url': 'gs://' + self.bucket_name + '/' + filename})

        table_id = f"{self.config['GCP']['PROJECT_ID']}.{self.BQ_dataset}.{table_name}"
        if remove_table:
            self.bigqwery_client.delete_table(table_id, not_found_ok=True)
        self.bigqwery_client.create_table(table_id, exists_ok=True)

        store = BigQueryVectorSearch(
            project_id=self.config['GCP']['PROJECT_ID'],
            dataset_name=self.BQ_dataset,
            table_name=table_name,
            location=self.config['GCP']['REGION'],
            embedding=self.model_emb_langchain,
            distance_strategy=d_strat
        )

        store.add_texts(texts, metadatas=metadata_list_of_dict)
        print('%d' % len(texts) + ' records added to BigQuery table ' + table_name)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def tbl_exists(self,table_ref):
        try:
            self.bigqwery_client.get_table(table_ref)
            return True
        except:
            return False
# ----------------------------------------------------------------------------------------------------------------------
    def summarize(self,text):
        summary = self.chain.run(question='Do very brief summary', input_documents=[Document(page_content=text, metadata={})])
        return summary
# ----------------------------------------------------------------------------------------------------------------------
    def from_list_of_dict(self,list_of_dict, select, as_df):
        if as_df:
            df = pd.DataFrame(list_of_dict)
            df = df.iloc[:, [c.find('@') < 0 for c in df.columns]]
            if select is not None and df.shape[0] > 0:
                df = df[[s for s in select if s in df.columns]]
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
    def search_vector(self,query,field=None, select=None, as_df=False,distance_strategy='cosine',search_filter_dict=None,limit=6):
        if distance_strategy is None:d_strat = 'COSINE'
        elif distance_strategy == 'cosine':d_strat = 'COSINE'
        elif distance_strategy == 'euclidean':d_strat = 'EUCLIDIAN'
        else:d_strat = 'COSINE'


        # query_vector = self.model_emb_langchain.embed_query(query)
        # store = BigQueryVectorSearch(
        #     project_id=self.config['GCP']['PROJECT_ID'],
        #     dataset_name=self.BQ_dataset,
        #     table_name=self.table_name,
        #     location=self.config['GCP']['REGION'],
        #     embedding=self.model_emb_langchain,
        #     distance_strategy=d_strat
        # )

        emb = str(self.model_emb_langchain.embed_query(query))
        #emb2 = str(self.model_emb_langchain.embed_documents([query])[0])
        #emb3  = str(self.model_emb_langchain.embed(query)[0])

        table_id = f"{self.config['GCP']['PROJECT_ID']}.{self.BQ_dataset}.{self.table_name}"
        SQl = f"SELECT distance, base.content, base.metadata FROM VECTOR_SEARCH(TABLE `{table_id}` , 'text_embedding', (select {emb} as text_embedding), distance_type => '{d_strat}')"

        if search_filter_dict is not None:
            filter_expressions = []
            for i in search_filter_dict.items():
                if isinstance(i[1], float):
                    expr = (
                        "ABS(CAST(JSON_VALUE("
                        f"base.`{self.metadata_field}`,'$.{i[0]}') "
                        f"AS FLOAT64) - {i[1]}) "
                        f"<= {sys.float_info.epsilon}"
                    )
                else:
                    val = str(i[1]).replace('"', '\\"')
                    expr = (
                        f"JSON_VALUE(base.`metadata`,'$.{i[0]}')"
                        f' = "{val}"'
                    )
                filter_expressions.append(expr)
            SQl += " WHERE " + " AND ".join(filter_expressions)

        df_res = self.bigqwery_client.query(SQl).result().to_dataframe(create_bqstorage_client=False)
        df_res = df_res[:limit]




        list_of_texts = [{'text':t, 'score':s} for t,s in zip(df_res['content'].values.tolist(),df_res['distance'].values.tolist())]
        list_of_metadata = [v for v in df_res['metadata'].values]
        for d1,d2 in zip(list_of_texts,list_of_metadata):
            d1.update(d2)

        res = self.from_list_of_dict(list_of_texts, select=select, as_df=as_df)
        return res
# ----------------------------------------------------------------------------------------------------------------------
    def get_stats(self,table_name):
        self.bigqwery_client.create_dataset(dataset=self.BQ_dataset, exists_ok=True)
        table_id = f"{self.config['GCP']['PROJECT_ID']}.{self.BQ_dataset}.{table_name}"

        self.bigqwery_client.create_table(table_id, exists_ok=True)
        xx = self.bigqwery_client.get_table(table_id).schema
        if len(xx)==0:
            return pd.DataFrame([])

        tbl = f"{self.config['GCP']['PROJECT_ID']}.{self.BQ_dataset}.{table_name}"
        df_filenames = self.bigqwery_client.query(f"SELECT metadata.filename FROM {tbl}").result().to_dataframe(create_bqstorage_client=False)
        df_filenames['id']=0
        df_agg = tools_DF.my_agg(df_filenames,'filename','filename',['count'],list_res_names=['#'])[['#', 'filename']]
        df_agg.dropna(inplace=True)
        df_agg.sort_values(by='#',ascending=False,inplace=True)
        return df_agg
# ----------------------------------------------------------------------------------------------------------------------
    def delete_items(self,table_name,filename):
        tbl = f"{self.config['GCP']['PROJECT_ID']}.{self.BQ_dataset}.{table_name}"
        df_doc_ids = self.bigqwery_client.query(f"SELECT doc_id, metadata.filename FROM {tbl}").result().to_dataframe(create_bqstorage_client=False)
        doc_ids = df_doc_ids[df_doc_ids['filename'] == filename]['doc_id'].values.tolist()
        lst = ','.join(['\''+str(doc_id)+'\'' for doc_id in doc_ids])
        exec_q = f'DELETE {tbl} WHERE doc_id in ({lst})'
        self.bigqwery_client.query(exec_q)
        return
# ----------------------------------------------------------------------------------------------------------------------
