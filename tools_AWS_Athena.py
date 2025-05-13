import uuid
import io
from retrying import retry
import numpy
import os
import boto3
import awswrangler as wr
import pandas as pd
from botocore.exceptions import ClientError
# --------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_Logger
import tools_SQL
# --------------------------------------------------------------------------------------------------------------------
class processor_S3(object):
    def __init__(self,filename_config,folder_out=None):
        self.name = "processor_S3"
        self.load_private_config(filename_config)
        self.folder_out = (folder_out if folder_out is not None else './')
        self.L = tools_Logger.Logger(self.folder_out + 'log.txt')
        boto3.setup_default_session(aws_access_key_id=self.aws_access_key_id,aws_secret_access_key=self.aws_secret_access_key, region_name=self.region)

        self.client_athena = boto3.airsim_client('athena')
        self.client_s3 = boto3.airsim_client('s3')
        self.service_resource = boto3.resource('s3')

        return
# ----------------------------------------------------------------------------------------------------------------------
    def load_private_config(self,filename_in):

        self.aws_access_key_id = None
        self.aws_secret_access_key = None
        self.region = 'us-east-1'

        if not os.path.isfile(filename_in):
            return

        with open(filename_in, 'r') as f:
            for i,line in enumerate(f):
                if i==0:self.aws_access_key_id = str(line.split()[0])
                if i==1:self.aws_secret_access_key = str(line.split()[0])
                if i==2:self.aws_bucket_name = str(line.split()[0])
                if i==3:self.region = str(line.split()[0])
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_all_buckets_stats(self,extended_stats = False,verbose=False):
        objects = self.client_s3.list_buckets()['Buckets']
        V = [[v for v in obj.values()] for obj in objects]
        K = [k for k in objects[0].keys()]
        df = pd.DataFrame(V, columns=K)

        if extended_stats:
            S,N=[],[]
            for v in V:
                df_bucket = self.get_bucket_stats(v[0])
                S.append(df_bucket.iloc[:,1].sum())
                N.append(df_bucket.shape[0])
            df = pd.concat([df,pd.DataFrame({'N':N,'Size':S})],axis=1)

        if verbose:
            print(tools_DF.prettify(df,showindex=False))

        return df
# ----------------------------------------------------------------------------------------------------------------------
    def get_bucket_stats(self,verbose=False,idx_sort=0,ascending=True):

        objects = self.service_resource.Bucket(self.aws_bucket_name).objects.all()
        columns = ['Key','Size','LastModified']
        df = pd.DataFrame([[object.meta.data[c] for c in columns] for object in objects], columns=columns)
        df['Size'] = df['Size'].apply(lambda x:tools_DF.pretty_size(x))
        if verbose:
            df = df.sort_values(by=df.columns[idx_sort],ascending=ascending)
            print(tools_DF.prettify(df, showindex=False))

        return df
# ----------------------------------------------------------------------------------------------------------------------
    def create_bucket(self,bucket_name):
        self.client_s3.create_bucket(Bucket=bucket_name)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def downdload_file(self,filename_out, aws_file_key):
        try:
            with open(filename_out, 'wb') as f:
                self.client_s3.download_fileobj(self.aws_bucket_name, aws_file_key, f)
        except:
            pass

        return True
# ----------------------------------------------------------------------------------------------------------------------
    def downdload_df(self,aws_file_key):
        obj = self.client_s3.get_object(Bucket=self.aws_bucket_name, Key=aws_file_key)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        return df
# ----------------------------------------------------------------------------------------------------------------------
    def upload_file(self, filename_in, aws_file_key):
        try:
            with open(filename_in, "rb") as f:
                self.client_s3.upload_fileobj(f, Bucket=self.aws_bucket_name, Key=aws_file_key)
                #self.client.Object(bucket_name, aws_file_key).put(Body=open(filename_in, 'rb'))
        except ClientError:
            return False
        return True
#----------------------------------------------------------------------------------------------------------------------
    def delete_file(self,aws_file_key):
        self.service_resource.Object(self.aws_bucket_name, aws_file_key).delete()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def delete_files_by_prefix(self,aws_file_prefix):
        bucket = self.service_resource.Bucket(self.aws_bucket_name)
        for obj in bucket.objects.filter(Prefix=aws_file_prefix):
            self.service_resource.Object(bucket.name, obj.key).delete()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def delete_all_files(self):
        df = self.get_bucket_stats()
        for a in numpy.unique([v[0] for v in df.iloc[:, 0].values]):
            self.delete_files_by_prefix(a)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_table(self,schema_name,table_name,limit=10,verbose=False):
        SQL = 'SELECT * FROM %s' % (schema_name+'.'+table_name) + (' limit %d'%limit if limit is not None else '')

        self.L.write(SQL)
        df = wr.athena.read_sql_query(SQL,database=schema_name,ctas_approach=True)
        self.L.write_time(SQL)
        if verbose:
            print(tools_DF.prettify(df.iloc[:limit]))

        return df
# ----------------------------------------------------------------------------------------------------------------------
    def create_view(self, schema_name, table_name, query):
        self.drop_table(schema_name, table_name)
        SQL = 'CREATE VIEW %s AS ' % (schema_name+'.'+table_name) + query
        aws_file_key_res = self.execute_query(SQL,schema_name)

        return aws_file_key_res
# ----------------------------------------------------------------------------------------------------------------------
    def drop_table(self, schema_name, table_name):
        SQL = 'delete_table_if_exists %s.%s' % (schema_name, table_name)
        self.L.write(SQL)
        wr.catalog.delete_table_if_exists(database=schema_name, table=table_name)
        self.L.write_time(SQL)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_databases(self,verbose=False):
        df = wr.catalog.databases()
        if verbose:
            print(tools_DF.prettify(df))
        return df
# ----------------------------------------------------------------------------------------------------------------------
    def get_tables(self,schema_name,verbose=False):
        df = wr.catalog.tables(database=schema_name)
        if verbose:
            print(tools_DF.prettify(df.iloc[:,:3]))
        return df
# ----------------------------------------------------------------------------------------------------------------------
    def transaction_filter(self,schema_name_out,table_name_out,schema_name_in,table_name_in,col_name_in,list_filter,filter_more_less=False,inverce=False):

        batch_size = 1000
        if len(list_filter)<batch_size:
            aws_file_key_res = self.create_view(schema_name_out, table_name_out, tools_SQL.SQL_apply_filter(schema_name_in, table_name_in, col_name_in, list_filter,filter_more_less, inverce))
        else:
            positions = numpy.arange(0, len(list_filter), batch_size)
            temp_scheme_in = schema_name_in
            temp_scheme_out = schema_name_out
            temp_table_in = table_name_in
            i=0
            names = []
            while i<positions.shape[0]:
                temp_table_out = 'v_' + uuid.uuid4().hex
                names.append(temp_table_out)
                if i==0:local_list_filter = list_filter[positions[i]:positions[i+1]]
                elif i<positions.shape[0]-1:local_list_filter = list_filter[positions[i]:positions[i+1]]
                else:local_list_filter = list_filter[positions[-1]:]
                self.create_view(temp_scheme_out, temp_table_out,tools_SQL.SQL_apply_filter(temp_scheme_in, temp_table_in, col_name_in,local_list_filter,filter_more_less,inverce))
                i+=1

            str_unions = ['SELECT * from (%s.%s)'%(schema_name_out,name) for name in names]
            str_union =  'union '.join(str_unions)
            SQL = 'SELECT * from (%s)'%str_union
            aws_file_key_res = self.create_view(schema_name_out, table_name_out, SQL)

        return aws_file_key_res
# ----------------------------------------------------------------------------------------------------------------------

    @retry(stop_max_attempt_number=10,wait_exponential_multiplier=100,wait_exponential_max=1 * 60 * 1000)
    def poll_status(self,query_exec_id):
        result = self.client_athena.get_query_execution(QueryExecutionId=query_exec_id)
        state = result['QueryExecution']['Status']['State']
        if state in ['SUCCEEDED','FAILED']:
            return result
        else:
            raise Exception
        return
# ----------------------------------------------------------------------------------------------------------------------
    def execute_query(self,SQL,schema_name,aws_file_key_out=None):

        do_cleanup = False
        if aws_file_key_out is None:
            aws_file_key_out = uuid.uuid4().hex
            do_cleanup = True

        s3_output = 's3://%s/%s' % (self.aws_bucket_name, aws_file_key_out)
        self.L.write(SQL)
        response = self.client_athena.start_query_execution(QueryString=SQL,QueryExecutionContext={'Database': schema_name},ResultConfiguration={'OutputLocation': s3_output})
        query_exec_id = response['QueryExecutionId']
        self.poll_status(query_exec_id)
        self.L.write_time(SQL)
        if do_cleanup:
            self.delete_files_by_prefix(aws_file_prefix=aws_file_key_out)
            aws_file_key_res = None
        else:
            aws_file_key_res = aws_file_key_out + '/' +response['QueryExecutionId'] + '.csv'

        return aws_file_key_res
# ----------------------------------------------------------------------------------------------------------------------
    def execute_query_wr(self,SQL,schema_name,s3_output=None,verbose=False):
        self.L.write(SQL)
        df = wr.athena.read_sql_query(SQL, database=schema_name, ctas_approach=True,s3_output=s3_output)
        self.L.write_time(SQL)
        if verbose:
            print(tools_DF.prettify(df.iloc[:10]))

        return df
# ----------------------------------------------------------------------------------------------------------------------
    def wrap(self,value,typ):
        value = '\''+str(value)+'\'' if typ in ['string'] else str(value)
        return value
# ----------------------------------------------------------------------------------------------------------------------
    def create_table_from_df_unstaable(self,schema_name,table_name,df):

        dct_typ = {'object':'string','int32':'int','int64':'int'}
        list_of_fields = [col+' '+dct_typ[str(typ)] for col,typ in zip(df.columns, df.dtypes)]
        str_fields = ', '.join(list_of_fields)
        str_prop = '\nROW FORMAT DELIMITED\nFIELDS TERMINATED BY \',\'\nLINES TERMINATED BY \'%cn\''%('\\')
        s3_loc = '\nLOCATION \'s3://%s/\'' % (self.aws_bucket_name)
        SQL = 'CREATE EXTERNAL TABLE IF NOT EXISTS %s.%s(%s) %s %s;'%(schema_name,table_name,str_fields,str_prop,s3_loc)
        #print(SQL)
        self.delete_all_files()
        self.execute_query(SQL, schema_name)
        self.delete_all_files()

        col_names = '('+', '.join([c for c in df.columns]) +')'
        values = []
        for r in range(df.shape[0]):
            row = df.iloc[r,:]
            V = [self.wrap(v,dct_typ[str(typ)]) for v,typ in zip(row.values,df.dtypes)]
            V = '(' + ', '.join(V) + ')'
            values.append(V)
            SQL = 'INSERT INTO %s.%s %s VALUES %s' % (schema_name, table_name,col_names,V)
            print(SQL)
            self.execute_query(SQL, schema_name)

        # all_values = ',\n'.join(values)
        # SQL = 'INSERT INTO %s.%s %s VALUES %s' % (schema_name, table_name,col_names,all_values)
        # print(SQL)
        # self.execute_query(SQL, schema_name)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def create_table_from_df_v2(self,schema_name,table_name,df):
        folder_out = './data/output/'
        df.to_csv(folder_out+'ttt.csv', index=False)
        self.upload_file(folder_out+'ttt.csv', aws_file_key=table_name+'/')
        #self.get_bucket_stats(verbose=True)

        dct_typ = {'object':'string','int32':'int','int64':'int'}
        list_of_fields = [col+' '+dct_typ[str(typ)] for col,typ in zip(df.columns, df.dtypes)]
        str_fields = ', '.join(list_of_fields)
        str_prop = '\nROW FORMAT SERDE \'org.apache.hadoop.hive.serde2.OpenCSVSerde\'\nFIELDS TERMINATED BY \',\'\nLINES TERMINATED BY \'%cn\''%('\\') #+ '\nTBLPROPERTIES(\'skip.header.line.count\'=\'1\')'
        s3_loc = '\nLOCATION \'s3://%s\'' % (self.aws_bucket_name+'/'+table_name)
        SQL = 'CREATE TABLE IF NOT EXISTS %s.%s(%s) %s %s;'%(schema_name,table_name,str_fields,str_prop,s3_loc)
        print(SQL)
        self.execute_query(SQL, schema_name)

        return
# ----------------------------------------------------------------------------------------------------------------------