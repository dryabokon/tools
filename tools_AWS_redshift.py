import numpy
import pandas as pd
import os
import yaml
# import redshift_connector
# from sqlalchemy import create_engine
import psycopg2
# --------------------------------------------------------------------------------------------------------------------
import tools_Logger
import tools_SQL
# --------------------------------------------------------------------------------------------------------------------
class processor(object):
    def __init__(self,filename_config_redshift,folder_out=None):
        self.folder_out = (folder_out if folder_out is not None else './')
        self.connection_wr = None
        self.connection_psycopg2 = None

        if filename_config_redshift is None:
            self.init_from_env_variables()
        else:
            self.init_from_private_config(filename_config_redshift)

        self.L = tools_Logger.Logger(self.folder_out + 'log.txt')
        self.dct_typ = {'object':'VARCHAR','datetime64[ns]':'DATE','int32':'int','int64':'int','float64':'float'}
        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_from_private_config(self, filename_in):

        if filename_in is None:
            return None

        if not os.path.isfile(filename_in):
            return None

        with open(filename_in, 'r') as config_file:
            config = yaml.safe_load(config_file)
            #self.engine = create_engine(f'postgresql://{user}:{password}@{host}/{database}')
            # self.connection_wr = redshift_connector.connect(database=config['database']['dbname'], user=config['database']['user'],
            #     password=config['database']['password'], host=config['database']['host'],port=config['database']['port'])
            #
            self.connection_psycopg2 = psycopg2.connect(dbname=config['database']['dbname'],user=config['database']['user'],password=config['database']['password'],host=config['database']['host'],port=config['database']['port'])

        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_from_env_variables(self):
        self.connection_psycopg2 = psycopg2.connect(dbname=os.environ["REDSHIFT_DBNAME"],
                                                    user=os.environ["REDSHIFT_USERNAME"],
                                                    password=os.environ["REDSHIFT_PASSWORD"],
                                                    host=os.environ["REDSHIFT_HOST"], port=os.environ["REDSHIFT_PORT"])
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_databases(self, verbose=False):
        SQL = 'select * from pg_namespace'
        return self.execute_query(SQL, verbose=verbose)
# ----------------------------------------------------------------------------------------------------------------------
    def get_table(self,schema_name,table_name,limit=10,verbose=False):
        SQL = 'SELECT * FROM %s' % (schema_name+'.'+table_name) + (' limit %d'%limit if limit is not None else '')
        return self.execute_query(SQL,verbose=verbose)
# ----------------------------------------------------------------------------------------------------------------------
    def get_tables(self, schema_name, verbose=False):
        SQL = 'select t.table_name from information_schema.tables t where table_schema=\'%s\''%schema_name
        return self.execute_query(SQL,verbose=verbose)
# ----------------------------------------------------------------------------------------------------------------------
    def get_table_structure(self, schema_name, table_name, verbose=False):
        df = self.execute_query('select ordinal_position as position, column_name,data_type from information_schema.columns where table_name = \'%s\' and table_schema = \'%s\' order by ordinal_position'%(table_name,schema_name), verbose=verbose)
        return df
# ----------------------------------------------------------------------------------------------------------------------
    def execute_query_wr(self,SQL,verbose=False):
        import awswrangler as wr
        self.L.write(SQL)
        df = wr.redshift.read_sql_query(sql=SQL,con=self.connection_wr)

        self.L.write_time(SQL)
        if verbose:
            #print(tools_DF.prettify(df.iloc[:10]))
            print(df.head())

        return df
# ----------------------------------------------------------------------------------------------------------------------
    def execute_query(self,SQL,verbose=False):
        self.L.write(SQL)
        #cursor = self.connection_psycopg2.cursor()
        df = pd.read_sql_query(SQL, self.connection_psycopg2)
        self.L.write_time(SQL)
        if verbose:
            #print(tools_DF.prettify(df.iloc[:10]))
            print((df.head()))

        return df
# ----------------------------------------------------------------------------------------------------------------------
    def update_table_from_df(self,df,str_typ,schema_name,table_name,create=False):
        if create:
            SQL1 = tools_SQL.drop_table_from_df(schema_name, table_name)
            SQL2 = tools_SQL.create_table_from_df(schema_name, table_name, df, str_typ)
            self.execute_transaction([SQL1,SQL2])

        chunk_size = 10000
        for offset in numpy.arange(0,df.shape[0],chunk_size):
            #print('%d / %d' % (offset,df.shape[0]))
            SQLs = tools_SQL.insert_to_table_from_df_v2(schema_name, table_name, df.iloc[offset:offset+chunk_size], str_typ)
            self.execute_transaction(SQLs)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def export_query(self,SQL,filename_out,chunk_size = 10000):

        #N = self.execute_query('select count(*) ' + SQL[SQL.lower().index('from'):]).iloc[0,0]
        offset = 0
        while True:
            #print('%.2f%%: %d/%d'%(offset/N,offset,N))
            print('%d'%(offset))
            df = self.execute_query(f"{SQL} LIMIT {chunk_size} OFFSET {offset}")
            if df.empty:break
            df.to_csv(self.folder_out+filename_out,index=False,mode=('w' if offset==0 else 'a'),header=(True if offset==0 else False))
            offset += chunk_size
        return
# ----------------------------------------------------------------------------------------------------------------------
    def execute_transaction(self,SQLs):

        cursor = self.connection_psycopg2.cursor()
        cursor.execute("BEGIN;")
        for SQL in SQLs:
            self.L.write(SQL)
            cursor.execute(SQL)
            self.L.write_time(SQL)

        cursor.execute("COMMIT;")

        return
# ----------------------------------------------------------------------------------------------------------------------
