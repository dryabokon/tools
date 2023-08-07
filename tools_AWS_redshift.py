import pandas as pd
import os
import yaml
import awswrangler as wr
import redshift_connector
import psycopg2
# --------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_Logger
# --------------------------------------------------------------------------------------------------------------------
class processor(object):
    def __init__(self,filename_config,folder_out=None):
        self.connection_wr = None
        self.connection_psycopg2 = None
        self.load_private_config(filename_config)
        self.folder_out = (folder_out if folder_out is not None else './')
        self.L = tools_Logger.Logger(self.folder_out + 'log.txt')

        return
# ----------------------------------------------------------------------------------------------------------------------
    def load_private_config(self,filename_in):

        if filename_in is None:
            return None

        if not os.path.isfile(filename_in):
            return None

        with open(filename_in, 'r') as config_file:
            config = yaml.safe_load(config_file)
            self.connection_wr = redshift_connector.connect(database=config['database']['dbname'], user=config['database']['user'],
                password=config['database']['password'], host=config['database']['host'],port=config['database']['port'])

            self.connection_psycopg2 = psycopg2.connect(dbname=config['database']['dbname'],user=config['database']['user'],
                                          password=config['database']['password'],host=config['database']['host'],port=config['database']['port'])

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
    def execute_query_wr(self,SQL,verbose=False):
        self.L.write(SQL)
        df = wr.redshift.read_sql_query(sql=SQL,con=self.connection_wr)

        self.L.write_time(SQL)
        if verbose:
            print(tools_DF.prettify(df.iloc[:10]))

        return df
# ----------------------------------------------------------------------------------------------------------------------
    def execute_query(self,SQL,verbose=False):
        self.L.write(SQL)
        #cursor = self.connection_psycopg2.cursor()
        df = pd.read_sql_query(SQL, self.connection_psycopg2)
        self.L.write_time(SQL)
        if verbose:
            print(tools_DF.prettify(df.iloc[:10]))

        return df
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