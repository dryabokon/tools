import numpy
import pandas as pd
import psycopg2
# ---------------------------------------------------------------------------------------------------------------------
import config
from tools import tools_SQL
from tools.tools_Logger import Logger
# ---------------------------------------------------------------------------------------------------------------------
settings = config.Settings()
# ---------------------------------------------------------------------------------------------------------------------
class Processor:
    def __init__(self) -> None:
        self.connection_wr = None
        self.connection_psycopg2 = psycopg2.connect(
            dbname=settings.redshift_database,
            user=settings.redshift_username,
            password=settings.redshift_password,
            host=settings.redshift_server,
            port=settings.redshift_port
        )

        self.L = Logger(settings.folder_out + 'log.txt')
        self.dct_typ = {
            'object': 'VARCHAR',
            'datetime64[ns]': 'DATE',
            'int32': 'int',
            'int64': 'int',
            'float64': 'float'
        }
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
    def get_percentiles(self,schema_name, table_name, column_name):
        SQL = f'SELECT\
                          PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY {column_name} ) AS percentile_01,\
                          PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY {column_name} ) AS percentile_05,\
                          PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY {column_name} ) AS percentile_95,\
                          PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY {column_name} ) AS percentile_99\
                        FROM {schema_name}.{table_name};'

        df = self.execute_query(SQL=SQL, verbose=False)
        p01,p05,p95,p99 = df.iloc[0,0],df.iloc[0,1],df.iloc[0,2],df.iloc[0,3]
        return p01,p05,p95,p99
# ----------------------------------------------------------------------------------------------------------------------
    def get_histo(self, schema_name, table_name, column_name,n_bins=50):

        p01,p05,p95,p99 = self.get_percentiles(schema_name, table_name, column_name)
        bin_width = (p95-p05)/n_bins

        # histogram_data0 = f'(SELECT {column_name}, \
        #                   FLOOR( {column_name}/{bin_width})*{bin_width} AS bin_start, \
        #                   FLOOR( {column_name}/{bin_width})*{bin_width}+{bin_width} AS bin_end \
        #                   FROM {schema_name}.{table_name} where {p05}<={column_name} and {column_name}< {p95})'

        histogram_data = f'(SELECT {column_name}, \
                          {p05} + FLOOR( ({column_name}-{p05})/{bin_width})*{bin_width} AS bin_start, \
                          {p05} + FLOOR( ({column_name}-{p05})/{bin_width})*{bin_width}+{bin_width} AS bin_end \
                          FROM {schema_name}.{table_name} where {p05}<={column_name} and {column_name}< {p95})'

        histogram_bins = f'(SELECT bin_start,bin_end,COUNT(*) AS {column_name}_count FROM {histogram_data} GROUP BY bin_start, bin_end )'
        SQL0 = f'(SELECT bin_start,bin_end,{column_name}_count FROM {histogram_bins} ORDER BY bin_start)'

        SQL1 = f'(select min({column_name}) as bin_start,{p01} as bin_end,count(*) from {schema_name}.{table_name} where                          {column_name}<{p01})'
        SQL2 = f'(select {p01} as bin_start,{p05} as bin_end,count(*) from {schema_name}.{table_name} where {p01}<={column_name} and {column_name}<{p05})'
        SQL3 = f'(select {p95} as bin_start,{p99} as bin_end,count(*) from {schema_name}.{table_name} where {p95}<={column_name} and {column_name}<{p99})'
        SQL4 = f'(select {p99} as bin_start,max({column_name})  as bin_end,count(*) from {schema_name}.{table_name} where {p99}<={column_name} )'

        SQL = f'select * from {SQL1} UNION ALL {SQL2} UNION ALL {SQL0} UNION ALL {SQL3} UNION ALL {SQL4} order by bin_start'
        df = self.execute_query(SQL)
        return df

# ----------------------------------------------------------------------------------------------------------------------
    def get_corr(self,schema_name, table_name,col1,col2):
        SQL = f'select correlation_coeff from (with t1 as (select {col1},avg({col1}) over() as avg_col1,{col2},avg({col2}) over() as avg_col2 from {schema_name}.{table_name})\
                select sum( ({col1}-avg_col1) *({col2}-avg_col2)) as numerator,sqrt(sum( ({col1}-avg_col1)^2)) * sqrt(sum( ({col2}-avg_col2)^2)) as denominator, numerator/denominator as correlation_coeff from t1)'
        return self.execute_query(SQL).iloc[0,0]
# ----------------------------------------------------------------------------------------------------------------------
    def export_query(self,SQL,filename_out,chunk_size = 10000):

        #N = self.execute_query('select count(*) ' + SQL[SQL.lower().index('from'):]).iloc[0,0]
        offset = 0
        while True:
            #print('%.2f%%: %d/%d'%(offset/N,offset,N))
            print('%d'%(offset))
            df = self.execute_query(f"{SQL} LIMIT {chunk_size} OFFSET {offset}")
            if df.empty:break
            df.to_csv(settings.folder_out+filename_out,index=False,mode=('w' if offset==0 else 'a'),header=(True if offset==0 else False))
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


# WITH histogram_data AS (
# SELECT
# trip_duration_hrs,
# -0.455 AS bin_start,
# 3.49111 AS bin_end
# FROM trip_anomaly.trip_fact),
# histogram_bins AS(SELECT -0.455,3.49111,COUNT(*) AS trip_duration_hrs_count FROM histogram_data GROUP BY bin_start, bin_end)
# SELECT bin_start,bin_end,trip_duration_hrs_count FROM histogram_bins ORDER BY bin_start;























