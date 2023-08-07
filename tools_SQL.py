import numpy
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
def SQL_my_agg(schema_name,table_name,cols_groupby,cols_value, aggs,list_res_names=None,order_idx=None,ascending=True):

    sep =' '
    dct_agg = {}
    for v, a in zip(cols_value, aggs):
        if a == 'top': a = (lambda x: x.value_counts().index[0] if any(x.value_counts()) else numpy.nan)
        if isinstance(a, list):
            dct_agg[v] = [item for item in a]
        else:
            dct_agg[v] = a

    list_agg = []
    for value, agg in zip(dct_agg.keys(), dct_agg.values()):
        for a in agg:
            list_agg.append('%s'%a +'(' +'%s'%value + ')')

    list_new_names = []

    if list_res_names is not None:
        for i,res_name in enumerate(list_res_names):
            list_new_names.append(res_name)
    else:
        for (value, agg) in zip(dct_agg.keys(), dct_agg.values()):
            for a in agg:
                list_new_names.append(value+a)

    for i, new_name in enumerate(list_new_names):
        list_agg[i] = list_agg[i] + ' as ' + new_name

    list_agg = cols_groupby + list_agg
    list_new_names = cols_groupby + list_new_names

    str_select =  ', '.join(list_agg)
    str_groupby = ', '.join(cols_groupby)
    str_order = sep + 'ORDER BY %s %s'%(list_new_names[order_idx],'asc' if ascending else 'desc') if order_idx is not None else ''

    SQL = 'SELECT ' + str_select + sep + 'FROM %s'%(schema_name+'.'+table_name) + sep + 'GROUP BY ' + str_groupby + str_order
    return SQL
# ----------------------------------------------------------------------------------------------------------------------
def SQL_construct_conditional_clause(col_name,filter,filter_more_less=False):
    sep = ' '
    if not isinstance(filter,list):
        filter = [filter]

    filter = ["'" + v + "'" if v is not None else None for v in filter]

    str_condition = ""
    if filter_more_less and len(filter) == 2:
        if filter[0] is not None:
            str_condition = '%s'%col_name + ' >= ' + filter[0]
        if filter[1] is not None:
            if len(str_condition)>0:
                str_condition+=' AND' + sep
            str_condition+= '%s'%col_name + ' < ' + filter[1]
    else:
        list_in = ', '.join(filter)
        str_condition = '%s'%col_name + ' in (' + list_in + ')'

    return str_condition
# ----------------------------------------------------------------------------------------------------------------------
def SQL_apply_filter(schema_name,table_name,list_col_name,list_filter,filter_more_less=False,inverce=False):
    sep = ' '
    if not isinstance(list_col_name, (list, tuple, numpy.ndarray, pd.Series)):
        list_col_name = [list_col_name]
        list_filter = [list_filter]

    str_yes_no = (' NOT ' if inverce else '')
    str_conditions = [SQL_construct_conditional_clause(col_name, filter, filter_more_less) for col_name, filter in zip(list_col_name,list_filter)]
    str_conditions = ' AND '.join(str_conditions)

    SQL = 'SELECT *' + sep + 'FROM %s'%(schema_name+'.'+table_name) + sep + 'WHERE' + str_yes_no + ' (' + str_conditions + ')'
    return SQL
# ----------------------------------------------------------------------------------------------------------------------
def SQL_where( schema_name, table_name, where_clause):
    SQL = 'SELECT * ' + 'FROM %s' % (schema_name + '.' + table_name) + ' WHERE ' + where_clause
    return SQL
# ----------------------------------------------------------------------------------------------------------------------
def SQL_fetch(self,schema_name_out,table_name_out,col_name_fetch_out,schema_name_in,table_name_in,col_name_fetch_in,col_value_in):
    sep = ' '
    str_what = '%s.%s.*,%s.%s.%s'%(schema_name_out,table_name_out,schema_name_in,table_name_in,col_value_in.lower())
    str_from = '%s.%s'%(schema_name_out,table_name_out)
    str_join = '%s.%s'%(schema_name_in,table_name_in)
    str_on = '%s.%s=%s.%s'%(table_name_out,col_name_fetch_out.lower(),table_name_in,col_name_fetch_in.lower())
    SQL = ('SELECT %s'+sep+'FROM %s' + sep+ 'LEFT OUTER JOIN %s'+sep+'ON %s')%(str_what,str_from,str_join,str_on)
    return SQL
# ----------------------------------------------------------------------------------------------------------------------
