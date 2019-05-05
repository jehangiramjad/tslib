from tslib.src.database.DBInterface import Interface
import psycopg2
from sqlalchemy import create_engine
import numpy as np
import io


class SqlImplementation(Interface):
    def __init__(self, driver="postgresql", host="localhost", database="querytime_test", user="aalomar",
                 password="AAmit32lids"):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.engine = create_engine(driver + '://' + user + ':' + password + '@' + host + '/' + database)

    def get_time_series(self, name, start, end, value_column="ts", index_col='"rowID"', Desc=False):
        """
        query time series table to return time series values from a certain range  [start to end]
        or all values with time stamp/index greter than start  (if end is None)
        :param name: (str) table (time series) name in database
        :param start: (int, timestamp) start index (timestamp) of the range query
        :param end: (int, timestamp) last index (timestamp) of the range query
        :param value_column: name of column than contain time series value
        :param index_col: name of column that contains time series index/timestamp
        :return:
        """
        if end is None:
            sql = 'Select ' + value_column + " from  " + name + " where " + index_col + " >= %s order by "+index_col
            result = self.engine.execute(sql, (start)).fetchall()
        else:
            if not Desc:
                sql = 'Select ' + value_column + " from  " + name + " where " + index_col + " >= %s and " + index_col + " <= %s order by " + index_col
            else:
                sql = 'Select ' + value_column + " from  " + name + " where " + index_col + " >= %s and " + index_col + " <= %s order by " + index_col + ' Desc'
            result = self.engine.execute(sql, (start, end)).fetchall()

        return np.array(result)

    def get_U_row(self, table_name, tsrow_range, models_range,k, return_modelno = False):

        """
        query the U matrix from the database table '... U_table' created via the index. the query depend on the ts_row
        range [tsrow_range[0] to tsrow_range[1]] and model range [models_range[0] to models_range[1]] (both inclusive)
        :param table_name: (str) table name in database
        :param tsrow_range:(list of length 2) start and end index  of the range query predicate on ts_row
        :param models_range: (list of length 2) start and end index  of the range query predicate on model_no
        :return: (numpy array) queried values for the selected range
        """
        columns = 'u'+ ',u'.join([str(i) for i in range(1, k + 1)])
        if return_modelno :
            columns = 'modelno, '+columns
        query = "SELECT "+ columns +" FROM " + table_name + " WHERE tsrow >= %s and tsrow <= %s and (modelno >= %s and modelno <= %s); "
        result = self.engine.execute(query,
                                     (tsrow_range[0], tsrow_range[1], models_range[0], models_range[1],)).fetchall()
        return np.array(result)
        pass

    def get_V_row(self, table_name, tscol_range,k,models_range = None, return_modelno = False):
        """
        query the V matrix from the database table '... V_table' created via the index. the query depend on the ts_col
        range [tscol_range[0] to tscol_range[1]]  (inclusive)
        :param table_name: (str) table name in database
        :param tscol_range:(list of length 2) start and end index  of the range query predicate on ts_col
        :return: (numpy array) queried values for the selected range
        """
        columns = 'v'+ ',v'.join([str(i) for i in range(1, k + 1)])
        if return_modelno :
            columns = 'modelno, '+columns
        if models_range is None:
            query = "SELECT "+ columns +" FROM " + table_name + " WHERE tscolumn >= %s and tscolumn <= %s ; "
            result = self.engine.execute(query, (tscol_range[0], tscol_range[1],)).fetchall()
        else:
            query = "SELECT " + columns + " FROM " + table_name + " WHERE tscolumn >= %s and tscolumn <= %s and (modelno >= %s and modelno <= %s); "
            result = self.engine.execute(query, (tscol_range[0], tscol_range[1],models_range[0], models_range[1],)).fetchall()
        return np.array(result)
        pass

    def get_S_row(self, table_name, models_range, k, return_modelno = False):
        """
        query the S matrix from the database table '... s_table' created via the index. the query depend on the model
        range [models_range[0] to models_range[1]] ( inclusive)
        :param table_name: (str) table name in database
        :param models_range: (list of length 2) start and end index  of the range query predicate on model_no
        :return: (numpy array) queried values for the selected range
        """
        columns = 's'+ ',s'.join([str(i) for i in range(1,k+1)])
        if return_modelno :
            columns = 'modelno, '+columns
        query = "SELECT "+ columns +" FROM " + table_name + " WHERE modelno >= %s and modelno <= %s;"
        result = self.engine.execute(query, (models_range[0], models_range[1],)).fetchall()
        return np.array(result)

        pass

    def get_coeff(self, table_name, column):

        """
        query the S matrix from the database materialized view (or table)  '... avergae_coefficients' created via the
        index. the query need to determine only the queried column
        :param table_name: (str) table name in database
        :param column: (str) column name
        :return: (numpy array) queried coefficients for the selected average
        """
        query = "SELECT %s from %s order by %s" %(column , table_name, column)
        result = self.engine.execute(query).fetchall()
        return np.array(result)

    def create_table(self, table_name, df, primary_key=None, load_data=True,replace_if_exists = True , include_index=True,
                     index_label="row_id"):
        """
        Create table in the database with the same fields as the given pandas dataframe. Rows in the df will be written to
        the newly created table.
        :param table_name: (str) name of the table to be created
        :param df: (Pandas data frame), to determine the schema of the table, as well as the data to be written in the new table
        :param primary_key: (str) primary key of the table
        :param load_data: (str) primary key of the table
        """
        # drop table if exists:
        if replace_if_exists:
            self.drop_table(table_name)

        # create table in database
        conn = self.engine.raw_connection()
        cur = conn.cursor()
        df.head(0).to_sql(table_name, self.engine, index=include_index, index_label=index_label)

        query = "ALTER TABLE  %s ADD PRIMARY KEY (%s);" % (table_name, primary_key)
        if primary_key is not None:
            self.engine.execute(query)
        # load content
        if load_data:
            output = io.BytesIO()
            df.to_csv(output, sep='\t', header=False, index=True, index_label=index_label)
            output.seek(0)
            cur.copy_from(output, table_name, null="")  # null values become ''
        conn.commit()
        conn.close()

    def drop_table(self, table_name):
        """
        Drop table from the database and all its dependents
        :param table_name: (str) name of the table to be created
        :param drop dependency : """

        query = " DROP TABLE IF EXISTS " +table_name + " Cascade; "

        self.engine.execute(query)

    def create_index(self, table_name, column, index_name='', ):
        """
        constructs an index on a specified column of the specified table
        :param table_name: (str) the name of the table to be indexed
        :param column: (str) the name of the column to be indexed on
        :param index_name: (str) the name of the index
        """
        query = 'CREATE INDEX %s ON %s (%s);' % (index_name  ,table_name,column)
        self.engine.execute( query)

    def create_coefficients_average_table(self, table_name, created_table_name, averages,max_model ):
        """
        first write the Query that caluculate  the coefficients average for the coefficient tables, and then use
        create_table_from_query to write table
        :param table_name: the name of the table to be queried
        :param created_table_name: the name of the table to be created
        :param average_windows:  (list) windows for averages to be calculated (e.g.: [10,20] calc. last ten and 20 models)
        :param max_model:  (int) index of the latest model
        """

        s1 =  'SELECT coeffpos, avg(coeffvalue) as average,'
        s_a =  'avg(coeffvalue) FILTER (WHERE modelno <= %s and modelno > %s -%s) as Last%s'
        predicates = (',').join([s_a %(max_model,max_model,i,i) for i in averages])
        query = s1+ predicates + ' FROM %s group by coeffpos' %table_name
        self.create_table_from_query(created_table_name, query)

    def create_table_from_query(self, table_name, query, ):
        """
        Create a new table using the output of a certain query. This is equivalent to a materialized view in
        PostgreSQL and Oracle
        :param table_name: the name of the table to be indexed
        :param query:  query to create table from
        """
        query = 'CREATE MATERIALIZED VIEW %s AS '%table_name+ query
        self.engine.execute( query)
        pass

    def execute_query(self, query):
        """
        function that simply passes queries to DB
        :param query: (str) query to be executed
        :return: query output
        """
        return self.engine.execute(query).fetchall()

    def insert(self, table_name, row):
        """
        create a new row in table_name
        :param table_name: name of an existing table
        :param row: (list) data to be added
        """
        pass

    def bulk_insert(self, table_name, df, include_index=True, index_label="row_id"):
        """
        create new rows in table_name The function assumes that the df headers are the same as the table's column names
        :param table_name: name of an existing table
        :param df: (pandas dataframe) data to be added
        """
        conn = self.engine.raw_connection()
        cur = conn.cursor()
        output = io.BytesIO()
        df.to_csv(output, sep='\t', header=False, index=include_index, index_label=index_label)
        output.seek(0)
        cur.copy_from(output, table_name, null="")