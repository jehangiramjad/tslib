import abc

class Interface(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_time_series(self, name, start, end, value_column="ts", index_col='"rowID"'):
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

    @abc.abstractmethod
    def get_U_row(self, table_name, tsrow_range, models_range):
        """
        query the U matrix from the database table '... U_table' created via the index. the query depend on the ts_row
        range [tsrow_range[0] to tsrow_range[1]] and model range [models_range[0] to models_range[1]] (both inclusive)
        :param table_name: (str) table name in database
        :param tsrow_range:(list of length 2) start and end index  of the range query predicate on ts_row
        :param models_range: (list of length 2) start and end index  of the range query predicate on model_no
        :return: (numpy array) queried values for the selected range
        """
        pass

    @abc.abstractmethod
    def get_V_row(self, table_name, tscol_range):
        """
        query the V matrix from the database table '... V_table' created via the index. the query depend on the ts_col
        range [tscol_range[0] to tscol_range[1]]  (inclusive)
        :param table_name: (str) table name in database
        :param tscol_range:(list of length 2) start and end index  of the range query predicate on ts_col
        :return: (numpy array) queried values for the selected range
        """
        pass

    @abc.abstractmethod
    def get_S_row(self, table_name, models_range):
        """
        query the S matrix from the database table '... s_table' created via the index. the query depend on the model
        range [models_range[0] to models_range[1]] ( inclusive)
        :param table_name: (str) table name in database
        :param models_range: (list of length 2) start and end index  of the range query predicate on model_no
        :return: (numpy array) queried values for the selected range
        """
        pass

    @abc.abstractmethod
    def get_coeff(self, table_name, column):
        """
        query the S matrix from the database materialized view (or table)  '... avergae_coefficients' created via the
        index. the query need to determine only the queried column
        :param table_name: (str) table name in database
        :param column: (str) column name
        :return: (numpy array) queried coefficients for the selected average
        """
        pass

    @abc.abstractmethod
    def create_table(self, table_name,df, primary_key):
        """
        Create table in the database with the same fields as the given pandas dataframe. Rows in the df will be written to
        the newly created table.
        :param table_name: (str) name of the table to be created
        :param df: (Pandas data frame), to determine the schema of the table, as well as the data to be written in the new table
        :param primary_key: (str) primary key of the table
        """
        pass

    def drop_table(self, table_name,):
        """
        Drop table from  database
        :param table_name: (str) name of the table to be created
        """
        pass

    @abc.abstractmethod
    def create_index(self, table_name, column, index_name='',):
        """
        constructs an index on a specified column of the specified table
        :param table_name: (str) the name of the table to be indexed
        :param column: (str) the name of the column to be indexed on
        :param index_name: (str) the name of the index
        """
        pass

    @abc.abstractmethod
    def create_table_from_query(self, table_name, query,):
        """
        Create a new table using the output of a certain query. This is equivalent to a materialized view in
        PostgreSQL and Oracle
        :param table_name: the name of the table to be indexed
        :param query:  query to create table from
        """
        pass

    @abc.abstractmethod
    def execute_query(self, query):
        """
        function that simply passes queries to DB
        :param query: (str) query to be executed
        :return: query output
        """
        pass

    @abc.abstractmethod
    def insert(self,table_name, row):
        """
        create a new row in table_name
        :param table_name: name of an existing table
        :param row: (list) data to be added

        """

    @abc.abstractmethod
    def create_coefficients_average_table(self, table_name, averages, ):
        """
        first write the Query that caluculate  the coefficients average for the coefficient tables, and then use
        create_table_from_query to write table
        :param table_name: the name of the table to be indexed
        :param average_windows:  (list) windows for averages to be calculated (e.g.: [10,20] calc. last ten and 20 models)
        """

        pass
    @abc.abstractmethod
    def bulk_insert(self, table_name, df):
        """
        create new rows in table_name
        :param table_name: name of an existing table
        :param df: (pandas dataframe) data to be added

        """