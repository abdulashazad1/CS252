'''data.py
Reads CSV files, stores data, access/filter data by variable name
Abdullah Shahzad
CS 251 Data Analysis and Visualization
Spring 2023
'''

import numpy as np
import csv

class Data:
    def __init__(self, filepath=None, headers=None, data=None, header2col=None):
        '''Data object constructor

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file
        headers: Python list of strings or None. List of strings that explain the name of each
            column of data.
        data: ndarray or None. shape=(N, M).
            N is the number of data samples (rows) in the dataset and M is the number of variables
            (cols) in the dataset.
            2D numpy array of the dataset’s values, all formatted as floats.
            NOTE: In Week 1, don't worry working with ndarrays yet. Assume it will be passed in
                  as None for now.
        header2col: Python dictionary or None.
                Maps header (var str name) to column index (int).
                Example: "sepal_length" -> 0

        TODO:
        - Declare/initialize the following instance variables:
            - filepath
            - headers
            - data
            - header2col
            - Any others you find helpful in your implementation
        - If `filepath` isn't None, call the `read` method.
        '''
        self.headers = headers
        self.data = data
        self.header2col = header2col
        self.filepath = filepath

        if filepath != None:
            self.filepath = filepath
            self.read(filepath=filepath)
        


    def read(self, filepath):
        '''Read in the .csv file `filepath` in 2D tabular format. Convert to numpy ndarray called
        `self.data` at the end (think of this as 2D array or table).

        Format of `self.data`:
            Rows should correspond to i-th data sample.
            Cols should correspond to j-th variable / feature.

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file

        Returns:
        -----------
        None. (No return value).
            NOTE: In the future, the Returns section will be omitted from docstrings if
            there should be nothing returned

        TODO:
        - Read in the .csv file `filepath` to set `self.data`. Parse the file to only store
        numeric columns of data in a 2D tabular format (ignore non-numeric ones). Make sure
        everything that you add is a float.
        - Represent `self.data` (after parsing your CSV file) as an numpy ndarray. To do this:
            - At the top of this file write: import numpy as np
            - Add this code before this method ends: self.data = np.array(self.data)
        - Be sure to fill in the fields: `self.headers`, `self.data`, `self.header2col`.

        NOTE: You may wish to leverage Python's built-in csv module. Check out the documentation here:
        https://docs.python.org/3/library/csv.html

        NOTE: In any CS251 project, you are welcome to create as many helper methods as you'd like.
        The crucial thing is to make sure that the provided method signatures work as advertised.

        NOTE: You should only use the basic Python library to do your parsing.
        (i.e. no Numpy or imports other than csv).
        Points will be taken off otherwise.

        TIPS:
        - If you're unsure of the data format, open up one of the provided CSV files in a text editor
        or check the project website for some guidelines.
        - Check out the test scripts for the desired outputs.
        '''
        # Opening the file from filepath and reading it
        file = open(filepath, "r") 
        file = file.readlines() 
        # Creating variables to hold different field vals
        cSepRows = []
        indexesOfNumerics = []
        finalData = []
        header2col = {}

        # Getting a list of lists with entries from each column
        for line in file:
            words = line.split(',')
            cSepRows.append(words)
   
        # Finding which rows are numerics
        for i in range(len(cSepRows[1])):
            cSepRows[1][i] = cSepRows[1][i].strip() # Stripping to account for bad spacing
            if cSepRows[1][i] == "numeric":
                indexesOfNumerics.append(i)

        if len(indexesOfNumerics) == 0:
            raise Exception('Data provided has no classification of data type.')
        # Removing all entries that aren't numerical
        for row in cSepRows:
            new_row = []
            for i, item in enumerate(row):
                if i in indexesOfNumerics:
                    new_row.append(item)
            row[:] = new_row  # replace the old row with the new row
            finalData.append(row)
        
        # Adding code to set header2col
        
        for i in range(len(cSepRows[0])):
            header2col[cSepRows[0][i].strip()] = i
        
        self.header2col = header2col # Setting the header2col field
        self.headers = finalData[0] # Setting the headers field
        
        #Removing the first two rows because 
        finalData.remove(finalData[0]) 
        finalData.remove(finalData[0])


        for i in range(len(finalData)):
            for j in range(len(row)):
                finalData[i][j] = float(finalData[i][j])

        self.data = finalData
        self.data = np.array(self.data) 
        #print(self.data)

    def __str__(self):
        '''toString method

        (For those who don't know, __str__ works like toString in Java...In this case, it's what's
        called to determine what gets shown when a `Data` object is printed.)

        Returns:
        -----------
        str. A nicely formatted string representation of the data in this Data object.
            Only show, at most, the 1st 5 rows of data
            See the test code for an example output.
        '''
        str1 = ''
        str2 = ''
        max_widths = []

        if self.headers is None:
            print("No data")

        else:
            for i in range(len(self.headers)):
                str1 += self.headers[i] + "        "
            
            if len(self.data)>5:
                for k in range(5):
                    for j in range(len(self.data[0])):
                        str2 += str(self.data[k][j]) + " "
                    str2 += "\n"
            else:
                for k in range(len(self.data)):
                    for j in range(len(self.data[0])):
                        str2 += str(self.data[k][j]) + " "
                    str2 += "\n"
            lines = str2.split('\n')

            # get the maximum width of each column
            
            for line in lines:
                max_w = 0
                for word in line.split(","):
                    if len(word) > max_w:
                        max_w = len(word)
                    
                max_widths.append(max_w)

            # format each line with evenly spaced values
            formatted_lines = ["".join(word.ljust(max_widths[i]) for i, word in enumerate(line.split())) for line in lines]

            # join the lines with newline characters
            grid = "\n".join(formatted_lines)

            # print the resulting grid

            return str1 + "\n" + grid
    def get_headers(self):
        '''Get method for headers

        Returns:
        -----------
        Python list of str.
        '''
        return self.headers

    def get_mappings(self):
        '''Get method for mapping between variable name and column index

        Returns:
        -----------
        Python dictionary. str -> int
        '''
        return self.header2col

    def get_num_dims(self):
        '''Get method for number of dimensions in each data sample

        Returns:
        -----------
        int. Number of dimensions in each data sample. Same thing as number of variables.
        '''
        return len(self.headers)

    def get_num_samples(self):
        '''Get method for number of data points (samples) in the dataset

        Returns:
        -----------
        int. Number of data samples in dataset.
        '''
        return len(self.data)

    def get_sample(self, rowInd):
        '''Gets the data sample at index `rowInd` (the `rowInd`-th sample)

        Returns:
        -----------
        ndarray. shape=(num_vars,) The data sample at index `rowInd`
        '''
        return self.data[rowInd]

    def get_header_indices(self, headers):
        '''Gets the variable (column) indices of the str variable names in `headers`.

        Parameters:
        -----------
        headers: Python list of str. Header names to take from self.data

        Returns:
        -----------
        Python list of nonnegative ints. shape=len(headers). The indices of the headers in `headers`
            list.
        '''
        header_indices = [] 
        for header in headers:
            if header in self.header2col: # if header is found in header2col map
                header_indices.append(self.header2col[header]) # append column index of header into list of header_indices
        return header_indices

    def get_all_data(self):
        '''Gets a copy of the entire dataset

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_data_samps, num_vars). A copy of the entire dataset.
            NOTE: This should be a COPY, not the data stored here itself.
            This can be accomplished with numpy's copy function.
        '''
        return np.copy(self.data)

    def head(self):
        '''Return the 1st five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). 1st five data samples.
        '''

        numRow = min(len(self.data), 5)
        return self.data[:numRow, :] 

    def tail(self):
        '''Return the last five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). Last five data samples.
        '''
        numRow = min(len(self.data), 5)
        return self.data[-numRow:, :] # start from numRowth row until the end of self.data

    def limit_samples(self, start_row, end_row):
        '''Update the data so that this `Data` object only stores samples in the contiguous range:
            `start_row` (inclusive), end_row (exclusive)
        Samples outside the specified range are no longer stored.

        (Week 2)

        '''
        self.data = self.data[start_row:end_row, :] # only rows in the specified range, all columns

    def select_data(self, headers, rows=[]):
        '''Return data samples corresponding to the variable names in `headers`.
        If `rows` is empty, return all samples, otherwise return samples at the indices specified
        by the `rows` list.

        (Week 2)

        For example, if self.headers = ['a', 'b', 'c'] and we pass in header = 'b', we return
        column #2 of self.data. If rows is not [] (say =[0, 2, 5]), then we do the same thing,
        but only return rows 0, 2, and 5 of column #2.

        Parameters:
        -----------
            headers: Python list of str. Header names to take from self.data
            rows: Python list of int. Indices of subset of data samples to select.
                Empty list [] means take all rows

        Returns:
        -----------
        ndarray. shape=(num_data_samps, len(headers)) if rows=[]
                 shape=(len(rows), len(headers)) otherwise
            Subset of data from the variables `headers` that have row indices `rows`.

        Hint: For selecting a subset of rows from the data ndarray, check out np.ix_
        '''
        
        cols = []
        headerList = self.header2col
        for header in headers:
            cols.append(headerList[header])
        if rows != []:
            
            return self.data[np.ix_(rows, cols)]
        else:
            return self.data[:, cols]
