'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
Abdullah Shahzad
CS 251 Data Analysis Visualization
Spring 2023
'''

from mpl_toolkits.mplot3d import Axes3D as mpl
import numpy as np
import matplotlib.pyplot as plt
import palettable
import analysis
import data


class Transformation(analysis.Analysis):

    def __init__(self, orig_dataset, data=None):
        '''Constructor for a Transformation object

        Parameters:
        -----------
        orig_dataset: Data object. shape=(N, num_vars).
            Contains the original dataset (only containing all the numeric variables,
            `num_vars` in total).
        data: Data object (or None). shape=(N, num_proj_vars).
            Contains all the data samples as the original, but ONLY A SUBSET of the variables.
            (`num_proj_vars` in total). `num_proj_vars` <= `num_vars`

        TODO:
        - Pass `data` to the superclass constructor.
        - Create an instance variable for `orig_dataset`.
        '''
        super().__init__(data) 
        self.OD = orig_dataset

    def project(self, headers):
        '''Project the original dataset onto the list of data variables specified by `headers`,
        i.e. select a subset of the variables from the original dataset.
        In other words, your goal is to populate the instance variable `self.data`.

        Parameters:
        -----------
        headers: Python list of str. len(headers) = `num_proj_vars`, usually 1-3 (inclusive), but
            there could be more.
            A list of headers (strings) specifying the feature to be projected onto each axis.
            For example: if headers = ['hi', 'there', 'cs251'], then the data variables
                'hi' becomes the 'x' variable,
                'there' becomes the 'y' variable,
                'cs251' becomes the 'z' variable.
            The length of the list matches the number of dimensions onto which the dataset is
            projected — having 'y' and 'z' variables is optional.

        TODO:
        - Create a new `Data` object that you assign to `self.data` (project data onto the `headers`
        variables). Determine and fill in 'valid' values for all the `Data` constructor
        keyword arguments (except you dont need `filepath` because it is not relevant here).
        '''
        h2c = {}
        for i in range(len(headers)):
            h2c[headers[i]] = i
        data1 = self.OD.select_data(headers)
        newData = data.Data(headers =headers, data = data1, header2col = h2c)
        self.data = newData

       
    def get_data_homogeneous(self):
        '''Helper method to get a version of the projected data array with an added homogeneous
        coordinate. Useful for homogeneous transformations.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected data array with an added 'fake variable'
        column of ones on the right-hand side.
            For example: If we have the data SAMPLE (just one row) in the projected data array:
            [3.3, 5.0, 2.0], this sample would become [3.3, 5.0, 2.0, 1] in the returned array.

        NOTE:
        - Do NOT update self.data with the homogenous coordinate.
        '''
        # getting a data set to edit
        dataSet = self.data.get_all_data()
        # making a column of ones
        homogeneousColumn = np.ones((dataSet.shape[0], 1))
        # stacking them together
        stacked = np.hstack((dataSet, homogeneousColumn))
        return stacked

    def translation_matrix(self, magnitudes):
        ''' Make an M-dimensional homogeneous transformation matrix for translation,
        where M is the number of features in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these
            amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The transformation matrix.

        NOTE: This method just creates the translation matrix. It does NOT actually PERFORM the
        translation!
        '''
        # Making a variable to store the length of the list of magnitudes
        lenMagnitudes = (len(magnitudes))
        # making an id matrix of dims m+1 and m
        identity = np.eye(len(magnitudes)+1,len(magnitudes))
        # looping over the magnitudes and
        for i in range(lenMagnitudes):
            magnitudes[i] = [magnitudes[i]]
        magnitudes.append([1])
        # Making an array of one col with magnitudes
        lastCol = np.array(magnitudes)
        # Stacking the identity with the lastCol of magnitudes
        final = np.hstack((identity,lastCol))
        return final



    def scale_matrix(self, magnitudes):
        '''Make an M-dimensional homogeneous scaling matrix for scaling, where M is the number of
        variables in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The scaling matrix.

        NOTE: This method just creates the scaling matrix. It does NOT actually PERFORM the scaling!
        '''
        # Storing the length of the magnitudes in a variable
        m = (len(magnitudes))
        final = np.eye(m+1, m+1)
        final[np.arange(m), np.arange(m)] = magnitudes
        return final


    def translate(self, magnitudes):
        '''Translates the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The translated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to translate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''
        # creating a matrix to translate the original with
        translater = self.translation_matrix(magnitudes=magnitudes)
        #creating a transpose of self.data
        transposed_data = self.get_data_homogeneous().T
        # performing the multiplication
        new_data = (translater @ transposed_data).T
        # deleting the homogenous coordinate
        new_data = np.delete(new_data, -1, 1)
        # setting self.data to the translated data
        headers = self.data.get_headers()
        dict = self.data.get_mappings()

        self.data = data.Data(data =new_data, headers = headers, header2col=dict)
        return self.data.get_all_data()


    def scale(self, magnitudes):
        '''Scales the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The scaled data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to scale the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        scaler = self.scale_matrix(magnitudes)
        #creating a transpose of self.data
        transposed_data = self.get_data_homogeneous().T
        # performing the multiplication
        new_data = (scaler @ transposed_data).T
        # deleting the homogeneous coordinate
        new_data = np.delete(new_data, len(magnitudes), 1)
        # setting self.data to the translated data
        headers = self.data.get_headers()
        mappings = self.data.get_mappings()
        self.data = data.Data(data = new_data, headers= headers, header2col=mappings)


    def transform(self, C):
        '''Transforms the PROJECTED dataset by applying the homogeneous transformation matrix `C`.

        Parameters:
        -----------
        C: ndarray. shape=(num_proj_vars+1, num_proj_vars+1).
            A homogeneous transformation matrix.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The projected dataset after it has been transformed by `C`

        TODO:
        - Use matrix multiplication to apply the compound transformation matix `C` to the projected
        dataset.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''
        transposed_data = self.get_data_homogeneous().T
        # multiplying the transposed data with C
        product = C @ transposed_data
        # undoing the transposition
        product = product.T
        # deleting the homogeneous coordinate
        product = np.delete(product, -1, 1)
        # returning the final matrix
        return product

        

    def normalize_together(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''
        headers = self.data.get_headers()
        minimum = min(self.min(headers))
        maximum= min(self.max(headers))
        mins = [-minimum]*len(headers)
        multipliers = [1/(maximum - minimum)]*len(headers)
        
        #making matrices for scaling and translation
        data_matrix = self.get_data_homogeneous().T
        t = self.translation_matrix(mins)
        s = self.scale_matrix(multipliers)
        
        #normalizing the data
        normalized = ((s@t)@ data_matrix).T
        final_data = np.delete(normalized, -1, 1)
        mappings = self.data.get_mappings()
        self.data = data.Data(data=final_data, headers=headers, header2col=mappings)

        return self.data.get_all_data()

    def normalize_separately(self):
        '''Normalize each variable separately by translating its local minimum to zero and scaling
        its local range to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''
        headers = self.data.get_headers()
        mins = self.min(headers)
        maxs = self.max(headers)
        scale_by = []

        # getting the diff between max and min for each col
        for i in range(len(mins)):
            difference = maxs[i] - mins[i]
            scale_by.append(1/difference)

        # creating a translation and scaling matrix
        data_matrix = self.get_data_homogeneous().T
        translater = self.translation_matrix([-1*element for element in mins])
        scaler = self.scale_matrix(scale_by)

        #normalizing the data
        normalized = ((scaler @ translater) @ data_matrix).T
        final_normalized = np.delete(normalized, -1, 1)
        h2c = self.data.get_mappings()

        self.data = data.Data(data=final_normalized, headers=headers, header2col=h2c)

        return final_normalized

    def rotation_matrix_3d(self, header, degrees):
        '''Make an 3-D homogeneous rotation matrix for rotating the projected data
        about the ONE axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(4, 4). The 3D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''
        radian = np.deg2rad(degrees)
        # rotation identity matrix
        r = np.eye(4)
        headers = self.data.get_headers()

        if header == headers[0]:
            r[1,1] = np.cos(radian)
            r[1,2] = -np.sin(radian)
            r[2,1] = np.sin(radian)
            r[2,2] = np.cos(radian)
        elif header == headers[1]:
            r[0,0] = np.cos(radian)
            r[2,0] = -np.sin(radian)
            r[0,2] = np.sin(radian)
            r[2,2] = np.cos(radian)
        elif header == headers[2]:
            r[0,0] = np.cos(radian)
            r[0,1] = -np.sin(radian)
            r[1,0] = np.sin(radian)
            r[1,1] = np.cos(radian)

        return r

    def rotate_3d(self, header, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to rotate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        # transposing the data
        transposed_data = self.get_data_homogeneous().T
        # making a rotation matrix to do matrix ops with
        r_matrix = self.rotation_matrix_3d(header, degrees)
        # getting the product of the rotation matrix and the transposed data
        product = r_matrix @ transposed_data
        transposed_product = product.T
        result = np.delete(transposed_product, -1, 1) 
        #setting self.data to the rotated matrix
        headers = self.data.get_headers()
        dict = self.data.get_mappings()
        self.data = data.Data(data = result, headers = headers, header2col = dict)

        return result
    
    def scatter_color(self, ind_var, dep_var, c_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Use a ColorBrewer color palette (e.g. from the `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''
        headers = self.data.get_headers()
        xindex = -10000
        yindex = -10000
        zindex = -10000
        for i in range(len(headers)):
            if headers[i] == ind_var:
                xindex = i
            if headers[i] == dep_var:
                yindex = i
            if headers[i] == c_var:
                zindex = i
        xData = self.data.data[:, xindex]
        yData = self.data.data[:, yindex]
        zData = self.data.data[:, zindex]
        
        figure, axes = plt.subplots()
        scatter = axes.scatter(xData, yData, c=zData, s=75, edgecolors='black')
        axes.set_title(title)
        axes.set_xlabel(ind_var)
        axes.set_ylabel(dep_var)
        colorbar = figure.colorbar(scatter)
        colorbar.set_label(str(c_var))
       
       # EXTENSION 1 
    def scatter3d_color(self, x, y, z, c, title):
        '''Creates a 3D scatter plot to visualize data.

        Axis labels are placed next to the POSITIVE direction of each axis.

        Parameters:
        -----------
        x: (str) variable represented on x axis
        y: (str) variable represented on y axis
        z: (str) variable represented on z axis
        c: (str) varibale represented by color 
        title = String, title of the plot

        Returns
        ----------
        figure: Fig with matplotlib scatterplot.

        '''

        headers = self.data.get_headers()
        x_ind = -10000
        y_ind = -10000
        z_ind = -10000
        c_ind = -10000

        for i in range(len(headers)):
            if headers[i] == x:
                x_ind = i
            elif headers[i] == y:
                y_ind = i
            elif headers[i] == z:
                z_ind = i
            elif headers[i] == c:
                c_ind = i

        x_data = self.data.data[:,x_ind]
        y_data = self.data.data[:,y_ind]
        z_data = self.data.data[:,z_ind]
        c_data = self.data.data[:,c_ind]

        figure = plt.figure(figsize=(12,12))
        axes = figure.add_subplot(111, projection='3d')

        scatter = axes.scatter(x_data, y_data, z_data, s = 100, c=c_data, cmap= plt.hot())
        axes.set_title('3D Plot With Color')
        axes.set_xlabel(str(x))
        axes.set_ylabel(str(y))
        axes.set_zlabel(str(z))

        colorbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.hot()))
        colorbar.set_label(str(c))
        
        return figure, axes

    # Extension 2
    def rotation_matrix_2d(self, degrees):
        ''' Takes an input in degrees and outputs a rotation matrix for the data to be multiplied by

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''
        rad = np.deg2rad(degrees)
        r = np.eye(3)
        headers = self.data.get_headers()

        r[0,0] = np.cos(rad)
        r[0,1] = -np.sin(rad)
        r[1,0] = np.sin(rad)
        r[1,1] = np.cos(rad)

        return r
    
    def rotate_2d(self, angle, center = [0, 0]):
        '''Rotates the projected data about the center by the angle in degrees.

        Parameters:
        -----------
        angle: float. Angle (in degrees) by which the projected dataset should be rotated.
        center: List or tuple [x,y] signifying the coordinates of the point around which the data 
        is rotated

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
        dataset.
        '''
        transposedData = self.get_data_homogeneous().T
        t1 = self.translation_matrix([-center[0], -center[1]])
        t2 = self.translation_matrix([center[0], center[1]])
        r = self.rotation_matrix_2d(angle)
        product = t2 @ r @ t1 @ transposedData
        transposedProduct = product.T
        final = np.delete(transposedProduct, -1, 1) 

        headers = self.data.get_headers()
        dict = self.data.get_mappings()
        self.data = data.Data(data = final, headers = headers, header2col = dict)

        return final