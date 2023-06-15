'''linear_regression.py
Subclass of Analysis that performs linear regression on data
Abdullah Shahzad
CS 252 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import math

import analysis


class LinearRegression(analysis.Analysis):
    '''
    Perform and store linear regression and related analyses
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        super().__init__(data)

        # ind_vars: Python list of strings.
        #   1+ Independent variables (predictors) entered in the regression.
        self.ind_vars = None
        # dep_var: string. Dependent variable predicted by the regression.
        self.dep_var = None

        # A: ndarray. shape=(num_data_samps, num_ind_vars)
        #   Matrix for independent (predictor) variables in linear regression
        self.A = None

        # y: ndarray. shape=(num_data_samps, 1)
        #   Vector for dependent variable predictions from linear regression
        self.y = None

        # R2: float. R^2 statistic
        self.R2 = None

        # Mean squared error (MSE). float. Measure of quality of fit
        self.mse = None

        # slope: ndarray. shape=(num_ind_vars, 1)
        #   Regression slope(s)
        self.slope = None
        # intercept: float. Regression intercept
        self.intercept = None
        # residuals: ndarray. shape=(num_data_samps, 1)
        #   Residuals from regression fit
        self.residuals = None

        # p: int. Polynomial degree of regression model (Week 2)
        self.p = 1

    def linear_regression(self, ind_vars, dep_var, method='scipy', p=1):
        '''Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var` using the method `method`.

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        method: str. Method used to compute the linear regression. Here are the options:
            'scipy': Use scipy's linregress function.
            'normal': Use normal equations.
            'qr': Use QR factorization

        TODO:
        - Use your data object to select the variable columns associated with the independent and
        dependent variable strings.
        - Perform linear regression using the appropriate method.
        - Compute R^2 on the fit and the residuals.
        - By the end of this method, all instance variables should be set (see constructor).

        NOTE: Use other methods in this class where ever possible (do not write the same code twice!)
        '''
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        #select data from dataset
        self.A = self.data.select_data(self.ind_vars)
        self.y = self.data.select_data([self.dep_var])

        #calculate linear regression based on method indicated
        if method == "scipy":
            result = self.linear_regression_scipy(self.A, self.y)
        elif method == "normal":
            print("norm")
            result = self.linear_regression_normal(self.A, self.y)
        elif method == "qr":
            result = self.linear_regression_qr(self.A, self.y)
        
        #assign intercept and slope
        self.intercept = np.squeeze(result[0])
        self.slope = result[1:, :]

        #create predictions using linear regression
        y_pred = self.predict(self.A)

        #calculate R2
        self.R2 = self.r_squared(y_pred)

        #calculate residuals
        self.residuals = self.compute_residuals(y_pred)

        #calculate MSE
        self.mse = self.compute_mse()
        

    def linear_regression_scipy(self, A, y):
        '''Performs a linear regression using scipy's built-in least squares solver (scipy.linalg.lstsq).
        Solves the equation y = Ac for the coefficient vector c.

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1, 1)
            Linear regression slope coefficients for each independent var PLUS the intercept term
        '''
        # A prime
        A_prime = np.hstack((np.ones((A.shape[0], 1)), A))
        # Using the scipy lin reg solver
        c, residues, rank, s= scipy.linalg.lstsq(A_prime, y)
        return c

    def linear_regression_normal(self, A, y):
        '''Performs a linear regression using the normal equations.
        Solves the equation y = Ac for the coefficient vector c.

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1, 1)
            Linear regression slope coefficients for each independent var AND the intercept term
        '''
        A_prime = np.hstack((np.ones((A.shape[0], 1)), A))
        A_transp = A_prime.T
        prod_A = A_transp @ A_prime

        prod_inverse = np.linalg.inv(prod_A)

        c = prod_inverse @ A_transp @ y

        return c

    def linear_regression_qr(self, A, y):
        '''Performs a linear regression using the QR decomposition

        (Week 2)

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1, 1)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: You should not compute any matrix inverses! Check out scipy.linalg.solve_triangular
        to backsubsitute to solve for the regression coefficients `c`.
        '''
        # eqn = Ac = y , QRc = y
        ones = np.ones((A.shape[0], 1))
        A_h = np.hstack((ones, A))

        Q, R = self.qr_decomposition(A_h)

        return scipy.linalg.solve_triangular(R, (Q.T @ y))

    def qr_decomposition(self, A):
        '''Performs a QR decomposition on the matrix A. Make column vectors orthogonal relative
        to each other. Uses the Gram–Schmidt algorithm

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars+1).
            Data matrix for independent variables.

        Returns:
        -----------
        Q: ndarray. shape=(num_data_samps, num_ind_vars+1)
            Orthonormal matrix (columns are orthogonal unit vectors — i.e. length = 1)
        R: ndarray. shape=(num_ind_vars+1, num_ind_vars+1)
            Upper triangular matrix

        TODO:
        - Q is found by the Gram–Schmidt orthogonalizing algorithm.
        Summary: Step thru columns of A left-to-right. You are making each newly visited column
        orthogonal to all the previous ones. You do this by projecting the current column onto each
        of the previous ones and subtracting each projection from the current column.
            - NOTE: Very important: Make sure that you make a COPY of your current column before
            subtracting (otherwise you might modify data in A!).
        Normalize each current column after orthogonalizing.
        - R is found by equation summarized in notebook
        '''
        N = A.shape[0]
        M = A.shape[1]
        Q = np.zeros((N, M))

        for i in range(M):
            col_i = np.copy(A[:, i])
            for j in range(0,i):
                col_j = Q[:, j]
                col_i = col_i - (np.dot(col_i, col_j) * col_j)
            col_i = col_i/np.linalg.norm(col_i)
            Q[:, i] = col_i

        R = Q.T @ A

        return Q, R
        


    def predict(self, X=None):
        '''Use fitted linear regression model to predict the values of data matrix self.A.
        Generates the predictions y_pred = mA + b, where (m, b) are the model fit slope and intercept,
        A is the data matrix.

        Parameters:
        -----------
        X: ndarray. shape=(num_data_samps, num_ind_vars).
            If None, use self.A for the "x values" when making predictions.
            If not None, use X as independent var data as "x values" used in making predictions.

        Returns
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1)
            Predicted y (dependent variable) values

        NOTE: You can write this method without any loops!
        '''
        if X is None:
            y_pred = (self.A @ self.slope) + self.intercept
        elif self.p>1:
            X = self.make_polynomial_matrix(X,self.p)
            y_pred = self.intercept + X @ self.slope
        else:
            y_pred = (X @ self.slope) + self.intercept

        return y_pred

    def r_squared(self, y_pred):
        '''Computes the R^2 quality of fit statistic

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps,).
            Dependent variable values predicted by the linear regression model

        Returns:
        -----------
        R2: float.
            The R^2 statistic
        '''
        y_mean = np.mean(self.y, axis = 0)
        y_diff = self.y - y_mean
        sq_y_diff = np.square(y_diff)
        smd = np.sum(sq_y_diff, axis=0)
        
        y_pred_diff = self.compute_residuals(y_pred)
        sq_y_pred_diff = np.square(y_pred_diff)
        sum_sq = np.sum(sq_y_pred_diff, axis=0)

        R2 = 1 - (sum_sq / smd)
        self.R2 = R2[0]
        return R2[0]

    def compute_residuals(self, y_pred):
        '''Determines the residual values from the linear regression model

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1).
            Data column for model predicted dependent variable values.

        Returns
        -----------
        residuals: ndarray. shape=(num_data_samps, 1)
            Difference between the y values and the ones predicted by the regression model at the
            data samples
        '''
        residuals = self.y - y_pred
        return residuals

    def compute_mse(self):
        '''Computes the mean squared error in the predicted y compared the actual y values.
        See notebook for equation.

        Returns:
        -----------
        float. Mean squared error

        Hint: Make use of self.compute_residuals
        '''
        y_mean_diff = self.compute_residuals(self.predict())
        squared_data = np.square(y_mean_diff)
        mean_sq_data = np.mean(squared_data, axis=0)
        return mean_sq_data[0]

    def scatter(self, ind_var, dep_var, title):
        '''Creates a scatter plot with a regression line to visualize the model fit.
        Assumes linear regression has been already run.

        Parameters:
        -----------
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        title: string. Title for the plot

        TODO:
        - Use your scatter() in Analysis to handle the plotting of points. Note that it returns
        the (x, y) coordinates of the points.
        - Sample evenly spaced x values for the regression line between the min and max x data values
        - Use your regression slope, intercept, and x sample points to solve for the y values on the
        regression line.
        - Plot the line on top of the scatterplot.
        - Make sure that your plot has a title (with R^2 value in it)
        '''
        # Setting the title to title parameter + R2 val
        title = title + " R2: " + str(self.R2)
    
        x, y = super().scatter(ind_var, dep_var, title)
        
        minX = self.A[:, 0].min()
        maxX = self.A[:, 0].max()
        spacedX = np.linspace(minX, maxX)

        if self.p == 1:
            yValues = self.slope[0] * spacedX + self.intercept
        else:
            spacedX_polyReg = self.make_polynomial_matrix(spacedX, self.p)
            yValues = self.intercept + (np.sum((spacedX_polyReg @ self.slope), axis = 1))

        plt.plot(spacedX, yValues)

    def pair_plot(self, data_vars, fig_sz=(15, 15), hists_on_diag=True):
        '''Makes a pair plot with regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        Parameters:
        -----------
        data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!
        hists_on_diag: bool. If true, draw a histogram of the variable along main diagonal of
            pairplot.

        TODO:
        - Use your pair_plot() in Analysis to take care of making the grid of scatter plots.
        Note that this method returns the figure and axes array that you will need to superimpose
        the regression lines on each subplot panel.
        - In each subpanel, plot a regression line of the ind and dep variable. Follow the approach
        that you used for self.scatter. Note that here you will need to fit a new regression for
        every ind and dep variable pair.
        - Make sure that each plot has a title (with R^2 value in it)
        '''
        fig, ax = super().pair_plot(data_vars, fig_sz=fig_sz)

        for i in range(len(data_vars)):
            for j in range(len(data_vars)):
                self.linear_regression([data_vars[j]], data_vars[i])
                minX = self.A[:, 0].min()
                maxX = self.A[:, 0].max()
                spacedX = np.linspace(minX, maxX)
                yValues = self.slope[0] * spacedX + self.intercept
                if hists_on_diag == True:
                    if (i==j):
                        numVars = len(data_vars)
                        ax[i, j].remove()
                        ax[i, j] = fig.add_subplot(numVars, numVars, i*numVars+j+1)
                        ax[i,j].hist(self.data.select_data([data_vars[j]]))
                        if j < numVars-1:
                            ax[i, j].set_xticks([])
                        else:
                            ax[i, j].set_xlabel(data_vars[i])
                        if i > 0:
                            ax[i, j].set_yticks([])
                        else:
                            ax[i, j].set_ylabel(data_vars[i])
                    else:    
                        ax[i,j].plot(spacedX, np.squeeze(yValues), 'g')
                        title = 'R2: ' + str(self.R2)
                        ax[i,j].set_title(title, fontsize = 10)
                else:
                    ax[i,j].plot(spacedX, np.squeeze(yValues), 'g')
                    title = 'R2: ' + str(self.R2) + '\nmsse: ' + str(self.mse)
                    ax[i,j].set_title(title, fontsize = 10)

        return fig, ax

    def make_polynomial_matrix(self, A, p):
        '''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
        for a polynomial regression model of degree `p`.

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, 1)
            Independent variable data column vector x
        p: int. Degree of polynomial regression model.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, p)
            Independent variable data transformed for polynomial model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10.

        NOTE: There should not be a intercept term ("x^0"), the linear regression solver method
        will take care of that.
        '''
        matrix = np.ones((A.shape[0], p))
        for i in range(p):
            matrix[:, i] = np.ravel(np.power(A,i+1))
        return matrix


    def poly_regression(self, ind_var, dep_var, p):
        '''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
        (Week 2)
        NOTE: For single linear regression only (one independent variable only)

        Parameters:
        -----------
        ind_var: str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        p: int. Degree of polynomial regression model.
             Example: if p=10, then the model should have terms in your regression model for
             x^1, x^2, ..., x^9, x^10, and a column of homogeneous coordinates (1s).

        TODO:
        - This method should mirror the structure of self.linear_regression (compute all the same things)
        - Differences are:
            - You create a matrix based on the independent variable data matrix (self.A) with columns
            appropriate for polynomial regresssion. Do this with self.make_polynomial_matrix.
            - You set the instance variable for the polynomial regression degree (self.p)
        '''
        self.ind_vars = ind_var
        self.dep_var = dep_var
        self.p = p

        data = self.data.select_data([ind_var])
        data = data.reshape(data.shape[0],1)
        self.A = self.make_polynomial_matrix(data, self.p)
        self.y = self.data.select_data([dep_var])
        self.Ahat = np.hstack((np.ones((self.A.shape[0], 1)), self.A))
        c, residues, rank, s = scipy.linalg.lstsq(self.Ahat, self.y)
        
        self.intercept = c[0][0]
        self.slope = c[1:]
        self.slope = self.slope.reshape(self.slope.shape[0],1)
        
        predictedValues = self.predict()
        self.R2 = self.r_squared(predictedValues)
        self.residuals = self.compute_residuals(predictedValues)
        self.m_sse = self.compute_mse() 


    def get_fitted_slope(self):
        '''Returns the fitted regression slope.
        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_ind_vars, 1). The fitted regression slope(s).
        '''
        return self.slope

    def get_fitted_intercept(self):
        '''Returns the fitted regression intercept.
        (Week 2)

        Returns:
        -----------
        float. The fitted regression intercept(s).
        '''
        return self.intercept

    def initialize(self, ind_vars, dep_var, slope, intercept, p):
        '''Sets fields based on parameter values.
        (Week 2)

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        slope: ndarray. shape=(num_ind_vars, 1)
            Slope coefficients for the linear regression fits for each independent var
        intercept: float.
            Intercept for the linear regression fit
        p: int. Degree of polynomial regression model.

        TODO:
        - Use parameters and call methods to set all instance variables defined in constructor. 
        '''
        self.ind_vars = ind_vars
        self.dep_var = [dep_var]
        self.p = p
        
        data = self.data.select_data(ind_vars)
        data = data.reshape(data.shape[0],1)
        self.A = self.make_polynomial_matrix(data, self.p)
        self.y = self.data.select_data([dep_var])

        self.intercept = intercept
        self.slope = slope

        predictedValues = self.predict()
        self.R2 = self.r_squared(predictedValues)
        self.residuals = self.compute_residuals(predictedValues)
        self.m_sse = self.compute_mse()

