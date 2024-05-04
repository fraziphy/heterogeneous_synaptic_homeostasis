import numpy as np
import scipy.stats as stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle
import sys

# Define bin sizes for different types of signals
bin_size = {"lfp": 0.5, "firing": 0.5, "v": 0.5}

# Function to compute Spectrogram
def Spectrogram(signal, taper, shift, sampling_rate):
    """
    Compute the spectrogram of a signal.

    Parameters:
        signal (numpy.ndarray): Input signal.
        taper (numpy.ndarray): Tapering function.
        shift (int): Shift value.
        sampling_rate (float): Sampling rate of the signal.

    Returns:
        numpy.ndarray: Spectrogram of the input signal.
    """
    # Compute spectrogram using Fourier transform
    N = taper.shape[0]
    M = 1 + (signal.shape[0] - N) // shift
    lfp = np.empty((M, N), dtype=float)
    for i in range(M):
        lfp[i, :] = signal[i * shift:i * shift + N]
    lfp = lfp - np.mean(lfp, axis=1).reshape(M, -1)
    lfp = np.fft.rfft(lfp * taper)
    spec = np.abs(lfp) ** 2 / N / sampling_rate
    spec[:, 1:] *= 2
    return spec.T

# Function to compute Autocorrelation
def AUTOCORR(x):
    """
    Compute the autocorrelation of a signal.

    Parameters:
        x (numpy.ndarray): Input signal.

    Returns:
        numpy.ndarray: Autocorrelation of the input signal.
    """
    # Compute autocorrelation of input signal
    xx = x - x.mean()
    auto = np.correlate(xx, xx, "same") / np.var(xx) / xx.size
    return auto

# Function to compute relative power spectrum
def REL_SPEC(ab_spec, frequency):
    """
    Compute the relative power spectrum.

    Parameters:
        ab_spec (numpy.ndarray): Absolute power spectrum.
        frequency (numpy.ndarray): Array of frequency values.

    Returns:
        numpy.ndarray: Relative power spectrum.
    """
    # Calculate power in different frequency bands
    slow_power = np.abs(ab_spec[1:np.argwhere(frequency > 1)[0][0]]).sum()
    delta_power = np.abs(ab_spec[np.argwhere(frequency > 1)[0][0]:np.argwhere(frequency > 4)[0][0]]).sum()
    theta_power = np.abs(ab_spec[np.argwhere(frequency > 4)[0][0]:np.argwhere(frequency > 7)[0][0]]).sum()
    alpha_power = np.abs(ab_spec[np.argwhere(frequency > 7)[0][0]:np.argwhere(frequency > 13)[0][0]]).sum()
    beta_power = np.abs(ab_spec[np.argwhere(frequency > 13)[0][0]:np.argwhere(frequency > 30)[0][0]]).sum()
    gamma_power = np.abs(ab_spec[np.argwhere(frequency > 30)[0][0]:]).sum()
    
    # Calculate total power
    total_power = slow_power + delta_power + theta_power + alpha_power + beta_power + gamma_power

    # Calculate relative power in each band
    rel_pow = [slow_power, delta_power, theta_power, alpha_power, beta_power, gamma_power]
    rel_pow = np.array(rel_pow) / total_power
    
    return rel_pow

# Function to compute high-low power ratio
def HIGH_LOW_POW(ab_spec, frequency):
    """
    Calculate the ratio of high to low power in a power spectrum.

    Parameters:
        ab_spec (numpy.ndarray): Absolute power spectrum.
        frequency (numpy.ndarray): Array of frequency values.

    Returns:
        float: Log10 ratio of high power to low power.
    """
    # Calculate low power by summing absolute power spectrum up to frequency > 4
    low_power = np.abs(ab_spec[1:np.argwhere(frequency > 4)[0][0]]).sum()
    
    # Calculate high power by summing absolute power spectrum from frequency > 30 onwards
    high_power = np.abs(ab_spec[np.argwhere(frequency > 30)[0][0]:]).sum()

    # Calculate the log10 ratio of high power to low power
    ratio = np.log10(high_power / low_power)

    return ratio

# Function for prestimulus analysis
def PRESTIMULUS_ANALYSIS(data, dt, rand_inst):
    """
    Perform analysis on prestimulus data.

    Parameters:
        data (numpy.ndarray): Array of prestimulus data.
        dt (float): Time step.
        rand_inst (int): Random instance index.

    Returns:
        dict: Dictionary containing analyzed data.
    """

    # Number of trials
    n_trial = data.shape[0]

    # Define analysis parameters
    time_window = 2000  # Size in ms
    overlap = 90
    sampling_rate = 1000 / dt  # Sampling rate in Hz
    taper_size = int(time_window / dt)  # Size in number of data points
    shift = (100 - overlap) * taper_size // 100
    hann = (np.sin(np.pi * np.arange(taper_size) / taper_size)) ** 2
    frequency = (sampling_rate * np.arange(taper_size) / taper_size)

    # Initialize variables to store analyzed data
    dataplot = {}
    dataplot["instance"] = data[rand_inst]

    bins = np.arange(0, 31)
    big_hist = np.array([])
    correlate = np.array([])
    spec_all = np.array([])
    REL_POW = np.array([])
    HIG_LOW = np.array([])

    # Loop through each trial
    for j in range(n_trial):
        # Compute histogram of data
        hist = np.histogram(data[j].ravel(), bins=bins)[0] * 100 / data[j].size
        big_hist = np.append(big_hist, hist)

        # Compute autocorrelation
        aa = AUTOCORR(data[j])
        correlate = np.append(correlate, aa)

        # Compute spectrogram
        aa_spec = Spectrogram(data[j], hann, shift, sampling_rate)
        spec_all = np.append(spec_all, aa_spec.sum(axis=1)[0:201])

        # Compute relative power spectrum
        rel_power_spec = REL_SPEC(aa_spec, frequency)
        REL_POW = np.append(REL_POW, rel_power_spec)

        # Compute high to low power ratio
        hig_low = HIGH_LOW_POW(aa_spec, frequency)
        HIG_LOW = np.append(HIG_LOW, hig_low)

    # Reshape data arrays
    big_hist = big_hist.reshape(n_trial, -1)
    correlate = correlate.reshape(n_trial, -1)
    spec_all = spec_all.reshape(n_trial, -1)
    REL_POW = REL_POW.reshape(n_trial, -1)

    # Compute means and standard deviations of analyzed data
    dataplot["hist"] = big_hist.mean(axis=0)
    dataplot["hist_std"] = big_hist.std(axis=0)
    dataplot["hist_bin"] = bins[:-1].copy()
    dataplot["corr"] = correlate.mean(axis=0)
    dataplot["corr_std"] = correlate.std(axis=0)
    dataplot["spec"] = np.log10(spec_all).mean(axis=0)
    dataplot["spec_std"] = np.log10(spec_all).std(axis=0)
    dataplot["spec_frequency"] = frequency[:201].copy()
    dataplot["rel_pow"] = REL_POW.mean(axis=0)
    dataplot["rel_pow_std"] = REL_POW.std(axis=0)

    # Compute distribution of high to low power ratios
    data_min = HIG_LOW.min()
    data_max = HIG_LOW.max()
    FFT_ratio_bins = np.arange(data_min, data_max + 0.05, 0.05)
    hist = np.histogram(HIG_LOW.ravel(), bins=FFT_ratio_bins)[0] * 100 / HIG_LOW.size
    dataplot["dist_high_low_pow"] = hist.copy()
    dataplot["dist_high_low_pow_bins"] = FFT_ratio_bins[:-1].copy()

    return dataplot

# Function for Stratified K-fold cross-validation
def Stratified_K_FOLD(rng, data, Classifier_Algorithm, k_fold=10):
    """
    Perform stratified k-fold cross-validation.

    Parameters:
        rng (numpy.random.Generator): Random number generator.
        data (numpy.ndarray): Input data.
        Classifier_Algorithm (class): Classifier algorithm class.
        k_fold (int): Number of folds for cross-validation (default is 10).

    Returns:
        tuple: Tuple containing arrays of train accuracies and test accuracies.
    """

    # Get the shape of the data
    n_trial, n_inputs = data.shape

    # Reshape data into a single array (column-major order)
    x1 = data.ravel(order="f")

    # Create labels for stratification
    y = np.arange(n_inputs)
    y = np.repeat(y, n_trial)

    # Permute the data
    p = rng.permutation(len(y))
    x1 = x1[p]
    y = y[p]

    # Initialize arrays to store accuracies for each fold
    accuracy_train = np.zeros(k_fold, dtype=float)
    accuracy_test = np.zeros(k_fold, dtype=float)

    # Initialize fold counter
    i = 0

    # Create stratified k-fold splitter
    skf = StratifiedKFold(n_splits=k_fold, random_state=1, shuffle=True)

    # Iterate over each fold
    for train_index, test_index in skf.split(x1, y):
        # Split data into training and testing sets
        X_train, X_test = x1[train_index], x1[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Instantiate the classifier algorithm
        Classifier = Classifier_Algorithm(X_train, y_train, X_test, y_test)

        # Train the classifier and get accuracies
        accuracy_train[i] = Classifier.accuracy_train
        accuracy_test[i] = Classifier.accuracy_test

        # Increment fold counter
        i += 1

    return accuracy_train, accuracy_test

# Class for Logistic Classification
class LOGISTIC_CLASSIFICATION:
    """
    Logistic regression classifier.
    """

    def __init__(self, data_train, label_train, data_test, label_test):
        """
        Initialize the logistic regression classifier.

        Parameters:
            data_train (numpy.ndarray): Training data.
            label_train (numpy.ndarray): Training labels.
            data_test (numpy.ndarray): Test data.
            label_test (numpy.ndarray): Test labels.
        """
        # Add bias term to training and test data
        self.data_train = np.column_stack([np.ones(len(label_train)), data_train])
        self.data_test = np.column_stack([np.ones(len(label_test)), data_test])

        # Train the classifier and calculate accuracies
        self.learning_parameters = self.Train(self.data_train, label_train)
        self.accuracy_train = self.Accuracy(self.learning_parameters, self.data_train, label_train)
        self.accuracy_test = self.Accuracy(self.learning_parameters, self.data_test, label_test)

    def Train(self, x, y):
        """
        Train the logistic regression classifier.

        Parameters:
            x (numpy.ndarray): Input data.
            y (numpy.ndarray): Labels.

        Returns:
            numpy.ndarray: Learned parameters.
        """
        # This method should implement the training of the logistic regression classifier
        pass

    def Predict(self, w, x):
        """
        Make predictions using the learned parameters.

        Parameters:
            w (numpy.ndarray): Learned parameters.
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Predicted labels.
        """
        # This method should implement the prediction using the learned parameters
        pass

    def Accuracy(self, w, x, y):
        """
        Calculate the accuracy of the classifier.

        Parameters:
            w (numpy.ndarray): Learned parameters.
            x (numpy.ndarray): Input data.
            y (numpy.ndarray): True labels.

        Returns:
            float: Accuracy of the classifier.
        """
        # Make predictions using learned parameters
        test_predict = self.Predict(w, x)

        # Calculate confusion matrix
        matrix = confusion_matrix(y, test_predict)

        # Calculate accuracy
        return matrix.trace() / matrix.sum()

# Class for Multinomial Classification
class MULTINOMIAL(LOGISTIC_CLASSIFICATION):
    """
    Multinomial logistic regression classifier.
    Inherits from the LOGISTIC_CLASSIFICATION class.
    """

    def __init__(self, data_train, label_train, data_test, label_test):
        """
        Initialize the multinomial logistic regression classifier.

        Parameters:
            data_train (numpy.ndarray): Training data.
            label_train (numpy.ndarray): Training labels.
            data_test (numpy.ndarray): Test data.
            label_test (numpy.ndarray): Test labels.
        """
        super().__init__(data_train, label_train, data_test, label_test)

    def Train(self, data, target):
        """
        Train the multinomial logistic regression classifier.

        Parameters:
            data (numpy.ndarray): Input data.
            target (numpy.ndarray): Target labels.

        Returns:
            numpy.ndarray: Learned parameters.
        """
        # Define hyperparameters
        lamda = 0
        tolerance = 1e-6

        # Get the number of classes
        n_c = len(np.unique(target))

        # Compute XI_XJ
        XI_XJ = [np.matmul(data[i].reshape(-1, 1), data[i].reshape(1, -1)) for i in range(len(target))]

        # Initialize parameters and variables
        THETA = np.zeros(((n_c - 1) * data.shape[1], 1))
        theta = THETA[:, [0]]
        z = data @ (THETA[:, [0]].reshape(data.shape[1], -1, order='F'))
        y_hat = self.Hypothesis(z)
        Y = self.One_Hot_Multinomial(target, n_c)[:, :-1]
        COST = np.array([])
        cost = -np.mean(np.log(y_hat[np.arange(len(target)), target]))
        COST = np.append(COST, cost)
        cond = True
        i = 0
        lamda_hes = lamda * np.eye((n_c - 1) * data.shape[1], dtype=float)

        # Newton's method for optimization
        while cond:
            H = [np.diag(i) - (i.reshape(-1, 1) @ i.reshape(1, -1)) for i in y_hat[:, :-1]]
            hes = list(map(lambda x, y: np.kron(x, y), H, XI_XJ))
            hes = sum(hes) - lamda_hes
            if np.linalg.det(hes) == 0:
                return theta, COST
            hesinv = np.linalg.inv(hes)

            step_part2 = [np.kron((Y - y_hat[:, :-1])[i], data[i]) for i in range(data.shape[0])]
            step_part2 = np.array(step_part2).sum(axis=0).reshape(-1, 1) - 2 * lamda * THETA[:, [i]]

            decrement = hesinv @ step_part2
            theta = THETA[:, [i]] + decrement
            THETA = np.hstack((THETA, theta))

            lambda_newton_decrement = (step_part2.reshape(1, -1) @ decrement) / 2

            z = data @ (THETA[:, [i + 1]].reshape(data.shape[1], -1, order='F'))
            y_hat = self.Hypothesis(z)
            cost = -np.mean(np.log(y_hat[np.arange(len(target)), target]))
            COST = np.append(COST, cost)

            cond = lambda_newton_decrement > tolerance
            i += 1

        return theta

    def One_Hot_Multinomial(self, y, c):
        """
        Convert labels to one-hot encoding.

        Parameters:
            y (numpy.ndarray): Label/ground truth.
            c (int): Number of classes.

        Returns:
            numpy.ndarray: One-hot encoded labels.
        """
        y_hot = np.zeros((len(y), c))
        y_hot[np.arange(len(y)), y] = 1
        return y_hot

    def Hypothesis(self, z):
        """
        Compute softmax function.

        Parameters:
            z (numpy.ndarray): Linear part.

        Returns:
            numpy.ndarray: Softmax probabilities.
        """
        exp = np.exp(z - np.max(z, axis=1).reshape(-1, 1))
        expnew = np.hstack((exp, np.exp(-np.max(z, axis=1)).reshape(-1, 1)))

        for i in range(len(z)):
            expnew[i] /= np.sum(expnew[i])

        return expnew

    def Predict(self, w, X_design):
        """
        Make predictions using learned parameters.

        Parameters:
            w (numpy.ndarray): Learned parameters.
            X_design (numpy.ndarray): Input.

        Returns:
            numpy.ndarray: Predicted labels.
        """
        z = X_design @ (w.reshape(X_design.shape[1], -1, order='F'))
        y_hat = self.Hypothesis(z)
        return np.argmax(y_hat, axis=1)

# Class for Binomial Classification
class BINOMIAL(LOGISTIC_CLASSIFICATION):
    """
    Binomial logistic regression classifier.
    Inherits from the LOGISTIC_CLASSIFICATION class.
    """

    def __init__(self, data_train, label_train, data_test, label_test):
        """
        Initialize the binomial logistic regression classifier.

        Parameters:
            data_train (numpy.ndarray): Training data.
            label_train (numpy.ndarray): Training labels.
            data_test (numpy.ndarray): Test data.
            label_test (numpy.ndarray): Test labels.
        """
        super().__init__(data_train, label_train, data_test, label_test)

    def Train(self, data, target):
        """
        Train the binomial logistic regression classifier.

        Parameters:
            data (numpy.ndarray): Input data.
            target (numpy.ndarray): Target labels.

        Returns:
            numpy.ndarray: Learned parameters.
        """
        # Define hyperparameters
        lamda = 0
        tolerance = 1e-6
        MINLIM = sys.float_info.min

        # Compute XI_XJ
        XI_XJ = [np.matmul(data[i].reshape(-1, 1), data[i].reshape(1, -1)) for i in range(len(target))]

        # Initialize parameters and variables
        THETA = np.zeros((1, data.shape[1]), dtype=float)
        theta = THETA[0]
        h = self.Hypothesis(THETA, data)
        COST = self.MLE_Logistic(h, target.reshape(-1, 1), MINLIM)
        cond = True
        i = 0
        lamda_hes = lamda * np.eye(data.shape[1], dtype=float)

        # Newton's method for optimization
        while cond:
            H = [i * (i - 1) for i in h]
            hes = list(map(lambda x, y: x * y, H, XI_XJ))
            hes = sum(hes) - lamda_hes
            if np.linalg.det(hes) == 0:
                return theta, COST
            hesinv = np.linalg.inv(hes)
            lprime = np.matmul((- h + target.reshape(-1, 1)).T, data) - 2 * lamda * THETA[i]
            hesinv_lprime = np.matmul(hesinv, lprime.reshape(-1, 1))
            theta = THETA[i] - hesinv_lprime.reshape(1, -1)
            THETA = np.vstack((THETA, theta))
            h = self.Hypothesis(theta, data)
            cost = self.MLE_Logistic(h, target.reshape(-1, 1), MINLIM)
            COST = np.append(COST, cost)
            lambda_newton_decrement = np.matmul(lprime.reshape(1, -1), hesinv_lprime) / 2
            cond = lambda_newton_decrement > tolerance
            i += 1

        return theta

    def MLE_Logistic(self, h, targ, MINLIM):
        """
        Compute the maximum likelihood estimate for logistic regression.

        Parameters:
            h (numpy.ndarray): Hypothesis values.
            targ (numpy.ndarray): Target labels.
            MINLIM (float): Minimum value for numerical stability.

        Returns:
            float: Maximum likelihood estimate.
        """
        h[h < MINLIM] = MINLIM
        B = 1 - h
        B[B < MINLIM] = MINLIM
        mle = targ * np.log(h) + (1 - targ) * np.log(B)
        return mle.sum()

    def Hypothesis(self, t, x):
        """
        Compute the hypothesis function.

        Parameters:
            t (numpy.ndarray): Parameters.
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Hypothesis values.
        """
        arg = -np.matmul(x, t.reshape(-1, 1))
        arg[arg >= np.log(sys.float_info.max)] = np.log(sys.float_info.max)
        h = 1 / (1 + np.exp(arg))
        return h

    def Predict(self, w, X_design):
        """
        Make predictions using learned parameters.

        Parameters:
            w (numpy.ndarray): Learned parameters.
            X_design (numpy.ndarray): Input.

        Returns:
            numpy.ndarray: Predicted labels.
        """
        h = self.Hypothesis(w, X_design)
        h[h >= 0.5] = 1
        h[h < 0.5] = 0
        return h

# Class for K-means Clustering
class K_MEANS_CLUSTERING:
    """
    K-Means clustering algorithm.
    """

    def __init__(self, data_train, label_train, data_test, label_test):
        """
        Initialize the K-Means clustering algorithm.

        Parameters:
            data_train (numpy.ndarray): Training data.
            label_train (numpy.ndarray): Training labels.
            data_test (numpy.ndarray): Test data.
            label_test (numpy.ndarray): Test labels.
        """
        # Train the model and compute accuracies
        learning_parameters = self.Train(data_train.reshape(len(label_train), -1), label_train)
        self.accuracy_train = self.Accuracy(learning_parameters, data_train.reshape(len(label_train), -1), label_train)
        self.accuracy_test = self.Accuracy(learning_parameters, data_test.reshape(len(label_test), -1), label_test)

    def Train(self, data, target):
        """
        Train the K-Means clustering algorithm.

        Parameters:
            data (numpy.ndarray): Input data.
            target (numpy.ndarray): Target labels.

        Returns:
            numpy.ndarray: Cluster centers.
        """
        dimension = data.shape[1]
        n_cluster = len(np.unique(target))
        n_trial = int(len(target) / n_cluster)

        # Initialize cluster centers
        center = np.zeros((n_cluster, dimension), dtype=float)

        # Initialize index
        index = np.arange(n_cluster)
        index = np.repeat(index, n_trial)

        cond = True

        # Update cluster centers until convergence
        while cond:
            for i in range(n_cluster):
                if (index == i).sum() != 0:
                    center[i] = data[index == i].mean(axis=0)
                else:
                    center[i] = data.mean(axis=0)

            index_new = self.Predict(center, data)

            cond = (index != index_new).any()
            index = index_new

        return center

    def Predict(self, w, x):
        """
        Make predictions using the cluster centers.

        Parameters:
            w (numpy.ndarray): Cluster centers.
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Predicted cluster indices.
        """
        # Calculate distance between each point and cluster centers
        distance = np.array([np.linalg.norm(i - w, axis=1) for i in x])

        # Return the index of the closest cluster center for each point
        return distance.argmin(axis=1)

    def Accuracy(self, w, x, y):
        """
        Compute accuracy of clustering.

        Parameters:
            w (numpy.ndarray): Cluster centers.
            x (numpy.ndarray): Input data.
            y (numpy.ndarray): True labels.

        Returns:
            float: Mutual Information (MI) between predicted clusters and true labels.
        """
        # Make predictions
        test_predict = self.Predict(w, x)

        # Compute Mutual Information (MI) for clustering validation
        mi_clustering_validation = self.MI_Clustering_Validation(test_predict, y)

        return mi_clustering_validation

    def MI_Clustering_Validation(self, new_pred, target):
        """
        Compute Mutual Information (MI) for clustering validation.

        Parameters:
            new_pred (numpy.ndarray): Predicted cluster indices.
            target (numpy.ndarray): True labels.

        Returns:
            float: Mutual Information (MI) between predicted clusters and true labels.
        """
        n_cluster1 = len(np.unique(new_pred))
        n_cluster2 = len(np.unique(target))

        # Compute probability distributions
        p_k1 = np.zeros(n_cluster1, dtype=float)
        p_k2 = np.zeros(n_cluster2, dtype=float)
        n_total = len(target)

        p_k1 = [len(new_pred[new_pred == i]) for i in range(n_cluster1)]
        p_k1 = np.array(p_k1) / n_total

        p_k2 = [len(target[target == i]) for i in range(n_cluster2)]
        p_k2 = np.array(p_k2) / n_total

        # Compute joint probability distribution
        aux_p_k1_p_k2 = (p_k1.reshape(-1, 1) @ p_k2.reshape(1, -1)).ravel()
        p_k1_k2 = np.zeros((n_cluster1, n_cluster2), dtype=float)

        for i in range(n_cluster1):
            p_k1_k2[i] = [((new_pred == i) & (target == j)).sum() for j in range(n_cluster2)]
        p_k1_k2 = p_k1_k2 / n_total
        p_k1_k2 = p_k1_k2.ravel()

        # Compute mutual information
        aux = p_k1_k2[p_k1_k2 != 0] / aux_p_k1_p_k2[p_k1_k2 != 0]
        mi = (p_k1_k2[p_k1_k2 != 0] * np.log2(aux)).sum()

        # Compute entropies
        H_C1 = - (p_k1[p_k1 != 0] * np.log2(p_k1[p_k1 != 0])).sum()
        H_C2 = - (p_k2[p_k2 != 0] * np.log2(p_k2[p_k2 != 0])).sum()

        mi = mi / np.sqrt(H_C1 * H_C2)

        return mi

# Class for Information Brain State Analysis
class INFORMATION_BRAIN_STATE:
    # Constructor
    def __init__(self, rngs_detect, rngs_diff, df, inputs, column, pop):
        self.n_trial = 500  # Number of trials
        
        # Information detection using logistic regression
        self.info_detec_logistic = self.Information_Detection(rngs_detect[0], df, inputs, column, pop, BINOMIAL)
        # Information detection using K-means clustering
        self.info_detec_k_means = self.Information_Detection(rngs_detect[1], df, inputs, column, pop, K_MEANS_CLUSTERING)
        
        # Information differentiation using logistic regression
        self.info_diff_logistic = self.Information_Differentiation(rngs_diff[0], df, inputs, column, pop, MULTINOMIAL)
        # Information differentiation using K-means clustering
        self.info_diff_k_means = self.Information_Differentiation(rngs_diff[1], df, inputs, column, pop, K_MEANS_CLUSTERING)

        # T-value calculation
        self.tvalue = self.T_Value(df, inputs, column, pop)
        # F-value calculation
        self.fvalue = self.F_Value(df, inputs, column, pop)

        # Mutual information analysis
        self.mi = self.Mutual_Information_4_Bins(df, inputs, column, pop)
        self.mi_detect = self.Mutual_Information_Detect_4_Bins(df, inputs, column, pop)

    # Method for information detection
    def Information_Detection(self, rng, df, inputs, column, pop, algorithm):
        # Arrays to store information and standard deviation
        info_detec = np.zeros((2, len(inputs)), dtype=float)
        info_detec_train = np.zeros((2, len(inputs)), dtype=float)
        
        # Loop over inputs
        for i, inpu in enumerate(inputs):
            X = np.zeros((self.n_trial, 2), dtype=float)
            X[:, 0] = df[column]["input:{}".format(inpu)][pop]["pre"]
            X[:, 1] = df[column]["input:{}".format(inpu)][pop]["post"]
            
            # Stratified K-fold cross-validation
            train, test = Stratified_K_FOLD(rng[i], X, algorithm, k_fold=10)
            
            # Calculate mean and standard deviation for test data
            info_detec[0, i] = test.mean()
            info_detec[1, i] = test.std()

            # Calculate mean and standard deviation for training data
            info_detec_train[0, i] = train.mean()
            info_detec_train[1, i] = train.std()
            
        return info_detec, info_detec_train

    # Method for information differentiation
    def Information_Differentiation(self, rng, df, inputs, column, pop, algorithm):
        # Arrays to store information and standard deviation
        info_diff = np.zeros(2, dtype=float)
        info_diff_train = np.zeros(2, dtype=float)
        
        X = np.zeros((self.n_trial, len(inputs)), dtype=float)
        
        for i, inpu in enumerate(inputs):
            X[:, i] = df[column]["input:{}".format(inpu)][pop]["post"]
            
        # Stratified K-fold cross-validation
        train, test = Stratified_K_FOLD(rng, X, algorithm, k_fold=10)

        # Calculate mean and standard deviation for test data
        info_diff[0] = test.mean()
        info_diff[1] = test.std()

        # Calculate mean and standard deviation for training data
        info_diff_train[0] = train.mean()
        info_diff_train[1] = train.std()
            
        return info_diff, info_diff_train

    # Method to compute probability distribution
    def Probability_Distribution(self, data, bins):
        # Initialize arrays
        p_R_S = np.empty((n_inputs, len(bins[:-1])), dtype=float)
        input_count = np.empty(n_inputs, dtype=float)
        R_all = np.array([])
        
        # Iterate over inputs
        for i in range(n_inputs):
            x = data[i]
            p_R_S_aux = np.histogram(x, bins=bins)[0]
            p_R_S_aux = p_R_S_aux / p_R_S_aux.sum()
            p_R_S[i] = p_R_S_aux.copy()
            input_count[i] = len(x)
            R_all = np.append(R_all, x)
        
        p_R = np.histogram(R_all, bins=bins)[0]
        p_R = p_R / p_R.sum()
        p_S = input_count / input_count.sum()
        non_index = np.where(p_R > 0)[0]
        
        p_S_R = np.zeros((len(bins[:-1]), n_inputs), dtype=float)
        for i in range(n_inputs):
            p_S_R_aux = np.zeros(len(bins[:-1]), dtype=float)
            p_S_R_aux[non_index] = p_S[i] * p_R_S[i, non_index] / p_R[non_index]
            p_S_R[:, i] = p_S_R_aux.copy()
            
        return p_R, p_S, p_R_S, p_S_R

    # Method to compute response-stimuli entropy
    def Response_Stimuli_Entropy(self, p_r, p_s, p_r_s, p_s_r):
        h_r = (-p_r[p_r > 0] * np.log2(p_r[p_r > 0])).sum()
        h_s = (-p_s[p_s > 0] * np.log2(p_s[p_s > 0])).sum()
        
        h_r_s_aux = np.zeros(p_r_s.shape, dtype=float)
        h_r_s_aux[p_r_s > 0] = p_r_s[p_r_s > 0] * np.log2(p_r_s[p_r_s > 0])
        h_r_s = (-p_s * h_r_s_aux.sum(axis=1)).sum()
        
        h_s_r_aux = np.zeros(p_s_r.shape, dtype=float)
        h_s_r_aux[p_s_r > 0] = p_s_r[p_s_r > 0] * np.log2(p_s_r[p_s_r > 0])
        h_s_r = (-p_r * h_s_r_aux.sum(axis=1)).sum()
        
        return h_r, h_r_s, h_s, h_s_r

    # Method to compute mutual information
    def Mutual_Information(self, data, n_bins):
        n_inputs = len(list(data.keys()))
        low_limit, high_limit = np.inf, -np.inf
        
        for i in range(n_inputs):
            low_limit = min(low_limit, data[i].min())
            high_limit = max(high_limit, data[i].max())
    
        bins = np.linspace(low_limit, high_limit, n_bins)
        
        if n_bins == 300:
            bins = np.linspace(0, 30, n_bins)
        else:
            bins = np.linspace(0, 60, n_bins)
        
        p_r, p_s, p_r_s, p_s_r = self.Probability_Distribution(data, bins)
        h_r, h_r_s, h_s, h_s_r = self.Response_Stimuli_Entropy(p_r, p_s, p_r_s, p_s_r)
        
        mi_per_stimuli = np.empty(n_inputs, dtype=float)
        for i in range(n_inputs):
            mi_per_stimuli[i] = (p_r_s[i][p_r_s[i] > 0] * np.log2(p_r_s[i][p_r_s[i] > 0])).sum() - (
                        p_r_s[i][p_r > 0] * np.log2(p_r[p_r > 0])).sum()
        
        return h_r - h_r_s, mi_per_stimuli

    # Method to compute mutual information for bins
    def Mutual_Information_4_Bins(self, df, inputs, column, pop):
        n_inputs = len(inputs)
        
        if pop == "pyr":
            n_bins = 300
        else:
            n_bins = 600
        
        MI = np.zeros(2, dtype=float)
        
        for i, cond in enumerate(["post", "pre"]):
            data_new = {}
            
            for iii in range(n_inputs):
                data_new[iii] = df[column]["input:{}".format(inputs[iii])][pop][cond]
                
            mi = self.Mutual_Information(data_new, n_bins)[0]
            MI[i] = mi
        
        return MI

    # Method to compute mutual information for detection for bins
    def Mutual_Information_Detect_4_Bins(self, df, inputs, column, pop):
        n_inputs = len(inputs)
        
        if pop == "pyr":
            n_bins = 300
        else:
            n_bins = 600
        
        MI = np.zeros(n_inputs, dtype=float)
        
        for i in range(n_inputs):
            data_new = {}
            data_new[0] = df[column]["input:{}".format(inputs[i])][pop]["pre"]
            data_new[1] = df[column]["input:{}".format(inputs[i])][pop]["post"]
            
            mi = self.Mutual_Information(data_new, n_bins)[0]
            MI[i] = mi
        
        return MI

    # Method to compute T-value
    def T_Value(self, df, inputs, column, pop):
        t_value = np.zeros(len(inputs), dtype=float)
        
        for i, inpu in enumerate(inputs):
            t_stat, p_value = stats.ttest_ind(df[column]["input:{}".format(inpu)][pop]["post"],
                                              df[column]["input:{}".format(inpu)][pop]["pre"])
            t_value[i] = t_stat
            
        return t_value

    # Method to compute F-value
    def F_Value(self, df, inputs, column, pop):
        f_value, p_value = stats.f_oneway(df[column]["input:{}".format(inputs[0])][pop]["post"],
                                          df[column]["input:{}".format(inputs[1])][pop]["post"],
                                          df[column]["input:{}".format(inputs[2])][pop]["post"],
                                          df[column]["input:{}".format(inputs[3])][pop]["post"],
                                          df[column]["input:{}".format(inputs[4])][pop]["post"])
            
        return f_value
