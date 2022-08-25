# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 17:37:02 2022

@author: austine
"""
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:48:30 2022

@author: austine
"""
from sortedcontainers import SortedList
from collections import deque
import math
import matplotlib.pyplot as plt

#######################
#Sequential Implementation of NUNC Local
#######################

#these are the functions needed to run NUNC Local without any wrappers 
#it just takes a state (the initial window of data) and then runs NUNC Local
#by updating the window with a new observation and performing NUNC

def argmax(l) :
    #this function is needed to compute the location of the max when 
    #performing each sequential update
    pos = max(range(len(l)),key=lambda i: l[i])
    return (l[pos],pos) 
    
class nunc_local_state:
    #this creates the initial sorted list and deque (window) for use with local
    #note we have length + 1 as we need to add the next point and start testing
    #when we use the nunc_local_update as per below
    
    '''
    Class for storing the window of data for use with nunc_local_update.
    
    Description
    -------------
    
    Instantiating this class takes inputted data and stores it in both an
    ordered tree and a deque. The deque is the window of points for use with
    NUNC Local, and the tree is that window of points in ordered search tree
    form so that quantiles can be computed in O(log w) time, where w is the 
    window size.
    
    The instantiation should take the first w-1 points, so that the window is
    almost full. This is because NUNC starts testing once w points have been
    observed, and so the algorithm should start testing from the next point
    onwards. For full details on the sequential implementation see the examples.
    
    The class comes with a built in method, update, that updates the window 
    and tree with a new observation.
    
    Attributes
    --------------
    tree: SortedList
        Ordered tree of observations in the window.
    window: deque
        Deque containing window of points. Initialises at length n + 1, where
        n is the length of the inputted data. This is so the algorithm starts
        testing from the next observed point, ie when the window is full.
    
    Methods
    -------------
    
    update(x):
        Method to update the tree and window with a new observation, x.
    '''

    def __init__(self, x):
        self.tree = SortedList(x)
        self.window = deque(x, maxlen = len(x) + 1)
        
    def update(self, x) :
        
        '''
        Method to update the tree and window with a new observation.
        
        Input
        ----------
        x: float
            The next observation to store in the window. (Can also be type int 
            but will be converted.)
        '''
        
        if len(self.window) == self.window.maxlen :
            self.tree.remove(self.window[0])
        self.window.append(x)
        self.tree.add(x)

def nunc_local_update(S,k,x) : 
    
    '''
    Function for performing a single iteration of NUNC Local.
    
    Description
    ---------------
    
    This function performs a single step of the NUNC Local algorithm. That is, 
    it tests a single window of data for a change in distribution.
    
    The function takes the window of data as a state, S, which is an object of
    class nunc_local_state. It then updates this state with a new data point x,
    and then performs NUNC. 
    
    Once NUNC has been carried out on the window, a set of costs are returned
    along with the updated window of data. In order to detect changes in the
    data stream, a user would check if the max of the returned costs exceed a
    given threshold. If so, then a change is detected by NUNC. For more details
    see the examples.
            
    Input
    ----------
    S: nunc_local_state
        The object storing the current window of data.
    k: int
        Number of quantiles used by NUNC.
    x: float
        The next observation to store in the window. (Can also be type int 
        but will be converted.)
    
    Returns
    ----------
    The updated state object, S, and a list of costs of length w-1 denoting
    the test statistic at the different points in the window. 
    '''
    
    if not isinstance(S, nunc_local_state):
        raise TypeError("S must be an object of class nunc_local_state")
    if not (isinstance(k, int)):
        raise TypeError("K must be an integer")   
    if k <= 0:
        raise ValueError("K must be a positive integer")
    try:
        float(x)
    except:
        raise TypeError("x must be a float")
    
    def quantiles(data,k,w) :
    #this function computes k quantiles in data from a window of size w
    #the expression for the probabilities is taken from Zou (2014) and 
    #Haynes (2017) and is designed to add emphasis to the tails of the 
    #probability distribution. Both authors show doing so increases test power.
    #data is the data to use for computing the quantiles
    #k is the number of quantiles and w is the window size (used for weights)
    

        def quantile(prob) :
            #this function works out the quantile value
            #it uses linear interpolation, ie quantile type = 7
            h = (len(data) - 1) * prob
            h_floor = int(h)
            if h_floor == h:
                return data[h]
            else:
                non_int_part = h - h_floor 
                lower = data[h_floor]
                upper = data[h_floor + 1]
                return lower + non_int_part * (upper - lower)
        c = math.log(2*w-1)   #weight as in Zou (2014)
        probs = [(1/(1+(2*(w-1)*math.exp((-c/k)*(2*i-1))))) for i in range(k)]
        return [quantile(p) for p in probs]    
    
    def eCDF_vals(data,quantile):    
        #used to return value of eCDF, not cost, at a given quantile  
        #data is the tree of data used to compute the ecdf
        #quantile is the numeric quantile value
        
        left = data.bisect_left(quantile)
        right = data.bisect_right(quantile)
        #value is number of points to left of quantile, plus 0.5 times
        #the points equal to the quantile
        val = (left+0.5*(right-left))/len(data)
        return val
    
    def one_point_emp_dist(data, quantile):
        #function for computing empirical CDF for data at a particular quantile
        #ie, is a point less than, equal to, or greater than the quantile
        #data is an array of numerics
        #quantile is the quantile to evaluate the eCDF at.
        if(data < quantile):
            return(1)
        elif (data == quantile):
            return(0.5)
        else: 
            return(0)
        
    def cdf_cost(cdf_val, seg_len):
        #function for computing the likelihood function using the value of the CDF
        #cdf_val is the value of the eCDF at a set quantile
        #seg_len is the length of the data used 
        if(cdf_val <= 0 or cdf_val >= 1):
            return(0) #avoids rounding error, does not affect result
        conj = 1 - cdf_val
        cost = seg_len * (cdf_val * math.log(cdf_val) - conj * math.log(conj))
        return(cost)
    
    def update_window_ecdf_removal(data_to_remove, quantiles, current_ecdf, current_len):
        #this function takes a set of K (number of quantiles) current eCDFs values
        #computed from a data set of length current_len, and removes a point
        #from this data. 
        #The function then returns the updated eCDF values following removal
        #of this point.
        num_quantiles = len(quantiles)
        for i in range(num_quantiles):
            current_ecdf[i] *= current_len
            current_ecdf[i] -= one_point_emp_dist(data_to_remove, quantiles[i]) 
            current_ecdf[i] /= (current_len - 1)  
        return current_ecdf
    
    S.update(x) #update state with new point
    tree = S.tree #extract tree and window of points
    window = S.window
    w = len(window)  #compute window size from inputted data
    Q = quantiles(tree,k,w) #update quantiles
    full_cdf_vals = [eCDF_vals(tree, q) for q in Q] #full data eCDF
    right_cdf_vals = full_cdf_vals.copy() #will update as we search for change in window
    full_cost = sum(cdf_cost(val, w) for val in full_cdf_vals) #full data cost
    segment_costs = list() #used for storing costs of segmenting window at different places
    current_len = w #this the current length of the right segment, it updates iteratively
    left_cdf_vals = [0] * len(Q) #again will update as we search window for points
    for i in range(0, w-1): #window updates are O(K)
    #as we loop over the window we "move" points from the right segment to the
    #left segment, and update the eCDFs as we go. This provides an O(K) cost
    #for updating the eCDFs for each segment.
        right_cdf_vals = update_window_ecdf_removal(window[i], Q, right_cdf_vals, 
                                                    current_len)
        #remove points from RHS iteratively and update eCDF
        current_len -= 1
        for j in range(len(Q)): #update LHS using RHS and full eCDFs
            left_cdf_vals[j] = (full_cdf_vals[j]*w - right_cdf_vals[j]*current_len) / (w - current_len)
        #compute costs of segmented data
        left_cost = sum([cdf_cost(val, w - current_len) for val in left_cdf_vals])
        right_cost = sum([cdf_cost(val, current_len) for val in right_cdf_vals])
        segment_costs.append(left_cost + right_cost)
    #return full costs for each location and also updated NUNC state S
    costs = [2*(cost - full_cost) for cost in segment_costs]
    return (S,costs)

#########
#NUNC Local
#########

def nunc_local(X,k,w,threshold) : 
    #wrapper function for nunc local - used as part of nunc_offline function
    #when method is equal to "local".
    
    #this function takes a set of data, X, and performs NUNC on the data stream
    #using a window of size w and k quantiles. 
    try:
        threshold = float(threshold)       
    except:
        raise TypeError("threshold must be a float")   
    if threshold <= 0:
        raise ValueError("threshold must be a positive numeric")          
    if not (isinstance(w, int)):
        raise TypeError("w must be a positive integer")
    if w <= 0:
        raise ValueError("w must be a positive integer")   
    if w > len(X):
        raise ValueError("w must not exceed length of data")   
    if not (isinstance(k, int)):
        raise TypeError("K must be an integer")   
    if k <= 0:
        raise ValueError("K must be a positive integer")
    if k > w:
        raise ValueError("K must be less than or equal to w")
    
    S = nunc_local_state(X[:w-1]) #initialise NUNC state
    cost_max_history = [] #for storing history of costs
    dtime = w-1 #keeps track of number of points seen
    for x in X[w-1:] : #begin NUNC algorithm
    #first update NUNC state with next point and perform NUNC test
        S, costs = nunc_local_update(S,k,x) 
        dtime += 1        
        cost_max,pos = argmax(costs) #select max, and location in window, of costs
        cost_max_history.append(cost_max) #store cost max
        if cost_max > threshold : #if true anomaly detected
            return(dtime-w+pos,dtime,cost_max, cost_max_history)
    #if no anomaly is detected, return max seen so far and -1 for detection
    cost_max = max(cost_max_history)
    return (-1, -1, cost_max, cost_max_history)

########################
#NUNC Global Code
#######################

def nunc_global(X,k,w,threshold) :
    #this function performs NUNC Global on a data set X.
    #it is used by the nunc_offline function when method is equal to "global".
    #this function is only for use sequentially, offline, as it is not 
    #unbounded.
    
    try:
        threshold = float(threshold)       
    except:
        raise TypeError("threshold must be a float")   
    if threshold <= 0:
        raise ValueError("threshold must be a positive numeric")          
    if not (isinstance(w, int)):
        raise TypeError("w must be a positive integer")
    if w <= 0:
        raise ValueError("w must be a positive integer")   
    if w > len(X):
        raise ValueError("w must not exceed length of data")   
    if not (isinstance(k, int)):
        raise TypeError("K must be an integer")   
    if k <= 0:
        raise ValueError("K must be a positive integer")
    if k > w:
        raise ValueError("K must be less than or equal to w")
    
    def quantiles(data,k,w) :
    #this function computes k quantiles in data from a window of size w
    #the expression for the probabilities is taken from Zou (2014) and 
    #Haynes (2017) and is designed to add emphasis to the tails of the 
    #probability distribution. Both authors show doing so increases test power.
    #data is the data to use for computing the quantiles
    #k is the number of quantiles and w is the window size (used for weights)
    
        def quantile(prob) :
            #this function works out the quantile value
            #it uses linear interpolation, ie quantile type = 7
            h = (len(data) - 1) * prob
            h_floor = int(h)
            if h_floor == h:
                return data[h]
            else:
                non_int_part = h - h_floor 
                lower = data[h_floor]
                upper = data[h_floor + 1]
                return lower + non_int_part * (upper - lower)
        c = math.log(2*w-1)   #weight as in Zou (2014)
        probs = [(1/(1+(2*(w-1)*math.exp((-c/k)*(2*i-1))))) for i in range(k)]
        return [quantile(p) for p in probs] 
    
    def cost(data,quantile) :
        #this function returns the cost function for NUNC given data and a
        #particular quantile.
        #data is the tree of data to use for cost computation.
        left = data.bisect_left(quantile)
        right = data.bisect_right(quantile)
        val = (left+0.5*(right-left))/len(data)
        conj = 1 - val
        return 0 if val <= 0 or val >= 1 else len(data)*(val*math.log(val)-conj* math.log(conj))
    
    def update(tree,window,x) :
        #this function updates the tree and window based on a new point x
        if len(window) == window.maxlen :
            tree.remove(window[0])
        window.append(x)
        tree.add(x)
        return (tree,window)
    
    def update_window(window, x):   
        #this function only updates the window based on a new point x
        window.append(x)
        return window
    
    
    def eCDF_vals(data,quantile):    
        #used to return value of eCDF, not cost, at a given quantile  
        #data is the tree of data used to compute the ecdf
        #quantile is the numeric quantile value
        
        left = data.bisect_left(quantile)
        right = data.bisect_right(quantile)
        #value is number of points to left of quantile, plus 0.5 times
        #the points equal to the quantile
        val = (left+0.5*(right-left))/len(data)
        return val
    
    def cdf_cost(cdf_val, seg_len):
        #function for computing the likelihood function using the value of the CDF
        #cdf_val is the value of the eCDF at a set quantile
        #seg_len is the length of the data used 
        if(cdf_val <= 0 or cdf_val >= 1):
            return(0) #avoids error, does not affect result
        conj = 1 - cdf_val
        cost = seg_len * (cdf_val * math.log(cdf_val) - conj * math.log(conj))
        return(cost)
    
    def one_point_emp_dist(data, quantile):
        #function for computing empirical CDF for data at a particular quantile
        #data is an array of numerics
        #quantile is the quantile to evaluate the eCDF at.
        ####Only for updating the z vals with a single point so O(1)#####
        if(data < quantile):
            return(1)
        elif (data == quantile):
            return(0.5)
        else:
            return(0)
        
    def update_z_vals(new_data, quantiles, current_z_vals, t, w):
        #updates the z values (eCDF computed from all data seen so far)
        #based on a new point 
        #takes as input the new data and the quantiles to update the 
        #current_z_vals at t is the number of points seen so far (hence it
        #is not an unbounded method), and w is the window size
        
        num_quantiles = len(quantiles)
        for i in range(num_quantiles):
            current_z_vals[i] *= (t-w)   #rescale existing z values
            current_z_vals[i] += one_point_emp_dist(new_data, quantiles[i])
            current_z_vals[i] /= (t - w + 1)
        #O(K) as we only use for new_data of a single point
        return(current_z_vals)
    
    def update_window_ecdf(new_data, old_data, quantiles, current_ecdf, w):
        #this function updates the ecdf for the current window of data
        #based on the first observed point leaving the window and a new point
        #being added to it.
        num_quantiles = len(quantiles)
        for i in range(num_quantiles):
            current_ecdf[i] *= w
            current_ecdf[i] -= one_point_emp_dist(old_data, quantiles[i]) 
            current_ecdf[i] += one_point_emp_dist(new_data, quantiles[i]) 
            current_ecdf[i] /= w  
        return current_ecdf
    
    #begin algorithm
    tree = SortedList()
    window = deque([],w)
    dtime = 0
    cost_history = []
    #get initial z vals (eCDF values for historic data) from first w points
    for x in X[:w] : #use tree to get O(log(w)) initial quantiles
        tree,window = update(tree,window,x)
        dtime += 1
    Q = quantiles(tree,k,len(window)) #compute fixed quantiles
    z_vals = [eCDF_vals(tree, q) for q in Q] #compute initial z_vals 
    window_ecdf_vals = z_vals.copy() #current ecdf
    
    #we start testing from index 2w-1 as this means we have seen w points to 
    #estimate the initial z values and then have w new points to test.
    #this means that from index w to 2w-2 we only update the current 
    #window ecdf. At 2w-1 onwards we update the z values with window[0], the 
    #point that is leaving the window and moving into the history and use
    #window[0] and x to update the current window eCDF:
        
    for x in X[w:2*w-1]:
        window_ecdf_vals = update_window_ecdf(x, window[0], Q,
                                              window_ecdf_vals, w)
        window = update_window(window,x)
        dtime += 1
        
        #no need to update z vals as we pop values already in initial estimate

    for x in X[2*w-1:] : 
        dtime += 1
        #update z vals with point that is going to leave the window,
        #and then update tree and window
        window_ecdf_vals = update_window_ecdf(x, window[0], Q,
                                              window_ecdf_vals, w)
        window = update_window(window,x)
        #compute eCDFs for weighted data           
        weighted_ecdf = [((dtime - w) / dtime)*z_vals[i] + (w / dtime)*window_ecdf_vals[i]
                         for i in range(k)]
        #compute cost functions and overall cost
        historic_cost = sum([cdf_cost(val, dtime - w) for val in z_vals])
        window_cost = sum([cdf_cost(val, w) for val in window_ecdf_vals])
        full_cost = sum([cdf_cost(val, dtime) for val in weighted_ecdf])
        overall_cost = 2*(historic_cost + window_cost - full_cost)
        #update z_vals with the point that is to leave window
        z_vals = update_z_vals(window[0], Q, z_vals, dtime, w)
        cost_history.append(overall_cost) #store cost
        if overall_cost > threshold : #if true anomaly detected
            return(dtime-w,dtime,overall_cost, cost_history)
        #update z vals with z val leaving window
    max_cost = max(cost_history)
    #if no anomaly is detected, return max seen so far and -1 for detection
    return (-1, -1, max_cost, cost_history)


############
#NUNC Offline Wrapper
############

def nunc_offline(data, k, w, threshold, method = "local"):
    
    '''
    Function for performing NUNC in an offline setting.
    
    Description
    ------------
    
    NUNC offline applies the NUNC algorithm to a pre-observed stream of data
    and searches this data for changes in distribution using a sliding window.
    
    Two different variants of NUNC exist: "NUNC Local" and "NUNC Global".
    Each of these three variants tests for changes in distribution through use
    of a cost function that makes a comparison between the pre and post change
    empirical CDFs for the data. This comparison is aggregated over K quantiles,
    as this enhances the power of the test by comparing both the centre, and
    tails, of the estimated distributions.
    
    The two different methods can be described as follows:

    NUNC Local
    This method searches for a change in distribution inside the points of
    data contained in the sliding window. An approximation for this algorithm
    can also be specified, that only searches a subset of the points in the
    sliding window for a change in order to enhance computational efficiency.

    NUNC Global
    This method tests if the data in the sliding window is drawn from a
    different distribution to the historic data.
    
    Parameters
    --------------
    
    data: list
        List of data to test using NUNC.
    threshold: float
        Threshold for the NUNC test.
    w: int
        Window size used by NUNC.
    k: int
        Number of quantiles used by NUNC. Must be less than the 
        size of the window.
    method: string
        To specify either "local" or "global" variant of NUNC.
        
    Returns
    ---------
    NUNC Object: NUNC_out
        An object of class NUNC_out containing the detection time, changepoint,
        max of the test statistics, list of test statistics for
        each window that is checked, and the data that was inputted.
    
    '''
    
    if not isinstance(method, str):
        raise TypeError("method must be a string of either local or global")
    
    if method.lower() == "local":
        (pos, dtime, cost_max, cost_history) = nunc_local(data, k, w, threshold)
    elif method.lower() == "global":
        (pos, dtime, cost_max, cost_history) = nunc_global(data, k, w, threshold)
    else:
        raise ValueError("method must be either local or global")
    
    res = NUNC_out(dtime, pos, cost_max, cost_history, data)
    return(res)
        
class NUNC_out:
    
    '''
    Class for storing the output of the NUNC offline algorithm.
    
    Attributes
    ------------
    changepoint: int
        The time NUNC identifies as the changepoint. -1 if no change
    detection_time: int
        The time NUNC identifies the changepoint. None if no change detected.
    cost_max: float
        The max of the costs observed - if a change is detected this will be
        the first value to exceed the threshold.
    cost_vec: list
        The list of max of NUNC test statistics for each window. 
    data: list
        The list of data inputted.
    
    Methods
    -----------
    
    plot(None):
        Method to plot the data, and (if detected the )changepoint and 
        time of detection
            
    '''
    
    def __init__(self, dtime, pos, cost_max, cost_history, data):
        self.detection_time = dtime
        self.changepoint = pos
        self.cost_max = cost_max
        self.cost_history = cost_history
        self.data = data
        
    def plot(self):
        data_length = len(self.data)
        data_length = len(self.data)
        x_points = list(range(data_length))
        plt.plot(x_points, self.data)
        plt.xlabel("Time")
        plt.ylabel("Value")
        
        if self.detection_time != -1:
            plt.axvline(x = self.changepoint, color = "red", 
                        label = "changepoint")
            plt.axvline(x = self.detection_time, color = "blue",
                        label = "Detection Time")
            plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
            
