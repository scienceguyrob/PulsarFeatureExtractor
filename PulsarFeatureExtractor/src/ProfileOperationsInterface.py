"""
This file is part of the PulsarFeatureExtractor.

PulsarFeatureExtractor is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PulsarFeatureExtractor is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PulsarFeatureExtractor.  If not, see <http://www.gnu.org/licenses/>.

File name:    ProfileOperationsInterface.py
Created:      February 7th, 2014
Author:       Rob Lyon
 
Contact:    rob@scienceguyrob.com or robert.lyon@postgrad.manchester.ac.uk
Web:        <http://www.scienceguyrob.com> or <http://www.cs.manchester.ac.uk> 
            or <http://www.jb.man.ac.uk>
            
This code runs on python 2.4 or later.

Here we define an interface for the operations
that can be run on profile data loaded in from phcx or pfd files.

"""

# Scipy/numpy imports.
from numpy import ceil
from scipy import stats

# Custom file imports.
from Utilities import Utilities

# ****************************************************************************************************
#
# CLASS DEFINITION
#
# ****************************************************************************************************
    
class ProfileOperationsInterface(Utilities):
    """
    Basically an interface that defines the functions which must be implemented in order
    to produce candidate scores.
    
    If you want to create a new score generation method simply create a sub-class of this file,
    and implement the required functions. This makes the code much more modular.
    """
        
    # ****************************************************************************************************
    #
    # Functions.
    #
    # ****************************************************************************************************
    
    def __init__(self,debugFlag):
        Utilities.__init__(self,debugFlag)
    
    # ****************************************************************************************************
    #
    # Sinusoid Fittings
    #
    # ****************************************************************************************************
    
    def getSinusoidFittings(self,profile):
        raise NotImplementedError("Please Implement this method")
    
    def fitSineSqr(self,yData,maxima):
        raise NotImplementedError("Please Implement this method")
    
    # ****************************************************************************************************
    #
    # Gaussian Fittings
    #
    # ****************************************************************************************************
    
    def getGaussianFittings(self,profile):
        raise NotImplementedError("Please Implement this method")
    
    def fitGaussian(self,xData,yData):
        raise NotImplementedError("Please Implement this method")
    
    def fitGaussianFixedWidthBins(self,xData,yData,bins):
        raise NotImplementedError("Please Implement this method")
    
    def fitGaussianWithBackground(self,xData,yData):
        raise NotImplementedError("Please Implement this method")
    
    def fitGaussianT1(self,yData):
        raise NotImplementedError("Please Implement this method")
    
    def fitDoubleGaussianT2(self,yData):
        raise NotImplementedError("Please Implement this method")
    
    def fitDoubleGaussian(self,yData):
        raise NotImplementedError("Please Implement this method")
    
    def fitDoubleGaussianWithBackground(self,yData,p0):
        raise NotImplementedError("Please Implement this method")
    
    # ****************************************************************************************************
    #
    # Candidate Parameter Functions
    #
    # ****************************************************************************************************
    
    def getCandidateParameters(self,profile):
        raise NotImplementedError("Please Implement this method")
    
    # ****************************************************************************************************
    #
    # DM Curve Fitting Functions
    #
    # ****************************************************************************************************
    
    def getDMFittings(self,data):
        raise NotImplementedError("Please Implement this method")
    
    # ****************************************************************************************************
    #
    # Sub-band Functions
    #
    # ****************************************************************************************************
    
    def getSubbandParameters(self,data=None,profile=None):
        raise NotImplementedError("Please Implement this method")
    
    # ****************************************************************************************************
    #
    # Utility Functions
    #
    # ****************************************************************************************************
    
    def freedmanDiaconisRule(self,data):
        """
        Calculate number of bins to use in histogram according to this rule.
        
        Parameters:
        data    -    a numpy.ndarray containing the data for which a histogram is to be computed.
        
        Returns:
        
        The 'optimal' number of bins for the histogram.   
        """
        # interquartile range, Q3-Q1....
        iqr = stats.scoreatpercentile(data, 75) - stats.scoreatpercentile(data, 25)
        binwidth = 2 * iqr * pow(len(data), -0.3333333)
        
        if(binwidth<=0):
            binwidth=60
            
        # calculate n bins
        rnge = max(data) - min(data)
        nbins = ceil( rnge / binwidth )
        
        if(self.debug):
            print "\tIQR: ",iqr
            print "\tBin Width: ",binwidth
            print "\tRange: ",rnge
            print "\tNumber of bins: ", nbins
            
        return int(nbins)
    
    # ****************************************************************************************************
    
    def getDerivative(self,yData):
        """
        Obtains the derivative for the y data points by simply performing,
        dy = y[i] - y[i+1] .
        
        Parameters:
        yData    -    a numpy.ndarray containing data (y-axis data).
        
        Returns:
        The changes in y, dy, for each point in yData as an array.
        
        """
        dy = []
        dataPoints = len(yData)-1 # Since there are n data points, with only n-1 line segments joining them.
        for i in range(dataPoints):
            dy.append(yData[i] - yData[i+1])
        return dy
    
    # ****************************************************************************************************
    
    def scale(self,x,min_,max_,newMin,newMax):
        """
        Re-scales a data value occurring in the range min and max, the
        a new data range given by newMin and newMax.
        
        Parameter:
        x        -    the data value to rescale.
        min_     -    the minimum value of the original data range for x.
        max_     -    the maximum value of the original data range for x.
        newMin   -    the minimum value of the new data range for x.
        newMax   -    the maximum value of the new data range for x.
        
        Returns:
        A new array with the data scaled to within the range [newMin,newMax].
        """
        
        x = (newMin * (1-( (x-min_) /( max_-min_ )))) + (newMax * ( (x-min_) /( max_-min_ ) ))
        return x
    
    # ****************************************************************************************************