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

File name:    ProfileOperations.py
Created:      February 7th, 2014
Author:       Rob Lyon
 
Contact:    rob@scienceguyrob.com or robert.lyon@postgrad.manchester.ac.uk
Web:        <http://www.scienceguyrob.com> or <http://www.cs.manchester.ac.uk> 
            or <http://www.jb.man.ac.uk>
            
This code runs on python 2.4 or later.

Contains implementations of the function used to generate scores.

+-----------------------------------------------------------------------------------------+
+                       PLEASE RECORD ANY MODIFICATIONS YOU MAKE BELOW                    +
+-----------------------------------------------------------------------------------------+
+ Revision |   Author    | Description                                       |    DATE    +
+-----------------------------------------------------------------------------------------+

 Revision:1    Rob Lyon    Initial version of the re-written code.            07/01/2014 
 
 Revision:2    Rob Lyon     Changes made to fit_sine(self,yData,maxima):      29/01/2014
 
                            a) Added MatLibPlot import, now plots sine curve 
                               fitted to the pulse profile.
                            b) Added an additional input to the __residuals
                               and __evaluate functions, the amplitude of
                               the peak and the background (fixed values).
                            c) Fixed the amplitude in the fit_sine function
                               so it now fixes amp=max(yData)-min(yData)/2.
                               This was done so that score 1 would not try
                               to fit just any sine curve, but one with the
                               max peak amplitude of the profile. Also fixed
                               the background value to the mean profile 
                               intensity.
                            d) Altered the parameters received by the call
                               to scipy.optimize.leastsq() in fit_sine()
                               function. Rather than just returning the
                               parameters of the fit, we now obtain the
                               covariance matrix too along with other variables.
                               This enables the R-squared value and other stats
                               to be computed.
                            e) Now produces plot of sine fit to pulse profile
                               in fit_sine(self,yData,maxima). Also computes
                               the R-squared value for the fit and other stats
                               i.e. the standard error and total error.
                            f) Placed original fit_sine(self,yData,maxima) under
                               new method name: fit_sine_original(), although
                               this code has the addition of matlibplot calls.
                               
Revision:3    Rob Lyon      Changes made to def fit_sine_sqr(self,yData,maxima): 29/01/2014 

                            a) Added an additional input to the __residuals
                               and __evaluate functions, the amplitude of
                               the peak (fixed value).
                            b) Fixed the amplitude in the fit_sine function
                               so it now fixes amp=max(yData)-min(yData)/2.
                               This was done so that score 1 would not try
                               to fit just any sine curve, but one with the
                               max peak amplitude of the profile. Also fixed
                               the background value to the mean profile 
                               intensity.
                            c) Altered the parameters received by the call
                               to scipy.optimize.leastsq() in fit_sine_sqr()
                               function. Rather than just returning the
                               parameters of the fit, we now obtain the
                               covarience matrix too along with other variables.
                               This enables the R-squared value and other stats
                               to be computed.
                            d) Now produces plot of sine squared fit to pulse proifle
                               in fit_sine_sqr(self,yData,maxima). Also computes
                               the R-squared value for the fit and other stats
                               i.e. the standard error and total error.
                            e) Placed original fit_sine_sqr(self,yData,maxima) under
                               new method name: fit_sine_sqr_original(), although
                               this code has the addition of matlibplot calls.

Revision:4    Rob Lyon      Added method that calculates number of bins needed  30/01/2014 
                            in a historgram, code provided by Sam Bates; uses
                            the freedman diaconis rule. Cheers Sam.
                            Look for def freedman_diaconis_rule(self,data)
                            
Revision:5    Rob Lyon      Created new version of the method:                  30/01/2014
                            def fit_gaussian_fixed_with_bins(self,xData,yData)
                            Changed the method so that it now uses the specific
                            number of bins for the histogram data passed in to
                            the method.

Revision:6    Rob Lyon      Changed Chi-squared calculation.                    30/01/2014
                            
                            a) In fit_sine(self,yData,maxima) changed chi
                            squared calculation (removed a restriction). 
                            
                            b) In fit_sine_sqr(self,yData,maxima) changed chi
                            squared calculation (removed a restriction).
                            
                            c) Added chisq /= len(yData) to the function:
                               fit_gaussian(self,xData,yData).
                               
                            d) In fit_gaussian_fixed(self,xData,yData) changed
                               chi-squared calculation (again removed a
                               restriction).
                            
Revision:7    Rob Lyon      Added background terms to __residuals and           30/01/2014
                            __evvaluate in fit_sine_sqr() function.
                            
Revision:8    Rob Lyon      Now take the absolute value of sigma,               30/01/2014
                            in def fit_gaussian_with_bg(self,xData,yData),
                            such that:

                            a) In __residuals(x, paras) we have:
                                err = y - ( abs(maximum) * exp( (-((x - mu) / abs(sigma) )**2) / 2) + (bg) )
                            
                            b) In __evaluate(x, paras) we have:
                                return ( abs(maximum) * exp( (-((x - mu) / abs(sigma) )**2) / 2) + (bg) )
                                
                            This change was to prevent errors caused by sigma
                            obtaining negative values.
                             
"""

# Numpy Imports:
from numpy import array
from numpy import ceil
from numpy import argmax
from numpy import delete
from numpy import sin
from numpy import pi
from numpy import exp
from numpy import sqrt
from numpy import log
from numpy import mean
from numpy import histogram
from numpy import corrcoef
from numpy import append

from scipy.optimize import leastsq
from scipy import std

import matplotlib.pyplot as plt # Revision:1

# Custom file Imports:
from ProfileOperationsInterface import ProfileOperationsInterface

# ****************************************************************************************************
#
# CLASS DEFINITION
#
# ****************************************************************************************************

class ProfileOperations(ProfileOperationsInterface):
    """                
    Contains the functions used to generate the scores that describe the key features of
    a pulsar candidate.
    
    """
    
    # ****************************************************************************************************
    #
    # Constructor.
    #
    # ****************************************************************************************************
    
    def __init__(self,debugFlag):
        ProfileOperationsInterface.__init__(self,debugFlag)
        # Set default bin width, won't be used since it is now dynamically recomputed.
        self.histogramBins = 60

    # ****************************************************************************************************
    #
    # Sinusoid Fittings
    #
    # ****************************************************************************************************
          
    def getSinusoidFittings(self,profile):
        """
        Computes the sinusoid fitting scores for the profile data. There are four scores computed:
        
        Score 1. Chi-Squared value for sine fit to raw profile. This score attempts to fit a sine curve
                 to the pulse profile. The reason for doing this is that many forms of RFI are sinusoidal.
                 Thus the chi-squared value for such a fit should be low for RFI (indicating
                 a close fit) and high for a signal of interest (indicating a poor fit).
                 
        Score 2. Chi-Squared value for sine-squared fit to amended profile. This score attempts to fit a sine
                 squared curve to the pulse profile, on the understanding that a sine-squared curve is similar
                 to legitimate pulsar emission. Thus the chi-squared value for such a fit should be low for
                 RFI (indicating a close fit) and high for a signal of interest (indicating a poor fit).
                 
        Score 3. Difference between maxima. This is the number of peaks the program identifies in the pulse
                 profile - 1. Too high a value may indicate that a candidate is caused by RFI. If there is only
                 one pulse in the profile this value should be zero.
                 
        Score 4. Sum over residuals.  Given a pulse profile represented by an array of profile intensities P,
                 the sum over residuals subtracts ( (max-min) /2) from each value in P. A larger sum generally
                 means a higher SNR and hence other scores will also be stronger, such as correlation between
                 sub-bands. Example,
                 
                 P = [ 10 , 13 , 17 , 50 , 20 , 10 , 5 ]
                 max = 50
                 min = 5
                 (abs(max-min))/2 = 22.5
                 so the sum over residuals is:
                 
                  = (22.5 - 10) + (22.5 - 13) + (22.5 - 17) + (22.5 - 50) + (22.5 - 20) + (22.5 - 10) + (22.5 - 5)
                  = 12.5 + 9.5 + 5.5 + (-27.5) + 2.5 + 12.5 + 17.5
                  = 32.5
        
        Parameters:
        profile    -    a numpy.ndarray containing profile data.
        
        Returns:
        the chi squared value of the sine fit.
        the chi squared value of the sine fit.
        the difference between maxima.
        the sum over the residuals.
        
        """
        profile_mean = profile.mean()
        profile_std = profile.std() # Note this is over n, not n-1.
        profile_max = profile.max()
        profile_min = profile.min()
        
        #print "mean:\t" , profile_mean
        #print "min:\t" , profile_min
        #print "max:\t" , profile_max
        #print "std:\t" , profile_std

        sumOverResiduals = 0
        
        # Calculate sum over residuals.
        for i in range( len(profile) ):
            sumOverResiduals += (abs( profile_max - profile_min ) / 2.)-profile[i]
        #print "Sum Over Residuals:\t",sumOverResiduals
        
        # Subtract background from profile. This is a type of feature scaling or
        # normalization of the data. I'm not sure why the standard score isn't calculated
        # here, i.e. (x-mean) / standard deviation , but there must be a good reason. If
        # you would like to use the standard score, just uncomment/comment as needed.
        #p = (profile - profile_mean) / profile_std
        normalisedProfile = profile - profile_mean - profile_std
        normalisedProfileLength = len(normalisedProfile)
        
        for i in range( normalisedProfileLength ):
            if normalisedProfile[i] < 0:
                normalisedProfile[i] = 0
        
        #print "Profile after normalization:\n",normalisedProfile
        
        # Find peaks in the normalized profile.
        # This code works by looking at small blocks of the normalized profile
        # for maximum values. The array indexes of the maximum values in the
        # normalized profile are then stored in 'peakIndexes'. The 'newProfile'
        # variable contains only those blocks from the normalized profile that
        # contain peaks. For example, a profile as follows:
        #
        #    index:      0   1   2    3   4   5   6   7   8   9   10  11   12  13  14  15
        # 
        #    profile = [ 0 , 0 , 5 , 10 , 5 , 0 , 0 , 0 , 0 , 0 , 0 , 15 , 0 , 0 , 0 , 0]
        #
        # would give:
        #
        #    peakIndexes = [3 , 11]
        #    newProfile  = [0 , 0 , 5 , 10 , 5 , 0 , 0 , 15 , 0 , 0 , 0 , 0]
        #                   |                        |   |                |
        #                   -------------------------    -----------------
        #                                |                        |
        #                                v                        v
        #                             Block 1                 Block 2
        #
        # Each block contains four zeros. So in this code a peak appears to be defined
        # as the maximum value occurring in a block of the normalized profile separated
        # by 4 bins with a normalized intensity of zero. Note that a block containing
        # intensities of only zero will be ignored. In this example the data in indexes
        # 7-10 was ignored.
        #
        # Note: I'm not sure why four zeroes was chosen, but I certainly won't change it!
        #       Changing it would give very different results.
    
        tempBinIndexes, tempBinValues, peakIndexes, newProfile = [],[],[],[] # 4 new array variables.
        zeroCounter = 0
        
        for i in range( normalisedProfileLength ):
            
            # If intensity at index i is not equal to zero, there is some signal.
            # This is not necessarily a peak.
            if normalisedProfile[i] != 0:
                tempBinValues.append(normalisedProfile[i])
                tempBinIndexes.append(i)
            
            # If four zeroes encountered, increment the counter.
            # This will cause the final else statement to be executed
            # if the next data item is another zero.    
            elif zeroCounter < 4:
                tempBinValues.append(normalisedProfile[i])
                tempBinIndexes.append(i)
                zeroCounter += 1
                
            else:              
                if max(tempBinValues) != 0:# If there is a peak...
                    peakIndexes.append(tempBinIndexes[argmax(tempBinValues)])
                    newProfile += list(tempBinValues)
                
                # Reset for next iteration.
                tempBinIndexes,tempBinValues = [],[]
                zeroCounter = 0
        
        # If there are leftover bins not processed in the loop above...
        if (tempBinValues != []):
            if (max(tempBinValues) != 0):# If there is a peak...
                peakIndexes.append(tempBinIndexes[argmax(tempBinValues)])
                newProfile += list(tempBinValues)# Add to the new profile.
        
        # The newProfile array will contain zero's at the start and end. This is
        # because on line 303 is 4 zeros haven't been seen, then they will be added
        # to the newProfile array. Just in case you wonder where the zeroes are coming
        # from.
        
        # Locate and count maxima.
        maxima = len(peakIndexes)
        
        # Calculate difference between maxima. This code simply subtracts
        # the peaks in the peakIndexes array at the indexes between 1 to n, from 
        # the peak values in the same array at indexes at 0 to (n-1).
        #
        # i.e. if peakIndexes = [ 1 , 2 , 3 , 4 , 5 , 6] , then:
        #    
        #    peakIndexs from 1 to n are   [ 2 , 3 , 4 , 5 , 6]
        #    peakIndexs from 0 to n-1 are [ 1 , 2 , 3 , 4 , 5]
        #
        #    So diff is given by,
        #    diff = [ (2-1) , (3-2) , (4-3) , (5-4) , (6-5) ]
        #         = [ 1 , 1 , 1 , 1 , 1 ]
        if maxima > 0:
            diff = delete(peakIndexes,0) - delete(peakIndexes,maxima-1)
        else:
            diff = []
        
        # Delete zeros in newProfile array. Does not delete all zero's however.
        # It leaves a single zero in between each block with data.
        
        finalProfile = []
        zeroCounter , i = 0,0
        while i < len(newProfile):
            if newProfile[i] != 0:
                finalProfile.append(newProfile[i])
                zeroCounter = 0
                
            elif zeroCounter < 1:
                finalProfile.append(newProfile[i])
                zeroCounter += 1
                
            i += 1 # Increment loop variable.
        
        #print "Final Profile for Sine fitting:\n",finalProfile
        
        # Perform fits to profile.
        # Divide chi-squared by maxima to reduce scores of data with many peaks.
        chisq_profile_sine_fit = self.fitSine(profile,maxima)/maxima # Fit sine curve to raw profile.
        chisq_finalProfile_sine_sqr_fit = self.fitSineSqr(profile,maxima)/maxima # Fit sine-squared curve to amended profile.
        
        return chisq_profile_sine_fit , chisq_finalProfile_sine_sqr_fit , float(len (diff)) , sumOverResiduals
    
    # ******************************************************************************************
    
    def fitSine(self,yData,maxima):
        """
        Fits a sine curve to data and returns the chi-squared value of the fit. Here
        the amplitude is fixed to max(yData) - min(yData) ) / 2 and the background term
        is fixed to the same value.
        
        Parameters:
        yData    -    a numpy.ndarray containing the data to fit the curve to (y-axis data).
        maxima   -    the number of maxima in the data.
         
        Returns:
        The chi-squared value of the fit.
        
        """
        
        # Obtain parameters for fitting.
        xData = array(range(len(yData)))
        amplitude = abs( max(yData) - min(yData) ) / 2.
        frequency = float( maxima / (len(yData) - 1.) )
        # The background terms decides where the middle of the sine curve will be,
        # i.e. smaller moves the curve down the y-axis, higher moves the curve up the
        # y-axis.
        background = abs( max(yData) - min(yData) ) / 2.
        
        # Calculates the residuals.
        def __residuals(paras, x, y,amp,bg): # Revision:2b
            # amp = the amplitude
            # f = the frequency
            # pi = Good old pi or 3.14159... mmmm pi.
            # phi = the phase.
            # bg = the mean of the data, center amplitude.
            # err = error.
            
            # Remember that here x and y are the data, such that,
            # x = bin number
            # y = intensity in bin number x
            f, phi = paras

            err = y - (abs(amp) * sin( 2 * pi * f * x + phi) + abs(bg))
            return err
        
        # Evaluates the function.
        def __evaluate(x, paras,amp,bg): # Revision:2b
            # Same variables as above.
            f, phi = paras
            return abs(amp) * sin( 2 * pi * f * x + phi) + abs(bg)
        
        if yData[0] == background:
            phi0 = 0
        elif yData[0] < background:
            try:
                phi0 = -1 / (4 * frequency)
            except ZeroDivisionError:
                phi0 = -1.0 / (4.0 * 0.00000000001)
        elif yData[0] > background:
            try:
                phi0 = +1 / (4 * frequency)
            except ZeroDivisionError:
                phi0 = +1.0 / (4.0 * 0.00000000001)
            
        # Perform sine fit.
        parameters = (frequency,phi0)
        
        # This call to leastsq() uses the full-output=True flag so that we can compute the 
        # R2 and other stats. This makes it easier to validate and debug the resulting fit
        # in other tools like Matlab. The original code is left below, just uncomment and
        # remove new code if necessary (1).
        #leastSquaresParameters = leastsq(__residuals, parameters, args=(xData,yData,amplitude),full_output=True)
        #fit = __evaluate(xData, leastSquaresParameters[0],amplitude)
        leastSquaresParameters,cov,infodict,mesg,ier = leastsq(__residuals, parameters, args=(xData,yData,amplitude,background),full_output=True) # @UnusedVariable
        fit = __evaluate(xData, leastSquaresParameters,amplitude,background) # Revision:2c
        
        # Chi-squared fit. Revision:6a
        chisq = 0
        for i in range(len(yData)):
            #if yData[i] >= 5.: # Not sure why this restriction is here. Removed for Revision:6a
            chisq += (yData[i]-fit[i])**2
        
        chisq /= len(yData)
        
        # Note leastSquaresParameters[0] contains the parameters of the fit obtained
        # by the least squares optimize call, [frequency,phi0,background].
        #print "Least squares parameters:\n", leastSquaresParameters[0]# This is used if the area below (1) above is uncommented.
        #print "Least squares parameters Full:\n", leastSquaresParameters
        #print "Chi Squared:\n", chisq*pow(float(maxima),4)/100000000.  
        #print "fit:\n", fit
        
        # This section should be commented out when testing is completed.
        if(self.debug):
            ssErr = (infodict['fvec']**2).sum() # 'fvec' is an array of residuals. 
            yData = array(yData)
            ssTot = ((yData-yData.mean())**2).sum()
            rsquared = 1-(ssErr/ssTot )
            
            print "\n\tSine fit to Pulse profile statistics:"
            print "\tStandard Error: ", ssErr
            print "\tTotal Error: ", ssTot
            print "\tR-Squared: ", rsquared
            print "\tAmplitude: ",amplitude
            print "\tFrequency: ",str(leastSquaresParameters[0])
            print "\tPhi: ",str(leastSquaresParameters[1])
            print "\tBackground: ",background
            plt.plot(xData,yData,'o', xData, __evaluate(xData, leastSquaresParameters,amplitude,background))
            plt.title("Sine fit to Profile")
            plt.show()
        
        #return leastSquaresParameters[0], chisq*pow(maxima,4)/100000000., fit, xData, yData
        # I've commented out the return statement above, as only the chi-squared value is used.
        # By not returning the extra items, the memory they use will be freed up when this
        # function terminates, reducing memory overhead.
        #return chisq*pow(float(maxima),4)/100000000.
        return chisq
    
    # ******************************************************************************************
    
    def fitSineSqr(self,yData,maxima):
        """
        Fits a sine-squared curve to data and returns the chi-squared value of the fit.
        
        Parameters:
        yData    -    a numpy.ndarray containing the data to fit the curve to (y-axis data).
        maxima  -    the number of maxima in the data.
        
        Returns:
        The chi-squared value of the fit.
        
        """
        
        # Calculates the residuals.
        def __residuals(paras, x, y,amp,bg): # Revision:3a
            # a = the amplitude
            # f = the frequency
            # pi = Good old pi or 3.14159... mmmm pi.
            # phi = the phase.
            # err = error.
            # bg = background term
            
            # Remembmer that here x and y are the data, such that,
            # x = bin number.
            # y = intensity in bin number x.
            f, phi = paras

            err = y - (abs(amp) * pow ( sin ( 2 * pi * f * x + phi),2)) + abs(bg)
            return err
        
        # Evaluates the function.
        def __evaluate(x, paras,amp,bg): # Revision:3a
            # Same variables as above.
            f, phi = paras
            return abs(amp) * pow ( sin ( 2 * pi * f * x + phi),2) + abs(bg)
        
        # Obtain parameters for fitting.
        xData = array(range(len(yData)))
        #amplitude = max(yData)
        amplitude = abs( max(yData) - min(yData) ) / 2. # Revision:3b
        frequency = float( maxima / (len(yData) - 1.) / 2. )
        background = abs( max(yData) - min(yData) ) / 2. # Revision:7
        
        if yData[0] == 0:
            phi0 = 0
        else:
            try:
                phi0 = -1 / (4 * frequency)
            except ZeroDivisionError:
                phi0 = -1.0 / (4.0 * 0.00000000001)
            
        # Perform sine fit.
        parameters = (frequency,phi0)
        # Revision:3c
        leastSquaresParameters,cov,infodict,mesg,ier = leastsq(__residuals, parameters, args=(xData,yData,amplitude,background),full_output=True)#@UnusedVariable
        fit = __evaluate(xData, leastSquaresParameters,amplitude,background)
        
        # Chi-squared fit. Revision:6b
        chisq = 0
        for i in range(len(yData)):
            #if yData[i] >= 5.: # Not sure why this restriction is here. Removed for Revision:6b
            chisq += (yData[i]-fit[i])**2
        
        chisq /= len(yData)
        
        #print "Least squares parameters:\n", leastSquaresParameters[0]
        #print "Chi Squared:\n", chisq / pow(float(maxima),4)  
        #print "fit:\n", fit
        
        # Revision:3d
        # This section should be commented out when testing is completed.
        if(self.debug):
            ssErr = (infodict['fvec']**2).sum() # 'fvec' is an array of residuals. 
            ssTot = ((yData-mean(yData))**2).sum()
            rsquared = 1-(ssErr/ssTot )
            
            print "\n\tSine Squared fit to Pulse profile statistics:"
            print "\tStandard Error: ", ssErr
            print "\tTotal Error: ", ssTot
            print "\tR-Squared: ", rsquared
            print "\tAmplitude: ",amplitude
            print "\tFrequency: ",str(leastSquaresParameters[0])
            print "\tPhi: ",str(leastSquaresParameters[1])
            plt.plot(xData,yData,'o', xData, __evaluate(xData, leastSquaresParameters,amplitude,background))
            plt.title("Sine Squared fit to Profile")
            plt.show()
        
        
        #return leastSquaresParameters[0], chisq / pow(float(maxima),4), fit, xData, yData
        # I've commented out return statement above, as only the chi-squared value is used.
        # By not returning the extra items, the memory they use will be freed up when this
        # function terminates, reducing memory overhead.
        return chisq
    
    # ****************************************************************************************************
    #
    # Gaussian Fittings
    #
    # ****************************************************************************************************
    
    def getGaussianFittings(self,profile):
        """
        Computes the Gaussian fitting scores for the profile data. There are seven scores computed:
        
        Score 5. Distance between expectation values of Gaussian and fixed Gaussian fits to profile histogram.
                 This scores fits a two Gaussian curves to a histogram of the profile data. One of these
                 Gaussian fits has its mean value set to the value in the centre bin of the histogram,
                 the other is not constrained. Thus it is expected that for a candidate arising from noise,
                 these two fits will be very similar - the distance between them will be zero. However a
                 legitimate signal should be different giving rise to a higher score value.
                 
        Score 6. Ratio of the maximum values of Gaussian and fixed Gaussian fits to profile histogram.
                 The score compute the maximum height of the fixed Gaussian curve (mean fixed to the centre
                 bin) to the profile histogram, and the maximum height of the non-fixed Gaussian curve
                 to the profile histogram. This ratio will be equal to 1 for perfect noise, or close to zero
                 for legitimate pulsar emission.
        
        Score 7. Distance between expectation values of derivative histogram and profile histogram. A histogram
                 of profile derivatives is computed. This score finds the absolute value of the mean of the 
                 derivative histogram, minus the mean of the profile histogram. A value close to zero indicates 
                 a candidate arising from noise, a value greater than zero some form of legitimate signal.
        
        Score 8. Full-width-half-maximum (FWHM) of Gaussian fit to pulse profile. Describes the width of the
                 pulse, i.e. the width of the Gaussian fit of the pulse profile. Equal to 2*sqrt( 2 ln(2) )*sigma.
                 Not clear whether a higher or lower value is desirable.
        
        Score 9. Chi squared value from Gaussian fit to pulse profile. Lower values are indicators of a close fit,
                 and a possible profile source.
        
        Score 10. Smallest FWHM of double-Gaussian fit to pulse profile. Some pulsars have a doubly peaked
                  profile. This score fits two Gaussians to the pulse profile, then computes the FWHM of this
                  double Gaussian fit. Not clear if higher or lower values are desired.
        
        Score 11. Chi squared value from double Gaussian fit to pulse profile. Smaller values are indicators
                  of a close fit and possible pulsar source.
        
        Parameters:
        profile    -    a numpy.ndarray containing profile data.
        
        Returns:
        the chi squared value of the sine fit.
        the chi squared value of the sine fit.
        the difference between maxima.
        the sum over the residuals.
        
        """
        
        # Stores the scores obtained just by this function.
        scores=[]
        
        # This is a deprecated call of a numpy library. The 'new' parameter is not
        # included in newer versions, but I haven't removed it here since the Jodrell
        # servers may be using an older version of numpy/scipy. Instead just uncomment or
        # comment as appropriate to use the original call. Here I believe that 60
        # is the number of bins used in the histogram by default.
        #histogram_dy = histogram(dy,60,new=True)
        #derivative_bins = ph.freedmanDiaconisRule(dy)
        #print "Bins for derivative histogram: ",derivative_bins
        
        self.histogramBins = self.freedmanDiaconisRule(profile)
        dy = self.getDerivative(profile)
        self.dy_histogramBins = self.freedmanDiaconisRule(dy)
        histogram_dy = histogram(dy,self.dy_histogramBins) # Calculates a histogram of the derivative dy.
        
        # Performs a gaussian fit on the derivative histogram.
        gaussianFitToDerivativeHistogram = self.fitGaussian(histogram_dy[1],histogram_dy[0])
        derivativeHistogram_sigma, derivativeHistogram_expect, derivativeHistogram_maximum = gaussianFitToDerivativeHistogram[0]
        
        if(self.debug==True):
            print "\n\tGaussian fit to Derivative Histogram details: " 
            print "\tSigma of derivative histogram = " , derivativeHistogram_sigma
            print "\tMu of derivative histogram = "    , derivativeHistogram_expect
            print "\tMax of derivative histogram = "   , derivativeHistogram_maximum
        
            # View histogram - for debugging only... uncomment matlibplot import at top if needed.
            
            hist, bins = histogram(dy,self.dy_histogramBins) # Calculates a histogram of the derivative.
            center = (bins[:-1] + bins[1:]) / 2
            plt.bar(center, hist, align='center')
            plt.title("Histogram of derivative dy")
            plt.show()
            
                
        histogram_profile = histogram(profile,self.histogramBins) # Calculates a histogram of the profile data.
                
        # Performs a gaussian fit on the profile histogram.
        gaussianFitToProfileHistogram = self.fitGaussian(histogram_profile[1],histogram_profile[0])
        profileHistogram_sigma, profileHistogram_expect, profileHistogram_maximum = gaussianFitToProfileHistogram[0]
        
        if(self.debug==True):
            print "\n\tGaussian fit to Profile Histogram details: " 
            print "\tSigma of profile histogram = " , profileHistogram_sigma
            print "\tMu of profile histogram = "    , profileHistogram_expect
            print "\tMax of profile histogram = "   , profileHistogram_maximum
            
            # View histogram - for debugging only... uncomment matlibplot import at top if needed.
            
            hist, bins = histogram(profile,self.histogramBins) # Calculates a histogram of the profile.
            center = (bins[:-1] + bins[1:]) / 2
            plt.bar(center, hist, align='center')
            plt.title("Histogram of profile")
            plt.show()
            
        
        # Here gf refers to Gaussian fit.
        # Performs a gaussian fit with fixed expectation value on the profile histogram.
        gf_ProfileHistogram_fixed_Expect = self.fitGaussianFixedWidthBins(histogram_profile[1],histogram_profile[0],self.histogramBins)
        gf_ProfileHistogram_fixed_sigma, gf_ProfileHistogram_fixed_maximum = gf_ProfileHistogram_fixed_Expect[0]
        gf_ProfileHistogram_fixed_fwhm = gf_ProfileHistogram_fixed_Expect[1]
        gf_ProfileHistogram_fixed_chi  = gf_ProfileHistogram_fixed_Expect[2] 
        gf_ProfileHistogram_fixed_xmax = gf_ProfileHistogram_fixed_Expect[4]
        
        if(self.debug==True):
            print "\n\tGaussian fits to Profile Historgram with fixed Mu details:" 
            print "\tSigma of Gaussian fit to Profile Historgram = "         , gf_ProfileHistogram_fixed_sigma
            print "\tMax of Gaussian fit to Profile Historgram = "           , gf_ProfileHistogram_fixed_maximum
            print "\tFWHM of Gaussian fit to Profile Historgram = "          , gf_ProfileHistogram_fixed_fwhm
            print "\tChi-squared of Gaussian fit to Profile Historgram = "   , gf_ProfileHistogram_fixed_chi
            print "\txmax of Gaussian fit to Profile Historgram = "  
        
        dexp_fix = abs(gf_ProfileHistogram_fixed_xmax - profileHistogram_expect)      # Score 5.
        amp_fix =  abs( gf_ProfileHistogram_fixed_maximum / profileHistogram_maximum) # Score 6.
        dexp = abs(derivativeHistogram_expect - profileHistogram_expect)              # Score 7.
        
        # Add scores.
        scores.append(float(dexp_fix)) # Score 5. Distance between expectation values of Gaussian and fixed Gaussian fits to profile histogram.
        scores.append(float(amp_fix))  # Score 6. Ratio of the maximum values of Gaussian and fixed Gaussian fits to profile histogram.
        scores.append(float(dexp))     # Score 7. Distance between expectation values of derivative histogram and profile histogram.
            
        minbg = min(profileHistogram_expect,profile.mean()) # Estimate background.                
        tempProfile = []

        if minbg > 0.:                      
                                
            for i in range(len(profile)):                
                newy = profile[i] - minbg + profile.std()          # Substract background from profile
                if newy < 0.:                          # and store the new profile in list temp
                    newy = 0.
                                        
                tempProfile.append(newy)
        else:                            
            tempProfile = profile                      
        
        # Here gf refers to Gaussian fit
        gf_profile_result = self.fitGaussianT1(tempProfile)
        gf_profile_fwhm, gf_profile_chi = gf_profile_result[1], gf_profile_result[2]
        
        # Add scores.
        scores.append(float(gf_profile_fwhm)) # Score 8. Full-width-half-maximum (FWHM) of Gaussian fit to pulse profile. 
        scores.append(float(gf_profile_chi))  # Score 9. Chi squared value from Gaussian fit to pulse profile.
    
        # dgf means double Gaussian fit
        try:
            dgf_profile_result = self.fitDoubleGaussianT2(profile) # Double gaussian fit around the maximum of the profile.
            dgf_profile_fwhm1  = dgf_profile_result[1]
            dgf_profile_chi    = dgf_profile_result[2]
            dgf_profile_fwhm2  = dgf_profile_result[6]
            
            # Here profile.std() is the standard deviation of the profile.
            # gf is Gaussian fit, dgf is double Gaussian fit.
            gf_dgf_diff = dgf_profile_result[3] - (gf_profile_result[3] + minbg - profile.std())    # Differences of gaussian fits t1 and t2.
            gf_dgf_std  = float(abs(gf_dgf_diff.std())) # Standard deviation of differences.
            
            if gf_dgf_std < 3.:
                dgf_fwhm = gf_profile_fwhm
            else:
                dgf_fwhm = float(min(dgf_profile_fwhm1 , dgf_profile_fwhm2))
        except IndexError:
            dgf_fwhm = 1000000
            dgf_profile_chi = 1000000
            
        # Add scores.
        scores.append(float(dgf_fwhm))         # Score 10. Smallest FWHM of double-Gaussian fit to pulse profile. 
        scores.append(float(dgf_profile_chi))  # Score 11. Chi squared value from double Gaussian fit to pulse profile.
        
        return scores
    
    # ******************************************************************************************
    
    def fitGaussian(self,xData,yData):
        """
        Fits a Gaussian to the supplied data. This should be histrogram data,
        that is the details of the bins (xData) and the frequencies (yData).
        
        Parameters:
        xData    -    a numpy.ndarray containing data (x-axis data).
        yData    -    a numpy.ndarray containing data (y-axis data).
        
        Returns:
        The parameters of the fit, one array and three other variables.
        
            leastSquaresParameters - array containing optimum three values for:
                                        * sigma
                                        * expect
                                        * maximum
            fwhm - the full width half maximum of the Gaussian.
            chisq - the chi-squared value of the fit.
            fit - the fit.
        
        """
        
        #print "xData (LENGTH=",len(xData),"):\n", xData
        #print "yData (LENGTH=",len(yData),"):\n", yData
        
        # Calculates the residuals.
        def __residuals(paras, x, y):
            # sigma = the standard deviation.
            # mu = the mean aka the expectation of the distribution.
            # maximum = .
            
            # Remembmer that here x and y are the data, such that,
            # x = bin number.
            # y = intensity in bin number x.
            sigma, mu, maximum = paras
            err = y - ( abs(maximum) * exp( (-((x - mu) / sigma )**2) / 2))
            return err
        
        # Evaluates the function.
        def __evaluate(x, paras):
            # Same variables as above.
            sigma, mu, maximum = paras
            return ( abs(maximum) * exp( (-((x - mu) / sigma )**2) / 2))
        
        # Reverses the order of the list entries.
        def __mirror(_list):
            reversedList = []
            listLength = len(_list)
            for i in range( listLength ):
                reversedList.append(_list[abs(i - listLength + 1)])
            return reversedList
        
        if xData == []:
            xData = range(len(yData))
        
        # Set up variables required to perfrom fit.
        _exit,counter = 0,0
        indexOfLargestValue_xAxis = argmax(yData) # First index of largest value along x-axis (highest frequency).
        expect = xData[indexOfLargestValue_xAxis]
        sigma = std(yData)
        maximum = max(yData)
        meansq = mean(yData)**2
        temp = yData
        
        #print "Index of largest value on x-axis:\t", indexOfLargestValue_xAxis
        #print "expect:\t", expect
        #print "sigma:\t", sigma
        #print "maximum:\t", maximum
        #print "meansq:\t", meansq
        
        # We are chopping off some the x-axis data here. This is because this function
        # is running on histogram data, i.e bin positions and frequncies. The xData array
        # holds details of the bins, yData the frequencies. So if the length of xData is n,
        # than the length of yData must be n-1.
        #     
        # For example, if the yData has been split across 6 bins then if we had:
        #
        #    xData = [ 0 , 10 , 20 , 30 , 40 , 50 ] # Bins
        #    yData = [ 1 , 2 , 5 , 1 , 0] # Frequencies
        # 
        # So here the last data point in xData is removed.
        if len(xData) == len(yData)+1:
            xData = xData[0:-1] # Chopping off last data point.
        xDatalength = len(xData)
        
        # Here check if maximum frequency is on the border. If it is then the data
        # is reversed. This code appears to throw data away, since when reversing
        # the data, the part not being reversed is discarded. Is this acceptable?
        #
        # For example, if we have data as follows:
        #
        #     yData = [10 , 0 , 3 , 1 , 1 , 1 , 0 , 1 , 3 , 0]
        #
        # then,
        #    
        #    indexOfLargestValue_xAxis = 0
        #
        # So,
        #
        #    cut = ceil( len(yData) / 2) = 5
        #    part1 = [10 , 0 , 3 , 1 , 1]
        #    part2 = [1 , 1 , 3 , 0 , 10]
        #
        # then yData is set to:
        #
        #    yData = part2+ part1 = [1 , 1 , 3 , 0 , 10 , 10 , 0 , 3 , 1 , 1]
        #
        # Obviously this isn't the data we started with, so is this a bug?
        #
        # BUG: If there are two bins with an equal MAXIMUM frequency, one of those *Could* be discarded here.
        #      This is because the indexOfLargestValue_xAxis variable from above, is obtained from
        #      the first bin with the maximum frequency, but there could be multiple bins with the
        #      maximum frequency in the histogram. So if,
        #      
        #      a) there is a max value in the first or last bin which we label b;
        #      b) 1 or more bins have the same frequency as bin b;
        #      c) those 1 or more bins which share the same frequency as b are not in the same half of the data;
        #
        #      Then the other bins with the the shared max frequency will be discarded.
        
        """
        print "xData length:",len(xData)
        print "xData:",xData
        print "yData length:",len(yData)
        print "yData:",yData
        
        part1,part2 = [],[]
        nearBorder = False
        if (indexOfLargestValue_xAxis == 0):# If the max value is at the begining.
            cut = ceil(len(yData)/2)        # Find midpoint.
            y_part1 = yData[:cut]             # Isolate first half of data.
            y_part2 = __mirror(y_part1)         # Reverse the order of the first half of data.
            yData = list(y_part2)+list(y_part1) # Data equals reversed data + first half of data.
            
            x_part1 = xData[:cut]             # Isolate first half of data.
            x_part2 = __mirror(x_part1)         # Reverse the order of the first half of data.
            xData = list(x_part2)+list(x_part1) # Data equals reversed data + first half of data.
            
            nearBorder = True
            
        elif (indexOfLargestValue_xAxis == xDatalength-1):# If the max value is at the end.
            cut = ceil(len(yData)/2)                     # Find midpoint.
            y_part1 = yData[cut:]                          # Isolate second half of data.
            y_part2 = __mirror(y_part1)                      # Reverse the order of the second half of data.
            yData = list(y_part1)+list(y_part2)              # Data equals second half of data + reversed data.
            
            x_part1 = xData[cut:]                          # Isolate second half of data.
            x_part2 = __mirror(x_part1)                      # Reverse the order of the second half of data.
            xData = list(x_part1)+list(x_part2)              # Data equals second half of data + reversed data.
            
            nearBorder = True
        
        print "post xData length:",len(xData)
        print "post xData:",xData
        print "post yData length:",len(yData)
        print "post yData:",yData
        print "Near border:",nearBorder
        # SO HERE WE ARE GETTING UNEQUAL X AND Y DATA LENGTHS.
        """
        
        # Perform the gaussian fit.       
        while _exit == 0:
            
            parameters = [sigma, expect, maximum]
            
            # Hackey solution to prevent situations where
            # there are more parameters than data points!
            # This causes the scipy least squares call to
            # fail.
            if(len(parameters)> len(xData)):
                lengthDifference = len(parameters)-len(xData)
                for i in range(0,lengthDifference):
                    xData=append(xData,0)
                    yData=append(yData,0)
                
            leastSquaresParameters = leastsq(__residuals, parameters, args=(xData,yData))
            
            """
            if nearBorder == True:
                leastSquaresParameters[0][1] -= xData[int(cut)]
                yData = temp
            """
                
            fwhm = abs(2 * sqrt(2 * log(2)) * leastSquaresParameters[0][0])
            fit = __evaluate(xData, leastSquaresParameters[0])
            
            # Compute Chi-squared value for fit.
            chisq = 0
            for i in range(xDatalength):
                chisq += (yData[i] - fit[i])**2
            
            chisq /= len(yData)# Revision:6c
            
            """
            if (chisq > meansq * xDatalength) & (leastSquaresParameters[0][0] < 0.2 * xDatalength) & (nearBorder==False):
            """
            if (chisq > meansq * xDatalength) & (leastSquaresParameters[0][0] < 0.2 * xDatalength):    
                counter += 1
                temp = delete(temp,indexOfLargestValue_xAxis)
                pos = argmax(temp)
                expect = xData[pos+counter]
                if counter > 5:
                    _exit += 1
            else:
                _exit += 1
        
        # I've commented this return statement out to avoid returning all the data (xData and yData),
        # since this data was passed in to the method in the first place.               
        #return leastSquaresParameters[0], fwhm, chisq, fit, xData, yData
        return leastSquaresParameters[0], fwhm, chisq, fit
    
    # ******************************************************************************************
    
    # Revision:5
    def fitGaussianFixedWidthBins(self,xData,yData,bins):
        """
        Fits a Gaussian to the supplied data under the constraint
        that the expectation value is fixed.
        
        Parameters:
        xData    -    a numpy.ndarray containing data (x-axis data).
        yData    -    a numpy.ndarray containing data (y-axis data).
        bins     -    the number of bins in the profile histogram.
        
        Returns:
        The parameters of the fit, one array and four other variables.
        
            leastSquaresParameters - array containing optimum three values for:
                                        * sigma
                                        * expect
                                        * maximum
            fwhm - the full width half maximum of the Gaussian.
            chisq - the chi-squared value of the fit.
            fit - the fit.
            xmax - the max expectation value.
        
        """
        
        #print "xData (LENGTH=",len(xData),"):\n", xData
        #print "yData (LENGTH=",len(yData),"):\n", yData
        
        # Calculates the residuals.
        def __residuals(paras, x, y, xmax):
            # sigma = the standard deviation.
            
            # Remembmer that here x and y are the data, such that,
            # x = bin number.
            # y = intensity in bin number x.
            sigma, maximum = paras
            err = y - ( abs(maximum) * exp( (-((x - xmax) / sigma )**2) / 2))
            return err
        
        # Evaluates the function.
        def __evaluate(x, paras, xmax):
            # Same variables as above.
            sigma, maximum = paras
            return ( abs(maximum) * exp( (-((x - xmax) / sigma )**2) / 2))
        
        if xData == []:
            xData = range(len(yData))
        if len(xData) == len(yData)+1:
            xData = xData[0:-1]
        
        # Set up variables required to perfrom fit.
        sigma = std(yData)
        maximum = max(yData)
        
        xmax = xData[int(bins/2)-1] # Made change here to ensure we start with centre bin.
        
        # perform fit ######
        parameters = [sigma, maximum]
        leastSquaresParameters = leastsq(__residuals, parameters, args=(xData,yData,xmax))
        fwhm = abs(2 * sqrt( 2 * log(2) ) * leastSquaresParameters[0][0])
        fit = __evaluate(xData, leastSquaresParameters[0],xmax)
        
        # Chi-squared fit. Revision:5
        chisq = 0
        for i in range(len(yData)):
            #if yData[i] >= 1.: # Not sure why this restriction is here. Removed for Revision:5
            chisq += (yData[i]-fit[i])**2
        
        chisq /= len(yData)
        
        return leastSquaresParameters[0], fwhm, chisq, fit, xmax
    
    # ******************************************************************************************
    
    def fitGaussianT1(self,yData):
        """
        Fits a Gaussian to the supplied data.
        
        Parameters:
        yData    -    a numpy.ndarray containing data (y-axis data).
        
        Returns:
        
        An object containing:
        
        the parameters of the fit, one array and three other variables.
        
            leastSquaresParameters - array containing optimum three values for:
                                        * sigma
                                        * expect
                                        * maximum
            fwhm - the full width half maximum of the Gaussian.
            chisq - the chi-squared value of the fit.
            fit - the fit.
            params - the parmeters of the fit.
        
        """
        
        xData =[]  # @UnusedVariable
        part1,part2 = [],[]
        yDataLength = len(yData)
        xData = range(yDataLength)
        xmax = argmax(yData) # Finds index of max value in yData.
        
        # Check if maximum is near borders of the interval.
        
        # The original script was hardcoded to look for peaks in bins 0-15
        # and 112-128 - if the peak was here the data would be processed further.
        # Since the length of the profile data may vary, we can't just hard code
        # literal values any more. So as a hack what we do is rescale the value
        # and check if it is less that 15 or greater than 112 and proceed as before.
        min_ = 0
        max_ = yDataLength
        newMin = 0
        newMax = 128
        tempXmax = self.scale(xmax, min_, max_, newMin, newMax)
        
        nearBorder = False
        if (tempXmax < 15) or (tempXmax >= 112):   # If index of max value is near begining or end.
            cut = int(ceil(yDataLength/2)) # Obtain midpoint.
            part1 = yData[:cut]            # Part 1 contains 1st half of data.
            part2 = yData[cut:]            # Part 2 contains 2nd half of data.
            yData = list(part2)+list(part1)# Swap the parts around. This is done differently in the function fit_gaussian(self,xData,yData) in this file.
            nearBorder = True
            
        """
        nearBorder = False
        if (xmax < 15) or (xmax >= 112):   # If index of max value is near begining or end.
            cut = int(ceil(yDataLength/2)) # Obtain midpoint.
            part1 = yData[:cut]            # Part 1 contains 1st half of data.
            part2 = yData[cut:]            # Part 2 contains 2nd half of data.
            yData = list(part2)+list(part1)# Swap the parts around. This is done differently in the function fit_gaussian(self,xData,yData) in this file.
            nearBorder = True
        """
            
        # Perform gaussian fit.
        result = self.fitGaussianWithBackground(xData,yData)
        
        # This bit increases the mu value for some reason. So if the peak
        # is near the border, then this code below will add the centre bin
        # number to the mean!?
        if nearBorder == True:
            result[0][1] = result[0][1]+cut
        
        
        return result
    
    # ******************************************************************************************
    
    def fitDoubleGaussianT2(self,yData):
        """
        Fits a double Gaussian to the supplied data.
        
        Parameters:
        yData    -    a numpy.ndarray containing data (y-axis data).
        
        Returns:
        
        An object containing the parameters of the fit.
        
        """
        
        part1,part2 = [],[]
        yDataLength = len(yData)
        xmax = argmax(yData) # Finds index of max value in yData.
        
        # The original script was hardcoded to look for peaks in bins 0-15
        # and 112-128 - if the peak was here the data would be processed further.
        # Since the length of the profile data may vary, we can't just hard code
        # literal values any more. So as a hack what we do is rescale the value
        # and check if it is less that 15 or greater than 112 and proceed as before.
        min_ = 0
        max_ = yDataLength
        newMin = 0
        newMax = 128
        tempXmax = self.scale(xmax, min_, max_, newMin, newMax)
        
        nearBorder = False
        if (tempXmax < 15) or (tempXmax >= 112):   # If index of max value is near begining or end.
            cut = int(ceil(yDataLength/2)) # Obtain midpoint.
            part1 = yData[:cut]            # Part 1 contains 1st half of data.
            part2 = yData[cut:]            # Part 2 contains 2nd half of data.
            yData = list(part2)+list(part1)# Swap the parts around. This is done differently in the function fit_gaussian(self,xData,yData) in this file.
            nearBorder = True
            
        """
        nearBorder = False
        if (xmax < 15) or (xmax >= 112):   # If index of max value is near begining or end.
            cut = int(ceil(yDataLength/2)) # Obtain midpoint.
            part1 = yData[:cut]            # Part 1 contains 1st half of data.
            part2 = yData[cut:]            # Part 2 contains 2nd half of data.
            yData = list(part2)+list(part1)# Swap the parts around. This is done differently in the function fit_gaussian(self,xData,yData) in this file.
            nearBorder = True
        """
            
        # Perform gaussian fit.
        result = self.fitDoubleGaussian(yData)
        
        if nearBorder == True:
            result[0][1] = result[0][1]+cut
            result[0][5] = result[0][5]+cut
        
        
        return result
    
    # ******************************************************************************************
    
    def fitGaussianWithBackground(self,xData,yData):
        """
        Fits a Gaussian to the supplied data, with a background term.
        
        Parameters:
        yData    -    a numpy.ndarray containing data (y-axis data).
        
        Returns:
        The parameters of the fit, one array and three other variables.
        
            leastSquaresParameters - array containing optimum three values for:
                                        * sigma
                                        * expect
                                        * maximum
            fwhm - the full width half maximum of the Gaussian.
            chisq - the chi-squared value of the fit.
            fit - the fit.
            params - the parmeters of the fit.
        
        """
        
        #print "xData (LENGTH=",len(xData),"):\n", xData
        #print "yData (LENGTH=",len(yData),"):\n", yData
        
        # Calculates the residuals.
        def __residuals(paras, x, y):
            # sigma = the standard deviation.
            # mu = the mean aka the expectation of the distribution.
            # maximum = .
            # bg = background term.
            
            sigma, mu, maximum, bg = paras
            err = y - ( abs(maximum) * exp( (-((x - mu) / abs(sigma) )**2) / 2) + (bg) ) # Revision:8a
            return err
        
        # Evaluates the function.
        def __evaluate(x, paras):
            # Same variables as above.
            sigma, mu, maximum, bg = paras
            return ( abs(maximum) * exp( (-((x - mu) / abs(sigma) )**2) / 2) + (bg) ) # Revision:8b
        
        ###### perform gaussian fit ######
        if xData == []:
            xData = range(len(yData))
        
        expect = argmax(yData)
        maximum = yData[expect]
        sigma = std(yData)
        bg = 1. 
        #mean(ydata) # Not sure why this was commented out?
        
        parameters = [sigma, expect, maximum, bg]
        leastSquaresParameters = leastsq(__residuals, parameters, args=(xData,yData))
        fwhm = abs(2 * sqrt(2 * log(2)) * leastSquaresParameters[0][0])
        fit = __evaluate(xData, leastSquaresParameters[0])
        
        #print "\tSigma chosen: ",leastSquaresParameters[0][0]
        #print "\tMu chosen: ",leastSquaresParameters[0][1]
        #print "\tMaximum chosen: ",leastSquaresParameters[0][2]
        #print "\tbg chosen: ",leastSquaresParameters[0][3]
        
        chisq = 0
        for i in range(len(yData)):
            chisq += (yData[i]-fit[i])**2
        
        chisq=chisq/len(yData)
        
        # I've commented this return statement out to avoid returning all the data.
        # Since all this data was passed in to the method in the first place.               
        #return leastSquaresParameters[0], fwhm, chisq/len(yData), fit, xData, yData, parameters    
        return leastSquaresParameters[0], fwhm, chisq, fit, parameters
    
    # ******************************************************************************************
    
    def fitDoubleGaussian(self,yData):
        """
        Fits a double Gaussian to the supplied data.
        
        Parameters:
        yData    -    a numpy.ndarray containing data (y-axis data).
        
        Returns:
        
        An object containing the parameters of the fit.
        
        """
        
        # I think this code needs another cleaning pass, as its still hard
        # to understand in places.
        
        #print "xData (LENGTH=",len(xData),"):\n", xData
        #print "yData (LENGTH=",len(yData),"):\n", yData
        
        # Calculates the residuals.
        def __residuals(paras, x, y):
            # sigma = the standard deviation.
            # mu = the mean aka the expectation of the distribution.
            # maximum = .
            # bg = background term.
            
            sigma, mu, maximum, bg = paras
            # Here sigma is zero?
            err = y - ( abs(maximum) * exp( (-((x - mu) / sigma )**2) / 2) + abs(bg) )
            return err
        
        # Evaluates the function.
        def __evaluate(x, paras):
            # Same variables as above.
            sigma, mu, maximum, bg = paras
            return ( abs(maximum) * exp( (-((x - mu) / sigma )**2) / 2) + abs(bg) )
        
        xData = range(len(yData))
        pos = argmax(yData) # indexOfLargestValue_xAxis
        newx,newy = xData,yData
        tolerance,limit = 0,5
        
        # Delete first peak.
        newx = delete(newx,pos)
        newy = delete(newy,pos)
        
        # I haven't understood how this part of the code works yet!
        # Once I've debugged it I will try and explain...
        i = 1   
        while i < len(yData):
            if ((pos-i) > 0) & ((pos+i) < len(yData)):
                if (yData[pos-i] < yData[pos-i+1]) & (yData[pos+i] < yData[pos+i-1]):
                    newx = delete(newx,pos-i)
                    newx = delete(newx,pos-i)
                    newy = delete(newy,pos-i)
                    newy = delete(newy,pos-i)
                elif (yData[pos-i]  >= yData[pos-i+1]) or (yData[pos+i] >= yData[pos+i-1]) & (tolerance < limit):
                    newx = delete(newx,pos-i)
                    newx = delete(newx,pos-i)
                    newy = delete(newy,pos-i)
                    newy = delete(newy,pos-i)
                    tolerance += 1
                else:
                    break
            elif ((pos-i) < 0):
                if (yData[pos+i] < yData[pos+i-1]):
                    newx = delete(newx,pos-i+1)
                    newy = delete(newy,pos-i+1)
                elif (yData[pos+i] >= yData[pos+i-1]) & (tolerance < limit):
                    newx = delete(newx,pos-i+1)
                    newy = delete(newy,pos-i+1)
                    tolerance += 1
                else:
                    break
            
            elif ((pos+i) > len(yData)):
                if (yData[pos-i] < yData[pos-i+1]):
                    newx = delete(newx,pos-i+1)
                    newy = delete(newy,pos-i+1)
                elif (yData[pos-i]  >= yData[pos-i+1]) & (tolerance < limit):
                    newx = delete(newx,pos-i)
                    newy = delete(newy,pos-i)
                    tolerance += 1
                else:
                    break
                
            i += 1 # Increment counter.
        
        counter = 0
        debugCounter=0
        while counter < 8:
            # New gaussian fit.
            debugCounter+=1
            npos = argmax(newy)
            nexpect = newx[npos]
            nsigma = std(newy)
            nmaximum = max(newy)
            nbg = mean(newy)
            
            np0 = [nsigma, nexpect, nmaximum, nbg]
            
            # Hackey solution to prevent situations where
            # there are more parameters than data points!
            # This causes the scipy least squares call to
            # fail.
            if(len(np0)> len(newx)):
                lengthDifference = len(np0)-len(newx)
                for i in range(0,lengthDifference):
                    newx=append(newx,0)
                    newy=append(newy,0)
                    
            plsq = leastsq(__residuals, np0, args=(newx,newy))
            nfwhm = abs(2 * sqrt(2*log(2)) * plsq[0][0])
            nfit = __evaluate(newx, plsq[0])
            
            nchisq = 0
            for i in range(len(newy)):
                nchisq += (newy[i]-nfit[i])**2/len(newy)
                #print "Entered My for loop: ", debugCounter+i
                
            # Substraction to data.
            newy = []
            for i in range(len(yData)):
                evaly = __evaluate(xData[i],plsq[0])
                if (evaly <= yData[i]):
                    newy.append(yData[i]-evaly+plsq[0][3])
                elif (evaly > yData[i]) & (xData[i] > (plsq[0][2]-(1.5*nfwhm)/2)) & (xData[i] < (plsq[0][2]+(1.5*nfwhm)/2)):
                    newy.append(plsq[0][3])
                else:
                    newy.append(yData[i])
                    
                newx = range(len(newy))
                    
            counter += 1
            if counter == 7:
                store_p2 = plsq[0]
            elif counter == 8:
                store_p1 = plsq[0]
        
        # Perform final gaussian fit.
        p = list(store_p1) + list(store_p2)
        
        # Data arriving here not the same.
        finalfit = self.fitDoubleGaussianWithBackground(yData,array(p))
        
        fit1 = __evaluate(xData,store_p1)
        fit2 = __evaluate(xData,store_p2)
        combifit = fit1+fit2-store_p1[3]-store_p2[3]+(store_p1[3]+store_p2[3])/2
        combi_chisq = 0
        
        for i in range(len(yData)):
            if combifit[i] >= 1.:
                combi_chisq += (yData[i]-combifit[i])**2/len(yData)
                
        combi_fwhm1 = abs(2 * sqrt(2*log(2)) * p[0])
        combi_fwhm2 = abs(2 * sqrt(2*log(2)) * p[4])
        
        if (finalfit[2] <= combi_chisq):
            return finalfit
        else:
            return [p,combi_fwhm2,combi_chisq,combifit,xData,yData,combi_fwhm1]
        
    # ******************************************************************************************
    
    def fitDoubleGaussianWithBackground(self,yData,p0):
        """
        Fits a double Gaussian to the supplied data with background.
        
        Parameters:
        yData    -    a numpy.ndarray containing data (y-axis data).
        
        Returns:
        
        An array object containing the parameters of the fit = [sigma1, mu1, maximum1, bg1, sigma2, mu2, maximum2, bg2].
        The fwhm of the first Gaussian.
        The chi-squared value of the first Gaussian fit.
        The double gaussian fit parameters.
        The xData.
        The yData yData.
        The fwhm of the second Gaussian.
        
        """

        # Calculates the residuals.
        def __residuals(paras, x, y):
            # sigma = the standard deviation.
            # mu = the mean aka the expectation of the distribution.
            # maximum = .
            # bg = background term.
            
            sigma1, mu1, maximum1, bg1, sigma2, mu2, maximum2, bg2 = paras
            err = y - ((abs(maximum1) * exp((-((x - mu1) / abs(sigma1))**2) / 2)) +
                       (abs(maximum2) * exp((-((x - mu2) / abs(sigma2))**2) / 2)) + (abs(bg1) + abs(bg2))/2)
            return err
        
        # Evaluates the function.
        def __evaluate(x, paras):
            # Same variables as above.
            sigma1, mu1, maximum1, bg1, sigma2, mu2, maximum2, bg2 = paras
            return ((abs(maximum1) * exp((-((x - mu1) / abs(sigma1))**2) / 2)) +
                       (abs(maximum2) * exp((-((x - mu2) / abs(sigma2))**2) / 2)) + (abs(bg1) + abs(bg2))/2)
            
        xData = range(len(yData))
        
        # Perform gaussian fit.
        leastSquaresParameters = leastsq(__residuals, p0, args=(xData,yData))
        fwhm1 = abs(2 * sqrt(2*log(2)) * leastSquaresParameters[0][0])
        fwhm2 = abs(2 * sqrt(2*log(2)) * leastSquaresParameters[0][4])
        fit = __evaluate(xData, leastSquaresParameters[0])

        chisq = 0
        for i in range(len(yData)):
            if fit[i] >= 1.:
                chisq += (yData[i]-fit[i])**2/len(yData)
                
        return leastSquaresParameters[0], fwhm1, chisq, fit, xData, yData, fwhm2
    
    
    # ****************************************************************************************************
    #
    # Candidate parameters
    #
    # ****************************************************************************************************
    
    def getCandidateParameters(self,profile):
        """
        Computes the candidate parameters. There are four scores computed:
        
        Score 12. The candidate period.
                 
        Score 13. The best signal-to-noise value obtained for the candidate. Higher values desired.
        
        Score 14. The best dispersion measure (dm) obtained for the candidate. Low DM values 
                  are assocaited with local RFI.
                 
        Score 15. The best pulse width.
        
        Parameters:
        profile    -    a numpy.ndarray containing profile data.
        
        Returns:
        The candidate period.
        The best signal-to-noise value obtained for the candidate. Higher values desired.
        The best dispersion measure (dm) obtained for the candidate.
        The best pulse width.
        
        """
        
        raise NotImplementedError("Please Implement this method")
        
    
    # ****************************************************************************************************
    #
    # DM Curve Fittings
    #
    # ****************************************************************************************************
    
    def getDMFittings(self,data):
        """
        Computes the dispersion measure curve fitting parameters. There are four scores computed:
        
        Score 16. This score computes SNR / SQRT( (P-W) / W ).
                 
        Score 17. Difference between fitting factor Prop, and 1. If the candidate is a pulsar,
                  then prop should be equal to 1.
        
        Score 18. Difference between best DM value and optimised DM value from fit. This difference
                  should be small for a legitimate pulsar signal. 
                 
        Score 19. Chi squared value from DM curve fit, smaller values indicate a smaller fit. Thus
                  smaller values will be possessed by legitimate signals.
        
        Parameters:
        data    -    the data.
        
        Returns:
        SNR / SQRT( (P-W) / W ).
        Difference between fitting factor Prop, and 1.
        Difference between best DM value and optimised DM value from fit.
        Chi squared value from DM curve fit, smaller values indicate a smaller fit.
        
        """
        raise NotImplementedError("Please Implement this method")
    
    # ****************************************************************************************************
    #
    # Sub-band scores
    #
    # ****************************************************************************************************
    
    def getSubbandParameters(self,data=None,profile=None):
        """
        Computes the sub-band scores. There are three scores computed:
        
        Score 20. RMS of peak positions in all sub-bands. Smaller values should be possessed by
                  legitimate pulsar signals.
                 
        Score 21. Average correlation coefficient for each pair of sub-bands. Larger values should be
                  possessed by legitimate pulsar signals.
        
        Score 22. Sum of correlation coefficients between sub-bands and profile. Larger values should be
                  possessed by legitimate pulsar signals.
        
        Parameters:
        data       -    the raw candidate data.
        profile    -    a numpy.ndarray containing profile data.
        
        Returns:
        RMS of peak positions in all sub-bands.
        Average correlation coefficient for each pair of sub-bands.
        Sum of correlation coefficients between sub-bands and profile.
        
        """
        raise NotImplementedError("Please Implement this method")
    
    # ******************************************************************************************
    
    def getSubband_scores(self,subbands, prof_bins, band_subbands, bestWidth):
        """
        Computes sub-band scores for a candidate. The function is based on
        a C-script by Aristeidis Noutsos.
        
        Parameters:
        subbands         -    the sub-band data.
        prof_bins        -    the number of bins in the profile.
        band_subbands    -    the number of sub-bands.
        bestWidth        -    the best pulse width.
        
        Returns:
        
        The root mean squared (RMS) of peak positions in all sub-bands.
        The average correlation coefficient for each pair of sub-bands.            
        
        """
        
        width_bins = int(ceil(bestWidth*prof_bins))
        subband_sums = []
        
        # CALCULATE THE AMPLITUDES FOR EACH SUBBAND USING A BOX-CAR EQUAL TO THE PULSE WIDTH.
        
        for i in range(band_subbands):
            
            sums_vec = []
            
            for j in range(prof_bins-width_bins+1):
                _sum = 0  
                for b in range(width_bins):
                    _sum += subbands[i][j+b]
                sums_vec.append(_sum)
            
            subband_sums.append(sums_vec)
            
        # FIND THE MAXIMA OF THE AMPLITUDES FOR EACH SUBBAND.
        
        max_bins, max_sums = [], []
        for i in range(len(subband_sums)):
            max_sum = -10000.0
            for j in range(len(subband_sums[i])):
                if (subband_sums[i][j]>max_sum):
                    max_sum = subband_sums[i][j]
                    max_bin = j+width_bins/2
                    
            max_bins.append(float(max_bin))
            max_sums.append(max_sum)
            
        med = array(max_bins).mean()
        
        # CHECK HOW CLOSE TO EACH OTHER ARE THE POSITIONS OF THE MAXIMA.
        
        count = 0
        var_med = 0.0
        
        for i in range(len(max_bins)):
            if (abs(max_bins[i]-med) <= float(width_bins)):
                count += 1
                var_med += pow(max_bins[i]-med,2)
                
        if (count > 1):
            var = var_med/float(count-1)
        else:
            mean,var = 0,0
            for i in range(len(max_bins)):
                mean += max_bins[i]
            
            mean /= float(len(max_bins))
            
            for i in range(len(max_bins)):
                var += pow(max_bins[i]-mean,2)
            
            var /= float(len(max_bins)-1)
            
        stdev = sqrt(var)
        
        #  Linear correlation.
        # Correlates the amplitudes across the pulse between subbands pairs.
        
        m = 0
        _sum = 0.0
        for i in range( len(subband_sums) ):
            k = i+1
            while k < len(subband_sums):
                # A RuntimeWarning is raised here when subband_sums[i]
                # and subband_sums[k] are completely empty. The code appears
                #subband_sums[i] to execute normally despite the error.
                cc = corrcoef(subband_sums[i], subband_sums[k])[0][1]
                if str(cc)=="nan":
                    k += 1
                else:
                    _sum += cc
                    k += 1
                    m += 1 
        
        # AVERAGE CORRELATION COEFFICIENT ACROSS ALL SUBBAND PAIRS.
        mean_corr = _sum/float(m)
        
        # RMS SCATTER OF THE MAXIMA NORMALISED TO THE PULSE WIDTH.
        rms = stdev/float(width_bins)
        
        return rms, mean_corr
    
    # ******************************************************************************************
    