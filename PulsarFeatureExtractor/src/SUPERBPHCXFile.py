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

File name:    SUPERBPHCXFile.py
Created:      February 6th, 2014
Author:       Rob Lyon
 
Contact:    rob@scienceguyrob.com or robert.lyon@postgrad.manchester.ac.uk
Web:        <http://www.scienceguyrob.com> or <http://www.cs.manchester.ac.uk> 
            or <http://www.jb.man.ac.uk>
            
This code runs on python 2.4 or later.

Script which manages SUPERB PHCX files.
 
"""

# Numpy Imports:
from numpy import array
from scipy.stats import skew
from scipy.stats import kurtosis
from numpy import std
from numpy import mean

# Standard library Imports:
import gzip,sys

# XML processing Imports:
from xml.dom import minidom

# Custom file Imports:
from CandidateFileInterface import CandidateFileInterface
from PHCXOperations import PHCXOperations

# ****************************************************************************************************
#
# CLASS DEFINITION
#
# ****************************************************************************************************

class SUPERBPHCX(CandidateFileInterface):
    """                
    Generates 22 scores that describe the key features of pulsar candidate.
    
    """
    
    # ****************************************************************************************************
    #
    # Constructor.
    #
    # ****************************************************************************************************
    
    def __init__(self,debugFlag,candidateName):
        """
        Default constructor.
        
        Parameters:
        
        debugFlag     -    the debugging flag. If set to True, then detailed
                           debugging messages will be printed to the terminal
                           during execution.
        candidateName -    the name for the candidate, typically the file path.
        """
        CandidateFileInterface.__init__(self,debugFlag)
        
        self.cand = candidateName # The name of the candidate.
        self.profileIndex = 0     # A phcx file specific variable, used to identify the section of the xml data to read.
        self.profile=[]           # The decimal profile data.
        self.rawdata=[]           # The raw data read in from the file, in this case xml.
        self.scores=[]
        self.profileOps = PHCXOperations(self.debug)
        self.setNumberOfScores(22)
        self.load()

    # ****************************************************************************************************
           
    def load(self):
        """
        Attempts to load candidate profile data from the file, performs file consistency checks if the
        debug flag is set to true.
        
        Parameters:
        N/A
        
        Return:
        N/A
        """
        
        # Read data directly from SUPERB phcx file.
        data = infile = open(self.cand, "rb")
        self.rawdata = minidom.parse(data) # strip off xml data
        data.close()
            
        # Explicit debugging required.
        if(self.debug):
            
            # If candidate file is invalid in some way...
            if(self.isValid()==False):
                
                print "Invalid SUPERB PHCX candidate: ",self.cand
                
                # Return only NaN values for scores.
                for n in range(0, self.numberOfScores):  # @UnusedVariable - this comment tells my IDE to ignore n being unused.
                    self.scores.append(float("nan"))
                return self.scores
            
            # Candidate file is valid.
            else:
                print "Candidate file valid."
                # Extracts data from this part of a candidate file. It contains details
                # of the profile in hexadecimal format. The data is extracted from the part
                # of the candidate file which resembles:
                # <Profile nBins='128' format='02X' min='-0.000310' max='0.000519'>
                #
                # Call to ph.getprofile() below will return a LIST data type of 128 integer data points.
                # Phcx files actually contain two profile sections (i.e. there are two <Profile>...</Profile> 
                # sections in the file) which can be read using the XML dom code by specifying the index of the
                # profile section to use. The first section profileIndex = 0 pertains to a profile obtained after the FFT,
                # the second, profileIndex = 1, to a profile that has been period and DM searched using PDMPD. We choose 1 here
                # as it should have a better SNR .... maybe.
                self.profile = array(self.getprofile())
                         
        # Just go directly to score generation without checks.
        else:
            self.out( "Candidate validity checks skipped.","")
            # See comment above to understand what happens with this call.
            self.profile = array(self.getprofile())
    
    # ****************************************************************************************************
    
    def getprofile(self):
        """
        Returns a list of 128 integer data points representing a pulse profile.
        Takes two parameters: the xml data and the profile index to use. 
        The xml data contains two distinct profile sections (i.e. there are two <Profile>...</Profile> 
        sections in the file) which are indexed. The first section with profileIndex = 0 pertains to a
        profile obtained after the FFT, the second, profileIndex = 1, to a profile that has been period
        and DM searched using PDMPD.
        
        Parameters:
        N/A
        
        Returns:
        A list data type containing 64 integer data points.
        """
        # First obtain desired block of xml data.
        block = self.rawdata.getElementsByTagName('Profile')
        
        # Get raw hexadecimal data from the block
        points = block[self.profileIndex].childNodes[0].data
        
        # The format of the hexadecimal data is 02X, i.e. hexadecimal value with 2 digits.
        decimal_profile = []
        index = 0 # The index at which hexadecimal conversion will be performed.
        
        while index < len(points):
            if points[index] != "\n":
                try:
                    hex_value = points[index:index+2]
                    #print "Hex value:\t", hex_value
                    decimal_profile.append(int(hex_value,16)) # now the profile (shape, unscaled) is stored in dec_value
                    #print "Decimal value:\t",int(hex_value,16)
                    index = index+2 # Skip two characters to next hexadecimal number since format is 02X.
                except ValueError:
                    if points[index] =="\t":# There is a tab at the end of the xml data. So break the loop normally here.
                        break
                    else: # Unexpected error, report to user. 
                        print "Unexpected value error obtaining profile data for: ",self.cand
                        break
            else:
                index = index+1
                
        return decimal_profile
                
    # ****************************************************************************************************
        
    def isValid(self):
        """
        Tests the xml data loaded from a phcx file for well-formedness, and invalid values.
        To understand the code here its best to take a look at a phcx xml file, to see the
        underlying structure. Alternatively I've generated a xml schema file which summarizes
        the structure (should be in same folder as this file) called: phcx.xsd.xml .
        
        Parameters:
        N/A
        
        Returns:
        True if the xml data is well formed and valid, else false.
        """
        
        # Read out data blocks.
        profile_block = self.rawdata.getElementsByTagName('Profile')
        subband_block = self.rawdata.getElementsByTagName('SubBands')
        datablock_block = self.rawdata.getElementsByTagName('DataBlock')
        
        # Test length of data in blocks. These should be equal to 2, since there
        # are two profile blocks, two sub-band blocks and two data blocks in the
        # xml file.
        if ( len(profile_block) == len(subband_block) == len(datablock_block) == 2 ):
            
            # There are two sections in the XML file:
            #<Section name='FFT'>...</Section>
            #<Section name='FFT-pdmpd'>...</Section>
            #
            # The first section (index=0) contains the raw FFT data, the second (index=1)
            # contains data that has been period and DM searched using a separate tool.
            # Mike Keith should know more about this tool called "pdmpd". Here
            # data from both these sections is extracted to determine its length.
            
            # From <Section name='FFT'>...</Section>
            subband_points_fft   = subband_block[0].childNodes[0].data
            datablock_points_fft = datablock_block[0].childNodes[0].data
            
            # From <Section name='FFT-pdmpd'>...</Section>
            profile_points_opt   = profile_block[1].childNodes[0].data
            subband_points_opt   = subband_block[1].childNodes[0].data
            datablock_points_opt = datablock_block[1].childNodes[0].data
            
            # Note sure if the checks here are valid, i.e. if there are 99 profile points is that bad?
            if ( len(profile_points_opt)>100) & (len(subband_points_opt)>1000) & (len(subband_points_fft)>1000) & (len(datablock_points_opt)>1000 ):
                
                subband_bins = int(subband_block[1].getAttribute("nBins"))
                subband_subbands = int(subband_block[1].getAttribute("nSub"))
                dmindex = list(self.rawdata.getElementsByTagName('DmIndex')[1].childNodes[0].data)
                
                # Stored here so call to len() made only once.
                lengthDMIndex = len(dmindex) # This is the DM index from the <Section name='FFT'>...</Section> part of the xml file.
                
                if (subband_bins == 64) & (subband_subbands == 16) & (lengthDMIndex > 100):
                    
                    # Now check for NaN values.
                    bestWidth      = float(self.rawdata.getElementsByTagName('Width')[1].childNodes[0].data)
                    bestSNR        = float(self.rawdata.getElementsByTagName('Snr')[1].childNodes[0].data)
                    bestDM         = float(self.rawdata.getElementsByTagName('Dm')[1].childNodes[0].data)
                    bestBaryPeriod = float(self.rawdata.getElementsByTagName('BaryPeriod')[1].childNodes[0].data)
                    
                    if (bestWidth != "nan") & (bestSNR != "nan") & (bestDM != "nan") & (bestBaryPeriod != "nan"):
                        return True
                    else:
                        print "\tPHCX check 4 failed, NaN's present in: ",self.cand
                        
                        # Extra debugging info for anybody encountering errors.
                        if (bestWidth != "nan") :
                            self.out("\t\"Width\" value found in <Section name='FFT-pdmpd'>...</> is NaN in: ",self.cand)
                        if (bestSNR != "nan") :
                            self.out("\t\"Snr\" value found in <Section name='FFT-pdmpd'>...</> is NaN in: ",self.cand)
                        if (bestDM != "nan"):
                            self.out("\t\"Dm\" value found in <Section name='FFT-pdmpd'>...</> is NaN in: ",self.cand)
                        if (bestBaryPeriod != "nan"):
                            self.out("\t\"BaryPeriod\" value found in <Section name='FFT-pdmpd'>...</> is NaN in: ",self.cand)
                            
                        return False
                else:
                    self.out("\tPHCX check 3 failed, wrong number of bins, sub-bands in: ",self.cand)
                    
                    # Extra debugging info for anybody encountering errors.
                    if(subband_bins!=64):
                        self.outMultiple("\tNumber of sub-band bins != 128 there are instead ",subband_bins, " in: ",self.cand)
                    if(subband_subbands!=16):
                        self.outMultiple("\tNumber of sub-bands != 16 there are instead ",subband_subbands, " in: ",self.cand)
                    if(lengthDMIndex<100):
                        self.outMultiple("\tNumber of DM indexes < 100 there are instead ",lengthDMIndex, " in: ",self.cand)
                        
                    return False
            else:
                self.out("\tPHCX check 2 failed, not enough profile points, sub-band points in: ",self.cand)
                self.out("\tPoints in <Section name='FFT'>...</>","")
                self.outMultiple("\tSub-band points: ",len(subband_points_fft)," Data block points: ", len(datablock_points_fft))
                self.out("\tPoints in <Section name='FFT-pdmpd'>...</>")
                self.outMultiple("\tProfile points: ",len(profile_points_opt)," Sub-band points: ",len(subband_points_opt)," Data block points: ", len(datablock_points_opt))
                return False
        else:
            self.out("\tPHCX check 1 failed, profile, sub-band and data blocks of unequal size in: ",self.cand)
            return False
    
    # ****************************************************************************************************
    
    def computeProfileScores(self):
        """
        Builds the scores using raw profile intensity data only. Returns the scores.
        
        Parameters:
        N/A
        
        Returns:
        An array of profile intensities as floating point values.
        """
        for intensity in self.profile:
            self.scores.append(float(intensity))
            
        return self.scores
    
    def getDMCurveData(self):
        """
        Returns a list of integer data points representing the candidate DM curve.
        
        Parameters:
        N/A
        
        Returns:
        A list data type containing data points.
        
        """
        
        return self.profileOps.getDMCurveData(self.rawdata,self.profileIndex)
    
    def computeProfileStatScores(self):
        """
        Builds the scores using raw profile intensity data only. Returns the scores.
        
        Parameters:
        N/A
        
        Returns:
        An array of profile intensities as floating point values.
        """
        
        try:
            
            bins =[] 
            for intensity in self.profile:
                bins.append(float(intensity))
            
            mn = mean(bins)
            stdev = std(bins)
            skw = skew(bins)
            kurt = kurtosis(bins)
            
            stats = [mn,stdev,skw,kurt]
            return stats
        
        except Exception as e: # catch *all* exceptions
            print "Error getting Profile stat scores from PHCX file\n\t", sys.exc_info()[0]
            print self.format_exception(e)
            raise Exception("Profile stat score extraction exception")
            return []
    
    def computeDMCurveStatScores(self):
        """
        Returns a list of integer data points representing the candidate DM curve.
        
        Parameters:
        N/A
        
        Returns:
        A list data type containing data points.
        
        """
        
        try:
            bins=[]
            bins=self.profileOps.getDMCurveData(self.rawdata,self.profileIndex)
            
            mn = mean(bins)
            stdev = std(bins)
            skw = skew(bins)
            kurt = kurtosis(bins)
            
            stats = [mn,stdev,skw,kurt]
            return stats  
        
        except Exception as e: # catch *all* exceptions
            print "Error getting DM curve stat scores from PHCX file\n\t", sys.exc_info()[0]
            print self.format_exception(e)
            raise Exception("DM curve stat score extraction exception")
            return []
     
    # ****************************************************************************************************
    
    def compute(self):
        """
        Builds the scores using the ProfileOperations.py file. Returns the scores.
        
        Parameters:
        N/A
        
        Returns:
        An array of 22 candidate scores as floating point values.
        """
        
        # Get scores 1-4
        self.computeSinusoidFittingScores()
        
        # Get scores 5-11
        self.computeGaussianFittingScores()

        # Get scores 12-15
        self.computeCandidateParameterScores()
        
        # Get scores 16-19
        self.computeDMCurveFittingScores()
        
        # Get scores 20-22
        self.computeSubBandScores()

        return self.scores
        
    # ****************************************************************************************************
    
    def computeSinusoidFittingScores(self):
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
        N/A
        
        Returns:
        
        Four candidate scores.
        """
        try:
            sin_fit = self.profileOps.getSinusoidFittings(self.profile)
            # Add first scores.
            self.scores.append(float(sin_fit[0])) # Score 1.  Chi-Squared value for sine fit to raw profile.
            self.scores.append(float(sin_fit[1])) # Score 2.  Chi-Squared value for sine-squared fit to amended profile.
            self.scores.append(float(sin_fit[2])) # Score 3.  Difference between maxima.
            self.scores.append(float(sin_fit[3])) # Score 4.  Sum over residuals.
            
            if(self.debug==True):
                print "\nScore 1. Chi-Squared value for sine fit to raw profile = ",sin_fit[0]
                print "Score 2. Chi-Squared value for sine-squared fit to amended profile = ",sin_fit[1]
                print "Score 3. Difference between maxima = ",sin_fit[2]
                print "Score 4. Sum over residuals = ",sin_fit[3]
        
        except Exception as e: # catch *all* exceptions
            print "Error computing scores 1-4 (Sinusoid Fitting) \n\t", sys.exc_info()[0]
            print self.format_exception(e)
            raise Exception("Sinusoid fitting exception")
            
    
    # ****************************************************************************************************
    
    def computeGaussianFittingScores(self):
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
        N/A
        
        Returns:
        
        Seven candidate scores.
        """
        
        try:
            guassian_fit = self.profileOps.getGaussianFittings(self.profile)
            
            self.scores.append(float(guassian_fit[0]))# Score 5. Distance between expectation values of Gaussian and fixed Gaussian fits to profile histogram.
            self.scores.append(float(guassian_fit[1]))# Score 6. Ratio of the maximum values of Gaussian and fixed Gaussian fits to profile histogram.
            self.scores.append(float(guassian_fit[2]))# Score 7. Distance between expectation values of derivative histogram and profile histogram.
            self.scores.append(float(guassian_fit[3]))# Score 8. Full-width-half-maximum (FWHM) of Gaussian fit to pulse profile. 
            self.scores.append(float(guassian_fit[4]))# Score 9. Chi squared value from Gaussian fit to pulse profile.
            self.scores.append(float(guassian_fit[5]))# Score 10. Smallest FWHM of double-Gaussian fit to pulse profile. 
            self.scores.append(float(guassian_fit[6]))# Score 11. Chi squared value from double Gaussian fit to pulse profile.
            
            if(self.debug==True):
                print "\nScore 5. Distance between expectation values of Gaussian and fixed Gaussian fits to profile histogram = ", guassian_fit[0]
                print "Score 6. Ratio of the maximum values of Gaussian and fixed Gaussian fits to profile histogram = ",guassian_fit[1]
                print "Score 7. Distance between expectation values of derivative histogram and profile histogram. = ",guassian_fit[2]
                print "Score 8. Full-width-half-maximum (FWHM) of Gaussian fit to pulse profile = ", guassian_fit[3]
                print "Score 9. Chi squared value from Gaussian fit to pulse profile = ",guassian_fit[4]
                print "Score 10. Smallest FWHM of double-Gaussian fit to pulse profile = ", guassian_fit[5]
                print "Score 11. Chi squared value from double Gaussian fit to pulse profile = ", guassian_fit[6]
        
        except Exception as e: # catch *all* exceptions
            print "Error computing scores 5-11 (Gaussian Fitting) \n\t", sys.exc_info()[0]
            print self.format_exception(e)
            raise Exception("Gaussian fitting exception")
    
    # ****************************************************************************************************
    
    def computeCandidateParameterScores(self):
        """
        Computes the candidate parameters. There are four scores computed:
        
        Score 12. The candidate period.
                 
        Score 13. The best signal-to-noise value obtained for the candidate. Higher values desired.
        
        Score 14. The best dispersion measure (dm) obtained for the candidate. Low DM values 
                  are assocaited with local RFI.
                 
        Score 15. The best pulse width.
                   
        Parameters:
        N/A
        
        Returns:
        
        Four candidate scores.
        """
        
        try:
            candidateParameters = self.profileOps.getCandidateParameters(self.rawdata,self.profileIndex)
            
            self.scores.append(float(candidateParameters[0]))# Score 12. Best period.
            self.scores.append(self.filterScore(13,float(candidateParameters[1])))# Score 13. Best S/N value.
            self.scores.append(self.filterScore(14,float(candidateParameters[2])))# Score 14. Best DM value.
            self.scores.append(float(candidateParameters[3]))# Score 15. Best pulse width.
            
            if(self.debug==True):
                print "\nScore 12. Best period = "         , candidateParameters[0]
                print "Score 13. Best S/N value = "        , candidateParameters[1], " Filtered value = ", self.filterScore(13,float(candidateParameters[1]))
                print "Score 14. Best DM value = "         , candidateParameters[2], " Filtered value = ", self.filterScore(14,float(candidateParameters[2]))
                print "Score 15. Best pulse width = "      , candidateParameters[3]
        
        except Exception as e: # catch *all* exceptions
            print "Error computing candidate parameters 12-15\n\t", sys.exc_info()[0]
            print self.format_exception(e)
            raise Exception("Candidate parameters exception")
    
    # ****************************************************************************************************
    
    def computeDMCurveFittingScores(self):
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
        N/A
        
        Returns:
        
        Four candidate scores.
        """
        
        try:
            DMCurveFitting = self.profileOps.getDMFittings(self.rawdata,self.profileIndex)
            
            self.scores.append(float(DMCurveFitting[0]))# Score 16. SNR / SQRT( (P-W)/W ).
            self.scores.append(float(DMCurveFitting[1]))# Score 17. Difference between fitting factor, Prop, and 1.
            self.scores.append(self.filterScore(18,float(DMCurveFitting[2])))# Score 18. Difference between best DM value and optimised DM value from fit, mod(DMfit - DMbest).
            self.scores.append(float(DMCurveFitting[3]))# Score 19. Chi squared value from DM curve fit.
            
            if(self.debug==True):
                print "\nScore 16. SNR / SQRT( (P-W) / W ) = " , DMCurveFitting[0]
                print "Score 17. Difference between fitting factor, Prop, and 1 = " , DMCurveFitting[1]
                print "Score 18. Difference between best DM value and optimised DM value from fit, mod(DMfit - DMbest) = ", DMCurveFitting[2], " Filtered value = ", self.filterScore(18,float(DMCurveFitting[2]))
                print "Score 19. Chi squared value from DM curve fit = " , DMCurveFitting[3]
        
        except Exception as e: # catch *all* exceptions
            print "Error computing DM curve fitting 16-19\n\t", sys.exc_info()[0]
            print self.format_exception(e)
            raise Exception("DM curve fitting exception")
    
    # ****************************************************************************************************
    
    def computeSubBandScores(self):
        """
        Computes the sub-band scores. There are three scores computed:
        
        Score 20. RMS of peak positions in all sub-bands. Smaller values should be possessed by
                  legitimate pulsar signals.
                 
        Score 21. Average correlation coefficient for each pair of sub-bands. Larger values should be
                  possessed by legitimate pulsar signals.
        
        Score 22. Sum of correlation coefficients between sub-bands and profile. Larger values should be
                  possessed by legitimate pulsar signals.
                   
        Parameters:
        N/A
        
        Returns:
        
        Three candidate scores.
        """
        try:
            subbandScores = self.profileOps.getSubbandParameters(self.profileIndex,self.rawdata,self.profile)
            
            self.scores.append(float(subbandScores[0]))# Score 20. RMS of peak positions in all sub-bands.
            self.scores.append(float(subbandScores[1]))# Score 21. Average correlation coefficient for each pair of sub-bands.
            self.scores.append(float(subbandScores[2]))# Score 22. Sum of correlation coefficients between sub-bands and profile.
            
            if(self.debug==True):
                print "\nScore 20. RMS of peak positions in all sub-bands = " , subbandScores[0]
                print "Score 21. Average correlation coefficient for each pair of sub-bands = " , subbandScores[1]
                print "Score 22. Sum of correlation coefficients between sub-bands and profile = " , subbandScores[2]
        
        except Exception as e: # catch *all* exceptions
            print "Error computing subband scores 20-22\n\t", sys.exc_info()[0]
            print self.format_exception(e)
            raise Exception("Subband scoring exception")
    
    # ****************************************************************************************************
    
    def getDMPlaneCurveData(self):
        """
        Returns a list of integer data points representing the candidate DM curve.
        
        Parameters:
        N/A
        
        Returns:
        A list data type containing data points.
        
        """
        
        return self.profileOps.getDMCurveData(self.rawdata,self.profileIndex)
        
    # ******************************************************************************************    
    
    def getSubintData(self):
        """
        Returns a list of integer data points representing the sub int data.
        
        Parameters:
        N/A
        
        Returns:
        A list data type containing data points.
        
        """
        
        return self.profileOps.getSubintData(self.rawdata,self.profileIndex)
        
    # ******************************************************************************************  
    
    def getSubbandData(self):
        """
        Returns a list of integer data points representing the sub band data.
        
        Parameters:
        N/A
        
        Returns:
        A list data type containing data points.
        
        """
        
        return self.profileOps.getSubbandData(self.rawdata,self.profileIndex)
        
    # ******************************************************************************************