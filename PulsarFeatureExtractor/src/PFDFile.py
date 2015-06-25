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

File name:    PFDFile.py
Created:      February 6th, 2014
Author:       Rob Lyon
 
Contact:    rob@scienceguyrob.com or robert.lyon@postgrad.manchester.ac.uk
Web:        <http://www.scienceguyrob.com> or <http://www.cs.manchester.ac.uk> 
            or <http://www.jb.man.ac.uk>
            
This code runs on python 2.4 or later.

Script which manages PFD files.

"""

# Standard library Imports:
import struct, sys

# Numpy Imports:
from numpy import array
from numpy import asarray
from numpy import concatenate
from numpy import floor
from numpy import fabs
from numpy import fromfile
from numpy import reshape
from numpy import float64
from numpy import arange
from numpy import add
from numpy import mean
from numpy import zeros
from numpy import shape
from scipy.stats import skew
from scipy.stats import kurtosis
from numpy import std

# Custom file Imports:
from CandidateFileInterface import CandidateFileInterface
from PFDOperations import PFDOperations

import matplotlib.pyplot as plt # Revision:1

# ****************************************************************************************************
#
# CLASS DEFINITION
#
# ****************************************************************************************************

class PFD(CandidateFileInterface):
    """                
    Represents a PFD file, generated during the LOFAR pulsar survey.
    
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
        self.cand = candidateName
        self.scores=[]
        self.profileOps = PFDOperations(self.debug)
        self.setNumberOfScores(22)
        self.load()

    # ****************************************************************************************************
           
    def load(self):
        """
        Attempts to load candidate data from the file, performs file consistency checks if the
        debug flag is set to true.
        
        Parameters:
        N/A
        
        Return:
        N/A
        """
        infile = open(self.cand, "rb")
        
        # The code below appears to have been taken from Presto. So it maybe
        # helpful to look at the Presto github repository to get a better feel
        # for what this code is doing. I certainly have no idea what is going on.
            
        swapchar = '<' # this is little-endian
        data = infile.read(5*4)
        testswap = struct.unpack(swapchar+"i"*5, data)
        # This is a hack to try and test the endianness of the data.
        # None of the 5 values should be a large positive number.
        
        if (fabs(asarray(testswap))).max() > 100000:
            swapchar = '>' # this is big-endian
            
        (self.numdms, self.numperiods, self.numpdots, self.nsub, self.npart) = struct.unpack(swapchar+"i"*5, data)
        (self.proflen, self.numchan, self.pstep, self.pdstep, self.dmstep, self.ndmfact, self.npfact) = struct.unpack(swapchar+"i"*7, infile.read(7*4))
        self.filenm = infile.read(struct.unpack(swapchar+"i", infile.read(4))[0])
        self.candnm = infile.read(struct.unpack(swapchar+"i", infile.read(4))[0])
        self.telescope = infile.read(struct.unpack(swapchar+"i", infile.read(4))[0])
        self.pgdev = infile.read(struct.unpack(swapchar+"i", infile.read(4))[0])
        
        test = infile.read(16)
        has_posn = 1
        for ii in range(16):
            if test[ii] not in '0123456789:.-\0':
                has_posn = 0
                break
            
        if has_posn:
            self.rastr = test[:test.find('\0')]
            test = infile.read(16)
            self.decstr = test[:test.find('\0')]
            (self.dt, self.startT) = struct.unpack(swapchar+"dd", infile.read(2*8))
        else:
            self.rastr = "Unknown"
            self.decstr = "Unknown"
            (self.dt, self.startT) = struct.unpack(swapchar+"dd", test)
            
        (self.endT, self.tepoch, self.bepoch, self.avgvoverc, self.lofreq,self.chan_wid, self.bestdm) = struct.unpack(swapchar+"d"*7, infile.read(7*8))
        (self.topo_pow, tmp) = struct.unpack(swapchar+"f"*2, infile.read(2*4))
        (self.topo_p1, self.topo_p2, self.topo_p3) = struct.unpack(swapchar+"d"*3,infile.read(3*8))
        (self.bary_pow, tmp) = struct.unpack(swapchar+"f"*2, infile.read(2*4))
        (self.bary_p1, self.bary_p2, self.bary_p3) = struct.unpack(swapchar+"d"*3,infile.read(3*8))
        (self.fold_pow, tmp) = struct.unpack(swapchar+"f"*2, infile.read(2*4))
        (self.fold_p1, self.fold_p2, self.fold_p3) = struct.unpack(swapchar+"d"*3,infile.read(3*8))
        (self.orb_p, self.orb_e, self.orb_x, self.orb_w, self.orb_t, self.orb_pd,self.orb_wd) = struct.unpack(swapchar+"d"*7, infile.read(7*8))
        self.dms = asarray(struct.unpack(swapchar+"d"*self.numdms,infile.read(self.numdms*8)))
        
        if self.numdms==1:
            self.dms = self.dms[0]
            
        self.periods = asarray(struct.unpack(swapchar + "d" * self.numperiods,infile.read(self.numperiods*8)))
        self.pdots = asarray(struct.unpack(swapchar + "d" * self.numpdots,infile.read(self.numpdots*8)))
        self.numprofs = self.nsub * self.npart
        
        if (swapchar=='<'):  # little endian
            self.profs = zeros((self.npart, self.nsub, self.proflen), dtype='d')
            for ii in range(self.npart):
                for jj in range(self.nsub):
                    try:
                        self.profs[ii,jj,:] = fromfile(infile, float64, self.proflen)
                    except Exception: # Catch *all* exceptions.
                        pass
                        #print ""
        else:
            self.profs = asarray(struct.unpack(swapchar+"d"*self.numprofs*self.proflen,infile.read(self.numprofs*self.proflen*8)))
            self.profs = reshape(self.profs, (self.npart, self.nsub, self.proflen))
                
        self.binspersec = self.fold_p1 * self.proflen
        self.chanpersub = self.numchan / self.nsub
        self.subdeltafreq = self.chan_wid * self.chanpersub
        self.hifreq = self.lofreq + (self.numchan-1) * self.chan_wid
        self.losubfreq = self.lofreq + self.subdeltafreq - self.chan_wid
        self.subfreqs = arange(self.nsub, dtype='d')*self.subdeltafreq + self.losubfreq
        self.subdelays_bins = zeros(self.nsub, dtype='d')
        self.killed_subbands = []
        self.killed_intervals = []
        self.pts_per_fold = []
        
        # Note: a foldstats struct is read in as a group of 7 doubles
        # the correspond to, in order:
        # numdata, data_avg, data_var, numprof, prof_avg, prof_var, redchi
        self.stats = zeros((self.npart, self.nsub, 7), dtype='d')
        
        for ii in range(self.npart):
            currentstats = self.stats[ii]
            
            for jj in range(self.nsub):
                if (swapchar=='<'):  # little endian
                    try:
                        currentstats[jj] = fromfile(infile, float64, 7)
                    except Exception: # Catch *all* exceptions.
                        pass
                        #print ""
                else:
                    try:
                        currentstats[jj] = asarray(struct.unpack(swapchar+"d"*7,infile.read(7*8)))
                    except Exception: # Catch *all* exceptions.
                        pass
                        #print ""
                    
            self.pts_per_fold.append(self.stats[ii][0][0])  # numdata from foldstats
            
        self.start_secs = add.accumulate([0]+self.pts_per_fold[:-1])*self.dt
        self.pts_per_fold = asarray(self.pts_per_fold)
        self.mid_secs = self.start_secs + 0.5*self.dt*self.pts_per_fold
        
        if (not self.tepoch==0.0):
            self.start_topo_MJDs = self.start_secs/86400.0 + self.tepoch
            self.mid_topo_MJDs = self.mid_secs/86400.0 + self.tepoch
        
        if (not self.bepoch==0.0):
            self.start_bary_MJDs = self.start_secs/86400.0 + self.bepoch
            self.mid_bary_MJDs = self.mid_secs/86400.0 + self.bepoch
            
        self.Nfolded = add.reduce(self.pts_per_fold)
        self.T = self.Nfolded*self.dt
        self.avgprof = (self.profs/self.proflen).sum()
        self.varprof = self.calc_varprof()
        self.barysubfreqs = self.subfreqs
        infile.close()
            
        # If explicit debugging required.
        if(self.debug):
            
            # If candidate file is invalid in some way...
            if(self.isValid()==False):
                
                print "Invalid PFD candidate: ",self.cand
                scores=[]
                
                # Return only NaN values for scores.
                for n in range(0, self.numberOfScores):
                    scores.append(float("nan"))
                return scores
            
            # Candidate file is valid.
            else:
                print "Candidate file valid."
                self.profile = array(self.getprofile())
            
        # Just go directly to score generation without checks.
        else:
            self.out( "Candidate validity checks skipped.","")
            self.profile = array(self.getprofile())
    
    # ****************************************************************************************************
    
    def getprofile(self):
        """
        Obtains the profile data from the candidate file.
        
        Parameters:
        N/A
        
        Returns:
        The candidate profile data (an array) scaled to within the range [0,255].
        """
        if not self.__dict__.has_key('subdelays'):
            self.dedisperse()
          
        normprof = self.sumprof - min(self.sumprof)
        
        s = normprof / mean(normprof)
        
        if(self.debug):
            plt.plot(s)
            plt.title("Profile.")
            plt.show()
            
        return self.scale(s)
    
    # ****************************************************************************************************
    
    def scale(self,data):
        """
        Scales the profile data for pfd files so that it is in the range 0-255.
        This is the same range used in the phcx files. So  by performing this scaling
        the scores for both type of candidates are directly comparable. Before it was
        harder to determine if the scores generated for pfd files were working correctly,
        since the phcx scores are our only point of reference. 
        
        Parameter:
        data    -    the data to scale to within the 0-255 range.
        
        Returns:
        A new array with the data scaled to within the range [0,255].
        """
        min_=min(data)
        max_=max(data)
        
        newMin=0;
        newMax=255
        
        newData=[]
        
        for n in range(len(data)):
            
            value=data[n]
            x = (newMin * (1-( (value-min_) /( max_-min_ )))) + (newMax * ( (value-min_) /( max_-min_ ) ))
            newData.append(x)
            
        return newData
    
    # ****************************************************************************************************
        
    def calc_varprof(self):
        """
        This function calculates the summed profile variance of the current pfd file.
        Killed profiles are ignored. I have no idea what a killed profile is. But it
        sounds fairly gruesome.
        """
        varprof = 0.0
        for part in range(self.npart):
            if part in self.killed_intervals: continue
            for sub in range(self.nsub):
                if sub in self.killed_subbands: continue
                varprof += self.stats[part][sub][5] # foldstats prof_var
        return varprof
    
    # ****************************************************************************************************
        
    def dedisperse(self, DM=None, interp=0):
        """
        Rotate (internally) the profiles so that they are de-dispersed
        at a dispersion measure of DM.  Use FFT-based interpolation if
        'interp' is non-zero (NOTE: It is off by default!).
        
        """

        if DM is None:
            DM = self.bestdm
            
        # Note:  Since TEMPO pler corrects observing frequencies, for
        #        TOAs, at least, we need to de-disperse using topocentric
        #        observing frequencies.
        self.subdelays = self.profileOps.delay_from_DM(DM, self.subfreqs)
        self.hifreqdelay = self.subdelays[-1]
        self.subdelays = self.subdelays-self.hifreqdelay
        delaybins = self.subdelays*self.binspersec - self.subdelays_bins
        
        if interp:
            
            new_subdelays_bins = delaybins
            
            for ii in range(self.npart):
                for jj in range(self.nsub):
                    tmp_prof = self.profs[ii,jj,:]
                    self.profs[ii,jj] = self.profileOps.fft_rotate(tmp_prof, delaybins[jj])
                    
            # Note: Since the rotation process slightly changes the values of the
            # profs, we need to re-calculate the average profile value
            self.avgprof = (self.profs/self.proflen).sum()
            
        else:
            
            new_subdelays_bins = floor(delaybins+0.5)
            
            for ii in range(self.nsub):
                
                rotbins = int(new_subdelays_bins[ii]) % self.proflen
                if rotbins:  # i.e. if not zero
                    subdata = self.profs[:,ii,:]
                    self.profs[:,ii] = concatenate((subdata[:,rotbins:],subdata[:,:rotbins]), 1)
                    
        self.subdelays_bins += new_subdelays_bins
        self.sumprof = self.profs.sum(0).sum(0)
    
    # ******************************************************************************************
    
    def plot_chi2_vs_DM(self, loDM, hiDM, N=100, interp=0):
        """
        Plot (and return) an array showing the reduced-chi^2 versus DM 
        (N DMs spanning loDM-hiDM). Use sinc_interpolation if 'interp' is non-zero.
        """

        # Sum the profiles in time
        sumprofs = self.profs.sum(0)
        
        if not interp:
            profs = sumprofs
        else:
            profs = zeros(shape(sumprofs), dtype='d')
            
        DMs = self.profileOps.span(loDM, hiDM, N)
        chis = zeros(N, dtype='f')
        subdelays_bins = self.subdelays_bins.copy()
        
        for ii, DM in enumerate(DMs):
            
            subdelays = self.profileOps.delay_from_DM(DM, self.barysubfreqs)
            hifreqdelay = subdelays[-1]
            subdelays = subdelays - hifreqdelay
            delaybins = subdelays*self.binspersec - subdelays_bins
            
            if interp:
                
                interp_factor = 16
                for jj in range(self.nsub):
                    profs[jj] = self.profileOps.interp_rotate(sumprofs[jj], delaybins[jj],zoomfact=interp_factor)
                # Note: Since the interpolation process slightly changes the values of the
                # profs, we need to re-calculate the average profile value
                avgprof = (profs/self.proflen).sum()
                
            else:
                
                new_subdelays_bins = floor(delaybins+0.5)
                for jj in range(self.nsub):
                    profs[jj] = self.profileOps.rotate(profs[jj], int(new_subdelays_bins[jj]))
                subdelays_bins += new_subdelays_bins
                avgprof = self.avgprof
                
            sumprof = profs.sum(0)        
            chis[ii] = self.calc_redchi2(prof=sumprof, avg=avgprof)

        return (chis, DMs)
    
    # ******************************************************************************************
    
    def calc_redchi2(self, prof=None, avg=None, var=None):
        """
        Return the calculated reduced-chi^2 of the current summed profile.
        """
        
        if not self.__dict__.has_key('subdelays'):
            self.dedisperse()
            
        if prof is None:  prof = self.sumprof
        if avg is None:  avg = self.avgprof
        if var is None:  var = self.varprof
        return ((prof-avg)**2.0/var).sum()/(len(prof)-1.0)
    
    # ******************************************************************************************
    
    def plot_subbands(self):
        """
        Plot the interval-summed profiles vs subband.  Restrict the bins
        in the plot to the (low:high) slice defined by the phasebins option
        if it is a tuple (low,high) instead of the string 'All'. 
        """
        if not self.__dict__.has_key('subdelays'):
            self.dedisperse()
        
        lo, hi = 0.0, self.proflen
        profs = self.profs.sum(0)
        lof = self.lofreq - 0.5*self.chan_wid
        hif = lof + self.chan_wid*self.numchan
        
        return profs
                        
    # ****************************************************************************************************
        
    def isValid(self):
        """
        Tests the data loaded from a pfd file.
        
        Parameters:
        
        Returns:
        True if the data is well formed and valid, else false.
        """
        
        # These are only basic checks, more in depth checks should be implemented
        # by someone more familiar with the pfd file format.
        if(self.proflen > 0 and self.numchan > 0):
            return True
        else:
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
        
        try:
            curve = self.profileOps.getDMCurveData(self)
            #curve = self.profileOps.getDMCurveDataNormalised(self)
            # Add first scores.
            
            if(self.debug==True):
                print "curve = ",curve
            
            return curve   
        
        except Exception as e: # catch *all* exceptions
            print "Error getting DM curve data from PFD file\n\t", sys.exc_info()[0]
            print self.format_exception(e)
            raise Exception("DM curve extraction exception")
            return []
    
    def computeProfileStatScores(self):
        """
        Builds the stat scores using raw profile intensity data only. Returns the scores.
        
        Parameters:
        N/A
        
        Returns:
        An array of profile intensities as floating point values.
        """
        
        try:
            
            bins=[] 
            for intensity in self.profile:
                bins.append(float(intensity))
            
            mn = mean(bins)
            stdev = std(bins)
            skw = skew(bins)
            kurt = kurtosis(bins)
            
            stats = [mn,stdev,skw,kurt]
            return stats
        
        except Exception as e: # catch *all* exceptions
            print "Error getting Profile stat scores from PFD file\n\t", sys.exc_info()[0]
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
            bins = self.profileOps.getDMCurveData(self)
            #curve = self.profileOps.getDMCurveDataNormalised(self)
            # Add first scores.
            
            mn = mean(bins)
            stdev = std(bins)
            skw = skew(bins)
            kurt = kurtosis(bins)
            
            stats = [mn,stdev,skw,kurt]
            return stats  
        
        except Exception as e: # catch *all* exceptions
            print "Error getting DM curve stat scores from PFD file\n\t", sys.exc_info()[0]
            print self.format_exception(e)
            raise Exception("DM curve stat score extraction exception")
            return []
        
    # ****************************************************************************************************
    
    def compute(self):
        """
        Builds the scores using the PFDOperations.py file. Returns the scores.
        
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
            
            candidateParameters = self.profileOps.getCandidateParameters(self)
            
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
            DMCurveFitting = self.profileOps.getDMFittings(self)
            
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
            subbandScores = self.profileOps.getSubbandParameters(self,self.profile)
            
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