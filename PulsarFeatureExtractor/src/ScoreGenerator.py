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

File name:    ScoreGenerator.py
Created:      February 7th, 2014
Author:       Rob Lyon
 
Contact:    rob@scienceguyrob.com or robert.lyon@postgrad.manchester.ac.uk
Web:        <http://www.scienceguyrob.com> or <http://www.cs.manchester.ac.uk> 
            or <http://www.jb.man.ac.uk>
            
This code runs on python 2.4 or later.

Script which generates scores for pulsar candidates. These scores are used as the
input features for machine learning classification algorithms. In total 22 scores
are generated from each individual candidate. Each score summarises a candidate
in some way.
  
This code runs on python 2.4 or later. I've tested the code to ensure that any changes
made here did not change the functionality of the original code. In other words the scores
output by this code are mathematically identical to those output by the original code
(unless improvements have been made). We know this for sure, since scores were recomputed
for a sample of candidates generated during the HTRU survey -
(score data stored at /local/scratch/cands). The scores generated using the original
script where exactly the same as those generated using this new script. 
 
"""

# Command Line processing Imports:
from optparse import OptionParser

# Custom file Imports:
import Utilities
import DataProcessor

# ******************************
#
# CLASS DEFINITION
#
# ******************************

class ScoreGenerator:
    """                
    Generates 22 scores that describe the key features of pulsar candidate, from the
    candidate's own phcx file. The scores generated are as follows:
    
    Score number    Description of score                                                                                Group
        1            Chi squared value from fitting since curve to pulse profile.                                    Sinusoid Fitting
        2            Chi squared value from fitting sine-squared curve to pulse profile.                             Sinusoid Fitting
        
        3            Number of peaks the program identifies in the pulse profile - 1.                                Pulse Profile Tests
        4            Sum over residuals.                                                                             Pulse Profile Tests
        
        5            Distance between expectation values of Gaussian and fixed Gaussian fits to profile histogram.   Gaussian Fitting
        6            Ratio of the maximum values of Gaussian and fixed Gaussian fits to profile histogram.           Gaussian Fitting
        7            Distance between expectation values of derivative histogram and profile histogram.              Gaussian Fitting    
        8            Full-width-half-maximum (FWHM) of Gaussian fit to pulse profile.                                Gaussian Fitting
        9            Chi squared value from Gaussian fit to pulse profile.                                           Gaussian Fitting
        10           Smallest FWHM of double-Gaussian fit to pulse profile.                                          Gaussian Fitting
        11           Chi squared value from double Gaussian fit to pulse profile.                                    Gaussian Fitting
        
        12           Best period.                                                                                    Candidate Parameters
        13           Best SNR value.                                                                                 Candidate Parameters
        14           Best DM value.                                                                                  Candidate Parameters
        15           Best pulse width (original reported as Duty cycle (pulse width / period)).                      Candidate Parameters
        
        16           SNR / SQRT( (P-W)/W ).                                                                          Dispersion Measure (DM) Curve Fitting
        17           Difference between fitting factor, Prop, and 1.                                                 Dispersion Measure (DM) Curve Fitting
        18           Difference between best DM value and optimised DM value from fit, mod(DMfit - DMbest).          Dispersion Measure (DM) Curve Fitting
        19           Chi squared value from DM curve fit.                                                            Dispersion Measure (DM) Curve Fitting
        
        20           RMS of peak positions in all sub-bands.                                                         Sub-band Scores
        21           Average correlation coefficient for each pair of sub-bands.                                     Sub-band Scores
        22           Sum of correlation coefficients between sub-bands and profile.                                  Sub-band Scores
        
    Check out Sam Bates' thesis for more information, "Surveys Of The Galactic Plane For Pulsars" 2011.
    
    """
    
    # ******************************
    #
    # MAIN METHOD AND ENTRY POINT.
    #
    # ******************************

    def main(self,argv=None):
        """
        Main entry point for the Application. Processes command line
        input and begins creating the scores.
    
        """
        
        # Python 2.4 argument processing.
        parser = OptionParser()

        # REQUIRED ARGUMENTS
        # None.
        
        # OPTIONAL ARGUMENTS
        parser.add_option("-c",       action="store",      dest="candDir",   help='the path to the directory containing candidates (required).',default="")
        parser.add_option("-v",       action="store_true", dest="verbose",   help='Verbose debugging flag (optional).',default=False)
        parser.add_option('-o',       action="store",      dest="outputPath",type="string",help='The path to write scores to (required).',default="")
        parser.add_option("--pfd",    action="store_true", dest="pfd",       help='Flag which indicates that ONLY .pfd files are being processed (optional).',default=False)
        parser.add_option("--phcx",   action="store_true", dest="phcx",      help='Flag which indicates that ONLY .phcx files are being processed (optional).',default=False)
        parser.add_option("--superb", action="store_true", dest="superb",    help='Flag which indicates that ONLY SUPERB .phcx files are being processed (optional).',default=False)
        parser.add_option("--arff",   action="store_true", dest="arff",      help='Flag which indicates that candidate data should be written to a ARFF file (optional).',default=False)
        parser.add_option("--profile",action="store_true", dest="profile",   help='Flag which indicates that profile, rather than score data should be generated (optional).',default=False)
        parser.add_option("--label",  action="store_true", dest="label",     help='Flag which indicates that user labelling of candidates will be done (optional).',default=False)
        parser.add_option("--dmprof", action="store_true", dest="dmprof",    help='Flag which indicates that DM and profile summary stats should be generated (optional).',default=False)

        (args,options) = parser.parse_args()# @UnusedVariable : Tells Eclipse IDE to ignore warning.
        
        # Update variables with command line parameters.
        
        self.debug = args.verbose
        self.outputPath = args.outputPath
        self.pfd = args.pfd
        self.phcx = args.phcx
        self.arff = args.arff
        self.superb = args.superb
        self.candidateDirectory = args.candDir
        self.genProfileData = args.profile
        self.processSingleCandidate = False
        self.label = args.label
        self.DM_PROFILE = args.dmprof
        
        # Helper files.
        utils = Utilities.Utilities(self.debug)
        dp = DataProcessor.DataProcessor(self.debug)
        
        # Process -s argument if provided, make sure file the user
        # want to write to exists - otherwise we default to 
        # writing candidates to separate files.
        if(utils.fileExists(self.outputPath)):
            self.singleFile = True
        else:
            try:
                output = open(self.outputPath, 'w') # First try to create file.
                output.close()
            except IOError:
                pass
            
            # Now check again if it exists.
            if(utils.fileExists(self.outputPath)):
                self.singleFile = True
            else:
                self.singleFile = False # Must be an invalid path.
        
        # Process -c argument if provided, make sure directory containing
        # the candidates the user wants to process exists - otherwise we default to 
        # looking for candidates in the local directory. 
        if(utils.dirExists(self.candidateDirectory)):
            self.searchLocalDirectory = False
        elif(utils.fileExists(self.candidateDirectory)):
            self.searchLocalDirectory = False
            self.processSingleCandidate = True
        else:
            self.searchLocalDirectory = True
            
        # We have to determine the directory we would like to process. 
        if(self.searchLocalDirectory):
            self.search = ""
        elif(self.processSingleCandidate==False):
            # We add a / here as the method we call next will then search the directory
            # by appending either *.pfd or *.phcx.gz. Without the additional / we would
            # only look for directories that end with .pfd etc.
            self.search = self.candidateDirectory+"/"
        elif(self.processSingleCandidate==True):
            self.search = self.candidateDirectory
            
        print "\n***********************************"
        print "| Executing score generation code |"
        print "***********************************"
        print "\tCommand line arguments:"
        print "\tDebug:",self.debug
        print "\tWrite to single file:",self.singleFile
        print "\tOutput path:",self.outputPath
        print "\tExpect PFD files:",self.pfd
        print "\tExpect PHCX files:",self.phcx
        print "\tExpect SUPERB PHCX files:",self.superb
        print "\tProduce ARFF file:",self.arff
        print "\tCandidate directory:",self.candidateDirectory
        print "\tProcess single candidate:",self.processSingleCandidate
        print "\tLabel candidates:",self.label
        print "\tGenerate DM and profile stats as scores only:",self.DM_PROFILE
        print "\tSearch local directory:",self.searchLocalDirectory,"\n\n" 
        
        
        # Based on the command line parameters there multiple possible execution paths:
        #
        # 1. a) Generate candidate scores for phcx files, each candidate scores written to a separate file.
        # 1. b) Generate candidate scores for phcx files, each candidate scores written to ONE file.
        #
        # 2. a) Generate candidate scores for pfd files, each candidate scores written to a separate file.
        # 2. b) Generate candidate scores for pfd files, each candidate scores written to ONE file.
        # 
        # 3. a) Generate candidate scores for pfd AND phcx files, each candidate scores written to a separate file.
        # 3. b) Generate candidate scores for pfd AND phcx files, each candidate scores written to ONE file.
        # 
        # 4. a) Generate candidate scores for SUPERB phcx files, each candidate score written to ONE file.
        #
        # 5. a) No details specified by user - look for pdf and phcx files and generate scores for them in separate files.
        #
        # So we need to implement each of these possible paths.
        
        # Path 0
        if(self.label):
            
            if(self.phcx and not self.pfd and not self.superb):# label phcx
                dp.labelPHCX(self.search, self.debug)
            elif(not self.phcx and self.pfd): # label pfd
                dp.labelPFD(self.search, self.debug)
                
        elif(self.DM_PROFILE):
            
            if(self.phcx and not self.pfd and not self.superb):# label phcx
                dp.dmprofPHCX(self.search, self.debug,self.outputPath,self.arff,self.processSingleCandidate)
            elif(not self.phcx and self.pfd and not self.superb): # label pfd
                dp.dmprofPFD(self.search, self.debug,self.outputPath,self.arff,self.processSingleCandidate)
            elif(not self.phcx and not self.pfd and self.superb): # label pfd
                dp.dmprofSUPERB(self.search, self.debug,self.outputPath,self.arff,self.processSingleCandidate)
        # Path 1
        elif(self.phcx and not self.pfd and not self.superb):
            # Path 1. a)
            if(not self.singleFile):
                # Process phcx files, written to separate files
                print "Processing .phcx files and writing their scores to separate files."
                dp.processPHCXSeparately(self.search,self.debug,self.processSingleCandidate)
                
            # Path 1. b)
            else:
                # Process phcx files, written to single file.
                print "Processing .phcx files and writing their scores to: ",self.outputPath
                dp.processPHCXCollectively(self.search,self.debug,self.outputPath,self.arff,self.genProfileData,self.processSingleCandidate)
                
        # Path 2
        elif(not self.phcx and self.pfd and not self.superb):
            # Path 2. a)
            if(not self.singleFile):
                # Process pfd files, written to separate files.
                print "Processing .pfd files and writing their scores to separate files."
                dp.processPFDSeparately(self.search,self.debug,self.processSingleCandidate)
                
            # Path 2. b)
            else:
                # Process pfd files, written to single file.
                print "Processing .pfd files and writing their scores to: ",self.outputPath
                dp.processPFDCollectively(self.search,self.debug,self.outputPath,self.arff,self.genProfileData,self.processSingleCandidate)
                
        # Path 3
        elif(self.phcx and self.pfd and not self.superb):
            # Path 3. a)
            if(not self.singleFile):
                # Process pfd and phcx files, written to separate files.
                print "Processing .pfd AND .phcx files and writing their scores to separate files."
                dp.processPFDAndPHCXSeparately(self.search,self.debug,self.processSingleCandidate)
                
            # Path 3. b)
            else:
                # Process pfd and phcx files, written to a single file.
                print "Processing .pfd AND .phcx  files and writing their scores to: ",self.outputPath
                dp.processPFDAndPHCXCollectively(self.search,self.debug,self.outputPath,self.arff,self.genProfileData,self.processSingleCandidate)
        
        # Path 4. a)
        elif(self.superb and not self.pfd and not self.phcx):
            print "Processing SUPERB .phcx  files and writing their scores to: ",self.outputPath
            dp.processSUPERBCollectively(self.search,self.debug,self.outputPath,self.arff,self.genProfileData,self.processSingleCandidate)
            
        # Path 5
        elif(not self.phcx and not self.pfd):
            if(not self.singleFile):
                dp.processPFDAndPHCXSeparately(self.search,self.debug,self.processSingleCandidate)
            else:
                dp.processPFDAndPHCXCollectively(self.search,self.debug,self.outputPath,self.genProfileData,self.processSingleCandidate)
        
        else:
            print "Didn't know what to do with your input."
            
        print "Done."
    
    # ****************************************************************************************************
      
if __name__ == '__main__':
    ScoreGenerator().main()