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

File name:    DataProcessor.py
Created:      February 7th, 2014
Author:       Rob Lyon
 
Contact:    rob@scienceguyrob.com or robert.lyon@postgrad.manchester.ac.uk
Web:        <http://www.scienceguyrob.com> or <http://www.cs.manchester.ac.uk> 
            or <http://www.jb.man.ac.uk>
            
This code runs on python 2.4 or later.

Script which collects candidate files and initiates score generation.
 
"""

# Standard library Imports:
import sys,os,fnmatch,datetime

from numpy import array
from numpy import concatenate

# Custom file Imports:
import Utilities
import Candidate

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ****************************************************************************************************
#
# CLASS DEFINITION
#
# ****************************************************************************************************

class DataProcessor(Utilities.Utilities):
    """                
    Searches for candidate files in the local directory, or a directory specified by the user.
    
    """
    
    # ****************************************************************************************************
    #
    # Constructor.
    #
    # ****************************************************************************************************
    
    def __init__(self,debugFlag):
        """
        Default constructor.
        
        Parameters:
        
        debugFlag    -    the debugging flag. If set to True, then detailed
                          debugging messages will be printed to the terminal
                          during execution.
        """
        Utilities.Utilities.__init__(self,debugFlag)
        self.phcxRegex = "*.phcx.gz" #
        self.superbRegex = "*.phcx"  #
        self.pfdRegex = "*.pfd"      #
        self.pfdScrunchedRegex = "*.pfd.36scrunch"      # 
        self.scoreStore = []         # Variable which stores the scores created for a candidate.
        self.candidateErrorLog="CandidateErrorLog.txt"
        self.arffCreated = False
        self.pfd = False
        self.phcx = False
        self.superb = False
        
        # Make sure error log exists.
        if(not self.fileExists(self.candidateErrorLog)):
            self.appendToFile(self.candidateErrorLog, "")
        
    # ****************************************************************************************************
        
    def processPHCXSeparately(self,directory,verbose,processSingleCandidate):
        """
        Reads PHCX files in from a specified directory, and initiates
        candidate score generation. Here the scores are written to 
        separate files.
        
        Parameters:
        directory    -    the directory to look for candidates in.
        verbose      -    the verbose logging flag.
        processSingleCandidate - a flag that when true, indicates that only a single 
                                 specified candidate should be processed.
        """
        self.phcx=True
        fileTypeRegexes = [self.phcxRegex]
        self.processSeparately(directory, verbose, fileTypeRegexes,processSingleCandidate)
    
    # ****************************************************************************************************
        
    def processPHCXCollectively(self,directory,verbose,outPath,arff,genProfileData,processSingleCandidate):
        """
        Reads PHCX files in from a specified directory, and initiates
        candidate score generation. Here scores are written to a single
        candidate file.
        
        Parameters:
        directory    -    the directory to look for candidates in.
        verbose      -    the verbose logging flag.
        outPath      -    the path to write candidate scores to.
        arff         -    flag that when true, indicates data should be written to an arff file.
        genProfileData     -    flag which indicates that profile, rather than score data should be generated
        processSingleCandidate - a flag that when true, indicates that only a single 
                                 specified candidate should be processed.
        """
        self.phcx=True
        fileTypeRegexes = [self.phcxRegex]
        self.processCollectively(directory, verbose, fileTypeRegexes,outPath,arff,genProfileData,processSingleCandidate)
    
    # ****************************************************************************************************
        
    def processSUPERBCollectively(self,directory,verbose,outPath,arff,genProfileData,processSingleCandidate):
        """
        Reads SUPERB survey PHCX files in from a specified directory, and initiates
        candidate score generation. Here scores are written to a single candidate file.
        
        Parameters:
        directory    -    the directory to look for candidates in.
        verbose      -    the verbose logging flag.
        outPath      -    the path to write candidate scores to.
        arff         -    flag that when true, indicates data should be written to an arff file.
        genProfileData     -    flag which indicates that profile, rather than score data should be generated.
        processSingleCandidate - a flag that when true, indicates that only a single 
                                 specified candidate should be processed.
        """
        self.superb=True
        fileTypeRegexes = [self.superbRegex]
        self.processCollectively(directory, verbose, fileTypeRegexes,outPath,arff,genProfileData,processSingleCandidate)
    
    # ****************************************************************************************************
    
    def processPFDSeparately(self,directory,verbose,processSingleCandidate):
        """
        Reads PFD files in from a specified directory, and initiates
        candidate score generation. Here scores are written to separate
        files.
        
        Parameters:
        directory    -    the directory to look for candidates in.
        verbose      -    the verbose logging flag.
        processSingleCandidate - a flag that when true, indicates that only a single 
                                 specified candidate should be processed.
        """
        self.pfd=True
        fileTypeRegexes = [self.pfdRegex,self.pfdScrunchedRegex]
        self.processSeparately(directory, verbose, fileTypeRegexes,processSingleCandidate)
    
    # ****************************************************************************************************
        
    def processPFDCollectively(self,directory,verbose,outPath,arff,genProfileData,processSingleCandidate):
        """
        Reads PFD files in from a specified directory, and initiates
        candidate score generation. Here scores are written to a single
        candidate file.
        
        Parameters:
        directory    -    the directory to look for candidates in.
        verbose      -    the verbose logging flag.
        outPath      -    the path to write candidate scores to.
        arff         -    flag that when true, indicates data should be written to an arff file.
        genProfileData     -    flag which indicates that profile, rather than score data should be generated
        processSingleCandidate - a flag that when true, indicates that only a single 
                                 specified candidate should be processed.
        """
        self.pfd=True
        fileTypeRegexes = [self.pfdRegex,self.pfdScrunchedRegex]
        self.processCollectively(directory, verbose, fileTypeRegexes,outPath,arff,False,processSingleCandidate)
    
    # ****************************************************************************************************
        
    def processPFDAndPHCXSeparately(self,directory,verbose,processSingleCandidate):
        """
        Reads both PFD and PHCX files in from a specified directory,
        and initiates candidate score generation. Here scores are
        written to separate files.
        
        Parameters:
        directory    -    the directory to look for candidates in.
        verbose      -    the verbose logging flag.
        outPath      -    the path to write candidate scores to.
        processSingleCandidate - a flag that when true, indicates that only a single 
                                 specified candidate should be processed.
        """
        self.pfd=True
        self.phcx=True
        fileTypeRegexes = [self.phcxRegex,self.pfdRegex,self.pfdScrunchedRegex]
        self.processSeparately(directory, verbose, fileTypeRegexes,processSingleCandidate)
    
    # ****************************************************************************************************
        
    def processPFDAndPHCXCollectively(self,directory,verbose,outPath,arff,genProfileData,processSingleCandidate):
        """
        Reads both PFD and PHCX files in from a specified directory,
        and initiates candidate score generation. Here scores are written
        to a single candidate file.
        
        Parameters:
        directory    -    the directory to look for candidates in.
        verbose      -    the verbose logging flag.
        outPath      -    the path to write candidate scores to.
        arff         -    flag that when true, indicates data should be written to an arff file.
        genProfileData     -    flag which indicates that profile, rather than score data should be generated
        processSingleCandidate - a flag that when true, indicates that only a single 
                                 specified candidate should be processed.
        """
        self.pfd=True
        self.phcx=True
        fileTypeRegexes = [self.phcxRegex,self.pfdRegex,self.pfdScrunchedRegex]
        self.processCollectively(directory, verbose, fileTypeRegexes,outPath,arff,False,processSingleCandidate)
    
    # ****************************************************************************************************
        
    def labelPHCX(self,directory,verbose):
        """
        Allows the user to label PHCX data.
        
        Parameters:
        directory    -    the directory to look for candidates in.
        verbose      -    the verbose logging flag.
        """
        self.phcx=True
        fileTypeRegexes = [self.phcxRegex]
        self.label(directory, verbose, fileTypeRegexes)
        
    def labelPFD(self,directory,verbose,outPath,arff,genProfileData,processSingleCandidate):
        """
        Allows the user to label PHCX data.
        
        Parameters:
        directory    -    the directory to look for candidates in.
        verbose      -    the verbose logging flag.
        """
        self.pfd=True
        fileTypeRegexes = [self.pfdRegex,self.pfdScrunchedRegex]
        self.label(directory, verbose, fileTypeRegexes)
        
    def dmprofPHCX(self,directory,verbose,outPath,arff,processSingleCandidate):
        """
        Generates DM and profile stat scores (no 22 candidate scores).
        
        Parameters:
        directory    -    the directory to look for candidates in.
        verbose      -    the verbose logging flag.
        outPath      -    the path to write candidate scores to.
        arff         -    flag that when true, indicates data should be written to an arff file.
        processSingleCandidate - a flag that when true, indicates that only a single 
                                 specified candidate should be processed.
        """
        self.phcx=True
        fileTypeRegexes = [self.phcxRegex]
        self.dmprof(directory, verbose, fileTypeRegexes,outPath,arff,processSingleCandidate)
        
    def dmprofPFD(self,directory,verbose,outPath,arff,processSingleCandidate):
        """
        Generates DM and profile stat scores (no 22 candidate scores).
        
        Parameters:
        directory    -    the directory to look for candidates in.
        verbose      -    the verbose logging flag.
        outPath      -    the path to write candidate scores to.
        arff         -    flag that when true, indicates data should be written to an arff file.
        processSingleCandidate - a flag that when true, indicates that only a single 
                                 specified candidate should be processed.
        """
        self.pfd=True
        fileTypeRegexes = [self.pfdRegex,self.pfdScrunchedRegex]
        self.dmprof(directory, verbose, fileTypeRegexes,outPath,arff,processSingleCandidate)
        
    def dmprofSUPERB(self,directory,verbose,outPath,arff,processSingleCandidate):
        """
        Generates DM and profile stat scores (no 22 candidate scores).
        
        Parameters:
        directory    -    the directory to look for candidates in.
        verbose      -    the verbose logging flag.
        outPath      -    the path to write candidate scores to.
        arff         -    flag that when true, indicates data should be written to an arff file.
        processSingleCandidate - a flag that when true, indicates that only a single 
                                 specified candidate should be processed.
        """
        self.superb=True
        fileTypeRegexes = [self.superbRegex]
        self.dmprof(directory, verbose, fileTypeRegexes,outPath,arff,processSingleCandidate)
    
    # ****************************************************************************************************
    
    def storeScore(self,candidate,scores,outPath):
        """
        Appends candidate scores to a list held by this object. This records 
        each score in memory as opposed to writing them out to a file each time.
        
        Parameters:
        
        candidate  -    The name of the candidate the scores belong to.
        scores     -    A float array of candidate scores.
        outputFile -    The file to write the scores to.
        
        Return:
        N/A
        """
        
        # Join scores into single comma separated line.
        allScores =  ",".join(map(str, scores))
        entry1 = candidate+","+allScores
        entry2 = entry1.replace("nan","0")
        entry3 = entry2.replace("inf","0")
        self.scoreStore.append(entry3)
        
    # ****************************************************************************************************
    
    def prepareARFFFile(self,path,genProfileData):
        """
        Creates an ARFF file with the appropriate headers, reader for data
        to be written to the file.
        
        Parameters:
        path               -    the path to the file to prepare
        genProfileData     -    flag which indicates that profile, rather than score data should be generated
        
        Returns:
        N/A
        """
        
        i = datetime.datetime.now()
        dt = i.isoformat()

        header = "@relation PulsarCandidates_"+dt+"\n"
        
        # Default when dealing with only 22 scores plus class label.
        attributes = 23
        
        if(genProfileData and self.superb):
            attributes=65 # SUPERB profile data has 64 bins, plus the class label.
        elif(genProfileData and self.phcx):
            attributes=129# HTRU profile data has 128 bins, plus the class label.
            
        for n in range(1, attributes):
            header += "@attribute Score"
            header += str(n)
            header += " numeric\n"
        
        header += "@attribute class {0,1}\n@data\n"
            
        if(self.fileExists(path) == False):
            output = open(path, 'w+')
            output.write(str(header))
            output.close()
        elif(self.fileExists(path) == True):
            self.appendToFile(path, header)
        
    def prepareDMProfileARFFFile(self,path):
        """
        Creates an ARFF file with the appropriate headers, reader for data
        to be written to the file.
        
        Parameters:
        path               -    the path to the file to prepare
        genProfileData     -    flag which indicates that profile, rather than score data should be generated
        
        Returns:
        N/A
        """
        
        i = datetime.datetime.now()
        dt = i.isoformat()

        header = "@relation PulsarCandidates_"+dt+"\n"
        header +="@attribute Profile_mean numeric\n"
        header +="@attribute Profile_stdev numeric\n"
        header +="@attribute Profile_skewness numeric\n"
        header +="@attribute Profile_kurtosis numeric\n"
        header +="@attribute DM_mean numeric\n"
        header +="@attribute DM_stdev numeric\n"
        header +="@attribute DM_skewness numeric\n"
        header +="@attribute DM_kurtosis numeric\n"
        header += "@attribute class {0,1}\n@data\n"
        
        if(self.fileExists(path) == False):
            output = open(path, 'w+')
            output.write(str(header))
            output.close()
        elif(self.fileExists(path) == True):
            self.appendToFile(path, header)
        
    # ****************************************************************************************************
    
    def storeScoreARFF(self,candidate,scores,outPath):
        """
        Appends candidate scores to a list held by this object. This records 
        each score in memory as opposed to writing them out to a file each time.
        
        Parameters:
        
        candidate  -    The name of the candidate the scores belong to.
        scores     -    A float array of candidate scores.
        outputFile -    The file to write the scores to.
        
        Return:
        N/A
        """
        
        # Join scores into single comma separated line.
        allScores =  ",".join(map(str, scores))
        entry = allScores+",?%"+candidate
        entry = entry.replace("nan","0")
        entry = entry.replace("inf","0")
        self.scoreStore.append(entry)
            
    # ****************************************************************************************************
        
    def outputScores(self,scores,outputFile):
        """
        Writes candidate scores to the specified file in CSV format.
        
        Parameters:
        
        scores     -    A float array of candidate scores.
        outputFile -    The file to write the scores to.
        
        Return:
        N/A
        """
        # Create path to output file where scores will be written.
        output = open(outputFile + ".dat", 'w') # @UnusedVariable - this silences IDE warnings
        allScores =  ",".join(map(str, scores))
        allScores1 = allScores.replace("nan","0")
        allScores2 = allScores1.replace("inf","0")
        output.write(str(allScores2))
        output.close()
    
    # ****************************************************************************************************
    
    def processCollectively(self,directory,verbose,fileTypeRegexes,outPath,arff,genProfileData,processSingleCandidate):
        """
        Processes pulsar candidates of all supported file types in the fileTypeRegexes array.
        Writes the scores for each candidate to a single file.
        
        Parameters:
        
        directory          -    the directory containing the candidates to process.
        verbose            -    debug logging flag, if true output statements will be verbose.
        fileTypeRegexes    -    an array containing the regular expressions that will be used
                                by the glob.glob() command to find files of interest.
        outPath            -    the file to where scores for all candidates will be written.
        arff               -    flag that when true, indicates data should be written to an arff file.
        genProfileData     -    flag which indicates that profile, rather than score data should be generated
        processSingleCandidate - a flag that when true, indicates that only a single 
                                 specified candidate should be processed.
        Return:
        
        N/A
        """
        
        # Prepare output ARFF file
        if(arff):
            self.prepareARFFFile(outPath,genProfileData)
            
        # Variables used to store stats on candidate scores generated,
        # and failure rate etc.
        candidatesProcessed = 0;
        successes = 0;
        failures = 0;
        
        # If user has provided no directory.
        if(directory ==""):
            directory=os.path.dirname(os.path.realpath(__file__))
            print "User has not provided a search directory - searching local directory"
        
        start = datetime.datetime.now()
        
        if(processSingleCandidate == False):
            
            for filetype in fileTypeRegexes:
                
                for root, subFolders, filenames in os.walk(directory):
                    
                    for filename in fnmatch.filter(filenames, filetype):
                        
                        cand = os.path.join(root, filename)
                        
                        candidatesProcessed+=1
                        
                        #print "Processing candidate:\t" , cand
                        
                        try:
                            
                            c = Candidate.Candidate(cand,str(directory+cand))
                            
                            if(genProfileData):
                                scores = c.calculateProfileScores(self.debug)
                            else:# Standard 22 scores
                                scores = c.calculateScores(self.debug)
                            
                            if(arff):
                                self.storeScoreARFF(cand, scores, outPath)
                            else:
                                self.storeScore(cand, scores, outPath)
                            
                        except Exception as e: # Catch *all* exceptions.
                            print "Error reading profile data :\n\t", sys.exc_info()[0]
                            print self.format_exception(e)
                            print cand, " did not have scores generated."
                            self.appendToFile(self.candidateErrorLog,cand+"\n")
                            failures+=1
                            continue
                        
                        successes+=1
        else:
            singleCand = directory
            # Added to allow code to deal with a single candidate     
            #print "Processing candidate:\t" , singleCand
            candidatesProcessed+=1
            
            # If we have a file containing paths to candidates.
            if(".txt" in singleCand):
                
                f = open(self.path)
                lines = f.readlines()
                f.close()
        
                if(len(lines)>0):
                    
                    for line in lines:
                
                        try:
                            c = Candidate.Candidate(line,line)
                            
                            if(genProfileData):
                                scores = c.calculateProfileScores(self.debug)
                            else:# Standard 22 scores
                                scores = c.calculateScores(self.debug)
                            
                            if(arff):
                                self.storeScoreARFF(singleCand, scores, outPath)
                            else:
                                self.storeScore(singleCand, scores, outPath)
                                
                            successes+=1
                            
                        except Exception as e: # Catch *all* exceptions.
                            print "Error reading profile data :\n\t", sys.exc_info()[0]
                            print self.format_exception(e)
                            print directory, " did not have scores generated."
                            self.appendToFile(self.candidateErrorLog,line+"\n")
                            failures+=1
            else:
                
                try:
                    c = Candidate.Candidate(singleCand,singleCand)
                
                    if(genProfileData):
                        scores = c.calculateProfileScores(self.debug)
                    else:# Standard 22 scores
                        scores = c.calculateScores(self.debug)
                
                    if(arff):
                        self.storeScoreARFF(singleCand, scores, outPath)
                    else:
                        self.storeScore(singleCand, scores, outPath)
                    
                    successes+=1
                    
                except Exception as e: # Catch *all* exceptions.
                    print "Error reading profile data :\n\t", sys.exc_info()[0]
                    print self.format_exception(e)
                    print directory, " did not have scores generated."
                    self.appendToFile(self.candidateErrorLog,singleCand+"\n")
                    failures+=1
        
        end = datetime.datetime.now()
        
        outputText=""
        for s in self.scoreStore:
            outputText+=s+"\n"
        
        self.appendToFile(outPath, outputText)
        
        print "\nCandidates processed:\t",candidatesProcessed
        print "Successes:\t", successes
        print "Failures:\t", failures
        print "Execution time: ", str(end - start)
        
    # ****************************************************************************************************
                        
    def processSeparately(self,directory,verbose,fileTypeRegexes,processSingleCandidate):
        """
        Processes the pulsar candidates of the type specified in the fileTypeRegexes array.
        Writes the scores for each candidate to a unique separate file.
        
        Parameters:
        
        directory          -    the directory containing the candidates to process.
        verbose            -    debug logging flag, if true output statements will be verbose.
        fileTypeRegexes    -    an array containing the regular expressions that will be used
                                by the glob.glob() command to find files of interest.
        processSingleCandidate - a flag that when true, indicates that only a single 
                                 specified candidate should be processed.
                                
        Return:
        
        N/A
        """
        
        # Variables used to store stats on candidate scores generated,
        # and failure rate etc.
        candidatesProcessed = 0;
        successes = 0;
        failures = 0;
        
        # If user has provided no directory.
        if(directory ==""):
            directory=os.path.dirname(os.path.realpath(__file__))
            print "User has not provided a search directory - searching local directory"
        
        start = datetime.datetime.now()
        
        if(processSingleCandidate == False):    
            for filetype in fileTypeRegexes:
                
                for root, subFolders, filenames in os.walk(directory):
                    
                    for filename in fnmatch.filter(filenames, filetype):
                        
                        cand = os.path.join(root, filename)
                        
                        candidatesProcessed+=1
                        
                        #print "Processing candidate:\t" , cand
                        
                        try:
                            
                            c = Candidate.Candidate(cand,str(directory+cand))
                            scores = c.calculateScores(self.debug)
                            self.outputScores(scores,cand)
                                
                        except Exception as e: # Catch *all* exceptions.
                            print "Error processing candidates :\n\t", sys.exc_info()[0]
                            print self.format_exception(e)
                            print cand, " did not have scores generated."
                            self.appendToFile(self.candidateErrorLog,cand+"\n")
                            failures+=1
                            continue
                        
                        successes+=1
        else:
            # Added to allow code to deal with a single candidate
            singleCand = directory   
            #print "Processing candidate:\t" , singleCand
            candidatesProcessed+=1
            try:
                
                c = Candidate.Candidate(singleCand,singleCand)
                scores = c.calculateScores(self.debug)
                self.outputScores(scores,singleCand)
                successes+=1
                    
            except Exception as e: # Catch *all* exceptions.
                print "Error processing candidates :\n\t", sys.exc_info()[0]
                print self.format_exception(e)
                print singleCand, " did not have scores generated."
                self.appendToFile(self.candidateErrorLog,directory+"\n")
                failures+=1
        
        end = datetime.datetime.now()
        
        print "\nCandidates processed:\t",candidatesProcessed
        print "Successes:\t", successes
        print "Failures:\t", failures
        print "Execution time: ", str(end - start)
    
    # ****************************************************************************************************
    
    def label(self,directory,verbose,fileTypeRegexes):
        """
        Labels pulsar candidates, generates scores and creates meta data.
        
        Parameters:
        
        directory          -    the directory containing the candidates to process.
        verbose            -    debug logging flag, if true output statements will be verbose.
        fileTypeRegexes    -    an array containing the regular expressions that will be used
                                by the glob.glob() command to find files of interest.
        Return:
        
        N/A
        """
            
        # Variables used to store stats on candidate scores generated,
        # and failure rate etc.
        candidatesProcessed = 0;
        successes = 0;
        failures = 0;
        
        # If user has provided no directory.
        if(directory ==""):
            directory=os.path.dirname(os.path.realpath(__file__))
            print "User has not provided a search directory - searching local directory"
        
        metaFile = directory + "/Cands.meta"
        scoresFile = directory + "/Scores.csv"
        profileFile = directory + "/Profile.csv"
        subbandFile = directory + "/Subband.csv"
        subintFile = directory + "/Subint.csv"
        dmcurveFile = directory + "/DMCurve.csv"
        
        self.positive = 0
        self.negative = 0
        
        start = datetime.datetime.now()
        
        for filetype in fileTypeRegexes:
            
            for root, subFolders, filenames in os.walk(directory):
                
                
                for filename in fnmatch.filter(filenames, filetype):
                    
                    cand = os.path.join(root, filename)
                    
                    candidatesProcessed+=1
                    
                    #print "Processing candidate:\t" , cand
                    
                    try:
                        
                        # Get scores.
                        label = "0"
                        c = Candidate.Candidate(cand,str(directory+cand))
                        twenty_two_scores = array(c.calculateScores(self.debug))
                        profileData = array(c.calculateProfileScores(self.debug))
                        #subbandData = array(c.getSubbandData(self.debug))
                        #subintData = array(c.getSubintData(self.debug))
                        dmCurveData = array(c.getDMCurveData(self.debug))
                            
                        #pngPath = cand+".png"
                        #fig=plt.figure(figsize=(10,8))# @UnusedVariable
                        #plt.ion() 
                             
                        #candidateImage = mpimg.imread(pngPath)
                        #plt.imshow(candidateImage, aspect='auto')
                        #plt.show()
                        
                        # Wait for user input. If an invalid input
                        # is provided, then prompt again.
                        #selection = ""
                        #while 'p' not in selection and 'n' not in selection and 'm' not in selection and 'x' not in selection:
                        #    selection = raw_input('Type p for known pulsar, n for not pulsar (x for exit):\n')
                         
                        #if 'p' in selection:
                        #    self.positive+=1
                        #    label = "1"
                        #elif 'n' in selection:
                        #    self.negative+=1
                        #    label = "0"
                            
                        #plt.clf()
                        #plt.close()
                        
                        # Now write data to files.
                        for v in twenty_two_scores:
                            self.appendToFile(scoresFile, str(v)+",")
                            
                        self.appendToFile(scoresFile,str(label))
                        self.appendToFile(scoresFile,",%"+cand+"\n")
                        
                        for v in profileData:
                            self.appendToFile(profileFile, str(v)+",")
                            
                        self.appendToFile(profileFile,str(label))
                        self.appendToFile(profileFile,",%"+cand+"\n")
                        
                        #for v in subbandData:
                        #    self.appendToFile(subbandFile, str(v)+",")
                            
                        #self.appendToFile(subbandFile,str(label))
                        #self.appendToFile(subbandFile,",%"+cand+"\n")
                        
                        #for v in subintData:
                        #    self.appendToFile(subintFile, str(v)+",")
                            
                        #self.appendToFile(subintFile,str(label))
                        #self.appendToFile(subintFile,",%"+cand+"\n")
                        
                        for v in dmCurveData:
                            self.appendToFile(dmcurveFile, str(v)+",")
                            
                        self.appendToFile(dmcurveFile,str(label))
                        self.appendToFile(dmcurveFile,",%"+cand+"\n")
                        
                        self.appendToFile(metaFile,cand+","+str(label)+"\n")
                    except Exception as e: # Catch *all* exceptions.
                        print "Error reading profile data :\n\t", sys.exc_info()[0]
                        print self.format_exception(e)
                        print cand, " did not have scores generated."
                        self.appendToFile(self.candidateErrorLog,cand+"\n")
                        failures+=1
                        continue
                    
                    successes+=1
        
        end = datetime.datetime.now()
        
        print "\nCandidates processed:\t",candidatesProcessed
        print "Successes:\t", successes
        print "Failures:\t", failures
        print "Positive:\t", self.positive
        print "Negative:\t", self.negative
        print "Execution time: ", str(end - start)
    
    # ******************************************************************************************
    
    def dmprof(self,directory,verbose,fileTypeRegexes,outPath,arff,processSingleCandidate):
        """
        Generates DM and profile stat scores (no 22 candidate scores).
        
        Parameters:
        
        directory          -    the directory containing the candidates to process.
        verbose            -    debug logging flag, if true output statements will be verbose.
        fileTypeRegexes    -    an array containing the regular expressions that will be used
                                by the glob.glob() command to find files of interest.
        outPath            -    the file to where scores for all candidates will be written.
        arff               -    flag that when true, indicates data should be written to an arff file.
        processSingleCandidate - a flag that when true, indicates that only a single 
                                 specified candidate should be processed.
        Return:
        
        N/A
        """
            
        # Prepare output ARFF file
        if(arff):
            self.prepareDMProfileARFFFile(outPath)
            
        # Variables used to store stats on candidate scores generated,
        # and failure rate etc.
        candidatesProcessed = 0;
        successes = 0;
        failures = 0;
        
        # If user has provided no directory.
        if(directory ==""):
            directory=os.path.dirname(os.path.realpath(__file__))
            print "User has not provided a search directory - searching local directory"
        
        start = datetime.datetime.now()
        
        if(processSingleCandidate == False):
            for filetype in fileTypeRegexes:
                
                for root, subFolders, filenames in os.walk(directory):
                    
                    
                    for filename in fnmatch.filter(filenames, filetype):
                        
                        cand = os.path.join(root, filename)
                        
                        candidatesProcessed+=1
                        
                        #print "Processing candidate:\t" , cand
                        
                        try:
                            
                            c = Candidate.Candidate(cand,str(directory+cand))
                            
                            profileStats = array(c.calculateProfileStatScores(self.debug))
                            DMCurveStats = array(c.calculateDMCurveStatScores(self.debug))
                            scores = concatenate([profileStats , DMCurveStats])
                            
                            if(arff):
                                self.storeScoreARFF(cand, scores, outPath)
                            else:
                                self.storeScore(cand, scores, outPath)
                            
                        except Exception as e: # Catch *all* exceptions.
                            print "Error reading profile data :\n\t", sys.exc_info()[0]
                            print self.format_exception(e)
                            print cand, " did not have scores generated."
                            self.appendToFile(self.candidateErrorLog,cand+"\n")
                            failures+=1
                            continue
                        
                        successes+=1
        else:
            singleCand = directory
            # Added to allow code to deal with a single candidate     
            #print "Processing candidate:\t" , singleCand
            candidatesProcessed+=1
            
            # If we have a file containing paths to candidates.
            if(".txt" not in singleCand):
                
                try:
                    c = Candidate.Candidate(singleCand,singleCand)
                    
                    profileStats = array(c.calculateProfileStatScores(self.debug))
                    DMCurveStats = array(c.calculateDMCurveStatScores(self.debug))
                    scores = concatenate([profileStats , DMCurveStats])
                    
                    # Order of scores for output:
                    # Profile_mean
                    # Profile_stdev
                    # Profile_skewness
                    # Profile_kurtosis
                    # DM_mean
                    # DM_stdev
                    # DM_skewness
                    # DM_kurtosis     
                    
                    if(arff):
                        self.storeScoreARFF(singleCand, scores, outPath)
                    else:
                        self.storeScore(singleCand, scores, outPath)
                        
                    successes+=1
                        
                except Exception as e: # Catch *all* exceptions.
                    print "Error reading profile data :\n\t", sys.exc_info()[0]
                    print self.format_exception(e)
                    print directory, " did not have scores generated."
                    self.appendToFile(self.candidateErrorLog,singleCand+"\n")
                    failures+=1
            else:
                
                f = open(self.path)
                lines = f.readlines()
                f.close()
        
                if(len(lines)>0):
                    
                    for line in lines:
                        
                        try:
                            c = Candidate.Candidate(line,line)
                            
                            profileStats = array(c.calculateProfileStatScores(self.debug))
                            DMCurveStats = array(c.calculateDMCurveStatScores(self.debug))
                            scores = concatenate([profileStats , DMCurveStats])
                            
                            # Order of scores for output:
                            # Profile_mean
                            # Profile_stdev
                            # Profile_skewness
                            # Profile_kurtosis
                            # DM_mean
                            # DM_stdev
                            # DM_skewness
                            # DM_kurtosis     
                            
                            if(arff):
                                self.storeScoreARFF(singleCand, scores, outPath)
                            else:
                                self.storeScore(singleCand, scores, outPath)
                                
                            successes+=1
                                
                        except Exception as e: # Catch *all* exceptions.
                            print "Error reading profile data :\n\t", sys.exc_info()[0]
                            print self.format_exception(e)
                            print directory, " did not have scores generated."
                            self.appendToFile(self.candidateErrorLog,line+"\n")
                            failures+=1
                
        
        end = datetime.datetime.now()
        
        outputText=""
        for s in self.scoreStore:
            outputText+=s+"\n"
        
        self.appendToFile(outPath, outputText)
        
        print "\nCandidates processed:\t",candidatesProcessed
        print "Successes:\t", successes
        print "Failures:\t", failures
        print "Execution time: ", str(end - start)
    
    # ******************************************************************************************