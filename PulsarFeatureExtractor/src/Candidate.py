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

File name:    Candidate.py
Created:      February 3rd, 2014
Author:       Rob Lyon
 
Contact:    rob@scienceguyrob.com or robert.lyon@postgrad.manchester.ac.uk
Web:        <http://www.scienceguyrob.com> or <http://www.cs.manchester.ac.uk> 
            or <http://www.jb.man.ac.uk>
            
This code runs on python 2.4 or later.

Represents a pulsar candidate.
 
"""

# Custom Imports
import PHCXFile as phcx
import PFDFile as pfd
import SUPERBPHCXFile as superb

# ****************************************************************************************************
#
# CLASS DEFINITION
#
# ****************************************************************************************************

class Candidate:
    """
    Represents a pulsar candidate. This class is used both to generate scores.
    
    """
    
    # ****************************************************************************************************
    #
    # INIT FUNCTION
    #
    # ****************************************************************************************************
    def __init__(self,name="Unknown",path=""):
        """
        Represents an individual Pulsar candidate.
        
        Parameters:
        
        name    -     the primary name for this individual candidate.
                      The file name typically used.
        path    -     the full path to the candidate file.
        
        """
        self.candidateName = name # Name of the candidate file, minus the path.
        self.candidatePath = path # The full path to the candidate.
        self.scores = []          # Stores all candidate scores.
        self.label = "Unknown"    # The label this candidate has received, either "POSITIVE" or "NEGATIVE".
        
        # Some candidates may be 'special' in that they possess extreme
        # values for one of their scores. This could be a minimum or max
        # value. This variable stores the index of that special score
        # (if it exists) with respect to the scores[] array.
        self.specialScore=-1      
        
        # If this candidate does have a special score, this variable
        # is used to provide a single word description of why it is
        # 'special'. So for example if specialScore=1 then special="MAX"
        # would indicate this candidate has the highest score 1 known.
        self.special="None"       
        
    # ****************************************************************************************************
    #
    # UTILITY FUNCTIONS.
    #
    # ****************************************************************************************************
    
    def addScores(self,lineFromFile):
        """
        Adds the scores read in from the candidate .dat file to this object.
        
        Parameters:
        lineFromFile    -    the string text from the file. This string will
                             be comma separated, e.g.
                             
                             1.0,2.0,3.0,4.0,5.0,...,22.0
        
        Returns:
        
        N/A
        
        """
        substrings = lineFromFile.split(",")

        counter = 1
        for s in substrings:
            if(s != "" or len(s)!=0):
                counter+=1
                self.scores.append(float(s))
            
    # ****************************************************************************************************
    # 
    # SCORE CALCULATIONS.
    #
    # ****************************************************************************************************
    
    def calculateScores(self,verbose):
        """
        Calculates the scores for this candidate. If the file name of
        this Candidate object contains .pfd, then the PFD file score generation
        code will be executed. Likewise if the file name ends in phcx, then
        PHCX file score generation code will be executed.
        
        If further data file formats need to be processed, then changes need
        to be made here to cope with them. For example, if a new file format
        called .x appears, then below a check must be added for .rob files,
        along with a new script to deal with these files.
        
        Parameters:
        verbose    -    the verbose logging flag.
        
        Returns:
        
        The candidate scores as an array of floats.
        """
        
        if(".pfd" in self.candidateName):
            #print "Computing PFD scores."
            c = pfd.PFD(verbose,self.candidateName)
            self.scores = c.compute()
            return self.scores
        elif(".gz" in self.candidateName):
            #print "Computing PHCX scores."
            c = phcx.PHCX(verbose,self.candidateName)
            self.scores = c.compute()
            return self.scores
        else:
            #print "Computing SUPERB PHCX scores."
            c = superb.SUPERBPHCX(verbose,self.candidateName)
            self.scores = c.compute()
            return self.scores
    
    # ****************************************************************************************************
        
    def calculateProfileScores(self,verbose):
        """
        Calculates the scores as profile data for this candidate. If the file name of
        this Candidate object contains .pfd, then the PFD file score generation
        code will be executed. Likewise if the file name ends in phcx, then
        PHCX file score generation code will be executed.
        
        If further data file formats need to be processed, then changes need
        to be made here to cope with them. For example, if a new file format
        called .x appears, then below a check must be added for .x files,
        along with a new script to deal with these files.
        
        Parameters:
        verbose    -    the verbose logging flag.
        
        Returns:
        
        The candidate scores as an array of floats.
        """
        
        if(".pfd" in self.candidateName):
            #print "Computing PFD scores."
            c = pfd.PFD(verbose,self.candidateName)
            self.scores = c.computeProfileScores()
            return self.scores
        elif(".gz" in self.candidateName):
            #print "Computing PHCX scores."
            c = phcx.PHCX(verbose,self.candidateName)
            self.scores = c.computeProfileScores()
            return self.scores
        else:
            #print "Computing SUPERB PHCX scores."
            c = superb.SUPERBPHCX(verbose,self.candidateName)
            self.scores = c.computeProfileScores()
            return self.scores
    
    def calculateProfileStatScores(self,verbose):
        """
        Calculates the stat scores of profile data for this candidate. If the file name of
        this Candidate object contains .pfd, then the PFD file score generation
        code will be executed. Likewise if the file name ends in phcx, then
        PHCX file score generation code will be executed.
        
        If further data file formats need to be processed, then changes need
        to be made here to cope with them. For example, if a new file format
        called .x appears, then below a check must be added for .x files,
        along with a new script to deal with these files.
        
        Parameters:
        verbose    -    the verbose logging flag.
        
        Returns:
        
        The candidate scores as an array of floats.
        """
        
        if(".pfd" in self.candidateName):
            #print "Computing PFD scores."
            c = pfd.PFD(verbose,self.candidateName)
            self.scores = [] # Clear any existing data
            self.scores = c.computeProfileStatScores()
            return self.scores
        elif(".gz" in self.candidateName):
            #print "Computing PHCX scores."
            c = phcx.PHCX(verbose,self.candidateName)
            self.scores = [] # Clear any existing data
            self.scores = c.computeProfileStatScores()
            return self.scores
        else:
            #print "Computing SUPERB PHCX scores."
            c = superb.SUPERBPHCX(verbose,self.candidateName)
            self.scores = [] # Clear any existing data
            self.scores = c.computeProfileStatScores()
            return self.scores
    
    # ****************************************************************************************************
        
    def getDMCurveData(self,verbose):
        """
        Gets the DM curve data belonging to this candidate.
        
        Parameters:
        verbose    -    the verbose logging flag.
        
        Returns:
        
        The DM curve data as an array of floats.
        """
        
        if(".pfd" in self.candidateName):
            #print "Computing PFD scores."
            c = pfd.PFD(verbose,self.candidateName)
            self.scores = c.getDMCurveData()
            return self.scores
        elif(".gz" in self.candidateName):
            #print "Computing PHCX scores."
            c = phcx.PHCX(verbose,self.candidateName)
            self.scores = c.getDMCurveData()
            return self.scores
        else:
            #print "Computing SUPERB PHCX scores."
            return []
        
    def calculateDMCurveStatScores(self,verbose):
        """
        Gets the DM curve data belonging to this candidate.
        
        Parameters:
        verbose    -    the verbose logging flag.
        
        Returns:
        
        The DM curve data as an array of floats.
        """
        
        if(".pfd" in self.candidateName):
            #print "Computing PFD scores."
            c = pfd.PFD(verbose,self.candidateName)
            self.scores = [] # Clear any existing data
            self.scores = c.computeDMCurveStatScores()
            return self.scores
        elif(".gz" in self.candidateName):
            #print "Computing PHCX scores."
            c = phcx.PHCX(verbose,self.candidateName)
            self.scores = [] # Clear any existing data
            self.scores = c.computeDMCurveStatScores()
            return self.scores
        else:
            #print "Computing SUPERB PHCX scores."
            return []
        
    # ****************************************************************************************************
        
    def getSubbandData(self,verbose):
        """
        Gets the sub band data belonging to this candidate.
        
        Parameters:
        verbose    -    the verbose logging flag.
        
        Returns:
        
        The sub band data as an array of floats.
        """
        
        if(".pfd" in self.candidateName):
            #print "Computing PFD scores."
            return []
        elif(".gz" in self.candidateName):
            #print "Computing PHCX scores."
            c = phcx.PHCX(verbose,self.candidateName)
            self.scores = c.getSubbandData()
            return self.scores
        else:
            #print "Computing SUPERB PHCX scores."
            return []
    
    # ****************************************************************************************************
        
    def getSubintData(self,verbose):
        """
        Gets the sub int data belonging to this candidate.
        
        Parameters:
        verbose    -    the verbose logging flag.
        
        Returns:
        
        The sub int data as an array of floats.
        """
        
        if(".pfd" in self.candidateName):
            #print "Computing PFD scores."
            return []
        elif(".gz" in self.candidateName):
            #print "Computing PHCX scores."
            c = phcx.PHCX(verbose,self.candidateName)
            self.scores = c.getSubintData()
            return self.scores
        else:
            #print "Computing SUPERB PHCX scores."
            return []
        
    # ****************************************************************************************************
        
    def getScore(self,index):
        """
        Obtains the specified score for this candidate. Compensates
        for zero indexing. So if score 1 is desired simply call
        getScore(1).
        
        Parameters:
        index    -    the index of the score to obtain.
        
        Returns:
        
        The floating point value of the desired score.
        """
        return float(self.scores[index-1])
    
    # ****************************************************************************************************
    
    def getName(self):
        """
        Obtains the name of the candidate file, not the full path.
        
        
        Returns:
        
        The name of the candidate file.
        """
        return self.candidateName
    
    # ****************************************************************************************************
    
    def getPath(self):
        """
        Obtains the full path to the candidate.
        
        
        Returns:
        
        The full path to the candidate.
        """
        return self.candidatePath
    
    # ****************************************************************************************************
    
    def setLabel(self,l):
        """
        Sets the label describing this candidate, i.e. positive or negative.
        To be clear the input should either be l="POSITIVE" or l="NEGATIVE".
        
        Parameters:
        l    -    the label for this candidate, i.e. l="POSITIVE" or l="NEGATIVE".
        
        Returns:
        
        N/A
        """
        self.label = l
    
    # ****************************************************************************************************
        
    def getLabel(self):
        """
        Gets the label describing this candidate, i.e. "POSITIVE" or "NEGATIVE".
        If the label is not known, it will be set to "Unknown" by default.
        
        Parameters:
        N/A
        
        Returns:
        
        The string label describing this candidate.
        """
        return self.label
    
    # ****************************************************************************************************
         
    def isPulsar(self):
        """
        Checks the label on this candidates, and determines if it
        represents a pulsar or not.
        
        Parameters:
        N/A
        
        Returns:
        
        True if this candidate represents a genuine pulsar, else False.
        """
        if(self.label=="POSITIVE"):
            return True
        else:
            return False
    
    # ****************************************************************************************************
        
    def setSpecialScore(self,special):
        """
        Sets the value of the score which makes this candidate unusual,
        i.e. score 1 may be the maximum observed or the minimum observed.
        
        Parameters:
        special    -    the score which makes this candidate is unique.
        
        Returns:
        
        N/A
        """
        try:
            self.specialScore=int(special)
        except Exception as e: # catch *all* exceptions @UnusedVariable
            self.specialScore=-1
    
    # ****************************************************************************************************
    
    def getSpecialScore(self):
        """
        Gets the value of the score which makes this candidate unusual.
        
        Parameters:
        
        N/A
        
        Returns:
        
        The integer value of the special score for this candidate.
        """
        return int(self.specialScore)
    
    # ****************************************************************************************************
    
    def setScores(self, data):
        """
        Sets the value of the scores for this candidate, stores them
        as an array of floating point values.
        
        Parameters:
        
        Data    -    the 22 candidate scores.
        
        Returns:
        
        N/A
        """
        
        self.scores=[float(i) for i in data]
    
    # ****************************************************************************************************
        
    def setSpecial(self, s):
        """
        Sets the value of the special description. This should
        be either MAX or MIN. This would indicate along with the
        specialScore why this candidate is unusual, e.g.
        
        If specialScore = 5 and special= MAX then this candidate
        would be unusual since it has the maximum value for score
        5. Since we also have access to the candidate's true label,
        we could go further and say that it has the MAX score 5 value
        for the positive or the negative class (here positive means
        legitimate pulsar, negative RFI etc).
        
        Parameters:
        
        s    -    the string special description.
        
        Returns:
        
        N/A
        """
        self.special = str(s)
    
    # ****************************************************************************************************
    
    def getSpecial(self):
        """
        Gets the value of the special description. This should
        be either MAX or MIN. This would indicate along with
        specialScore why this candidate is unusual, e.g.
        
        If specialScore = 5 and special= MAX then this candidate
        would be unusual since it has the maximum value for score
        5. Since we also have access to the candidate's true label,
        we could go further and say that it has the MAX score 5 value
        for the positive or the negative class (here positive means
        legitimate pulsar, negative RFI etc).
        
        Parameters:
        
        N/A
        
        Returns:
        
        Gets the string value of the special description.
        """
        return str(self.special)
        
    # ****************************************************************************************************
    
    def __str__(self):
        """
        Overridden method that provides a neater string representation
        of this class. This is useful when writing these objects to a file
        or the terminal.
        
        """
            
        return self.candidateName + "," + self.candidatePath
    