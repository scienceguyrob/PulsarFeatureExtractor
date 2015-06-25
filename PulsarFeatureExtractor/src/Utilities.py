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

File name:    Utilities.py
Created:      February 1st, 2014
Author:       Rob Lyon
 
Contact:    rob@scienceguyrob.com or robert.lyon@postgrad.manchester.ac.uk
Web:        <http://www.scienceguyrob.com> or <http://www.cs.manchester.ac.uk> 
            or <http://www.jb.man.ac.uk>
            
This code runs on python 2.4 or later.

"""

# Python 2.4 imports.
import traceback
import sys, os

# ******************************************************************************************
#
# CLASS DEFINITION
#
# ******************************************************************************************

class Utilities(object):
    """
    Provides utility functions used when computing scores.
    
    """
    
    # ******************************************************************************************
    #
    # Constructor.
    #
    # ******************************************************************************************
    
    def __init__(self,debugFlag):
        self.debug = debugFlag
        
    # ******************************************************************************************
    #
    # Functions.
    #
    # ******************************************************************************************
    
    def appendToFile(self,path,text):
        """
        Appends the provided text to the file at the specified path.
        
        Parameters:
        path    -    the path to the file to append text to.
        text    -    the text to append to the file.
        
        Returns:
        N/A
        """
        
        destinationFile = open(path,'a')
        destinationFile.write(str(text))
        destinationFile.close()
    
    # ******************************************************************************************
    
    def fileExists(self,path):
        """
        Checks a file exists, returns true if it does, else false.
        
        Parameters:
        path    -    the path to the file to look for.
        
        Returns:
        True if the file exists, else false.
        """
        
        try:
            fh = open(path)
            fh.close()
            return True
        except IOError:
            return False
    
    # ******************************************************************************************
    
    def dirExists(self,path):
        """
        Checks a directory exists, returns true if it does, else false.
        
        Parameters:
        path    -    the path to the directory to look for.
        
        Returns:
        True if the file exists, else false.
        """
        
        try:
            if(os.path.isdir(path)):
                return True
            else:
                return False
        except IOError:
            return False
    
    # ******************************************************************************************
            
    def format_exception(self,e):
        """
        Formats error messages.
        
        Parameters:
        e    -    the exception.
        
        Returns:
        
        The formatted exception string.
        """
        exception_list = traceback.format_stack()
        exception_list = exception_list[:-2]
        exception_list.extend(traceback.format_tb(sys.exc_info()[2]))
        exception_list.extend(traceback.format_exception_only(sys.exc_info()[0], sys.exc_info()[1]))
        
        exception_str = "\nTraceback (most recent call last):\n"
        exception_str += "".join(exception_list)
        
        # Removing the last \n
        exception_str = exception_str[:-1]
        
        return exception_str
    
    # ******************************************************************************************
    
    def out(self,message,parameter):
        """
        Writes a debug statement out if the debug flag is set to true.
        
        Parameters:
        message    -    the string message to write out
        parameter  -    an accompanying parameter to write out.
        
        Returns:
        N/A
        """
        
        if(self.debug):
            print message , parameter
            
    # ******************************************************************************************
    
    def outMutiple(self,parameters):
        """
        Writes a debug statement out if the debug flag is set to true.
        
        Parameters:
        parameters  -    the values to write out.
        
        Returns:
        N/A
        """
        
        if(self.debug):
            
            output =""
            for p in parameters:
                output+=str(p)
                
            print output
            
    # ******************************************************************************************
            