# PulsarFeatureExtractor
Extracts features from PHCX and PFD pulsar candidate files. Not to be
confused with the PulsarFeatureLab, which is used for feature extraction
and experimentation.

This is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Its distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

See <http://www.gnu.org/licenses/> for more license details.

Author:       Rob Lyon
 
Contact:    rob@scienceguyrob.com or robert.lyon@postgrad.manchester.ac.uk

Web:        http://www.scienceguyrob.com

1.	Overview

Script which extracts feature data from pulsar candidates. These features are used as the inputs to machine
learning classification algorithms. The code can extract two different types of features:

	i. 22 Scores described in Sam Bates' thesis, "Surveys Of The Galactic Plane For Pulsars" 2011.
		The scores generated are as follows:
		
<table>
	<tr>
		<th>Number</th>
		<th>Description of feature </th>
		<th>Type</th>
	</tr>
	<tr>
		<th>1</th>
		<th>Chi squared value from fitting since curve to pulse profile. </th>
		<th>Sinusoid Fitting</th>
	</tr>
	<tr>
		<th>2</th>
		<th>Chi squared value from fitting sine-squared curve to pulse profile.</th>
		<th>Sinusoid Fitting</th>
	</tr>
	<tr>
		<th>3</th>
		<th>Number of peaks the program identifies in the pulse profile - 1.</th>
		<th>Pulse Profile Tests</th>
	</tr>
	<tr>
		<th>4</th>
		<th>Sum over residuals. </th>
		<th>Pulse Profile Tests</th>
	</tr>
	<tr>
		<th>5</th>
		<th>Distance between expectation values of Gaussian and fixed Gaussian fits to profile histogram.</th>
		<th>Gaussian Fitting</th>
	</tr>
	<tr>
		<th>6</th>
		<th>Ratio of the maximum values of Gaussian and fixed Gaussian fits to profile histogram.</th>
		<th>Gaussian Fitting</th>
	</tr>
	<tr>
		<th>7</th>
		<th>Distance between expectation values of derivative histogram and profile histogram.</th>
		<th>Gaussian Fitting</th>
	</tr>
	<tr>
		<th>8</th>
		<th>Full-width-half-maximum (FWHM) of Gaussian fit to pulse profile.</th>
		<th>Gaussian Fitting</th>
	</tr>
	<tr>
		<th>9</th>
		<th>Chi squared value from Gaussian fit to pulse profile. </th>
		<th>Gaussian Fitting</th>
	</tr>
	<tr>
		<th>10</th>
		<th>Smallest FWHM of double-Gaussian fit to pulse profile. </th>
		<th>Gaussian Fitting</th>
	</tr>
	<tr>
		<th>11</th>
		<th>Chi squared value from double Gaussian fit to pulse profile.</th>
		<th>Gaussian Fitting</th>
	</tr>
	<tr>
		<th>12</th>
		<th>Best period.</th>
		<th>Candidate Parameters</th>
	</tr>
	<tr>
		<th>13</th>
		<th>Best SNR value.</th>
		<th>Candidate Parameters</th>
	</tr>
	<tr>
		<th>14</th>
		<th>Best DM value.</th>
		<th>Candidate Parameters</th>
	</tr>
	<tr>
		<th>15</th>
		<th>Best pulse width (original reported as Duty cycle (pulse width / period)).</th>
		<th>Candidate Parameters</th>
	</tr>
	<tr>
		<th>16</th>
		<th>SNR / SQRT( (P-W)/W ).</th>
		<th>Dispersion Measure (DM) Curve Fitting</th>
	</tr>
	<tr>
		<th>17</th>
		<th>Difference between fitting factor, Prop, and 1.</th>
		<th>Dispersion Measure (DM) Curve Fitting</th>
	</tr>
	<tr>
		<th>18</th>
		<th>Difference between best DM value and optimised DM value from fit, mod(DMfit - DMbest).</th>
		<th>Dispersion Measure (DM) Curve Fitting</th>
	</tr>
	<tr>
		<th>19</th>
		<th>Chi squared value from DM curve fit.</th>
		<th>Dispersion Measure (DM) Curve Fitting</th>
	</tr>
	<tr>
		<th>20</th>
		<th>RMS of peak positions in all sub-bands.</th>
		<th>Sub-band Scores</th>
	</tr>
	<tr>
		<th>21</th>
		<th>Average correlation coefficient for each pair of sub-bands.</th>
		<th>Sub-band Scores</th>
	</tr>
	<tr>
		<th>22</th>
		<th>Sum of correlation coefficients between sub-bands and profile.</th>
		<th>Sub-band Scores</th>
	</tr>
</table>
  	
  	ii. 8 Scores described in my own paper, "Fifty Years of Pulsar Candidate Selection: From simple filters to a new
  		principled real-time classification approach"
		
<table>
	<tr>
		<th>Number</th>
		<th>Description of feature </th>
	</tr>
	<tr>
		<th>1</th>
		<th>Mean of the integrated profile.</th>
	</tr>
	<tr>
		<th>2</th>
		<th>Standard deviation of the integrated profile.</th>
	</tr>
	<tr>
		<th>3</th>
		<th>Excess kurtosis of the integrated profile.</th>
	</tr>
	<tr>
		<th>4</th>
		<th>Skewness of the integrated profile.</th>
	</tr>
	<tr>
		<th>5</th>
		<th>Mean of the DM-SNR curve.</th>
	</tr>
	<tr>
		<th>6</th>
		<th>Standard deviation of the DM-SNR curve.</th>
	</tr>
	<tr>
		<th>7</th>
		<th>Excess kurtosis of the DM-SNR curve.</th>
	</tr>
	<tr>
		<th>8</th>
		<th>Skewness of the DM-SNR curve.</th>
	</tr>
</table>	
  	
2. Requirements

	The PulsarFeatureExtractor files have the following system requirements:

	Python 2.4 or later.
	[SciPy](http://www.scipy.org/)
	[NumPy](http://www.numpy.org/)
	[matplotlib library] (http://matplotlib.org/)

3. Usage

The main application script ScoreGenerator.py can be executed via:
	
<i>python ScoreGenerator.py</i>
	
The script accepts a number of arguments. It requires two of these to execute, and accepts another eight as optional.
	
Required Arguments
	
<table>
  <tr>
    <th>Flag</th>
    <th>Type</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>−c</td>
    <td>string</td>
    <td>Path to the directory containing PHCX or PFD candidates to extract features from.</td>
  </tr>
  <tr>
    <td>−o</td>
    <td>string</td>
    <td>Full path to the output file to write extracted feature data to.</td>
  </tr>
</table>

Optional Arguments

<table>
  <tr>
    <th>Flag</th>
    <th>Type</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>--pfd</td>
    <td>boolean</td>
    <td>Flag which indicates that ONLY .pfd files are to be processed.</td>
  </tr>
  <tr>
    <td>--phx</td>
    <td>boolean</td>
    <td>Flag which indicates that ONLY HTRU .phcx files are to be processed.</td>
  </tr>
  <tr>
    <td>--superb</td>
    <td>boolean</td>
    <td>Flag which indicates that ONLY SUPERB .phcx files are to be processed.</td>
  </tr>
  <tr>
    <td>--arff</td>
    <td>boolean</td>
    <td>Flag which indicates that feature data should be written to an ARFF file.</td>
  </tr>
  <tr>
    <td>--profile</td>
    <td>boolean</td>
    <td>Flag which indicates that profile, rather than score data should be generated as features.</td>
  </tr>
  <tr>
    <td>--dmprof</td>
    <td>boolean</td>
    <td>Flag which indicates that DM and profile data should be extracted as features.</td>
  </tr>
  <tr>
    <td>-v</td>
    <td>boolean</td>
    <td>Verbose debugging flag.</td>
  </tr>
</table>

4. Citing this work

	Please use the following citation if you make use of tool:
	
	@misc{PulsarFeatureExtractor,
	author = {Lyon, R. J.},
	title  = {{Pulsar Feature Extractor}},
	affiliation = {University of Manchester},
	month  = {November},
	year   = {2014},
	howpublished = {World Wide Web Accessed (19/11/2014), \newline \url{https://github.com/scienceguyrob/PulsarFeatureExtractor}},
	notes  = {Accessed 19/11/2014}
	}
	
5. Acknowledgements

	This work was supported by grant EP/I028099/1 for the University of Manchester Centre for
	Doctoral Training in Computer Science, from the UK Engineering and Physical Sciences Research
	Council (EPSRC).