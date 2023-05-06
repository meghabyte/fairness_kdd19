# Mathematical notions vs. human perception of fairness

This repository contains the code for the KDD 2019 paper "Mathematical notions vs. human perception of fairness: A descriptive approach to fairness for machine learning". 

Code for running the server locally and the full adaptive test selection procedure is in the `Server` subdirectory. 
Analysis of our data for all reported results is in the `Analysis` subdirectory. 

If you find this respository useful, please cite:

```
@InProceedings{srivastava2019fairness,
  title = 	 {Mathematical notions vs. human perception of fairness: A descriptive approach to fairness for machine learning},
  author = 	 {Srivastava, Megha and Heidari, Hoda and Krause, Andreas},
  booktitle	=   {Knowledge Discovery and Data Mining (KDD)},
  year = 	 {2019},
}
```


***Installation Requirements***

In order to run the code found in this repository, the following are needed:

1. Python 3
2. Cython (pip install Cython)
2. cherrypy (pip install cherrypy)
3. pip install numpy scipy matplotlib ipython jupyter pandas sympy nose

***Server***

To run the server locally, you must be in the Server folder. Make sure to run make. Then, run python fairnessAppServer.py --local and go to localhost::8080. 

To run the code without the interface, just call python fairnessAppServer.py. 
A list of command line flags can be found below:

--timesteps 20 : number of tests, default 20
--rgseed 30 : seed to control test generation, effects which is the 1st test shown, defeault 30 
--nprgseed 30: seed to control the noise when modeling a user's response, only applicable in test mode, default 30
--testmode : simulate user response rather than running interactively
--hypothesis 0 : control which hypothesis to simulate a noisy user response for. 0 is EA, 1 is FD, 2 is FN, 3 is DP, 100 is random answer. 
