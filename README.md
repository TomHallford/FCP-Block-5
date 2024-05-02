##READ ME##

FCP_assignment.py read me file

Enter the folder containing FCP_assignment.py using git bash

IMPORTRANT Tips: Before Running the program, the user should import these libraries:
--numpy, argparse, matplotlib.pyplot, matplotlib.cm, random, math--

The Flags for the python file are as follows:
=====task 1=====
-ising_model
-test_ising
-external
-alpha
=====task 2=====
-defuant
-test_defuant
-threshold
-beta
-n_count
-time_step
=====task 3=====
-network
-test_network
=====task 5=====
-use_network

The explination for each task function is as follows:

=====task 1=====

###-ising_model flag be inputted on the terminal will play an animation for showing the ising_model. The value of external and alpha are default which be setted in 0 and 1.

###-ising_model  follows one or two flags (-external) (-alpha). these two flags will affect the agreement function and create a new animation.

###-test_ising flag is used to test whether the agreement function is correct.    

###-external and -alpha will follow two numbers, They will change value of external and alpha in the agreement function.


=====task 2=====

-defuant flag activates the defuant model
-test_defuant flag runs the test for defuant model
-beta flag that when followed by a float between 0 and 1 changes the beta value
-threshold flag that when followed by a float between 0 and 1 changes the threshold value
-n_count flag when followed by an integer changes the amount of nieghbours
-time_step flag when followed by an integer changes the length of time the model runs for

=====task 3=====
Using the network flag followed by an integer will create a random network with that many nodes, the network will then be displayed and the terminal will output the mean degree, average pathl ength and the mean clustering co-efficient.
Using the test network flag will run tests included in the code to verify the mean degree, average path length and mean clustering co-efficient are calculated correctly.
=====task 5=====
Use this flag followed by an integer after the main ising flag to run the ising model using a network. This will create a random network with a number of nodes equal to the integer put in, an animation will then play showing the ising model acting on the network. 

=====task 4=====
The 'm_r_network' generates a ring network with N nodes. Each node is assigned a random value of 1 or -1 and connects to its neighbors within a set range.  The 'm_s_w_network' creates a ring network  then re-wires it to create a small world network. The rewiring process involves randomly selecting conections and reassigning them to create a new connection based on a specified probability. The main function utilizes argument parsing to take the users inputs for creating either a ring or a small world network. Depending on the user's choice, it makes the corresponding network, calculates and displays the mean path length, the mean degree, and the mean clustering coefficient of the network, and then plots the network.
