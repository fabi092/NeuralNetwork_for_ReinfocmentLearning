Fabian Albertskirchinger / GS17M015
Programme for training a neural network with the Iris training dataset.
The programme can be controlled via the console, with the most important settings being displayed again after each command.

The following commands are available:

end
train					Starts the training of the neural network with current settings
check float float float float		Lets the neural network assign an input to a flower type
accuracy  	integer			Sets a new target precision
generations	integer			Sets the maximum training generation number
learnrate 	float			Sets the step size of the weight changes
momentum 	float			Sets momentum, which takes into account the previous change in the weighting changes.
filepath 	string			Set path of the training set
