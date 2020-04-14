# chaotic_neural_behavior_and_plasticity

Simulation of a recurrent neural network with excitatory and inhibitory neurons

Dependencies:
- tensorflow
- matplotlib.pyplot
- seaborn
- numpy


snnClass.py provides the class for a spiking neural layer.

recurrent.py runs a simulation of a spiking neural network with to layers:
one layer of excitatory neurons (E), one layer of inhibitory neurons (I).

The simulation tracks the spiking activity in the network after an single random activation of a fraction of the neurons. After this initial external activation, the activity in the two layers are entirely driven by their recurrent connections.
At each time step, E send excitatory signals to I and to itself, and I send inhibitory signals to E and to itself.

The simulation can be run with or without spike-timing dependent plasticity.
Thus the simulation can show the stabilizing effect of having spike-timing dependent plasticity on the number of active neurons.

See comparisons in the .jpg files in this folder.

Note, that minor changes in the hyperparameters might lead to major changes in the trajectory of network activity, and in many cases, the activity will not be able to sustain itself autonomously.

Note also, that even with the default settings provided, the activity might in some cases die out within the first 10 time steps due to randomness in the initial activation.


