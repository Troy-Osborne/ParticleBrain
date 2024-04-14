# ParticleBrain
Feed Forward Artificial Neural Networks optimised with Particle Swarm Optimisers. 
This isn't made for evolving giant neural networks all at once, but piece by piece.
The currently defined class is called brainlobe "BrainLobe" as its only a subset of the full network and is for use with other other ANNs or Brainlobe classes.
When run in a heirachy or network each subsection can be pre-optimised or switched in and out in a modular fashion.
Large Neural networks will be computationally expensive to train so you break the task into smaller networks which you train one at a time.

To evolve the Neural Network a few copies are made and each one will start shifting all variables in a random direction,
this direction will change as its values alternate between the personal best of that candidate,
and the best of the all candidates, looking for optimal configurations between them.
Whenever a new personal best is found it will be written to that candidate and when a new global best is found it will be written to the network.

While there are some speed sacrifices the benefit of the method is that different particles will orbit at different distances from the optima,
the closest particles more precisely to fine tune the best known answer and converge on the local extrema,
while the furthest out particles orbit around their personal best and global best at great speeds and distances
jumping around rapidly sweeping the surrounding space for different inputs with potential better solutions,
If one is found all particles from the prior best will accelerate in the direction of the new best, continuing to travel down the graident if it offers continual improvement
Another plus is that it supports all activaiton functions without requiring a derivative of the function for backpropegation.
It fact it can learn without backprop. (though you could still implement it, it would just be harder with all the varied types of activation functions)
