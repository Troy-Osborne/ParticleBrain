# ParticleBrain

Disclaimer: 
I'm not always the most legible or concise so I'll probably start making a habit of generating AI readmes. 
I just don't want people to assume my code is AI generated. I will still be including my original Readme at the bottom.

This part of the readme was partially generated with AI, the code was not:

# Feed Forward Artificial Neural Networks with Particle Swarm Optimization

This project implements feed-forward artificial neural networks (ANNs) optimized with particle swarm optimizers (PSO). 
The current class, called "BrainLobe," is designed as a modular subset of a full network and can be used alongside other ANNs or BrainLobe classes in a hierarchical or networked architecture.

Overview:
    Modular Design: BrainLobe is a subset of the full network, allowing you to train and optimize individual sections independently.
    Particle Swarm Optimization: The network uses PSO to optimize weights and biases, exploring the search space for optimal configurations.
    Activation Functions: Supports various activation functions, including non-standard ones like sine waves, without requiring derivatives for backpropagation.

Features
    Optimized Training: Train networks piece by piece to reduce computational expense and improve efficiency.
    Flexible Configuration: BrainLobe classes can be pre-optimized or swapped in and out as needed.
    Particle Swarm Dynamics: Particles orbit at different distances from the optima, providing a balance between fine-tuning and broad exploration of the search space.

How Particle Swarm Optimization Works
    Initialization: Create multiple copies of the network, each with random initial weights and biases.
    Exploration: Each particle shifts its variables in a random direction and changes direction based on its personal best and the global best across all particles.
    Acceleration: When a new personal or global best is found, particles accelerate in that direction, continuing to move down the gradient if it offers improvement.
    Orbiting: Particles orbit around their personal and global best positions, fine-tuning close to the local optima while exploring further out for new solutions.
    https://en.wikipedia.org/wiki/Particle_swarm_optimization
Advantages
    No Backpropagation Required: The PSO approach supports all activation functions without the need for derivatives.
    Modular Training: Train individual parts of a larger network, allowing you to tackle complex problems more efficiently.
    Simplicity: Uses only standard operations, mostly imperative style programming, with minimal python specific libraries except for plotting in the demo(s). 
      Avoids use of vectorisation so that the code is easily transpilable into C family languages or other fast compilable languages. Once I finish updating I plan to create an in-program C++ transpiler

Getting Started
    Clone the Repository: Clone the project repository to your local machine.
    Install Matplotlib if you want to run the current demo code. Everything else is vanilla python
    Import the Brainlobe Class into any projects that needs it!


# My Original Readme (Now you see why I got the AI to structure it for me):
Feed Forward Artificial Neural Networks optimised with Particle Swarm Optimisers. 
This isn't made for evolving giant neural networks all at once, but piece by piece.
The currently defined class is called brainlobe "BrainLobe" as its only a subset of the full network and is for use with other other ANNs or Brainlobe classes.
When run in a heirachy or network each subsection can be pre-optimised or switched in and out in a modular fashion.
Large Neural networks will be computationally expensive to train so you break the task into smaller networks which you train one at a time.

To evolve the Neural Network a few copies are made and each one will start shifting all variables in a random direction,
this direction will change as its values alternate between the personal best of that candidate,
and the best of the all candidates, looking for optimal configurations between them.
Whenever a new personal best is found it will be written to that candidate and when a new global best is found it will be written to the network.

While there are some speed sacrifices compared to traditional backpropegation learning the benefit of the PSO is that different particles will orbit at different distances from the optima,
the closest particles more precisely to fine tune the best known answer and converge on the local extrema, while the furthest out particles orbit around their personal best and global best at great speeds and distances
jumping around rapidly sweeping the surrounding space for different inputs with potential better solutions,
If a new best candidate is found all other candidates will converge on the new best while, continuing to travel down their path if it offers continual improvement
Another plus is that it supports all activaiton functions without requiring a derivative of the function for backpropegation.
It fact it can learn without backprop. (though you could still implement it, it would just be harder with all the varied types of activation functions)
