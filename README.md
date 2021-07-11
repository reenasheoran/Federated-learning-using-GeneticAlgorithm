# Federated learning using Genetic Algorithm
This project in the impelementation of 
## Motivation
Most of the data in the world are contained not publicly on servers, but on client-side devices and sources (i.e. mobile phones, edge devices). Typically, users don’t end up directly sharing their data anywhere outside their devices. And if they do, there’s no guarantee of privacy. In this way, ML models are typically trained on data that’s been collected and stored in a centralized server.
Federated Learning (FL) is a new paradigm for building machine learning (ML) models that keeps user data private. Through this project, I'll apply the concepts of FL in Python to build a demonstration of how a system like this might work.
## How it works
The interaction between client and server is implemented by using programming sockets with the socket Python library. The client-server architecture will handle multiple clients simultaneously. Using PyGAD, I’ll also create a feed-forward neural network (FFNN), also using a genetic algorithm (GA).
 Using FL, a model can be created directly out of such private data. 
