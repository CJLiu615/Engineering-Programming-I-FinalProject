**Neural Network from Scratch in C++**

This project implements a neural network from scratch in C++, without using any external machine learning libraries. It focuses on understanding core principles of neural networks, building all components manually, and visualizing training performance using Gnuplot.

The neural network is trained to perform the Exclusive OR (XOR) function, and supports adjustable topology and training epochs to observe performance trends.

**🧠 Project Highlights**

* Built entirely in C++ from the ground up

* Simulates feedforward and backpropagation learning

* Dynamically configurable topologies: e.g., {2, 4, 1}, {2, 4, 4, 1}, etc.
![image](https://github.com/user-attachments/assets/51817947-2db9-43ed-b245-f668c3353ecc)


* Visual performance metrics using Gnuplot

* Fully modular class design for neurons, layers, training data, and network

**📦 Features**

* Implemented classes:

   * TrainingData: manages dynamic XOR input generation and network configuration

   * Neuron: defines weights, activations, gradients, and learning updates

   * Net: links neurons across layers and manages training logic

* Tracks:

   * Net recent error

   * Average error for Target 0 and 1

   * Overall average error

   * Recent accuracy

* Supports user input inference (post-training)

* Epoch and architecture-based performance comparison

**🏗️ Network Architecture**

Example topology:

Input Layer (2 neurons + bias)
↓
Hidden Layer(s) (e.g., 4 neurons + bias per layer)
↓
Output Layer (1 neuron)

Topology can be modified with:

std::vector<unsigned> topology = {2, 4, 4, 1};

**📈 Visualization**

Performance trends are graphed using Gnuplot and include:

* Training error vs. epoch

* Comparison across different topologies

* Effects of increasing hidden layers or epochs

* Inference results under various configurations

**🔧 Tools Used**

* C++ (tested with Cygwin, Sublime Text, Xcode)

* Gnuplot (via Cygwin integration)

* Standard C++ libraries only (no external ML libraries)

**📁 Project Structure**

├── src/                    
All class definitions (Net, Neuron, TrainingData)

├── data/                   
Training samples (if extended to external files)

├── plots/                  
Gnuplot scripts and generated images

├── main.cpp               
Entry point and network execution logic

├── Makefile                
For building on Unix systems

└── README.md


**📊 Sample Output (for XOR)**

![image](https://github.com/user-attachments/assets/12ecefc3-4bd5-4c54-adb2-7e570a1d8546)


**🔍 Observations**

* Increased epochs → lower average error and higher accuracy

* Too many hidden layers → longer training and risk of plateau

* Random weight initialization → variability in results across runs

**🚀 Future Work**

* Standardize initial inputs and weights for reproducibility

* Add runtime tracking for performance evaluation

* Extend to multi-class classification problems

* Implement alternate activation functions (ReLU, Sigmoid)

**👨‍💻 Authors**

Chao-Jia Liu – chao-jia.liu@my.utsa.edu

Department of Electrical and Computer Engineering

The University of Texas at San Antonio
