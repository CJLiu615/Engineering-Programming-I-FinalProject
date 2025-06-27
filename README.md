**Neural Network from Scratch in C++**

This project implements a neural network from scratch in C++, without using any external machine learning libraries. It focuses on understanding core principles of neural networks, building all components manually, and visualizing training performance using Gnuplot.

The neural network is trained to perform the Exclusive OR (XOR) function, and supports adjustable topology and training epochs to observe performance trends.

**🧠 Project Highlights**

Built entirely in C++ from the ground up

Simulates feedforward and backpropagation learning

Dynamically configurable topologies: e.g., {2, 4, 1}, {2, 4, 4, 1}, etc.

Visual performance metrics using Gnuplot

Fully modular class design for neurons, layers, training data, and network

**📦 Features**

Implemented classes:

TrainingData: manages dynamic XOR input generation and network configuration

Neuron: defines weights, activations, gradients, and learning updates

Net: links neurons across layers and manages training logic

Tracks:

Net recent error

Average error for Target 0 and 1

Overall average error

Recent accuracy

Supports user input inference (post-training)

Epoch and architecture-based performance comparison

🏗️ Network Architecture
Example topology:

scss
Copy
Edit
Input Layer (2 neurons + bias)
↓
Hidden Layer(s) (e.g., 4 neurons + bias per layer)
↓
Output Layer (1 neuron)
Topology can be modified with:

cpp
Copy
Edit
std::vector<unsigned> topology = {2, 4, 4, 1};
📈 Visualization
Performance trends are graphed using Gnuplot and include:

Training error vs. epoch

Comparison across different topologies

Effects of increasing hidden layers or epochs

Inference results under various configurations

🔧 Tools Used
C++ (tested with Cygwin, Sublime Text, Xcode)

Gnuplot (via Cygwin integration)

Standard C++ libraries only (no external ML libraries)

📁 Project Structure
bash
Copy
Edit
.
├── src/                    # All class definitions (Net, Neuron, TrainingData)
├── data/                   # Training samples (if extended to external files)
├── plots/                  # Gnuplot scripts and generated images
├── main.cpp                # Entry point and network execution logic
├── Makefile                # For building on Unix systems
└── README.md
▶️ How to Run
Compile:

bash
Copy
Edit
g++ -o xor_nn main.cpp -std=c++11
Run:

bash
Copy
Edit
./xor_nn
To plot graphs (ensure Gnuplot is installed):

bash
Copy
Edit
gnuplot plot_error.gp
📊 Sample Output (for XOR)
Input A	Input B	Target	Output
0	0	0	0.01
1	0	1	0.94
0	1	1	0.92
1	1	0	0.07

🔍 Observations
Increased epochs → lower average error and higher accuracy

Too many hidden layers → longer training and risk of plateau

Random weight initialization → variability in results across runs

🚀 Future Work
Standardize initial inputs and weights for reproducibility

Add runtime tracking for performance evaluation

Extend to multi-class classification problems

Implement alternate activation functions (ReLU, Sigmoid)

👨‍💻 Authors
Chao-Jia Liu – chao-jia.liu@my.utsa.edu

Abdulaziz Alshehri – abdulaziz.alshehri@my.utsa.edu

Department of Electrical and Computer Engineering
The University of Texas at San Antonio
