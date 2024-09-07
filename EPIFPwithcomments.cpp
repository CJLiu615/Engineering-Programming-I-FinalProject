#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <cassert>
#include <iomanip> 

using namespace std;


// ******************** class Training Data ********************//
// Simulate the TrainingData class behavior
class TrainingData
{
public:
    bool isEndOfFile() { return CurrentIndex >= MyTrainingData.size(); }
    void GetTopology(vector<unsigned> &topology) { topology = {2, 4, 1}; }
    unsigned GetNextInputs(vector<double> &InputValues);
    unsigned GetTargetOutputs(vector<double> &TargetOutputValues);

private:
    vector<pair<vector<double>, vector<double>>> MyTrainingData;
    unsigned CurrentIndex = 0;

public:
    TrainingData()
    {
        // Generate random training sets for XOR -- two inputs and one output
        for (int i = 2000; i >= 0; --i)
        {
            int n1 = (int)(2.0 * rand() / double(RAND_MAX));
            int n2 = (int)(2.0 * rand() / double(RAND_MAX));
            int t = n1 ^ n2; 
            MyTrainingData.push_back({{static_cast<double>(n1), static_cast<double>(n2)}, {static_cast<double>(t)}});
        }
    }
};

unsigned TrainingData::GetNextInputs(vector<double> &InputValues)
{
    if (isEndOfFile())
        return 0;

    InputValues = MyTrainingData[CurrentIndex].first;
    CurrentIndex++;
    return InputValues.size();
}

unsigned TrainingData::GetTargetOutputs(vector<double> &TargetOutputValues)
{
    if (isEndOfFile())
        return 0;

    TargetOutputValues = MyTrainingData[CurrentIndex - 1].second;
    return TargetOutputValues.size();
}

struct Connection // make it easier to address. We can also seperate them to two containers
{
    double weight;
    double deltaWeight;
};

class Neuron; // need to leave this for the use of the type for layers, it needs to have that for reference.
              // but we can't defined all the neurons up here, the neurons and layers are mutually dependence

typedef vector<Neuron> Layer;

class Neuron
{
public:
    Neuron(unsigned NumberofOutputs, unsigned MyIndex); // in the neuron constructor, we need a for loop to construct these connections
                                                        // the neuron constructor doesn't know anything about the next layer unless we tell it!
                                                        // so, the minimum info we need to tell it about the neext layer is, The Number of Neuron of next layer
    void SetOutputValue(double value) { MyOutputValue = value; } // sets the output value
    double GetOutputValue(void) const { return MyOutputValue; } // return the output value
    void FeedForward(const Layer &previousLayer); // allow neurons to feedForward under feedForward function
    void ComputeOutputGradients(double TargetValues);
    void ComputeHiddenGradients(const Layer &nextLayer); // it's a handle to the next layer, so can be const
    void UpdateInputWeights(Layer &previousLayer); // updates its weight, this one will be modified

private:
    // overall net training rate
    static double eta; //[0.0..1.0] // just simple tunable parameters, only class Neuron needs to have them and all the neuron can use the same value, so it is static rather than dynamic
    static double alpha; //[0.0..n] multiplier of last weight change (momentum)
    static double ActivationFunction(double x);
    static double ActivationFunctionDerivative(double x); // need this in back propagation learning
    static double RandomWeight(void) { return rand() / double(RAND_MAX); } // returning a random number between 0 ~ 1, RAND_MAX means the maximum of the random number
    double sumDOW(const Layer &nextLayer) const; // need to read the next layer

    double MyOutputValue;
    vector<Connection> MyOutputWeights; // ** in this tutorial, each layer is fully connected to others, for spareslly connected, need more study
                                        // We will also need to store one more piece of info for each weight, the changing weights!
    unsigned m_MyIndex;
    double MyGradient;
};

// give some initial values here
double Neuron::eta = 0.15; // overall net learning rate [0.0..1.0]
double Neuron::alpha = 0.5; // momentum, multiplier of last deltaWeight [0.0..n]

double Neuron::ActivationFunction(double x)
{
    // in the tutorial, we use hypobolic tangent function which gives an output from a range of -1 ~ 1
    // tanh - output range [-1.0..1.0]
    return tanh(x);
}

double Neuron::ActivationFunctionDerivative(double x)
{
    //tanh derivative
    return 1.0 - x * x; // quick way, the actual derivative is (d/dx)tanhx = 1 - (tanh^2)x
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;
    // Sum our contributions of the errors at the nodes we feed
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
    {   
        // for loop that goes through all the neuron that is in the next layer
        sum += MyOutputWeights[n].weight * nextLayer[n].MyGradient; // take the sum of the connection weight that goes from us to it
                                                                    // m_outputWeights index by the other neuron's index number
    }

    return sum;
}

Neuron::Neuron(unsigned NumberofOutputs, unsigned MyIndex) // write the constructor
{
    for (unsigned c = 0; c < NumberofOutputs; ++c) // c for connection
    {
        MyOutputWeights.push_back(Connection());
        MyOutputWeights.back().weight = RandomWeight(); // also set that weight to something random
    }

    m_MyIndex = MyIndex; // move the incoming augment into a member
}

void Neuron::FeedForward(const Layer &previousLayer) // output = f( "sigma" i * w )
{
    double sum = 0.0; // need a variable to hold the sum

    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer
    for (unsigned n = 0; n < previousLayer.size(); ++n) // includes bias neuron
    {
        sum += previousLayer[n].GetOutputValue() * previousLayer[n].MyOutputWeights[m_MyIndex].weight;
    }

    // activation (transfer) function
    MyOutputValue = Neuron::ActivationFunction(sum); // shapes its output
}

void Neuron::ComputeOutputGradients(double TargetValues)
{
    // reduce the training net error
    double delta = TargetValues - MyOutputValue; // to check the target value it suppose to have and the actual value it does have
    MyGradient = delta * Neuron::ActivationFunctionDerivative(MyOutputValue); // and multiply the difference by the derivative of its output value
}

void Neuron::ComputeHiddenGradients(const Layer &nextLayer)
{
    // the error calculation is a little bit different than output gradients
    double dow = sumDOW(nextLayer); // *dow : derivative of weights
    MyGradient = dow * Neuron::ActivationFunctionDerivative(MyOutputValue); // multiply the derivative of transfer function output value
}

void Neuron::UpdateInputWeights(Layer &previousLayer) // this is the weights are stored in the previous layers
{
    // The weights to be updated are in the Connection container in the neurons in the preceding layer
    for (unsigned n = 0; n < previousLayer.size(); ++n) // for loop goes through all the previous layer, including the bias
    {
        Neuron &neuron = previousLayer[n]; // the "OTHER" neuron that we are updating
        double OldDeltaWeight = neuron.MyOutputWeights[m_MyIndex].deltaWeight; // need to remember that other neuron's connection weight from it to us

        double NewDeltaWeight = // can modified in different terms of what we want
            // Individual input, magnified by the gradient and train rate:
            eta // the learning rate, it's the term when we talks about the overall training rate 
            * neuron.GetOutputValue() 
            * MyGradient 
            // Also add momentum = a fraction of the previous delta weight
            +alpha // the momentum rate, and thats a multiplier of the old changing rate for the last training sample
            * OldDeltaWeight;

            // eta: 0.0 - slow learner; 0.2 - medium learner; 1.0 - reckless learner
            // alpha: 0.0 - no momentum; 0.5 - moderate momentum

        neuron.MyOutputWeights[m_MyIndex].deltaWeight = NewDeltaWeight; // set it to the new delta weights
        neuron.MyOutputWeights[m_MyIndex].weight += NewDeltaWeight; // changing it by adding the new delta weights
    }
}

// ******************** class Net ********************//

class Net //makes a bunch of neurons and arrange them into 2-D per layer, and all the neurons will be arranged in layers
{
public:
    Net(const vector<unsigned> &topology); //feedForward only reads the values from inputVals and transfer into input neuron
    void FeedForward(const vector<double> &InputValues);
    void BackPropagation(const vector<double> &TargetValues);
    void GetResults(vector<double> &ResultValues) const; //aware of const crackness! //getResults does only take results back and put into a 
                                                         //container, without changing the values
    double GetNetRecentError(void) const { return MyNetRecentError; }
    double GetAverageErrorForTarget(double target) const;

private:
    vector<Layer> MyLayers; //m_layers[layerNum][neuronNum]
    double MyError;
    double MyNetRecentError;
    double MyRecentAverageSmoothingFactor;
    vector<pair<double, double>> MyRecentErrors; // Stores recent errors for each target value
};

Net::Net(const vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    for (unsigned LayerNumber = 0; LayerNumber < numLayers; ++LayerNumber)
    {
        MyLayers.push_back(Layer());
        unsigned NumberofOutputs = LayerNumber == topology.size() - 1 ? 0 : topology[LayerNumber + 1]; // define the number of output
        // "numOutputs = layerNum == topology.size - 1" means the output layer, and the number of output is 0.
        // "all the other cases, the number of output is whatever is in that element of topology for the next layer over."

        //we have made a new Layer, now fill it each neurons, and add a bias to the layer(By using an inner loop)
        for (unsigned neuronNum = 0; neuronNum <= topology[LayerNumber]; ++neuronNum)
        {
            MyLayers.back().push_back(Neuron(NumberofOutputs, neuronNum)); // .back is the std member function that gives you lastest element in the container
                                                                           // push_back is to pin the neuron
                                                                           // each neuron knows its own index for the purpose of accessing the weight array
        }

        // Force the bias node's output value to 1.0. It's the last neuron created above
        MyLayers.back().back().SetOutputValue(1.0);
    }
}

void Net::FeedForward(const vector<double> &InputValues)
{
    assert(InputValues.size() == MyLayers[0].size() - 1); // assert statement: to assert what you believe to be true at this point, and if it is not true, then there will be run-time error. 
                                                          // .size gives the number of neuron; -1 is to subtract bias neuron 

    // Assign (latch) the input values into the input neurons
    for (unsigned i = 0; i < InputValues.size(); ++i)
    {
        // give class net a legal way to set the output value
        MyLayers[0][i].SetOutputValue(InputValues[i]); // can't use outputVal here because it is a private member for class neuron
    }

    //Forward propagate: looping through each layers and the neurons in each layer, and tell each neurons to feedForward
    for (unsigned LayerNumber = 1; LayerNumber < MyLayers.size(); ++LayerNumber) // input layer already been set, so start from the first hidden layer
    {
        Layer &previousLayer = MyLayers[LayerNumber - 1];
        for (unsigned n = 0; n < MyLayers[LayerNumber].size() - 1; ++n)
        {
            MyLayers[LayerNumber][n].FeedForward(previousLayer); // it has its own feedForward
        }
    }
}

void Net::BackPropagation(const vector<double> &TargetValues) // where the Net learn
{
    // Calculate overall net error (RMS of output neuron errors) *RMS = Root Mean Square Error

    Layer &outputLayer = MyLayers.back();
    MyError = 0.0; // accumulate the overall net error, now just set 0 because it's new training path

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
    {
        double delta = TargetValues[n] - outputLayer[n].GetOutputValue();
        MyError += delta * delta;
    }
    MyError /= outputLayer.size() - 1; // get average error squared
    MyError = sqrt(MyError); // RMS

    // Implement a recent average measurement:
    // gives how well our net is being trained
    MyNetRecentError =
        (MyNetRecentError * MyRecentAverageSmoothingFactor + MyError) /
        (MyRecentAverageSmoothingFactor + 1.0);

    MyRecentErrors.emplace_back(TargetValues[0], MyError); 

    if (MyRecentErrors.size() > 100)
    {
        MyRecentErrors.erase(MyRecentErrors.begin());
    }

    // Calculate output layer gradients

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
    {
        outputLayer[n].ComputeOutputGradients(TargetValues[n]); // class Neuron will know how to do the math, not class Net
    }

    // Calculate gradients on hidden layers

    for (unsigned LayerNumber = MyLayers.size() - 2; LayerNumber > 0; --LayerNumber) // start with the right most layer until reach the first hidden layer
    {
        Layer &hiddenLayer = MyLayers[LayerNumber]; // for documentation purpose
        Layer &nextLayer = MyLayers[LayerNumber + 1]; // for documentation purpose

        for (unsigned n = 0; n < hiddenLayer.size(); ++n)
        {
            hiddenLayer[n].ComputeHiddenGradients(nextLayer);
        }
    }

    // For all layers from outputs to first hidden layer, update connection weights

    for (unsigned LayerNumber = MyLayers.size() - 1; LayerNumber > 0; --LayerNumber) // go through all the layers, start from the right most one, and don't need to count input layer
    {
        Layer &layer = MyLayers[LayerNumber];
        Layer &previousLayer = MyLayers[LayerNumber - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n) // for each neuron we will index each neuron, and update the inputweights
        {
            layer[n].UpdateInputWeights(previousLayer); // we just need to the the previous layer
        }
    }
}

void Net::GetResults(vector<double> &ResultValues) const
{
    ResultValues.clear(); // clears out the container that is passed to it

    for (unsigned n = 0; n < MyLayers.back().size() - 1; ++n) // loop through all the neurons and output layers and moves their output values onto resultsVal
    {
        ResultValues.push_back(MyLayers.back()[n].GetOutputValue());
    }
}

double Net::GetAverageErrorForTarget(double target) const
{
    double sumError = 0.0;
    int countError = 0;

    for (auto &pair : MyRecentErrors)
    {
        if (pair.first == target)
        {
            sumError += pair.second;
            ++countError;
        }
    }

    return countError > 0 ? sumError / countError : 0.0;
}

void showVectorValues(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i)
    {
        cout << v[i] << " ";
    }

    cout << endl;
}

int main()
{
    TrainingData trainData;

    vector<unsigned> topology;
    trainData.GetTopology(topology);
    Net myNet(topology); //construct the NN

    // Train the neural network
    vector<double> InputValues, TargetValues, ResultValues;
    int trainingPass = 0;

    // Vectors to store training data
    vector<double> averageErrorTarget0;
    vector<double> averageErrorTarget1;
    vector<double> overallAverageError;
    vector<double> recentAccuracy;

    while (!trainData.isEndOfFile())
    {
        ++trainingPass;
        cout << endl
             << "Pass " << trainingPass;

        // Get new input data and feed it forward
        if (trainData.GetNextInputs(InputValues) != topology[0])
        {
            break;
        }
        showVectorValues(": Input:", InputValues);
        myNet.FeedForward(InputValues);

        // Collect the net's actual results
        myNet.GetResults(ResultValues);
        showVectorValues("Outputs:", ResultValues);

        // Train the net what the outputs should have been
        trainData.GetTargetOutputs(TargetValues);
        showVectorValues("Target:", TargetValues);
        assert(TargetValues.size() == topology.back());

        myNet.BackPropagation(TargetValues);

        // Calculate and store average error for Target 0
        averageErrorTarget0.push_back(myNet.GetAverageErrorForTarget(0.0));

        // Calculate and store average error for Target 1
        averageErrorTarget1.push_back(myNet.GetAverageErrorForTarget(1.0));

        // Calculate and store overall average error
        overallAverageError.push_back((myNet.GetAverageErrorForTarget(0.0) + myNet.GetAverageErrorForTarget(1.0)) / 2);

        // Calculate and store recent accuracy
        recentAccuracy.push_back(100 - (((myNet.GetAverageErrorForTarget(0.0) + myNet.GetAverageErrorForTarget(1.0)) / 2)*100));

        // Report how well the training is working
        cout << "Net recent error: " << myNet.GetNetRecentError() << endl;
        cout << "Average error for Target 0: " << myNet.GetAverageErrorForTarget(0.0) << endl;
        cout << "Average error for Target 1: " << myNet.GetAverageErrorForTarget(1.0) << endl;
        cout << "Overall average error: " << (myNet.GetAverageErrorForTarget(0.0) + myNet.GetAverageErrorForTarget(1.0)) / 2 << endl;
        cout << "Recent accuracy: ";
        cout << fixed << setprecision(8) << 100 - ((myNet.GetAverageErrorForTarget(0.0) + myNet.GetAverageErrorForTarget(1.0)) / 2);
        cout << defaultfloat << endl; 
    }

    // Generate Gnuplot scripts for each graph
    ofstream scriptFile;

    // Gnuplot script for average error for Target 0
    scriptFile.open("average_error_target0.gp");
    scriptFile << "set terminal png" << endl;
    scriptFile << "set output 'average_error_target0.png'" << endl;
    scriptFile << "plot '-' with lines title 'Average Error for Target 0'" << endl;
    for (size_t i = 0; i < averageErrorTarget0.size(); ++i)
    {
        scriptFile << i << " " << averageErrorTarget0[i] << endl;
    }
    scriptFile << "e" << endl;
    scriptFile.close();

    // Gnuplot script for average error for Target 1
    scriptFile.open("average_error_target1.gp");
    scriptFile << "set terminal png" << endl;
    scriptFile << "set output 'average_error_target1.png'" << endl;
    scriptFile << "plot '-' with lines title 'Average Error for Target 1'" << endl;
    for (size_t i = 0; i < averageErrorTarget1.size(); ++i)
    {
        scriptFile << i << " " << averageErrorTarget1[i] << endl;
    }
    scriptFile << "e" << endl;
    scriptFile.close();

    // Gnuplot script for overall average error
    scriptFile.open("overall_average_error.gp");
    scriptFile << "set terminal png" << endl;
    scriptFile << "set output 'overall_average_error.png'" << endl;
    scriptFile << "plot '-' with lines title 'Overall Average Error'" << endl;
    for (size_t i = 0; i < overallAverageError.size(); ++i)
    {
        scriptFile << i << " " << overallAverageError[i] << endl;
    }
    scriptFile << "e" << endl;
    scriptFile.close();

    // Gnuplot script for recent accuracy
    scriptFile.open("recent_accuracy.gp");
    scriptFile << "set terminal png" << endl;
    scriptFile << "set output 'recent_accuracy.png'" << endl;
    scriptFile << "plot '-' with lines title 'Recent Accuracy'" << endl;
    for (size_t i = 0; i < recentAccuracy.size(); ++i)
    {
        scriptFile << i << " " << recentAccuracy[i] << endl;
    }
    scriptFile << "e" << endl;
    scriptFile.close();

    // Execute Gnuplot scripts
    system("gnuplot average_error_target0.gp");
    system("gnuplot average_error_target1.gp");
    system("gnuplot overall_average_error.gp");
    system("gnuplot recent_accuracy.gp");

    // Allow user input and test the neural network in inference mode
    char choice;
    do
    {
        // Prompt the user for input
        double input1, input2;
        cout << "\nEnter two inputs separated by space: ";
        cin >> input1 >> input2;

        // Feed the user input through the neural network
        InputValues = {input1, input2};
        myNet.FeedForward(InputValues);
        myNet.GetResults(ResultValues);

        // Display the output predicted by the neural network
        cout << "Output: " << ResultValues[0] << endl;

        // Ask if the user wants to continue
        cout << "Do you want to continue? (y/n): ";
        cin >> choice;
    } while (choice == 'y' || choice == 'Y');

    cout << endl
         << "Done" << endl;

    return 0;
}