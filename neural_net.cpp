#include <cmath>
#include <random>
#include <vector>

struct DataPoint {
    std::vector<double> inputs;
    std::vector<double> expected_outputs;

    DataPoint(std::vector<double> in, int correct_output) {
        inputs = in;
        expected_outputs = std::vector<double>(10);
        expected_outputs[correct_output] = 1;
    }
};

struct Layer {
	int input_size, output_size;
    // weights[i][j] is the weight from the ith input node to the jth output node
	std::vector<std::vector<double> > weights;
	std::vector<double> biases;
    std::vector<double> outputs;
	
	// constructor	
	Layer(int in, int out) {
		input_size = in;
		output_size = out;
		
		weights = initialize_random_weights(in, out); 
		biases = std::vector<double>(out);
        outputs = std::vector<double>(out);
	}

    // Function to initialize random weights for a ReLU network
    std::vector<std::vector<double>> initialize_random_weights(int input_size, int output_size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, 0.1); 

        // Initialize weights matrix
        std::vector<std::vector<double>> weights(input_size, std::vector<double>(output_size));
        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                weights[i][j] = dist(gen); // Sample from normal distribution
            }
        }

        return weights;
    }

	std::vector<double> calc_outputs(std::vector<double> inputs, bool output_layer) {
		for (int i = 0; i < output_size; i++) {
			double output = biases[i];
			for (int j = 0; j < input_size; j++) {
				output += (inputs[j] * weights[j][i]);
			}
			outputs[i] = relu(output);
		}
		
        if (output_layer) {
            softmax(outputs);
        }

		return outputs;
	}

    double relu(double activation) {
        return (activation > 0) ? activation : 0;
    }

    double relu_derivative(double activation) {
        return (activation > 0) ? 1 : 0;
    }

    std::vector<double> softmax(std::vector<double>& activations) {
        double sum = 0;

        for (double activation : activations) {
            sum += exp(activation);
        }

        for (double& activation : activations) {
            activation = exp(activation) / sum;
        }

        return activations;
    }

    std::vector<std::vector<double> > softmax_derivative(std::vector<double> inputs) {
        std::vector<std::vector<double> > jacobian;
        std::vector<double> probabilities = softmax(inputs);

        for (int i = 0; i < inputs.size(); i++) {
            std::vector<double> row;
            for (int j = 0; j < inputs.size(); j++) {
                double probability = probabilities[j];
                row.push_back(probability * exp(inputs[j]) * ((i == j) ? 1.0 - probability : -probability));
            }
            jacobian.push_back(row);
        }
        
        return jacobian;
    }

    double node_cost(double activation, double target) {
        double error = target - activation;
        return error * error;
    }
    
    double node_cost_derivative(double activation, double target) {
        return 2 * (target - activation);
    }

    std::vector<double> node_gradient(std::vector<double> expected_output) {
        std::vector<double> node_gradient = std::vector<double>(expected_output.size());
        for (int i = 0; i < expected_output.size(); i++) {
            double cost_derivative = node_cost_derivative(outputs[i], expected_output[i]);
            double activation_derivative = relu_derivative(outputs[i]);
            node_gradient[i] = cost_derivative * activation_derivative;
        }

        return node_gradient;
    }
};

struct Net {
	std::vector<Layer> layers;
	
	// constructor
	Net(std::vector<int> layer_sizes) {
		for (int i = 0; i < layer_sizes.size() - 1; i++) {
			layers[i] = Layer(layer_sizes[i], layer_sizes[i + 1]);
		}
	}	
	
	std::vector<double> forward_prop(std::vector<double> inputs) {
		for (int i = 0; i < inputs.size(); i++) {
			inputs = layers[i].calc_outputs(inputs, (i == inputs.size() - 1));
		}
		return inputs;
	}

	int result(std::vector<double> inputs) {
		std::vector<double> outputs = forward_prop(inputs);
		int res = -1;

		for (int i = 0; i < 10; i++) {
			if (outputs[i] > outputs[res]) {
				res = i;
			}	
		}
		return res;
	}

    double net_cost(DataPoint point) {
        std::vector<double> outputs = forward_prop(point.inputs);
        Layer output_layer = layers[layers.size() - 1];
        double cost = 0;
        for (int cur_node = 0; cur_node < outputs.size(); cur_node++) {
            cost += output_layer.node_cost(outputs[cur_node], point.expected_outputs[cur_node]);
        }

        return cost;
    }

    double avg_cost(std::vector<DataPoint> data) {
        double total_cost = 0;

        for (DataPoint point : data) {
            total_cost += net_cost(point);
        }

        return total_cost / data.size();
    }
};

int main() {
    return 0;
}
