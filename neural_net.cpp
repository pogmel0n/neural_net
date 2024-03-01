#include <vector>

struct Layer {
	int input_size, output_size;
	std::vector<std::vector<double> > weights;
	std::vector<double> biases;
	
	// constructor	
	Layer(int in, int out) {
		input_size = in;
		output_size = out;
		
		weights = std::vector<std::vector<double> > w(in, std::vector<double>(out));
		biases = std::vector<double>(out);
	}

	std::vector<double> calc_outputs(std::vector<double> inputs) {
		std::vector<double> outputs(output_size);

		for (int i = 0; i < output_size; i++) {
			double output = biases[i];
			for (int j = 0; j < input_size; j++) {
				output += (inputs[j] * weights[j, i]);
			}
			outputs[i] = output;
		}
		
		return outputs;
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
		for (Layer layer : layers) {
			inputs = layer.calc_outputs(inputs);
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
};
