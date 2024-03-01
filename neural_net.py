import numpy as np

class Layer:
	def __init__(in: int, out: int):
		self.input_size = in
		self.outpt_size = out
		
		weight = np.zeros[in, out]
