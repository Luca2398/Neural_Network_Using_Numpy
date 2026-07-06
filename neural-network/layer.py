class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input) -> float:
        pass

    def backward(self, output, learn_rate) -> float:
        pass