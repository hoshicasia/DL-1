import numpy as np
from .base import Criterion
from .activations import LogSoftmax
from scipy.special import softmax, log_softmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        loss = ((input - target) ** 2).mean()
        return loss

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        return 2 / (input.shape[0] * input.shape[1]) * (input - target)


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        soft = self.log_softmax(input)
        return -1 * np.mean(soft[np.arange(input.shape[0]), np.ravel(target)])

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        #Special thanks to chatgpt for helping me debug it because it won't work
        tmp = np.full_like(input, 0)
        np.add.at(tmp, (np.arange(input.shape[0]), target), -1 / input.shape[0])
        return self.log_softmax.compute_grad_input(input, tmp)
