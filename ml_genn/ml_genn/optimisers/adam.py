from pygenn import VarAccessMode

from .optimiser import Optimiser
from ..utils.model import CustomUpdateModel
from ..utils.snippet import ConstantValueDescriptor

from copy import deepcopy


genn_model = {
    "vars": [("M", "scalar"), ("V", "scalar")],
    "params": [("Beta1", "scalar"), ("Beta2", "scalar"),
               ("Epsilon", "scalar"), ("Alpha", "scalar"),
               ("MomentScale1", "scalar"), ("MomentScale2", "scalar")],
    "var_refs": [("Gradient", "scalar", VarAccessMode.READ_ONLY),
                 ("Variable", "scalar")],
    "update_code":
        """
        // Update biased first moment estimate
        M = (Beta1 * M) + ((1.0 - Beta1) * Gradient);

        // Update biased second moment estimate
        V = (Beta2 * V) + ((1.0 - Beta2) * Gradient * Gradient);

        // Add gradient to variable, scaled by learning rate
        Variable -= (Alpha * M * MomentScale1) / (sqrt(V * MomentScale2) + Epsilon);
        """}


class Adam(Optimiser):
    """Optimizer that implements the Adam algorithm [Kingma2014]_.
    Adam optimization is a stochastic gradient descent method that 
    is based on adaptive estimation of first-order and second-order moments.
    """
    alpha = ConstantValueDescriptor()
    """ Learning rate"""
    
    beta1 = ConstantValueDescriptor()
    """ The exponential decay rate for the 1st moment estimates. """
    
    beta2 = ConstantValueDescriptor()
    """ The exponential decay rate for the 2nd moment estimates. """
    
    epsilon = ConstantValueDescriptor()
    """A small constant for numerical stability. This 
    is the epsilon in Algorithm 1 of the [Kingma2014]_.
    """

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def set_step(self, genn_cu, step):
        assert step >= 0
        moment_scale_1 = 1.0 / (1.0 - (self.beta1 ** (step + 1)))
        moment_scale_2 = 1.0 / (1.0 - (self.beta2 ** (step + 1)))

        genn_cu.set_dynamic_param_value("Alpha", self.alpha)
        genn_cu.set_dynamic_param_value("MomentScale1", moment_scale_1)
        genn_cu.set_dynamic_param_value("MomentScale2", moment_scale_2)

    def get_model(self, gradient_ref, var_ref, 
                  zero_gradient: bool) -> CustomUpdateModel:
        model = CustomUpdateModel(
            deepcopy(genn_model),
            {"Beta1": self.beta1, "Beta2": self.beta2,
             "Epsilon": self.epsilon, "Alpha": self.alpha, 
             "MomentScale1": 0.0, "MomentScale2": 0.0},
            {"M": 0.0, "V": 0.0},
            {"Gradient": gradient_ref, "Variable": var_ref})

        # Make parameters dynamic
        model.set_param_dynamic("Alpha")
        model.set_param_dynamic("MomentScale1")
        model.set_param_dynamic("MomentScale2")

        # If a optimiser than automatically zeros
        # gradients should be provided
        if zero_gradient:
            # Change variable access model of gradient to read-write
            model.set_var_ref_access_mode("Gradient",
                                          VarAccessMode.READ_WRITE)

            # Add update code to zero the gradient
            model.append_update_code(
                """
                // Zero gradient
                Gradient = 0.0;
                """)

        # Return model
        return model
