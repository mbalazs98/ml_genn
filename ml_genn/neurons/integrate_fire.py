from . import Neuron
from ..utils import InitValue, Value

genn_model = {
    "vars": [("V", "scalar")],
    "params": [("Vthr", "scalar")],
    "sim_code":
        """
        $(V) += $(Isyn);
        """,
    "threshold_condition_code":
        """
        $(V) >= $(Vthr)
        """,
    "reset_code":
        """
        $(V) = 0.0;
        """,
    "is_auto_refractory_required": False}

class IntegrateFire(Neuron):
    def __init__(self, threshold=1.0, v=0.0):
        super(IntegrateFire, self).__init__()

        self.threshold = Value(threshold)
        self.v = Value(v)

    def get_model(self, population):
        return genn_model
                    
    @property
    def params(self):
        return {"Vthr": self.threshold}
    
    @property
    def vars(self):
        return {"V": self.v}