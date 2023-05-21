import numpy as np

from evametrics import Cardinalities

class _TrackedQuantity:
    pass


class TrackedScalar(_TrackedQuantity):

    def __init__(self) -> None:
        self.cumulative = 0
        self.N = 0
        self.current = None

    def update(self, value: float, size: int) -> None:
        self.cumulative += value
        self.N += size
        self.current = self.cumulative / self.N

    def reset(self) -> None:
        self.__init__()

    @property
    def value(self) -> float:
        return self.current


class TrackedCardinalities(_TrackedQuantity):

    basal_identifiers = {'TP', 'TN', 'FP', 'FN'}
    derived_identifiers = {'TPR', 'TNR', 'ACC', 'F1'}
    joint_identifiers = basal_identifiers | derived_identifiers

    def __init__(self) -> None:
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.TPR = None
        self.TNR = None
        self.ACC = None
        self.F1 = None


    def update(self, values: Cardinalities) -> None:
        for cardinality in self.basal_identifiers:
            # wow, such Python :D
            update = getattr(values, cardinality).item()
            current = getattr(self, cardinality)
            setattr(self, cardinality, current + update)

        self.TPR = self.TP / (self.TP + self.TN)
        self.TNR = self.TN / (self.TN + self.FP)
        numerator = self.TP + self.TN
        denominator = self.TP + self.TN + self.FP + self.FN
        self.ACC = numerator / denominator
        self.F1 = 2 * self.TP / (2*self.TP + self.FP + self.FN)
        

    def reset(self) -> None:
        self.__init__()