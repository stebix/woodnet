import numpy as np
from numbers import Number

from woodnet.evaluation.metrics import (compute_TPR, compute_TNR, compute_ACC,
                                        compute_MCC, compute_F1, Cardinalities)


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

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.value:.5f}, N={self.N})'
    
    def __repr__(self) -> str:
        return self.__str__()



class TrackedCardinalities(_TrackedQuantity):

    basal_identifiers = {'TP', 'TN', 'FP', 'FN'}
    derived_identifiers = {'TPR', 'TNR', 'ACC', 'F1', 'MCC'}
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
        self.MCC = None


    def update(self, values: Cardinalities) -> None:
        for cardinality in self.basal_identifiers:
            # wow, such Python :D
            update = getattr(values, cardinality).item()
            current = getattr(self, cardinality)
            setattr(self, cardinality, current + update)

        self.TPR = compute_TPR(TP=self.TP, FN=self.FN)
        self.TNR = compute_TNR(TN=self.TN, FP=self.FP)
        self.ACC = compute_ACC(TP=self.TP, TN=self.TN, FP=self.FP, FN=self.FN)
        self.F1 = compute_F1(TP=self.TP, FP=self.FP, FN=self.FN)
        self.MCC = compute_MCC(TP=self.TP, TN=self.TN, FP=self.FP, FN=self.FN)
        

    def reset(self) -> None:
        self.__init__()


    def __str__(self) -> str:
        s = f'{self.__class__.__name__}('
        values = [
            '='.join((identifier, self._format_number(getattr(self, identifier))))
            for identifier in self.joint_identifiers
        ]
        values = ', '.join(values)
        return ''.join((s, values, ')'))

    
    def __repr__(self) -> str:
        return self.__str__()
    

    def state_dict(self) -> dict[str, Number | None]:
        """
        Current state dict of the tracked cardinalities, i.e. mapping
        from name to current values.
        """
        state_dict = {
            name : getattr(self, name)
            for name in self.joint_identifiers
        }
        return state_dict
            
    
    @staticmethod
    def _format_number(num: int | float) -> str:
        """Conditional formatting of number string representation."""
        if isinstance(num, int):
            return str(num)
        elif isinstance(num, float):
            return f'{num:.5f}'
        return str(num)





