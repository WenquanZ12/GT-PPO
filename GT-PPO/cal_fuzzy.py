import numpy as np


class TFN:
    def __init__(self, left, peak, right):
        self.left = left
        self.peak = peak
        self.right = right

    def __add__(self, other):
        if isinstance(other, TFN):
            return TFN(self.left + other.left, self.peak + other.peak, self.right + other.right)
        else:
            raise ValueError("Addition is only supported between TFN objects.")

    def __repr__(self):
        return f"TFN({self.left}, {self.peak}, {self.right})"

    def compare(self, other):
        # Comparison based on the center value, then peak, then spread
        if self.center() > other.center():
            return 1
        elif self.center() < other.center():
            return -1
        else:  # If center values are equal
            if self.peak_value() > other.peak_value():
                return 1
            elif self.peak_value() < other.peak_value():
                return -1
            else:  # If peak values are also equal
                if self.spread() > other.spread():
                    return 1
                elif self.spread() < other.spread():
                    return -1
                else:
                    return 0

    def max(self, other):
        # Return the larger of the two TFNs based on comparison
        if self.compare(other) > 0:
            return self
        else:
            return other
# Create two TFNs
tfn1 = TFN(1, 5, 9)
tfn2 = TFN(2, 6, 10)

# Addition
tfn_sum = tfn1 + tfn2
print(f"Sum: {tfn_sum}")

# Comparison
result = tfn1.compare(tfn2)
if result == 1:
    print(f"{tfn1} is greater than {tfn2}")
elif result == -1:
    print(f"{tfn1} is less than {tfn2}")
else:
    print(f"{tfn1} is equal to {tfn2}")

# Maximum
max_tfn = tfn1.max(tfn2)
print(f"Maximum: {max_tfn}")

