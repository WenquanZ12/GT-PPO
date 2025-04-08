class TFN:
    """
    Triangular Fuzzy Number (TFN): (a, b, c), where a <= b <= c.
    """
    def __init__(self, a: float, b: float, c: float):
        assert a <= b <= c, "Invalid TFN: Must satisfy a <= b <= c"
        self.a = a
        self.b = b
        self.c = c

    def __repr__(self):
        return f"({self.a}, {self.b}, {self.c})"

    # TFN 加法
    def __add__(self, other):
        return TFN(self.a + other.a, self.b + other.b, self.c + other.c)

    # TFN 比较（基于 Sakawa）
    def compare(self, other) -> int:
        """
        Compare two TFNs using Sakawa's method.
        Return:
            1 if self > other
            -1 if self < other
            0 if equal
        """
        f1_self = (self.a + 2 * self.b + self.c) / 4
        f1_other = (other.a + 2 * other.b + other.c) / 4
        if f1_self > f1_other:
            return 1
        elif f1_self < f1_other:
            return -1
        else:
            if self.b > other.b:
                return 1
            elif self.b < other.b:
                return -1
            else:
                range_self = self.c - self.a
                range_other = other.c - other.a
                if range_self > range_other:
                    return 1
                elif range_self < range_other:
                    return -1
                else:
                    return 0

    # TFN 最大操作
    def max(self, other):
        return self if self.compare(other) >= 0 else other

    # 支持 < 运算符
    def __lt__(self, other):
        return self.compare(other) == -1

    # 支持 == 运算符
    def __eq__(self, other):
        return self.compare(other) == 0
if __name__ == "__main__":
    A = TFN(3, 4, 6)
    B = TFN(2, 5, 7)

    print("A:", A)
    print("B:", B)

    # 加法
    C = A + B
    print("A + B =", C)

    # 比较
    if A > B:
        print("A > B")
    elif A < B:
        print("A < B")
    else:
        print("A == B")

    # 最大
    print("Max(A, B):", A.max(B))

    # 排序测试
    tfns = [TFN(2, 3, 5), TFN(1, 4, 7), TFN(3, 4, 6)]
    print("Before sorting:", tfns)
    tfns.sort()
    print("After sorting:", tfns)
