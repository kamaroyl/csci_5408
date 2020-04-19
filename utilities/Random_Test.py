from numpy.random import default_rng
from numpy import ndarray
from numpy import zeros


def main():
    k = 10
    A = zeros(100)
    print(A)
    A[:k] = 1
    print(A)
    rg = default_rng()
    rg.shuffle(A)
    print(A)


if __name__ == "__main__":
       # execute only if run as a script
    main()
