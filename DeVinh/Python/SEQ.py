import math
import time
import numpy as np


def check_equivalent_arrays(n: int, array_A: list, array_B: list):
    print(n, "", array_A, "", array_B)
    # check conditions
    if n not in range(1, int(5e4 + 1)):
        return -1
    if (min(array_A) < 1 or min(array_B) < 1) or (max(array_A) > 1e9 or max(array_B) > 1e9):
        return -1

    dpA = np.zeros((n, n))

    for i in range(len(array_A)):
        for j in range(i, len(array_A)):
            dpA[i][j] = array_A[i] / array_A[j] == 1

    dpB = np.zeros((n, n))
    for i in range(len(array_B)):
        for j in range(i, len(array_B)):
            dpB[i][j] = array_B[i] / array_B[j] == 1

    return int((dpA == dpB).all())


if __name__ == "__main__":
    with open('./INP/SEQ.INP') as f:
        data = f.readlines()

    n = data[0]
    array_A = [int(x) for x in data[1].split(" ")]
    array_B = [int(x) for x in data[2].split(" ")]

    result = check_equivalent_arrays(int(n), array_A, array_B)
    print(result)

    with open("./OUT/SEQ.OUT", 'w') as wf:
        wf.write(str(result))
