import numpy as np


def density(array):
    """
    Weighted average of a segment.
    """
    return np.average(array[0, :], weights=array[1, :])


def phi(array, x, y):
    """
    phi(segment, x, y) is defined to be the largest index i ∈ [x, y]
    that minimizes density(segment[i:y + 1]).
    """
    if x == y:
        return x

    densities = [density(array[:, x:i + 1]) for i in range(x, y)]
    return np.where(densities == np.min(densities))[-1][0] + x


def best(array, x, y, z):
    """
    Given three indices x <= y <= z, it returns the largest
    index i ∈ [x, y] that maximizes density(segment[i, z]).
    """
    if x == y:
        return x

    i = x
    while i < y and density(array[:, i:phi(array, i, y - 1) + 1]) <= density(
            array[:, i:z + 1]):
        i = phi(array, i, y - 1) + 1
    return i


class Segment:

    def __init__(self, values, weights):
        self.values = values
        self.weights = weights

    def as_array(self):
        return np.array([self.values, self.weights])

    def __getitem__(self, index):
        if not self.values:
            return

        if isinstance(index, slice):
            return self.as_array()[:, index]

        raise ValueError("Invalid index")

    @property
    def length(self):
        return len(self.values)

    @property
    def density(self):
        return density(self[:])

    def maximum_density_subsegment(self, w_min: int = 0, w_max: int = np.inf):
        """
        Finds the subsegment with maximum density.

        Args:
            w_min (int, optional): minimal width of subsegment. Defaults to 0.
            w_max (int, optional): maximal width of subsegment. Defaults to np.inf.
        """
        j0 = next(idx for idx, v in enumerate(np.cumsum(self.weights))
                  if v >= w_min)
        i = np.zeros(self.length + 1, dtype=int)
        res = []

        for j in range(j0, self.length):
            w_inv = enumerate(np.cumsum(np.flip(self.weights[:j + 1])))
            candidates = [
                j - idx for idx, v in w_inv if v >= w_min and v <= w_max
            ]
            x, y = candidates[-1], candidates[0]
            i[j] = best(self[:], max(i[j - 1], x), y, j)
            res = np.append(res, [i[j], j, density(self[i[j]:j + 1])])

        res = res.reshape((-1, 3))

        return res[np.argmax(res[:, 2]), :], res
