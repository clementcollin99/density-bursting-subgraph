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


def maximum_density_subsegment(
    s: "Segment",
    w_min: int = 2,
    w_max: int = np.inf,
):
    """
    Finds the subsegment with maximum density.

    Args:
        s (Segment): the segment whose subsegment with maximum density is sought
        w_min (int, optional): minimal width of subsegment. Defaults to 2.
        w_max (int, optional): maximal width of subsegment. Defaults to np.inf.

    Returns:
        tuple: indices delimiting the maximum density segment and its density
    """
    assert w_min >= 2 and w_min <= w_max
    j0 = next(idx for idx, v in enumerate(np.cumsum(s.weights)) if v >= w_min)
    i = np.zeros(s.length + 1, dtype=int)
    res = []

    for j in range(j0, s.length):
        w_inv = enumerate(np.cumsum(np.flip(s.weights[:j + 1])))
        candidates = [j - idx for idx, v in w_inv if v >= w_min and v <= w_max]
        x, y = candidates[-1], candidates[0]
        i[j] = best(s[:], max(i[j - 1], x), y, j)
        res = np.append(res, [i[j], j, density(s[i[j]:j + 1])])

    res = res.reshape((-1, 3))

    return res[np.argmax(res[:, 2]), :]


class Segment:
    """_summary_
    """

    def __init__(self, values, weights):
        self._values = values
        self._weights = weights

    @property
    def values(self):
        return self._values

    @property
    def weights(self):
        return self._weights

    @property
    def length(self):
        return len(self.values)

    @property
    def density(self):
        return density(self[:])

    def as_array(self):
        return np.array([self._values, self._weights])

    def __getitem__(self, index):
        if self.values is None or len(self.values) == 0:
            return

        if isinstance(index, slice):
            return self.as_array()[:, index]

        raise ValueError("Invalid index")

    def __str__(self):
        return str(self.as_array())
