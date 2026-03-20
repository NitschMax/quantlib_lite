class Path:
    def __init__(self, times, values):
        if not len(times) == len(values):
            raise ValueError('Times and values of the Path must have the same length!')
        self.__times = tuple(times)
        self.__values = tuple(values)

    @property
    def times(self):
        return self.__times

    @property
    def values(self):
        return self.__values

    def __len__(self):
        return len(self.times)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Path(self.times[key], self.values[key])
        else:
            return (self.times[key], self.values[key])

    def __repr__(self):
        return f"Path({self.times}, {self.values})"
