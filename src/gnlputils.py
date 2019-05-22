from numpy import dot
from numpy.linalg import norm


def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def extract_keys(lines, key: str):
    return [json[key] for json in lines]


def get_from_rankings(rankings, dictionary):
    return [dictionary[index] for _, index in rankings]


def split_data(data, dev_start: float, test_start: float, is_test: bool):
    train, dev, test = split_all_data(data, dev_start, test_start)
    if is_test:
        return (train, test)
    else:
        return (train, dev)

def split_all_data(data, dev_start: float, test_start: float):
    return (data[:int(dev_start * len(data))],
            data[int(dev_start * len(data)): int(test_start * len(data))],
            data[int(test_start * len(data)):])

