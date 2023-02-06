from typing import Callable

def filter_by(items, criterion):
    assert callable(criterion)
    return [item for item,next_item in zip(items, items[1:]) if criterion(item, next_item)]

# def start_with_a(word):
#     return word[0] == 'a'


if __name__ == '__main__':
    start_with_a = lambda x, y: x[0] == 'a' and y[1] == 'p'
    fruits = ['apple', 'apricot', 'pear', 'plum', 'orange', 'banana', 'watermelon', 'lemon']
    fruits_a = filter_by(fruits, start_with_a)
    # fruits_b = filter_by(fruits, lambda x: 't' in x)
    print(fruits_a)
    # print(fruits_b)





