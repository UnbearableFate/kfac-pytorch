from functools import reduce

vote_store = {12 : [[1,2,],[1,2],[1,3]], 13 : [[1,2,3],[4,5,6],[7,8,9]]}

widely_accepted_slow = reduce(set.intersection, map(set,vote_store[12]))
print([widely_accepted_slow])