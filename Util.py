from numpy import empty
import numpy as np
from numpy import transpose, roll, reshape, meshgrid, arange
import itertools



def cartprod(arrays):
    N = len(arrays)
    return transpose(meshgrid(*arrays, indexing='ij'),
                     roll(arange(N + 1), -1)).reshape(-1, N)


def gen_all_states(num_actions, fixed):
    action_list = []
    for x in range(0, len(num_actions)):
        if x != fixed:
            action_list.append([p for p in range(0, num_actions[x])])
        else:
            action_list.append([0])  # a dummy since we will change it

    return cartprod(action_list)



def id_generator():
    counter = 1
    while True:
        yield counter
        counter += 1


def inner_product(lst):
    outp = list(itertools.product(*lst))
    out = []
    for i in outp:
        temp = []
        for j in i:
            if isinstance(j, list):
                for k in j:
                    temp.append(k)
            else:
                temp.append(j)
        out.append(temp)
    return out


def findMaxs(listCand, currPlayer):
    maxval = -1
    #print(inner_product(listCand))
    ret_list = []
    list_combinations = [[]]
    for x in listCand:
        list_combinations =  [i + [y] for y in x for i in list_combinations ]
    for myl in list_combinations:
        for subl in myl:
            maxval = max(maxval, subl["data"][currPlayer])

        retlist = []
        for subl in myl:
            if subl["data"][currPlayer] == maxval:
                retlist.append(subl)

        ret_list+=retlist
   
    
    return ret_list

def find_max_val(listcand, currPlayer):
    maxval = -1
    for subl in listcand:
       
        maxval = max(maxval, subl["data"][currPlayer])

    retlist = []
    for subl in listcand:
        if subl["data"][currPlayer] == maxval:
            retlist.append(subl)
    return retlist

def remove_dup_id(list_dict):
    hashtable = dict()
    for elem in list_dict:
        if elem["id"] in hashtable:
            continue
        else:
            
            hashtable[elem["id"]] = (elem["data"], elem["path"])
    retlist = []
    ix_list = []
    for k,v in hashtable.items():
        retlist.append(v[0])
        ix_list.append(v[1])
    return ix_list, retlist
