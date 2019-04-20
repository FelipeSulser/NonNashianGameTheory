import itertools
import json
import random
from numpy import transpose, roll, reshape, meshgrid, arange, empty
from functools import reduce
import copy
import numpy as np
import math
import argparse
import os

max_num_actions_1 = 2
max_num_actions_2 = 2
min_outcome_val = 0
max_outcome_val = max_num_actions_1* max_num_actions_2 



'''

 Public functions used to generate Extensive Form Games

'''

def sample_extensive_form_random_dup(max_num_actions, max_depth):
    num_players = 2
    total_num_actions = max_num_actions ** max_depth
    action_set = [list(range(0,total_num_actions)) for act in range(0,num_players)]
    dictval = []
    while dictval == []:
        num_leaves, dictval =  __gen_random_dict(max_num_actions,max_depth, avoid_ties=False, curr_depth=0)
    action_set = [list(range(0,num_leaves)) for act in range(0, num_players)]

    #traverse dict and put outcomes
    __modify_leaves(dictval, action_set, avoid_ties=False)

    ret_dict = dict()
    ret_dict['x'] = 0
    ret_dict['y'] = dictval

    return ret_dict

def sample_extensive_form_random_nodup(max_num_actions, max_depth):
    num_players = 2
    total_num_actions = max_num_actions ** max_depth
    action_set = [list(range(0,total_num_actions)) for act in range(0,num_players)]
    dictval = []
    while dictval == []:
        num_leaves, dictval =  __gen_random_dict(max_num_actions,max_depth, avoid_ties=True, curr_depth=0)
    action_set = [list(range(0,num_leaves)) for act in range(0, num_players)]

    #traverse dict and put outcomes 
    __modify_leaves(dictval, action_set, avoid_ties=True)
    ret_dict = dict()
    ret_dict['x'] = 0
    ret_dict['y'] = dictval
    return ret_dict

def sample_extensive_form_alternating_dup(num_actions, depth):
    '''
    Generates a single game where outcomes may contain duplicates. 
    Furthermore the game is sequential meaning that p1 plays then p2 alternating between them
    '''
   
    #total_num_actions = reduce(lambda x, y: x*y, num_actions)
    num_players = len(num_actions)
    total_num_actions = num_actions[0] ** depth
    action_set = [list(range(0,total_num_actions)) for act in range(0,num_players)]
    ret_dict = dict()
    ret_dict['x'] = 0
    ret_dict['y'] = __gen_dict_content(num_actions,depth,action_set, avoid_ties=False, curr_depth=0, curr_player=0)
    return ret_dict

def sample_extensive_form_alternating_nodup(num_actions, depth):
    '''
    Generates a single game where outcomes may not contain duplicates. 
    Furthermore the game is sequential meaning that p1 plays then p2 alternating between them
    '''
   
    #total_num_actions = reduce(lambda x, y: x*y, num_actions)
    num_players = len(num_actions)
    total_num_actions = num_actions[0] ** depth
    action_set = [list(range(0,total_num_actions)) for act in range(0,num_players)]
    ret_dict = dict()
    ret_dict['x'] = 0
    ret_dict['y'] = __gen_dict_content(num_actions,depth,action_set, avoid_ties=True, curr_depth=0, curr_player=0)
    return ret_dict


def convert_two_player_ef_to_nf(game):
    generator = id_generator()
    game_content = copy.deepcopy(game["y"])
    __annotate_nodes(game_content, generator)
    __get_indices(game_content, id_generator())
   
    dict_val = dict()
    num_actions_dict = dict()
   
    __find_pure_strategies(game_content, dict_val, num_actions_dict)
   
    list_strategies = __cartprod_pure_strategies(dict_val)
   
    game_matrix, id_map = __get_nf_list(game_content, list_strategies, num_actions_dict)

    res_dict = dict()
    res_dict["x"] = game["x"]
    res_dict["z"] = 2
    res_dict["y"] = game_matrix
    res_dict["idmap"] = id_map
    res_dict["annotated"] = game_content
    return res_dict




'''

 Private functions used to generate Extensive Form Games
'''



def __cartprod(arrays):
    N = len(arrays)
    return transpose(meshgrid(*arrays, indexing='ij'),
                     roll(arange(N + 1), -1)).reshape(-1, N)


# function to generate all __permutations of a list
def __perm(xs) :
    if xs == [] :
        yield []
    for x in xs :
        ys = [y for y in xs if not y==x]
        for p in __perm(ys) :
            yield ([x] + p)
 
#function to generate all possible assignments of <item> balls into <ks> bins
def __combinations_generator(items, ks):
	if len(ks) == 1:
		for c in itertools.combinations(items, ks[0]):
			yield (c)

	else:
		for c_first in itertools.combinations(items, ks[0]):
			items_remaining= set(items) - set(c_first)
			for c_other in __combinations_generator(items_remaining, ks[1:]):
				yield (c_first) + c_other



def __generate_random_tree(nodelist=[], idx=0, parent=None, fixed_children=True, depth=0, max_children=2, max_depth=2):
    """
    Build a list of nodes in a random tree up to a maximum depth.
        :param:    nodelist     list, the nodes in the tree; each node is a list with elements [idx, parent, depth]
        :param:    idx          int, the index of a node
        :param:    parent       int, the index of the node's parent
        :param:    depth        int, the distance of a node from the root
        :param:    max_children int, the maximum number of children a node can have
        :param:    max_depth    int, the maximum distance from the tree to the root

    """
    if depth < max_depth and depth >= 0:
        # add a random number of children
       
        n = max_children
        if not fixed_children:
            n = random.randint(0, max_children)

        nodelist.extend([[idx+i, parent, depth, "INT"] for i in range(0,n)])  

        # for each new child, add new children
        for i in range(0, n):
        	gen_leaves = __generate_random_tree(nodelist, len(nodelist), idx + i, depth + 1, max_children, max_depth)
        	
        	if gen_leaves and len(gen_leaves) == 0:
        		
        		nodelist[idx+i-1][3] = "LEAF"

    elif depth == max_depth:
        # add a random number of leaves
        n = max_children
        if not fixed_children:
            n = random.randint(0, max_children)
        leaves = [[idx+i, parent, depth, "LEAF"] for i in range(0,n)]
        nodelist.extend(leaves)  
        return leaves

    else:  # this should never happen
        raise ValueError('Function stopped -> depth > max_depth or depth < 0.')


def __gen_random_dict(max_num_actions, max_depth, avoid_ties=True, curr_depth=0, leaf_threshold=0.2):
    num_player = 2

    prob_leaf = 0
    if curr_depth == max_depth:
        prob_leaf = 0
    else:
        prob_leaf = random.uniform(0, 1)

    #start with player 0
    if curr_depth == 0:
        new_dict = dict()
        new_dict["p"] = 0
        new_dict["c"] = []
        sumval = 0
        #num actions is random too
        num_actions = random.choice([i for i in range(1, max_num_actions)])
        for act in range(0, num_actions):
            act_val, subgame = __gen_random_dict(max_num_actions,max_depth, avoid_ties=avoid_ties, curr_depth=curr_depth+1)
            sumval += act_val
            new_dict["c"].append(subgame)
        return (sumval,new_dict)

    if prob_leaf < leaf_threshold:

        #its a leaf
        return (1,[])
    else:
        #its a node
        new_dict = dict()
        new_dict["p"] = random.choice([0,1])
        new_dict["c"] = []
        sumval = 0
        #num actions is random too
        num_actions = random.choice([i for i in range(1, max_num_actions)])
        for act in range(0, num_actions):
            act_val, subgame = __gen_random_dict(max_num_actions,max_depth, avoid_ties=avoid_ties, curr_depth=curr_depth+1)
            sumval += act_val
            new_dict["c"].append(subgame)


        return (sumval,new_dict)
   
def __modify_leaves(d, action_set, avoid_ties=False):
    if isinstance(d, list) and len(d) == 0:
        myL = []
        if avoid_ties:
            chosenval = random.choice(action_set[0])
            action_set[0].remove(chosenval)
            myL.append(chosenval)
            chosenval = random.choice(action_set[1])
            action_set[1].remove(chosenval)
            myL.append(chosenval)
        else:
            chosenval = random.choice(action_set[0])
            myL.append(chosenval)
            chosenval = random.choice(action_set[1])
            myL.append(chosenval)
        d = myL
        return d
    elif isinstance(d, list) and len(d) > 0:
        for i in range(0, len(d)):
            d[i] = __modify_leaves(d[i], action_set, avoid_ties=avoid_ties)
        return d
    elif isinstance(d, dict):
        for k, v in d.items():
            d[k] = __modify_leaves(v, action_set, avoid_ties=avoid_ties)
        return d
    else:
        return d
    


def __gen_dict_content(num_actions, depth, action_set, avoid_ties=True, curr_depth=0, curr_player=0):
    num_player = len(num_actions)

    if curr_depth == depth:
        myL = []
        for i in range(0, num_player):
            #put it in the list
            if avoid_ties:
                chosenval = random.choice(action_set[i])
                action_set[i].remove(chosenval)
                myL.append(chosenval)
            else:
                chosenval = random.choice(action_set[i])
                myL.append(chosenval)
        return myL
        
    else:
        new_dict = dict()
        new_dict["p"] = curr_player
        new_dict["c"] = []
        new_player = (curr_player + 1) % num_player # alternate 0, 1,... ,p
        
        for act in range(0, num_actions[curr_player]):
            new_dict["c"].append(__gen_dict_content(num_actions,depth, action_set, avoid_ties=avoid_ties, curr_depth=curr_depth+1, curr_player=new_player))
        
        return new_dict


    return None


def __annotate_nodes(root, id_generator):
    if isinstance(root, list):
        return

    root["id"] = next(id_generator)
   
    for child in root['c']:
        __annotate_nodes(child, id_generator)
    return

def __find_pure_strategies(root, dictval, num_actions_dict):
    if isinstance(root, dict) and "idleaf" in root:
        return
    player = root['p']
    if player not in dictval:
        dictval[player] = dict()
        
    idval = root["id"]
    if idval not in dictval[player]:
        dictval[player][idval] = []

    to_append = [{"id": idval, "action": act} for act in range(0, len(root["c"]))]
    if player not in num_actions_dict:
        num_actions_dict[player] = 1
    num_actions_dict[player] *= len(to_append)
    dictval[player][idval].append(to_append)
    for child in root["c"]:
        __find_pure_strategies(child, dictval, num_actions_dict)
    return

def __cartprod_pure_strategies(dict_strategies):
    list_of_lists = []
   
    for k,v in dict_strategies.items():
        # v contains a list of dict, expand it
        for  k2, v2 in v.items():
            list_of_lists.append(v2)
    
    return  __cartprod(list_of_lists)


def __find_action(strategy, currid):
    for elem in strategy:
        if elem["id"] == currid:
            return elem["action"]
    return -1

def __get_nf_list(root, ix_strategies, num_actions):
    nf_list = []
    id_map = []
    starting_player = root['p']
    for strategy in ix_strategies:
        itertree = root
        while not (isinstance(itertree, dict) and "idleaf" in itertree):
            currid = itertree["id"]
            action = __find_action(strategy, currid)
            itertree = itertree["c"][action]
        nf_list.append(itertree['data'])
        id_map.append(itertree['idleaf'])

    nf_list = [tuple(x) for x in nf_list]
    num_actions_tup = [v for k,v in num_actions.items()]
    num_actions_id = [v for k,v in num_actions.items()]
    if len(num_actions_tup) < 2:
        num_actions_tup.append(1)
        num_actions_id.append(1)
    num_actions_tup.append(2)
    # num_actions_id.append(1)
    num_actions_tup = tuple(num_actions_tup)
    
    nf_list = np.asarray(nf_list).reshape(num_actions_tup)  
    id_map = np.asarray(id_map).reshape(num_actions_id)

    return (nf_list.tolist(), id_map.tolist())

def __get_indices(root, genval):
    if isinstance(root, list):
        return
    ix = 0
    for ix in range(0, len(root['c'])):
        if isinstance(root['c'][ix], list):
            root['c'][ix] = {"idleaf": next(genval), "data": root['c'][ix]}
        else:
            __get_indices(root['c'][ix], genval)




def __find_path(root, idval, path):
    if isinstance(root, dict) and "idleaf" in root:
       
        if root['idleaf'] == idval:
            return True
        else:
            return False

    children = root['c']
    for ix_child in range(0, len(children)):
        path.append(ix_child)
        if __find_path(root['c'][ix_child], idval, path):
            return True
        del path[-1]
    return False

def translate_nf_ix_ef(nf_ix, id_map, annotated_tree):
    idmap = np.asarray(id_map)
   
    res_list = []
    for ix in nf_ix:
       
        idval = idmap[tuple(ix)]
        
        #find this id in the tree and return the path to get its ix
        path = []
        tree_node = annotated_tree
        __find_path(tree_node, idval, path)
        res_list.append(copy.deepcopy(path))
    return [tuple(x) for x in res_list]

def __nf_equilibria_to_ef(nf_ix, id_map, annotated_tree,name,  return_dict):
    list_res = translate_nf_ix_ef(nf_ix, id_map, annotated_tree)
    return_dict[name] = list(set(list_res))
    return return_dict

def __get_ix_currpos(player_list, pos, ixval):
    posmatch = 0
    for currIx in player_list:
        if currIx[pos] == ixval:
            return posmatch
        posmatch += 1
    return posmatch

def __annotate_nf_data(game):
    nf_game = convert_two_player_ef_to_nf(game)
    pte(nf_game)
    nash(nf_game)
    minimax(nf_game)
    individual(nf_game)
    rationalizability(nf_game)
    game = __nf_equilibria_to_ef(nf_game['N'], nf_game['idmap'], nf_game['annotated'], "NNF", game)
    game = __nf_equilibria_to_ef(nf_game['P'], nf_game['idmap'], nf_game['annotated'], "PNF", game)
    game = __nf_equilibria_to_ef(nf_game['M'], nf_game['idmap'], nf_game['annotated'], "MNF", game)
    game = __nf_equilibria_to_ef(nf_game['I'], nf_game['idmap'], nf_game['annotated'], "INF", game)
    game = __nf_equilibria_to_ef(nf_game['R'], nf_game['idmap'], nf_game['annotated'], "RNF", game)
    return game
 

def __get_outcome(tuple_ix, game_dict, pos=0):
    if pos >= len(tuple_ix):
        return tuple(game_dict)

    return __get_outcome(tuple_ix, game_dict['c'][tuple_ix[pos]], pos+1)

