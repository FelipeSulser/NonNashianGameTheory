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


def generate_games(num_actions, num_players, duplicates=False, symmetric=False, sample=None):
    if duplicates and symmetric:
        if sample:
            return __sample_normal_form_symmetric_dup(num_actions, num_players, sample)
        else:
            return __gen_normal_form_symmetric_dup(num_actions, num_players)

    if duplicates and not symmetric:
        if sample:
            return __sample_normal_form_dup(num_actions, num_players, sample)
        else:
            return __gen_normal_form_dup(num_actions, num_players)

    if not duplicates and symmetric:
        if sample:
            return __sample_normal_form_symmetric_nodup(num_actions, num_players, sample)
        else:
            return __gen_normal_form_symmetric_nodup(num_actions, num_players)

    #no dup and not sym
    if sample: 
        return __sample_normal_form_nodup(num_actions, num_players, sample)
    else:
        return __gen_normal_form_nodup(num_actions, num_players)


def game_generation(num_actions, num_players, duplicates, symmetric, sample, save_route=None, the_file=None):
    num_games = 0
    for game in generate_games(num_actions, num_players, duplicates=duplicates, symmetric=symmetric, sample=sample):
        if save_route:
            the_file.write(json.dumps(game)+"\n") 
        else:
            print(game)
        num_games += 1

def __all_possible_games_dup(action_arr):
    '''
    All possible games with repetitions 

    - <action_arr> indicates the actions that each player may make
    ''' 
    num_eq_classes = reduce((lambda x, y: x * y), action_arr)
    num_range = list(range(0, num_eq_classes))
    range_values = [num_range for x in range(0, len(action_arr))]

    # keeps all in memory, not efficient. Find a solution with generators
    outcomes = [itertools.product(p, repeat=num_eq_classes) for p in range_values] 
    return itertools.product(*outcomes)


def __all_possible_games_nodup(action_arr):
    '''
    All possible games without repetitions

    - <action_arr> indicates the actions that each player may make
    '''
    num_eq_classes = reduce((lambda x, y: x * y), action_arr)
    num_range = list(range(0, num_eq_classes))
    range_values = [num_range for x in range(0, len(action_arr))]
    return itertools.product(*map(itertools.permutations, range_values))


def __all_possible_symmetric_nodup(action_arr):
    '''
    All possible symmetric games without repetitions

    - <action_arr> indicates the actions that each player may make
    '''
    num_eq_classes = reduce((lambda x, y: x * y), action_arr)
    num_range = list(range(0, num_eq_classes))
    range_values = num_range
    return itertools.permutations(range_values)


def __all_possible_symmetric_dup(action_arr):
    '''
    All possible symmetric games with repetitions

    - <action_arr> indicates the actions that each player may make
    '''
    num_eq_classes = reduce((lambda x, y: x * y), action_arr)
    num_range = list(range(0, num_eq_classes))
    range_values = num_range
    return itertools.product(range_values, repeat=num_eq_classes)

def __sample_normal_form_nodup(num_actions, num_players, num_samples):
        ix_gen = 0
        num_games = 0
        eq_count = 0
        outcome_perm = []
        action_combination = []
        if isinstance(num_actions, list): 
            action_combination = [num_actions[i] for i in range(0, num_players)]
        else:
            action_combination = [num_actions for i in range(0, num_players)]

        num_eq_classes = reduce((lambda x, y: x * y), action_combination)
        # for action_combination in __gen_player_actions(num_players, max_num_actions):

        # print("Generating game for: " + str(action_combination))
        # allgames = __all_possible_games_nodup(action_combination)
        finish = False
        for sample in range(0, num_samples):
            currgame = dict()
            currgame["x"] = ix_gen
            currgame["z"] = num_players  # fixed to two players for now
            currgame["y"] = []

            dimensionality = action_combination + [num_players]
            index_val = [list(range(0, action_combination[i])) for i in range(0, num_players)]
            indexer_list = __cartprod(index_val)

            a = np.empty(dimensionality, dtype=np.int16)
            available_classes = [list(range(0,num_eq_classes)) for i in range(0, num_players)]

            for index_v in indexer_list:
                assigned_list = []
                for pl_ix in range(0, num_players):
                    chosenval = random.choice(available_classes[pl_ix])
                    available_classes[pl_ix].remove(chosenval)
                    assigned_list.append(chosenval)
                a[tuple(index_v)] = assigned_list

            currgame["y"] = a.tolist()
            num_games += 1
            ix_gen += 1
            yield currgame


def __sample_normal_form_dup(num_actions, num_players, num_samples):
        ix_gen = 0
        num_games = 0
        eq_count = 0
        outcome_perm = []
        action_combination = []
        if isinstance(num_actions, list): 
            action_combination = [num_actions[i] for i in range(0, num_players)]
        else:
            action_combination = [num_actions for i in range(0, num_players)]
        
        num_eq_classes = reduce((lambda x, y: x * y), action_combination)
        # for action_combination in __gen_player_actions(num_players, max_num_actions):

        # print("Generating game for: " + str(action_combination))
        # allgames = __all_possible_games_nodup(action_combination)
        finish = False
        for sample in range(0, num_samples):
            currgame = dict()
            currgame["x"] = ix_gen
            currgame["z"] = num_players  # fixed to two players for now
            currgame["y"] = []

            dimensionality = action_combination + [num_players]
            index_val = [list(range(0, action_combination[i])) for i in range(0, num_players)]
            indexer_list = __cartprod(index_val)

            a = np.empty(dimensionality, dtype=np.int16)

            for index_v in indexer_list:
                val = [random.randint(0,num_eq_classes-1) for i in range(0, num_players)]
                a[tuple(index_v)] = val

            currgame["y"] = a.tolist()
            num_games += 1
            ix_gen += 1
            yield currgame

def __gen_normal_form_nodup(num_actions, num_players):
    '''
    Generator function that returns a json object with a unique representation of a game
    The game will be non-symmetric and without duplicates

    - <max_num_actions> defines the maximum number of actions available per players, therefore 
    the actions will range from (0, max_num_actions) for all players

    - <num_players> specifies the number of players in the game
    '''
    ix_gen = 0
    num_games = 0
    eq_count = 0
    outcome_perm = []
    action_combination = []
    if isinstance(num_actions, list): 
        action_combination = [num_actions[i] for i in range(0, num_players)]
    else:
        action_combination = [num_actions for i in range(0, num_players)]
    # for action_combination in __gen_player_actions(num_players, max_num_actions):
    numOutcomes = reduce((lambda x, y: x * y), action_combination)
    min_outcome_val = 1
    max_outcome_val = numOutcomes
    range_outcome = range(min_outcome_val, max_outcome_val + 1)
    value_list = [range_outcome for x in range(0, num_players)]

    # print("Generating game for: " + str(action_combination))
    allgames = __all_possible_games_nodup(action_combination)
    finish = False
    while not finish:
        currgame = dict()
        currgame["x"] = ix_gen
        currgame["z"] = num_players  # fixed to two players for now
        currgame["y"] = []

        outcome_perm = []
        try:
            outcome_perm = list(next(allgames))
            zipper = zip(*outcome_perm)
            list_outcomes = []
            for x in zipper:
                list_outcomes.append(list(x))
            
        except StopIteration:
            break

        ix = 0
        ix_gen += 1
        if not outcome_perm:
            break

        dimensionality = action_combination + [num_players]
        index_val = [list(range(0, action_combination[i])) for i in range(0, num_players)]
        indexer_list = __cartprod(index_val)

        a = np.empty(dimensionality, dtype=np.int16)

        for index_v in indexer_list:
            val = list_outcomes[ix]
            a[tuple(index_v)] = val
            ix += 1

        currgame["y"] = a.tolist()
        num_games += 1
        yield currgame


def __gen_normal_form_dup(num_actions, num_players):
    '''
    Generator function that returns a json object with a unique representation of a game
    The game will be non-symmetric and WITH duplicates

    - <max_num_actions> defines the maximum number of actions available per players, therefore 
    the actions will range from (0, max_num_actions) for all players

    - <num_players> specifies the number of players in the game
    '''
    ix_gen = 0

    num_games = 0
    outcome_perm = []
    action_combination = []
    if isinstance(num_actions, list): 
        action_combination = [num_actions[i] for i in range(0, num_players)]
    else:
        action_combination = [num_actions for i in range(0, num_players)]
        
    numOutcomes = reduce((lambda x, y: x * y), action_combination)
    min_outcome_val = 1
    max_outcome_val = numOutcomes
    range_outcome = range(min_outcome_val, max_outcome_val + 1)
    value_list = [range_outcome for x in range(0, num_players)]
    outcome_possibilities = list(itertools.product(*value_list))
    # print("Generating game for: " + str(action_combination))
    matrix_weight = [1 for x in range(0, numOutcomes)]
    allcombs = __all_possible_games_dup(action_combination)
    finish = False
    while not finish:
        currgame = dict()
        currgame["x"] = ix_gen
        currgame["z"] = num_players  # fixed to two players for now
        currgame["y"] = []

        outcome_perm = []
        try:
            outcome_perm = list(next(allcombs))
            zipper = zip(*outcome_perm)
            list_outcomes = []
            for x in zipper:
                list_outcomes.append(list(x))
            
        except StopIteration:
            break

        ix = 0
        ix_gen += 1
        if outcome_perm == []:
            break

        dimensionality = action_combination + [num_players]

        index_val = [list(range(0, action_combination[i])) for i in range(0, num_players)]
        indexer_list = __cartprod(index_val)

        a = np.empty(dimensionality, dtype=np.int16)
        for index_v in indexer_list:
            val = list_outcomes[ix]
            a[tuple(index_v)] = val
            ix += 1

        currgame["y"] = a.tolist()

        num_games += 1
        # x = mycol.insert_one(currgame)
        yield currgame


def __sample_normal_form_symmetric_nodup(num_actions, num_players, num_samples):
    ix_gen = 0
    num_games = 0
    outcome_perm = []
    action_combination = [num_actions for i in range(0, num_players)]
    # for action_combination in __gen_player_actions_symmetric(num_players, max_num_actions):
    numOutcomes = reduce((lambda x, y: x * y), action_combination)
    min_outcome_val = 1
    max_outcome_val = numOutcomes
    range_outcome = range(min_outcome_val, max_outcome_val + 1)
    value_list = [range_outcome for x in range(0, num_players)]
    num_eq_classes = reduce((lambda x, y: x * y), action_combination)

    for sample in range(0, num_samples):
        currgame = dict()
        currgame["x"] = ix_gen
        currgame["z"] = num_players  # fixed to two players for now
        currgame["y"] = []

        outcome_perm = []
        
        ix = 0
        ix_gen += 1

        dimensionality = action_combination + [num_players]
        num_actions_curr = action_combination[0]

        num_var = reduce((lambda x, y: x * y), action_combination)
        all_values = [list(range(0, val)) for val in action_combination]
        all_ix = __cartprod(all_values)

        a = np.empty(dimensionality, dtype=np.int16)
        processed = dict()
        ix_outcome = 0

        available_classes = list(range(0,num_eq_classes))
        for ix in all_ix:
            if tuple(ix) in processed:
                continue
            # if diagonal
            if len(set(ix)) == 1:
                # print("Diag: "+str(ix))
                chosenval = random.choice(available_classes)
                available_classes.remove(chosenval)
                val = [chosenval for x in range(0, num_players)]
                processed[tuple(ix)] = True
                a[tuple(ix)] = val
                ix_outcome += 1
            else:
                allperm = list(itertools.permutations(ix))
                allpermtup = [tuple(val) for val in allperm]
                allpermset = set(allperm)
                allpermset.remove(tuple(ix))
                # currvec = []
                # for i in range(0, len(action_combination)):
                #     currvec.append(list_outcomes[ix_outcome])
                #     ix_outcome += 1
                currvec = []
                for pl_ix in range(0, num_players):
                    chosenval = random.choice(available_classes)
                    currvec.append(chosenval)
                    available_classes.remove(chosenval)

                # currvec = [random.randint(0, num_eq_classes) for x in range(0, num_players)]
                a[tuple(ix)] = currvec
 
                orig_perm = ix
                for perm in allpermset:
                    newvec = [0 for x in range(0, len(currvec))]
                    # which values does this guy have?
                    change_ix = -1
                    change_val = -1
                    for ix_iter in range(0, len(ix)):
                        if ix[ix_iter] == perm[ix_iter]:
                            newvec[ix_iter] = currvec[ix_iter]
                        else:
                            if change_ix != -1:
                                newvec[ix_iter] = currvec[change_ix]
                                newvec[change_ix] = currvec[ix_iter]
                                change_ix = -1
                            else:
                                change_ix = ix_iter
                    a[perm] = newvec
                    processed[perm] = True

        currgame["y"] = a.tolist()
        num_games += 1
        yield currgame

 
def __gen_normal_form_symmetric_nodup(num_actions, num_players):
    '''
    Generator function that returns a json object with a unique representation of a game
    The game will be SYMMETRIC and WITHOUT duplicates

    - <max_num_actions> defines the maximum number of actions available per players, therefore 
    the actions will range from (0, max_num_actions) for all players

    - <num_players> specifies the number of players in the game
    '''
    ix_gen = 0
    num_games = 0
    outcome_perm = []
    action_combination = [num_actions for i in range(0, num_players)] 
    # for action_combination in __gen_player_actions_symmetric(num_players, max_num_actions):
    numOutcomes = reduce((lambda x, y: x * y), action_combination)
    min_outcome_val = 1
    max_outcome_val = numOutcomes
    range_outcome = range(min_outcome_val, max_outcome_val + 1)
    value_list = [range_outcome for x in range(0, num_players)]
    # print("Generating game for: " + str(action_combination))
    allgames = __all_possible_symmetric_nodup(action_combination)
    finish = False
    while not finish:
        currgame = dict() 
        currgame["x"] = ix_gen
        currgame["z"] = num_players  # fixed to two players for now
        currgame["y"] = []

        outcome_perm = []
        try:
            outcome_perm = list(next(allgames))
            list_outcomes = outcome_perm
            
        except StopIteration:
            break

        ix = 0 
        ix_gen += 1
        if outcome_perm == []:
            break

        dimensionality = action_combination + [num_players]

        num_var = reduce((lambda x, y: x * y), action_combination)
        # print(num_var)
        all_values = [list(range(0, val)) for val in action_combination]
        all_ix = __cartprod(all_values)
        # print(all_ix)
        a = np.empty(dimensionality, dtype=np.int16)
        processed = dict()
        ix_outcome = 0
        for ix in all_ix:
            if tuple(ix) in processed:
                continue
            # if diagonal
            if len(set(ix)) == 1:
                # print("Diag: "+str(ix))
                val = [list_outcomes[ix_outcome] for x in range(0, len(action_combination))]
                processed[tuple(ix)] = True
                a[tuple(ix)] = val
                ix_outcome += 1
            else:
                allperm = list(itertools.permutations(ix))
                allpermtup = [tuple(val) for val in allperm]
                allpermset = set(allperm)
                allpermset.remove(tuple(ix))
                currvec = []
                for i in range(0, len(action_combination)):
                    currvec.append(list_outcomes[ix_outcome])
                    ix_outcome += 1
                a[tuple(ix)] = currvec

                orig_perm = ix
                for perm in allpermset:
                    newvec = [0 for x in range(0, len(currvec))]
                    # which values does this guy have?
                    change_ix = -1
                    change_val = -1
                    for ix_iter in range(0, len(ix)):
                        if ix[ix_iter] == perm[ix_iter]:
                            newvec[ix_iter] = currvec[ix_iter]
                        else:
                            if change_ix != -1:
                                newvec[ix_iter] = currvec[change_ix]
                                newvec[change_ix] = currvec[ix_iter]
                                change_ix = -1
                            else:
                                change_ix = ix_iter
                    a[perm] = newvec
                    processed[perm] = True

        currgame["y"] = a.tolist()
        num_games += 1
        yield currgame


def __sample_normal_form_symmetric_dup(num_actions, num_players, num_samples):
    ix_gen = 0
    num_games = 0
    outcome_perm = []
    action_combination = [num_actions for i in range(0, num_players)]
    # for action_combination in __gen_player_actions_symmetric(num_players, max_num_actions):
    numOutcomes = reduce((lambda x, y: x * y), action_combination)
    min_outcome_val = 1
    max_outcome_val = numOutcomes
    range_outcome = range(min_outcome_val, max_outcome_val + 1)
    value_list = [range_outcome for x in range(0, num_players)]
    num_eq_classes = reduce((lambda x, y: x * y), action_combination)

    for sample in range(0, num_samples):
        currgame = dict()
        currgame["x"] = ix_gen
        currgame["z"] = num_players  # fixed to two players for now
        currgame["y"] = []

        outcome_perm = []
        
        ix = 0
        ix_gen += 1

        dimensionality = action_combination + [num_players]
        num_actions_curr = action_combination[0]

        num_var = reduce((lambda x, y: x * y), action_combination)
        all_values = [list(range(0, val)) for val in action_combination]
        all_ix = __cartprod(all_values)

        a = np.empty(dimensionality, dtype=np.int16)
        processed = dict()
        ix_outcome = 0
        for ix in all_ix:
            if tuple(ix) in processed:
                continue
            # if diagonal
            if len(set(ix)) == 1:
                # print("Diag: "+str(ix))
                randomNum = random.randint(0,num_eq_classes-1)
                val = [randomNum for x in range(0, num_players)]
                processed[tuple(ix)] = True
                a[tuple(ix)] = val
                ix_outcome += 1
            else:
                allperm = list(itertools.permutations(ix))
                allpermtup = [tuple(val) for val in allperm]
                allpermset = set(allperm)
                allpermset.remove(tuple(ix))
                # currvec = []
                # for i in range(0, len(action_combination)):
                #     currvec.append(list_outcomes[ix_outcome])
                #     ix_outcome += 1
                currvec = [random.randint(0, num_eq_classes-1) for x in range(0, num_players)]
                a[tuple(ix)] = currvec
 
                orig_perm = ix
                for perm in allpermset:
                    newvec = [0 for x in range(0, len(currvec))]
                    # which values does this guy have?
                    change_ix = -1
                    change_val = -1
                    for ix_iter in range(0, len(ix)):
                        if ix[ix_iter] == perm[ix_iter]:
                            newvec[ix_iter] = currvec[ix_iter]
                        else:
                            if change_ix != -1:
                                newvec[ix_iter] = currvec[change_ix]
                                newvec[change_ix] = currvec[ix_iter]
                                change_ix = -1
                            else:
                                change_ix = ix_iter
                    a[perm] = newvec
                    processed[perm] = True

        currgame["y"] = a.tolist()
        num_games += 1
        yield currgame


def __gen_normal_form_symmetric_dup(num_actions, num_players):
    '''
    Generator function that returns a json object with a unique representation of a game
    The game will be SYMMETRIC and WITH duplicates

    - <max_num_actions> defines the maximum number of actions available per players, therefore 
    the actions will range from (0, max_num_actions) for all players

    - <num_players> specifies the number of players in the game
    '''
    ix_gen = 0
    num_games = 0
    outcome_perm = []
    action_combination = [num_actions for i in range(0, num_players)]
    # for action_combination in __gen_player_actions_symmetric(num_players, max_num_actions):
    numOutcomes = reduce((lambda x, y: x * y), action_combination)
    min_outcome_val = 1
    max_outcome_val = numOutcomes
    range_outcome = range(min_outcome_val, max_outcome_val + 1)
    value_list = [range_outcome for x in range(0, num_players)]

    # print("Generating game for: " + str(action_combination))
    allgames = __all_possible_symmetric_dup(action_combination)
    finish = False
    while not finish:
        currgame = dict()
        currgame["x"] = ix_gen
        currgame["z"] = num_players  # fixed to two players for now
        currgame["y"] = []

        outcome_perm = []
        try:
            outcome_perm = list(next(allgames))
            list_outcomes = outcome_perm
            

        except StopIteration:
            break
        
        ix = 0
        ix_gen += 1
        if outcome_perm == []:
            break

        dimensionality = action_combination + [num_players]
        num_actions_curr = action_combination[0]

        # print(list_outcomes)
        num_var = reduce((lambda x, y: x * y), action_combination)
        # print(num_var)
        all_values = [list(range(0, val)) for val in action_combination]
        all_ix = __cartprod(all_values)
        # print(all_ix)
        a = np.empty(dimensionality, dtype=np.int16)
        processed = dict()
        ix_outcome = 0
        for ix in all_ix:
            if tuple(ix) in processed:
                continue
            # if diagonal
            if len(set(ix)) == 1:
                # print("Diag: "+str(ix))
                val = [list_outcomes[ix_outcome] for x in range(0, len(action_combination))]
                processed[tuple(ix)] = True
                a[tuple(ix)] = val
                ix_outcome += 1
            else:
                allperm = list(itertools.permutations(ix))
                allpermtup = [tuple(val) for val in allperm]
                allpermset = set(allperm)
                allpermset.remove(tuple(ix))
                currvec = []
                for i in range(0, len(action_combination)):
                    currvec.append(list_outcomes[ix_outcome])
                    ix_outcome += 1
                a[tuple(ix)] = currvec

                orig_perm = ix
                for perm in allpermset:
                    newvec = [0 for x in range(0, len(currvec))]
                    # which values does this guy have?
                    change_ix = -1
                    change_val = -1
                    for ix_iter in range(0, len(ix)):
                        if ix[ix_iter] == perm[ix_iter]:
                            newvec[ix_iter] = currvec[ix_iter]
                        else:
                            if change_ix != -1:
                                newvec[ix_iter] = currvec[change_ix]
                                newvec[change_ix] = currvec[ix_iter]
                                change_ix = -1
                            else:
                                change_ix = ix_iter
                    a[perm] = newvec
                    processed[perm] = True

        currgame["y"] = a.tolist()
        num_games += 1
        yield currgame



def __cartprod(arrays):
    N = len(arrays)
    return transpose(meshgrid(*arrays, indexing='ij'),
                     roll(arange(N + 1), -1)).reshape(-1, N)

def __gen_player_actions(num_players, max_actions):
    '''
    Generator function that generates all possible combination of actions
    for <num_players>

    Actions go from 1 to <max_actions>
    '''

    range_actions = list(range(1, max_actions + 1))
   
    list_val = [range_actions for x in range(0, num_players)]

    for element in itertools.product(*list_val):
        yield (list(element))


def __gen_player_actions_symmetric(num_players, max_actions):
    '''
    Generates all possible symmetric action configurations

    - <num_players> is the number of player for the game
    - <max_actions> is the maximum number of actions per player
    '''
    range_actions = list(range(1, max_actions + 1))
    for action in range(1, max_actions + 1):
        currAct = [action for x in range(0, num_players)]
        yield (list(currAct))



        
def __get_outcome(tuple_ix, game_list):
    return tuple(np.asarray(game_list)[tuple_ix])