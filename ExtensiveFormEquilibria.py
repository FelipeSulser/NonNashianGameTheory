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

'''

 Public functions used to compute Extensive Form Equilibria

'''

def nash(gameDict):
    nash_ix = __find_nash_eq(gameDict['y'])
    gameDict['N'] = nash_ix
    return gameDict


def rationalizability(gameDict):
    rat_ix = __iterated_elim_strictly_dominated_actions(gameDict['y'])
    gameDict['R'] = rat_ix
    return gameDict


def individual(gameDict):
    individual_ix = __individually_rational(gameDict['y'])
    gameDict['I'] = individual_ix
    return gameDict 


def spe(gameDict):
    spe_ix = __get_spe_equilibria(gameDict['y'])
    tupler = [tuple(subl) for subl in spe_ix]
    gameDict['S'] = tupler
    return gameDict

def hofstader(gameDict):
    hofstader_ix = __find_hofstader_eq(gameDict['y'])
    gameDict['H'] = hofstader_ix
    return gameDict

def pte(gameDict):
    pte_ix = __find_perfectly_transparent_eq(gameDict['y'])
    gameDict['P'] = pte_ix
    return gameDict

def minimax(gameDict):
    minimax_ix = __minimax_rationalizability(gameDict['y'])
    gameDict['M'] = minimax_ix
    return gameDict

def ppe(gameDict):
    ppe_ix = __get_ppe_equilibria(gameDict['y'])
    tupler = [tuple(subl) for subl in ppe_ix]
    gameDict['P'] = tupler
    return gameDict


def shiffrin(gameDict):
    shiffrin_ix = __get_shiffrin_equilibria(gameDict['y'])
    gameDict['Sh'] = shiffrin_ix
    return gameDict

def get_equilibria(equilibria, game_list):
    if equilibria == "H":
        return __find_hofstader_eq(game_list)
    elif equilibria == "P":
        return __find_perfectly_transparent_eq(game_list)
    elif equilibria == "M":
        return __minimax_rationalizability(game_list)
    elif equilibria == "N":
        return __find_nash_eq(game_list)
    elif equilibria == "R":
        return __iterated_elim_strictly_dominated_actions(game_list)
    elif equilibria == "I":
        return __individually_rational(game_list)

    raise Exception('Error, equilibria not defined. Try: [hofstader|pte|minimax]')


'''

 Private functions used to compute Extensive Form Equilibria

'''


def __spe_eq_rec(game_content, genval, curr_path):
    if isinstance(game_content, list):
        
        return [{"data":game_content, "id": next(genval), "path": curr_path[:]}]
    #if its a dict
    if "cand" in game_content:
        currPlayer = game_content["p"]
        return game_content["cand"]
    else:
        game_content["cand"] = []
        allCands = []
        ixpos = 0
        for elem in game_content["c"]:
            curr_path.append(ixpos)
            allCands.append(__spe_eq_rec(elem, genval, curr_path))
            del curr_path[-1]
            ixpos += 1

        currPlayer = game_content["p"]
        maxIxs = findMaxs(allCands, currPlayer)
        #print("maxs : "+str(maxIxs))
        maxIxs = find_max_val(maxIxs, currPlayer)

        #resIx = [{"data": cont, "id": next(genval)} for cont in maxIxs]
        game_content["cand"] = maxIxs
        return maxIxs



def __get_spe_equilibria(game_dict):
    game_content = game_dict
    spe_eq = []
    possibilities = []
    copy_game = copy.deepcopy(game_content)
    generator = id_generator()
    path = []
    res = __spe_eq_rec(copy_game, generator, path)
    
    res = remove_dup_id(res)
   
    return res[0]


def __find_nash_eq(game_list):
    arr_game = np.asarray(game_list)
    num_players = arr_game.ndim - 1
    num_actions = list(arr_game.shape)[:-1]
    # print("DIM: "+str(num_actions))
    list_max_ix = []
    for i in range(0, num_players):
        max_values_player = []
        # print("player "+str(i))
        allPossibleStates = gen_all_states(num_actions, i)
        for state in allPossibleStates:
            currmax = -1
            max_ix_list = []

            for pdata in range(0, num_actions[i]):
                actualState = state
                actualState[i] = pdata
                ixVal = tuple(actualState)
                selectedVal = arr_game[ixVal][i]
                # print(selectedVal)
                if selectedVal > currmax:
                    max_ix_list = [ixVal]
                    currmax = selectedVal

                elif selectedVal == currmax:
                    max_ix_list.append(ixVal)

            max_values_player.append(max_ix_list)
        list_max_ix.append(max_values_player)

    flattened_list = []
    for currlist in list_max_ix:
        flattened_list.append([item for sublist in currlist for item in sublist])
    res_ix = list(reduce(set.intersection, [set(item) for item in flattened_list]))
    # print(res_ix)
    res_ix = [[int(val) for val in x] for x in res_ix]
    return res_ix



def __iterated_elim_strictly_dominated_actions(game_list):
    '''
    Rationalizability based on set of not strictly dominated actions
    IMPORTANT: This method does not take into account mixed strategies but only pure strategies
    '''


    arr_game = np.asarray(game_list)
    num_players = arr_game.ndim - 1
    num_actions = list(arr_game.shape)[:-1]
    maximin_utility = []

    available_ix = cartprod([list(range(0, num_actions[x])) for x in range(0, num_players)])
    available_ix = [tuple(listed) for listed in available_ix]
    prev_len = -1
    curr_len = len(available_ix)
    while curr_len and prev_len != curr_len:

        for player in range(0, num_players):
            min_list = []
            all_ixs = available_ix  # gen_all_states(num_actions, player)
            # print("FIX PLAYER: "+str(player))

            for action in range(0, num_actions[player]):

               
                isCovered = True
                for compAction in range(0, num_actions[player]):
                    if action == compAction:
                        continue

                    for curr_ix in available_ix:
                        ix_action = list(curr_ix)
                        oldval = ix_action[player]
                        ix_action[player] = action

                        val_action = arr_game[tuple(ix_action)][player]
                        ix_action[player] = oldval
                        oldval = curr_ix[player]
                        ix_compaction = list(curr_ix)
                        ix_compaction[player] = compAction

                        val_compaction = arr_game[tuple(ix_compaction)][player]
                        ix_compaction[player] = oldval
                       
                        if val_action >= val_compaction:
                            isCovered = False
                            break
                    if isCovered:
                        break

                if isCovered and num_actions[player] > 1:
                    # reduce it
                    toDel = []
                    for ix_iter in available_ix:
                        if ix_iter[player] == action:
                            toDel.append(tuple(ix_iter))
                    # print("Action : "+str(action)+ " of player: "+str(player)+" is covered")
                    available_ix = list(set(available_ix) - set(toDel))
                    # print("Avail: "+str(available_ix))

        prev_len = curr_len
        curr_len = len(available_ix)

    available_ix = [[int(val) for val in x] for x in available_ix]
    return available_ix

def __individually_rational(game_list):
    arr_game = np.asarray(game_list)
    num_players = arr_game.ndim - 1
    num_actions = list(arr_game.shape)[:-1]
    player_choices = [list(range(0, num_actions[x])) for x in range(0, num_players)]

    all_ix = cartprod(player_choices)
    pte_candidates = all_ix

    maximin_utility = __find_maximin(arr_game, pte_candidates)
    newlist_candidates = []

    for cand in pte_candidates:
        val = arr_game[tuple(cand)]
        if __satisfies_maximin(val, maximin_utility):
            newlist_candidates.append(cand)

    newlist_candidates = [tuple([int(val) for val in x]) for x in newlist_candidates]
    return newlist_candidates


def __satisfies_maximin(list_val, maximin_utility):
    for ix in range(0, len(list_val)):
        if list_val[ix] < maximin_utility[ix]:
            return False
    return True


def __find_maximin(arr_game, pte_candidates):
    num_players = arr_game.ndim - 1
    num_actions = list(arr_game.shape)[:-1]
    maximin_utility = []
    for player in range(0, num_players):
        # for each action of the player, fix it and find min of all
        # then find the max across actions for each player
        min_list = []
        for action in range(0, num_actions[player]):
            minval = 9999  # arbitrary large number
            for state in pte_candidates:#list(filter(lambda x: x[player] == action, pte_candidates)):
                if state[player] != action:
                    continue
                actualState = state
                ixVal = tuple(actualState)
                selectedVal = arr_game[ixVal][player]
                minval = min(minval, selectedVal)

            if minval != 9999:
                min_list.append(minval)

        maximin_utility.append(max(min_list))
    return maximin_utility


def __get_game_by_line(filename):
    fp = open(filename)
    for i, line in enumerate(fp):
        yield (i,line)


def __validate_equilibria(equilibria):
    if equilibria == "nash" or equilibria == "rationalizability" or equilibria == "individual" or equilibria == "hofstader" or equilibria == "pte" or equilibria == "minimax":
        return equilibria.title()[0]

    raise Exception('Error, equilibria not defined. Try: [nash|rationalizability|individual]')





def __get_indices(root, genval):
    if isinstance(root, list):
        return
    ix = 0
    for ix in range(0, len(root['c'])):
        if isinstance(root['c'][ix], list):
            root['c'][ix] = {"id": next(genval), "data": root['c'][ix]}
        else:
            __get_indices(root['c'][ix], genval)

def __annotate_nodes(root, genval):
    
    if isinstance(root, dict) and "id" in root:
        return [root]

    #for all children, annotate them flat
    candidates = []
    for child in root['c']:
        currSublist = __annotate_nodes(child, genval)
        if not (isinstance(currSublist[0], dict) and "id" in currSublist[0]):
            #flatten it
            currSublist = [x for y in currSublist for x in y]
        candidates.append(currSublist)
    root["cand"] = candidates
    return candidates




def __getMaxList(list_outcomes, currPlayer):
    maxval = -1
    id_loc = -1
    for outcome in list_outcomes:
        if outcome["data"][currPlayer] >= maxval:
            maxval = outcome["data"][currPlayer]
            id_loc = outcome["id"]
    return (maxval, id_loc)


def __maxPayoff(list_trees_payoff, currPlayer):
    maxval = -1
    curr_max_ix = 0
    #print("__MaxPayoff: "+str(list_trees_payoff)+" currPlayer: "+str(currPlayer))
    for i in range(0, len(list_trees_payoff)):
        currMax, idmax = __getMaxList(list_trees_payoff[i], currPlayer)
        if currMax >= maxval:
            curr_max_ix = i
            maxval = currMax
    return curr_max_ix


def __remove_less_than(listElem, threshold, currPlayer):
    retList = []
   
    for subtree in listElem:
        subList = []
        for elem in subtree:
            if elem["data"][currPlayer] >= threshold:
                subList.append(elem)
        retList.append(subList)
    return retList


def __dup_max_values(cands, currPlayer):
    maxval = -1
    for i in range(0, len(cands)):
        for elem in cands[i]:
           
            if elem["data"][currPlayer] >= maxval:

                maxval = elem["data"][currPlayer]
    #now find if there is more than one
    ret_list = []
   
    for i in range(0, len(cands)):
        for elem in cands[i]:
           
            if elem["data"][currPlayer] == maxval:
                ret_list.append(elem)

    return ret_list

def __duplicated(cands, currPlayer):
    maxval = -1
    for i in range(0, len(cands)):
        for elem in cands[i]:
            if elem["data"][currPlayer] >= maxval:
                maxval = elem["data"][currPlayer]
    #now find if there is more than one
    found = 0
    for i in range(0, len(cands)):
        for elem in cands[i]:
            if elem["data"][currPlayer] == maxval:
                found += 1
    if found > 1:
        return True
    return False


def __ppe_eq_loop(root):
    iterval = root
    while not (isinstance(iterval, dict) and "id" in iterval):
       
        currPlayer = iterval['p']
        subtree_ix = __maxPayoff(iterval['cand'], currPlayer)
        maxSubPayoff = -1
        maxSubPayoff_ix = -1
        if __duplicated(iterval['cand'], currPlayer):
          
            return __dup_max_values(iterval['cand'], currPlayer)

        for i in range(0, len(iterval['cand'])):
            if i != subtree_ix:
                currMax, currMaxIx = __getMaxList(iterval['cand'][i], currPlayer)
                if currMax >= maxSubPayoff:
                    maxSubPayoff = currMax
                    maxSubPayoff_ix = currMaxIx
                #maxSubPayoff = max(maxSubPayoff, __getMaxList(iterval['cand'][i], currPlayer))
        #clean cand
        cand_list = iterval['c'][subtree_ix]
        if isinstance(cand_list, dict) and "id" in cand_list:
            return [cand_list]
        
        cand_list = cand_list['cand']
       
        cand_list = __remove_less_than(cand_list, maxSubPayoff, currPlayer)
        iterval = iterval['c'][subtree_ix]
        iterval['cand'] = cand_list
    
    return [tuple(iterval)] #returns here if no dups


def __encapsulate_find_ix(root, res_node, curr_path):
    all_paths = []
    
    for res_val in res_node:

        curr_path = []
        ppe_ix = __find_ix(root, res_val, curr_path)
        all_paths.append(curr_path)
    return all_paths



def __find_ix(root, res_node, curr_path): 
   
    if isinstance(root, dict) and "id" in root:
        if tuple(root["data"]) == tuple(res_node["data"]) and root["id"] == res_node["id"]:
            return curr_path
        else:
            return None
    for pos_ix in range(0, len(root['c'])):
        curr_path.append(pos_ix)
       
        if __find_ix(root['c'][pos_ix], res_node, curr_path):
            return curr_path
       
        del curr_path[-1]
    return None

def __get_ppe_equilibria(game_dict):
    game_content = game_dict
    ppe_eq = []
    copy_game = copy.deepcopy(game_content)
    generator = id_generator()
    __get_indices(copy_game, generator)
   
    annotation = __annotate_nodes(copy_game, generator)
   
    res_node = __ppe_eq_loop(copy_game)
   
    path = []
    path = __encapsulate_find_ix(copy_game, res_node, path)
    # print("PPE is: "+str(res_node))
    # print("PATH: "+str(path))
   
    return path

def __shiffrin_annotation(root, parent, threshold=None):
    if isinstance(root, list):
        return tuple(root)

    #for all children, annotate them flat
    candidates = []
    allOutcomes = True
    mapDone = []
    for child in root['c']:
        currSublist = __shiffrin_annotation(child, root)

        if not isinstance(child, list):
            allOutcomes = False
           
            mapDone.append(False)
        
        else:
            mapDone.append(True)

        candidates.append(currSublist)
  

    newL = []
    for elem in candidates:
        if not isinstance(elem, tuple):
            newL += elem
        else:
            newL.append(elem)
    candidates = newL
   
    if threshold:
        clean_cand = []
        currPlayer = root['p']
        for elem in candidates:
           
            if elem[currPlayer] > threshold[currPlayer]:
                clean_cand.append(elem)

        if not clean_cand:
            clean_cand = [threshold]

        candidates = clean_cand


    root["choices"] = candidates
    root["Q"] = []
    if allOutcomes:
       
        root["Q"] = selfish_max(candidates, root['p'])

    root['parent'] = parent
    root['allDone'] = allOutcomes
    root['mapDone'] = mapDone
    return candidates

def __all_done(bitList):
    for elem in bitList:
        if not elem:
            return False
    return True

def __selfish_best(list_out, player):
    maxIx = -1
    maxVal = -1
    for i in range(0, len(list_out)):
        if list_out[i][player] >= maxVal:
            maxVal = list_out[i][player]
            maxIx = i
    return list_out[maxIx]

def __shiffrin_eq_rec(root, top_Q=(-1,-1)):
    #Base case
    if isinstance(root, list):
        if root[0] > top_Q[0] and root[1] > top_Q[1]:
            return tuple(root)
        else:
            return top_Q

    allDone = False
    bestQ = (-1,-1)
    currPlayer = root['p']
    mapDone = [False for x in root['c']]
    firstTime = [True for x in root['c']]
    while not allDone:
        listQ = []
        for ix in range(0,len(mapDone)):
            if not mapDone[ix]:
                child = root['c'][ix]
                #compute for first time
                newQ = (-1,-1)
                if firstTime[ix]:
                    newQ = __shiffrin_eq_rec(child)
                    listQ.append(newQ)
                else:
                    newQ = __shiffrin_eq_rec(child, bestQ)
                    listQ.append(newQ)
                   
                firstTime[ix] = False

                if newQ == bestQ:
                    mapDone[ix] = True
       
        newQ = __selfish_best(listQ, currPlayer)
        if newQ[currPlayer] >= bestQ[currPlayer] and newQ[0] >= top_Q[0] and newQ[1] >= top_Q[1]:
            bestQ = newQ
        else: 
            if top_Q[0] > bestQ[0] and top_Q[1] > bestQ[1]:
                bestQ = top_Q
                        #mapDone = [False for x in root['c']] #reset the bitmap
                    
                #print("At node: "+str(root)+" NEWQ="+str(newQ)+" topQ="+str(top_Q)+" and bestQ="+str(bestQ))
        
        allDone = __all_done(mapDone)

    # all Done, return selfish the largest
   
    return bestQ


def __get_shiffrin_ix(root, res_node, curr_path): 
   
    if isinstance(root, list):
        if tuple(root) == tuple(res_node):
            return curr_path
        else:
            return None
    for pos_ix in range(0, len(root['c'])):
        curr_path.append(pos_ix)
       
        if __get_shiffrin_ix(root['c'][pos_ix], res_node, curr_path):
            return curr_path
       
        del curr_path[-1]
    return None

def __get_shiffrin_equilibria(game_dict):
    game_content = copy.deepcopy(game_dict)
    shiffrin_eq = []
    #__shiffrin_annotation(game_content, None)
    #print(game_content)
    outcome = __shiffrin_eq_rec(game_content)
    outcome_ix = []

    outcome_ix = __get_shiffrin_ix(game_content, outcome, outcome_ix)

    return [tuple(outcome_ix)]



def __gen_diagonal_states(num_actions):
    ''' 
    Used in the hofstader equilibrium,
    returns the index of all elements on the diagonal
    '''
    num_players = len(num_actions)
    action_range = range(0, num_actions[0])
    diagonal_list = []
    for x in action_range:
        diagonal_list.append(tuple([x for i in range(0, num_players)]))
    return diagonal_list


def __find_hofstader_eq(game_list):
    arr_game = np.asarray(game_list)
    num_players = arr_game.ndim - 1
    num_actions = list(arr_game.shape)[:-1]
    if len(set(num_actions)) != 1:
        # all symmetric games have same actions for each player
        raise ValueError("Error: Game is not symmetric")

    list_max_ix = []

    allPossibleStates = __gen_diagonal_states(num_actions)
    currmax = -1
    for ix_val in allPossibleStates:
        currval = arr_game[ix_val][0]  # symmetric, all equal in diagonal
        if currval > currmax:
            currmax = currval
            list_max_ix = [ix_val]

        elif currval == currmax:
            list_max_ix.append(ix_val)

    return list_max_ix


def __minimax_rationalizability(game_list):
    arr_game = np.asarray(game_list)
    num_players = arr_game.ndim - 1
    num_actions = list(arr_game.shape)[:-1]
    maximin_utility = []

    available_ix = cartprod([list(range(0, num_actions[x])) for x in range(0, num_players)])
    available_ix = [tuple([int(x) for x in listed]) for listed in available_ix]

    prev_len = -1
    curr_len = len(available_ix)

    while available_ix and prev_len != curr_len:

        for player in range(0, num_players):
            # for each action of the player, fix it and find min of all
            # then find the max across actions for each player
            min_list = []
            all_ixs = available_ix  # gen_all_states(num_actions, player)
            # print("FIX PLAYER: "+str(player))
            for action in range(0, num_actions[player]):
                isCovered = False
                # find max value of action
                action_max = -1

                for curr_ix in available_ix:
                    ix_action = list(curr_ix)
                    oldval = ix_action[player]
                    ix_action[player] = action

                    val_action = arr_game[tuple(ix_action)][player]
                    ix_action[player] = oldval
                    action_max = max(action_max, val_action)

                for compAction in range(0, num_actions[player]):
                    if action == compAction:
                        continue

                    compActionMin = 9999
                    for curr_ix in available_ix:
                        ix_compaction = list(curr_ix)
                        oldval = ix_compaction[player]
                        ix_compaction[player] = compAction

                        val_compaction = arr_game[tuple(ix_compaction)][player]
                        ix_compaction[player] = oldval
                        compActionMin = min(compActionMin, val_compaction)

                    if compActionMin != 9999 and compActionMin > action_max:
                        # can safely remove action
                        isCovered = True
                        break

                if isCovered and num_actions[player] > 1:
                    # reduce it
                    toDel = []
                    for ix_iter in available_ix:
                        if ix_iter[player] == action:
                            toDel.append(tuple(ix_iter))

                    # print("Action : "+str(action)+ " of player: "+str(player)+" is covered")
                    available_ix = list(set(available_ix) - set(toDel))
                    # print("Avail: "+str(available_ix))
        prev_len = curr_len
        curr_len = len(available_ix)

    # print(available_ix)
    return available_ix






def __find_perfectly_transparent_eq(game_list):
    '''
    PTE is defined as the set of strategies that are immune against its common knowledge
    Related to Superrationality for non-symmetric games

    Given a <game_list>, returns the count of strategies that are PTE
    '''
    arr_game = np.asarray(game_list)
    num_players = arr_game.ndim - 1
    num_actions = list(arr_game.shape)[:-1]
    player_choices = [list(range(0, num_actions[x])) for x in range(0, num_players)]
    
    all_ix = cartprod(player_choices)
    pte_candidates = [[int(val) for val in x] for x in all_ix] #make it serializable
    iterval = 0
    #limit it to number of payoffs, so it works for games with ties too
    num_payoffs = reduce((lambda x, y: x * y), num_actions)
    for i in range(0, num_payoffs):
        if len(pte_candidates) == 0:
            break
    #while len(pte_candidates) > 1:  # due to pte uniqueness
        maximin_utility = __find_maximin(arr_game, pte_candidates)

        newlist_candidates = []

        for cand in pte_candidates:
            val = arr_game[tuple(cand)]
            if __satisfies_maximin(val, maximin_utility):
                newlist_candidates.append(cand)
        pte_candidates = newlist_candidates
        iterval+=1

    pte_candidates = [tuple([x for x in elem]) for elem in pte_candidates]
    return pte_candidates 




	