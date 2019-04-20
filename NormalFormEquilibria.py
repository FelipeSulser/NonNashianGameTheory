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
from Util import cartprod, gen_all_states, id_generator


'''

 Public functions used to compute the equilibria on Normal Form Games

'''

def nash(gameDict):
    nash_ix = __find_nash_eq(gameDict['y'])
    tupler = [tuple(subl) for subl in nash_ix]
    gameDict['N'] = tupler
    return gameDict

 
def rationalizability(gameDict):
    rat_ix = __iterated_elim_strictly_dominated_actions(gameDict['y'])
    gameDict['R'] = [tuple(subl) for subl in rat_ix]
    return gameDict

 
def individual(gameDict):
    individual_ix = __individually_rational(gameDict['y'])
    gameDict['I'] = individual_ix
    return gameDict 


def idip(gameDict):
    idip_ix = __iterated_elim_weakly_dominated_actions(gameDict['y'])
    gameDict['ID'] = idip_ix
    return gameDict


def hofstader(gameDict):
    hofstader_ix = __find_hofstader_eq(gameDict['y'])
    gameDict['H'] = hofstader_ix
    return gameDict


def pte(gameDict):
    pte_ix = __find_perfectly_transparent_eq(gameDict['y'])
    tupler = [tuple(subl) for subl in pte_ix]
    gameDict['P'] = tupler  
    return gameDict


def minimax(gameDict):
    minimax_ix = __minimax_rationalizability(gameDict['y'])
    gameDict['M'] = minimax_ix
    return gameDict


def translucent(gameDict):
    translucent_ix = __translucent_equilibrium(gameDict['y'])
    gameDict['T'] = translucent_ix
    return gameDict


def pce(gameDict):
    pce_ix = __perfect_cooperative_equilibrium(gameDict['y'])
    gameDict['PC'] = pce_ix
    return gameDict


def mpce(gameDict):
    mpce_ix = __max_perfect_cooperative_equilibrium(gameDict['y'])
    gameDict['MP'] = mpce_ix
    return gameDict


def irmme(gameDict):
    minimax(gameDict)
    if "M" not in gameDict:
        raise ValueError("Error, compute minimax first")

    game_list = gameDict['y']
    available_states = gameDict['M']
    res_ix = __individual_after_minimax(game_list, available_states)
    gameDict['C'] = res_ix
    return gameDict


def get_equilibria(equilibria, game_list):
    if equilibria == "N":
        return __find_nash_eq(game_list)
    elif equilibria == "R":
        return __iterated_elim_strictly_dominated_actions(game_list)
    elif equilibria == "I":
        return __individually_rational(game_list)
    elif equilibria == "H":
        return __find_hofstader_eq(game_list)
    elif equilibria == "P":
        return __find_perfectly_transparent_eq(game_list)
    elif equilibria == "M":
        return __minimax_rationalizability(game_list)

    raise Exception('Error, equilibria not defined. Try: [nash|rationalizability|individual]')


'''

	Private Functions

'''

def __satisfies_maximin(list_val, maximin_utility):
    for ix in range(0, len(list_val)):
        if list_val[ix] < maximin_utility[ix]:
            return False
    return True



def __translucent_equilibrium(game_list):
    arr_game = np.asarray(game_list)
    num_players = arr_game.ndim - 1
    num_actions = list(arr_game.shape)[:-1]
    player_choices = [list(range(0, num_actions[x])) for x in range(0, num_players)]

    all_ix = cartprod(player_choices)
    pte_candidates = all_ix

    maximin_utility = __find_secondminimin(arr_game, pte_candidates)
   # print("utility="+str(maximin_utility))
    newlist_candidates = []

    for cand in pte_candidates:
        val = arr_game[tuple(cand)]
        if __satisfies_maximin(val, maximin_utility):
            newlist_candidates.append(cand)

    newlist_candidates = [tuple([int(val) for val in x]) for x in newlist_candidates]
    return newlist_candidates



def __annotate_nodes(root):
    if isinstance(root, list):
        return [tuple(root)] 

    #for all children, annotate them flat
    candidates = []
    for child in root['c']:
        currSublist = __annotate_nodes(child)
        if not isinstance(currSublist[0], tuple):
            #flatten it
            currSublist = [x for y in currSublist for x in y]
        candidates.append(currSublist)
    root["cand"] = candidates
    return candidates

def __perfect_cooperative_equilibrium(game_list):
    arr_game = np.asarray(game_list)
    num_players = arr_game.ndim - 1
    num_actions = list(arr_game.shape)[:-1]
    player_choices = [list(range(0, num_actions[x])) for x in range(0, num_players)]

    all_ix = cartprod(player_choices)
    pte_candidates = all_ix

    maximin_utility = __find_maximax(arr_game, pte_candidates)
    #print("utility="+str(maximin_utility))
    newlist_candidates = []

    for cand in pte_candidates:
        val = arr_game[tuple(cand)]
        if __satisfies_maximin(val, maximin_utility):
            newlist_candidates.append(cand)

    newlist_candidates = [tuple([int(val) for val in x]) for x in newlist_candidates]
    return newlist_candidates


def __max_perfect_cooperative_equilibrium(game_list):
    arr_game = np.asarray(game_list)
    num_players = arr_game.ndim - 1
    num_actions = list(arr_game.shape)[:-1]
    player_choices = [list(range(0, num_actions[x])) for x in range(0, num_players)]

    all_ix = cartprod(player_choices)
    pte_candidates = all_ix

    maximin_utility = __find_maximax(arr_game, pte_candidates)
    #print("utility="+str(maximin_utility))
    newlist_candidates = []

    for cand in pte_candidates:
        val = arr_game[tuple(cand)]
        if __satisfies_maximin(val, maximin_utility):
            newlist_candidates.append(cand)

    newlist_candidates = [tuple([int(val) for val in x]) for x in newlist_candidates]


    while len(newlist_candidates) > 0:
        maximin_utility = [(x+1) for x in maximin_utility]
        newlist_candidates = []
 
        for cand in pte_candidates:
            val = arr_game[tuple(cand)]
            if __satisfies_maximin(val, maximin_utility):
                newlist_candidates.append(cand)

        newlist_candidates = [tuple([int(val) for val in x]) for x in newlist_candidates]


    while len(newlist_candidates) == 0:
        maximin_utility = [(x-1) for x in maximin_utility]
        newlist_candidates = []

        for cand in pte_candidates:
            val = arr_game[tuple(cand)]
            if __satisfies_maximin(val, maximin_utility):
                newlist_candidates.append(cand)

        newlist_candidates = [tuple([int(val) for val in x]) for x in newlist_candidates]

    return newlist_candidates




def __getMaxList(list_outcomes, currPlayer):
    maxval = -1

    for outcome in list_outcomes:
        maxval = max(maxval, outcome[currPlayer])
    return maxval


def __maxPayoff(list_trees_payoff, currPlayer):
    maxval = -1
    curr_max_ix = 0
    #print("__MaxPayoff: "+str(list_trees_payoff)+" currPlayer: "+str(currPlayer))
    for i in range(0, len(list_trees_payoff)):
        currMax = __getMaxList(list_trees_payoff[i], currPlayer)
        if currMax >= maxval:
            curr_max_ix = i
            maxval = currMax
    return curr_max_ix


def __remove_less_than(listElem, threshold, currPlayer):
    retList = []
   
    for subtree in listElem:
        subList = []
        for elem in subtree:
            if elem[currPlayer] >= threshold:
                subList.append(elem)
        retList.append(subList)
    return retList


def __ppe_eq_loop(root):
    iterval = root
    while not isinstance(iterval, tuple):
        #print("I am iter: "+str(iterval))
        currPlayer = iterval['p']
        subtree_ix = __maxPayoff(iterval['cand'], currPlayer)
        maxSubPayoff = -1
        for i in range(0, len(iterval['cand'])):
            if i != subtree_ix:
               maxSubPayoff = max(maxSubPayoff, __getMaxList(iterval['cand'][i], currPlayer))
        #clean cand
        cand_list = iterval['c'][subtree_ix]
        if isinstance(cand_list, list):
            return cand_list
        
        cand_list = cand_list['cand']
       
        cand_list = __remove_less_than(cand_list, maxSubPayoff, currPlayer)
        iterval = iterval['c'][subtree_ix]
        iterval['cand'] = cand_list

    return iterval



def __find_ix(root, res_node, curr_path):
    if isinstance(root, list):
        if root == res_node:
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
    annotation = __annotate_nodes(copy_game)
    res_node = __ppe_eq_loop(copy_game)
    path = []
    ppe_ix = __find_ix(game_content, res_node, path)
    # print("PPE is: "+str(res_node))
    # print("PATH: "+str(path))
   
    return [path]


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


def __filter_unavailable(all_ix, avail_ix):
   
    all_ix_clean = []
    for elem in all_ix:
        tupled = tuple(elem)
        if tupled in avail_ix:
            all_ix_clean.append(tupled)

    #put back in original state
    all_ix_clean = np.asarray(all_ix_clean)
   
    return all_ix_clean

def __individual_after_minimax(game_list, available_states):
    # print("Game is: "+str(game_list))
    # print("Avail ix: "+str(available_states))
    avail_ix = set(available_states)
    arr_game = np.asarray(game_list)
    num_players = arr_game.ndim - 1
    num_actions = list(arr_game.shape)[:-1]
    player_choices = [list(range(0, num_actions[x])) for x in range(0, num_players)]

    all_ix = cartprod(player_choices)
    pte_candidates = __filter_unavailable(all_ix, avail_ix)

    maximin_utility = __find_maximin(arr_game, pte_candidates)
    newlist_candidates = []

    for cand in pte_candidates:
        val = arr_game[tuple(cand)]
        if __satisfies_maximin(val, maximin_utility):
            newlist_candidates.append(cand)

    newlist_candidates = [tuple([int(val) for val in x]) for x in newlist_candidates]
    return newlist_candidates
   

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




def __satisfies_maximin(list_val, maximin_utility):
   
    for ix in range(0, len(list_val)):
        if list_val[ix] < maximin_utility[ix]:
            return False
    return True


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



def __spe_eq_rec(game_content, genval, curr_path):
    if isinstance(game_content, list):
        return [{"data":game_content, "id": next(genval), "path": curr_path.copy()}]
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

    avail_action = dict()

    for player in range(0, num_players):
        avail_action[player] = [i for i in range(0, num_actions[player])]

    prev_len = -1
    curr_len = len(available_ix)
    while curr_len and prev_len != curr_len:
      
        for player in range(0, num_players):
            min_list = []
            all_ixs = available_ix  
            for action in range(0, num_actions[player]):
               
                
                for compAction in range(0, num_actions[player]):
                   
                    if action == compAction:
                        continue
                    isCovered = True
                   
                    for curr_ix in available_ix:
                        ix_action = list(curr_ix)
                        oldval = ix_action[player]
                        ix_action[player] = action
                        # print(str(ix_action)+ " and "+str(available_ix))
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


def __iterated_elim_weakly_dominated_actions(game_list):
    '''
    IDIP based on set of not strictly dominated actions
    IMPORTANT: This method does not take into account mixed strategies but only pure strategies
    '''


    arr_game = np.asarray(game_list)
    num_players = arr_game.ndim - 1
    num_actions = list(arr_game.shape)[:-1]
    maximin_utility = []

    available_ix = cartprod([list(range(0, num_actions[x])) for x in range(0, num_players)])
    available_ix = [tuple(listed) for listed in available_ix]

    avail_action = dict()

    for player in range(0, num_players):
        avail_action[player] = [i for i in range(0, num_actions[player])]

    prev_len = -1
    curr_len = len(available_ix)
    while curr_len and prev_len != curr_len:
      
        for player in range(0, num_players):
           
            min_list = []
            all_ixs = available_ix  
            for action in range(0, num_actions[player]):
              

                for compAction in range(0, num_actions[player]):
                  
                    if action == compAction:
                        continue
                   
                   
                    isCovered = True
                    init = True
                    index_eq = []
                    index_lt = []
                    islt = 0
                    isEq = 0
                    # print("available_ix "+str(available_ix))
                    #action iter
                   
                    for counteract in range(0, num_actions[abs(1-player)]):
                        action_ix = [0,0]
                        action_ix[player] = action
                        action_ix[abs(1-player)] = counteract
                       
                        if tuple(action_ix) not in available_ix:
                            continue

                        # print("Action_ix "+str(action_ix))
                       
                        counteract_ix = [0,0]
                        counteract_ix[player] = compAction
                        counteract_ix[abs(1-player)] = counteract
                        # print("counteract_ix "+str(counteract_ix))
                        if tuple(counteract_ix) not in available_ix:
                            continue
                        
                        init = False
                        val_action = arr_game[tuple(action_ix)][player]
                        val_compaction = arr_game[tuple(counteract_ix)][player]
                        # print("val action "+ str(val_action)+" and "+ str(val_compaction))
                        if val_action > val_compaction:

                            isCovered = False
                            # print("action gt compaction")
                            break
                        if val_action < val_compaction:
                            islt += 1
                            index_lt.append(tuple(action_ix))
                        if val_action == val_compaction:
                            index_eq.append(tuple(action_ix))
                            isEq += 1
                           
                    if isCovered  and islt == 1 and num_actions[player] > 1:

                        break

                # print("Is it covered? "+str(isCovered))
                if isCovered and islt == 1 and num_actions[player] > 1:
                  
                    toDel = []
                    for index_tr in index_lt:
                        toDel.append(tuple(index_tr))
                    available_ix = list(set(available_ix) - set(toDel))

                elif isCovered and isEq == 0 and num_actions[player] > 1:
                  
                    toDel = []
                    for index_tr in index_lt:
                        toDel.append(tuple(index_tr))
                    available_ix = list(set(available_ix) - set(toDel))
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
        if ____satisfies_maximin(val, maximin_utility):
            newlist_candidates.append(cand)

    newlist_candidates = [tuple([int(val) for val in x]) for x in newlist_candidates]
    return newlist_candidates


def ____satisfies_maximin(list_val, maximin_utility):
    for ix in range(0, len(list_val)):
        if list_val[ix] < maximin_utility[ix]:
            return False
    return True


def __find_maximax(arr_game, pte_candidates):
    num_players = arr_game.ndim - 1
    num_actions = list(arr_game.shape)[:-1]
    maximin_utility = []
    for player in range(0, num_players):
        # for each action of the player, fix it and find min of all
        # then find the max across actions for each player
        max_list = []
        max_myplayer_list = []
        for action in range(0, num_actions[player]):
            maxval = -1  # arbitrary large number
          
            maxmyplayer = -1
            for state in pte_candidates:#list(filter(lambda x: x[player] == action, pte_candidates)):
                if state[player] != action:
                    continue
                actualState = state
                ixVal = tuple(actualState)
                selectedVal = arr_game[ixVal][abs(1-player)] #opponent
                myPlayerVal = arr_game[ixVal][player]
                if selectedVal >= maxval:
                    maxval = selectedVal
                    maxmyplayer = myPlayerVal
                   

            if maxval != -1:
                max_list.append(maxval)
                max_myplayer_list.append(maxmyplayer)
        
        maximin_utility.append(max(max_myplayer_list))
    return maximin_utility

def __find_secondminimin(arr_game, pte_candidates):
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
        #find second min
        selectedVal = 0
        sortedmin = sorted(min_list)
        if len(sortedmin) == 1:
            selectedVal = sortedmin[0]
        else:
            selectedVal = sortedmin[1]

        maximin_utility.append(selectedVal)
    return maximin_utility


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



