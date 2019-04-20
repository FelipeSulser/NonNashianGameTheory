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


class ExtensiveFormEquilibria:
	def nash(gameDict):
	    nash_ix = find_nash_eq(gameDict['y'])
	    gameDict['N'] = nash_ix
	    return gameDict


	def rationalizability(gameDict):
	    rat_ix = iterated_elim_strictly_dominated_actions(gameDict['y'])
	    gameDict['R'] = rat_ix
	    return gameDict


	def individual(gameDict):
	    individual_ix = individually_rational(gameDict['y'])
	    gameDict['I'] = individual_ix
	    return gameDict 


	def spe(gameDict):
	    spe_ix = get_spe_equilibria(gameDict['y'])
	    tupler = [tuple(subl) for subl in spe_ix]
	    gameDict['S'] = tupler
	    return gameDict



	def spe_eq_rec(game_content, genval, curr_path):
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
	            allCands.append(spe_eq_rec(elem, genval, curr_path))
	            del curr_path[-1]
	            ixpos += 1

	        currPlayer = game_content["p"]
	        maxIxs = findMaxs(allCands, currPlayer)
	        #print("maxs : "+str(maxIxs))
	        maxIxs = find_max_val(maxIxs, currPlayer)

	        #resIx = [{"data": cont, "id": next(genval)} for cont in maxIxs]
	        game_content["cand"] = maxIxs
	        return maxIxs



	def get_spe_equilibria(game_dict):
	    game_content = game_dict
	    spe_eq = []
	    possibilities = []
	    copy_game = copy.deepcopy(game_content)
	    generator = id_generator()
	    path = []
	    res = spe_eq_rec(copy_game, generator, path)
	    
	    res = remove_dup_id(res)
	   
	    return res[0]


	def find_nash_eq(game_list):
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



	def iterated_elim_strictly_dominated_actions(game_list):
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

	def individually_rational(game_list):
	    arr_game = np.asarray(game_list)
	    num_players = arr_game.ndim - 1
	    num_actions = list(arr_game.shape)[:-1]
	    player_choices = [list(range(0, num_actions[x])) for x in range(0, num_players)]

	    all_ix = cartprod(player_choices)
	    pte_candidates = all_ix

	    maximin_utility = find_maximin(arr_game, pte_candidates)
	    newlist_candidates = []

	    for cand in pte_candidates:
	        val = arr_game[tuple(cand)]
	        if satisfies_maximin(val, maximin_utility):
	            newlist_candidates.append(cand)

	    newlist_candidates = [tuple([int(val) for val in x]) for x in newlist_candidates]
	    return newlist_candidates


	def satisfies_maximin(list_val, maximin_utility):
	    for ix in range(0, len(list_val)):
	        if list_val[ix] < maximin_utility[ix]:
	            return False
	    return True


	def find_maximin(arr_game, pte_candidates):
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


	def get_game_by_line(filename):
	    fp = open(filename)
	    for i, line in enumerate(fp):
	        yield (i,line)


	def validate_equilibria(equilibria):
	    if equilibria == "nash" or equilibria == "rationalizability" or equilibria == "individual" or equilibria == "hofstader" or equilibria == "pte" or equilibria == "minimax":
	        return equilibria.title()[0]

	    raise Exception('Error, equilibria not defined. Try: [nash|rationalizability|individual]')


	def hofstader(gameDict):
	    hofstader_ix = find_hofstader_eq(gameDict['y'])
	    gameDict['H'] = hofstader_ix
	    return gameDict

	def pte(gameDict):
	    pte_ix = find_perfectly_transparent_eq(gameDict['y'])
	    gameDict['P'] = pte_ix
	    return gameDict

	def minimax(gameDict):
	    minimax_ix = minimax_rationalizability(gameDict['y'])
	    gameDict['M'] = minimax_ix
	    return gameDict

	def ppe(gameDict):
	    ppe_ix = get_ppe_equilibria(gameDict['y'])
	    tupler = [tuple(subl) for subl in ppe_ix]
	    gameDict['P'] = tupler
	    return gameDict


	def shiffrin(gameDict):
	    shiffrin_ix = get_shiffrin_equilibria(gameDict['y'])
	    gameDict['Sh'] = shiffrin_ix
	    return gameDict



	def get_indices(root, genval):
	    if isinstance(root, list):
	        return
	    ix = 0
	    for ix in range(0, len(root['c'])):
	        if isinstance(root['c'][ix], list):
	            root['c'][ix] = {"id": next(genval), "data": root['c'][ix]}
	        else:
	            get_indices(root['c'][ix], genval)

	def annotate_nodes(root, genval):
	    
	    if isinstance(root, dict) and "id" in root:
	        return [root]

	    #for all children, annotate them flat
	    candidates = []
	    for child in root['c']:
	        currSublist = annotate_nodes(child, genval)
	        if not (isinstance(currSublist[0], dict) and "id" in currSublist[0]):
	            #flatten it
	            currSublist = [x for y in currSublist for x in y]
	        candidates.append(currSublist)
	    root["cand"] = candidates
	    return candidates




	def getMaxList(list_outcomes, currPlayer):
	    maxval = -1
	    id_loc = -1
	    for outcome in list_outcomes:
	        if outcome["data"][currPlayer] >= maxval:
	            maxval = outcome["data"][currPlayer]
	            id_loc = outcome["id"]
	    return (maxval, id_loc)


	def maxPayoff(list_trees_payoff, currPlayer):
	    maxval = -1
	    curr_max_ix = 0
	    #print("MaxPayoff: "+str(list_trees_payoff)+" currPlayer: "+str(currPlayer))
	    for i in range(0, len(list_trees_payoff)):
	        currMax, idmax = getMaxList(list_trees_payoff[i], currPlayer)
	        if currMax >= maxval:
	            curr_max_ix = i
	            maxval = currMax
	    return curr_max_ix


	def remove_less_than(listElem, threshold, currPlayer):
	    retList = []
	   
	    for subtree in listElem:
	        subList = []
	        for elem in subtree:
	            if elem["data"][currPlayer] >= threshold:
	                subList.append(elem)
	        retList.append(subList)
	    return retList


	def dup_max_values(cands, currPlayer):
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

	def duplicated(cands, currPlayer):
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


	def ppe_eq_loop(root):
	    iterval = root
	    while not (isinstance(iterval, dict) and "id" in iterval):
	       
	        currPlayer = iterval['p']
	        subtree_ix = maxPayoff(iterval['cand'], currPlayer)
	        maxSubPayoff = -1
	        maxSubPayoff_ix = -1
	        if duplicated(iterval['cand'], currPlayer):
	          
	            return dup_max_values(iterval['cand'], currPlayer)

	        for i in range(0, len(iterval['cand'])):
	            if i != subtree_ix:
	                currMax, currMaxIx = getMaxList(iterval['cand'][i], currPlayer)
	                if currMax >= maxSubPayoff:
	                    maxSubPayoff = currMax
	                    maxSubPayoff_ix = currMaxIx
	                #maxSubPayoff = max(maxSubPayoff, getMaxList(iterval['cand'][i], currPlayer))
	        #clean cand
	        cand_list = iterval['c'][subtree_ix]
	        if isinstance(cand_list, dict) and "id" in cand_list:
	            return [cand_list]
	        
	        cand_list = cand_list['cand']
	       
	        cand_list = remove_less_than(cand_list, maxSubPayoff, currPlayer)
	        iterval = iterval['c'][subtree_ix]
	        iterval['cand'] = cand_list
	    
	    return [tuple(iterval)] #returns here if no dups


	def encapsulate_find_ix(root, res_node, curr_path):
	    all_paths = []
	    
	    for res_val in res_node:

	        curr_path = []
	        ppe_ix = find_ix(root, res_val, curr_path)
	        all_paths.append(curr_path)
	    return all_paths



	def find_ix(root, res_node, curr_path): 
	   
	    if isinstance(root, dict) and "id" in root:
	        if tuple(root["data"]) == tuple(res_node["data"]) and root["id"] == res_node["id"]:
	            return curr_path
	        else:
	            return None
	    for pos_ix in range(0, len(root['c'])):
	        curr_path.append(pos_ix)
	       
	        if find_ix(root['c'][pos_ix], res_node, curr_path):
	            return curr_path
	       
	        del curr_path[-1]
	    return None

	def get_ppe_equilibria(game_dict):
	    game_content = game_dict
	    ppe_eq = []
	    copy_game = copy.deepcopy(game_content)
	    generator = id_generator()
	    get_indices(copy_game, generator)
	   
	    annotation = annotate_nodes(copy_game, generator)
	   
	    res_node = ppe_eq_loop(copy_game)
	   
	    path = []
	    path = encapsulate_find_ix(copy_game, res_node, path)
	    # print("PPE is: "+str(res_node))
	    # print("PATH: "+str(path))
	   
	    return path

	def shiffrin_annotation(root, parent, threshold=None):
	    if isinstance(root, list):
	        return tuple(root)

	    #for all children, annotate them flat
	    candidates = []
	    allOutcomes = True
	    mapDone = []
	    for child in root['c']:
	        currSublist = shiffrin_annotation(child, root)

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

	def all_done(bitList):
	    for elem in bitList:
	        if not elem:
	            return False
	    return True

	def selfish_best(list_out, player):
	    maxIx = -1
	    maxVal = -1
	    for i in range(0, len(list_out)):
	        if list_out[i][player] >= maxVal:
	            maxVal = list_out[i][player]
	            maxIx = i
	    return list_out[maxIx]

	def shiffrin_eq_rec(root, top_Q=(-1,-1)):
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
	                    newQ = shiffrin_eq_rec(child)
	                    listQ.append(newQ)
	                else:
	                    newQ = shiffrin_eq_rec(child, bestQ)
	                    listQ.append(newQ)
	                   
	                firstTime[ix] = False

	                if newQ == bestQ:
	                    mapDone[ix] = True
	       
	        newQ = selfish_best(listQ, currPlayer)
	        if newQ[currPlayer] >= bestQ[currPlayer] and newQ[0] >= top_Q[0] and newQ[1] >= top_Q[1]:
	            bestQ = newQ
	        else: 
	            if top_Q[0] > bestQ[0] and top_Q[1] > bestQ[1]:
	                bestQ = top_Q
	                        #mapDone = [False for x in root['c']] #reset the bitmap
	                    
	                #print("At node: "+str(root)+" NEWQ="+str(newQ)+" topQ="+str(top_Q)+" and bestQ="+str(bestQ))
	        
	        allDone = all_done(mapDone)

	    # all Done, return selfish the largest
	   
	    return bestQ


	def get_shiffrin_ix(root, res_node, curr_path): 
	   
	    if isinstance(root, list):
	        if tuple(root) == tuple(res_node):
	            return curr_path
	        else:
	            return None
	    for pos_ix in range(0, len(root['c'])):
	        curr_path.append(pos_ix)
	       
	        if get_shiffrin_ix(root['c'][pos_ix], res_node, curr_path):
	            return curr_path
	       
	        del curr_path[-1]
	    return None

	def get_shiffrin_equilibria(game_dict):
	    game_content = copy.deepcopy(game_dict)
	    shiffrin_eq = []
	    #shiffrin_annotation(game_content, None)
	    #print(game_content)
	    outcome = shiffrin_eq_rec(game_content)
	    outcome_ix = []

	    outcome_ix = get_shiffrin_ix(game_content, outcome, outcome_ix)

	    return [tuple(outcome_ix)]



	def gen_diagonal_states(num_actions):
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


	def find_hofstader_eq(game_list):
	    arr_game = np.asarray(game_list)
	    num_players = arr_game.ndim - 1
	    num_actions = list(arr_game.shape)[:-1]
	    if len(set(num_actions)) != 1:
	        # all symmetric games have same actions for each player
	        raise ValueError("Error: Game is not symmetric")

	    list_max_ix = []

	    allPossibleStates = gen_diagonal_states(num_actions)
	    currmax = -1
	    for ix_val in allPossibleStates:
	        currval = arr_game[ix_val][0]  # symmetric, all equal in diagonal
	        if currval > currmax:
	            currmax = currval
	            list_max_ix = [ix_val]

	        elif currval == currmax:
	            list_max_ix.append(ix_val)

	    return list_max_ix


	def minimax_rationalizability(game_list):
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



	def find_maximin(arr_game, pte_candidates):
	    num_players = arr_game.ndim - 1
	    num_actions = list(arr_game.shape)[:-1]
	    maximin_utility = []
	    # print("Candidates: "+str(pte_candidates))
	    for player in range(0, num_players):
	        # for each action of the player, fix it and find min of all
	        # then find the max across actions for each player
	        # print("player: "+str(player))
	        min_list = []
	        for action in range(0, num_actions[player]):
	            minval = 9999  # arbitrary large number
	            # print("find min for : "+str(action))
	            for state in pte_candidates:#list(filter(lambda x: x[player] == action, pte_candidates)):
	                if state[player] != action:
	                    continue
	                actualState = state
	                ixVal = tuple(actualState)
	                selectedVal = arr_game[ixVal][player]
	                # print("Ix: "+str(ixVal) + " value: "+str(selectedVal))
	                minval = min(minval, selectedVal)

	            # print("minval "+str(minval))
	            if minval != 9999:
	                min_list.append(minval)

	        maximin_utility.append(max(min_list))
	    return maximin_utility


	def satisfies_maximin(list_val, maximin_utility):
	   
	    for ix in range(0, len(list_val)):
	        if list_val[ix] < maximin_utility[ix]:
	            return False
	    return True


	def find_perfectly_transparent_eq(game_list):
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
	        maximin_utility = find_maximin(arr_game, pte_candidates)

	        newlist_candidates = []

	        for cand in pte_candidates:
	            val = arr_game[tuple(cand)]
	            if satisfies_maximin(val, maximin_utility):
	                newlist_candidates.append(cand)
	        pte_candidates = newlist_candidates
	        iterval+=1

	    pte_candidates = [tuple([x for x in elem]) for elem in pte_candidates]
	    return pte_candidates 


	def get_equilibria(equilibria, game_list):
	    if equilibria == "H":
	        return find_hofstader_eq(game_list)
	    elif equilibria == "P":
	        return find_perfectly_transparent_eq(game_list)
	    elif equilibria == "M":
	        return minimax_rationalizability(game_list)
	    elif equilibria == "N":
	        return find_nash_eq(game_list)
	    elif equilibria == "R":
	        return iterated_elim_strictly_dominated_actions(game_list)
	    elif equilibria == "I":
	        return individually_rational(game_list)

	    raise Exception('Error, equilibria not defined. Try: [hofstader|pte|minimax]')


class ExtensiveFormGenerator:
	max_num_actions_1 = 2
	max_num_actions_2 = 2
	min_outcome_val = 0
	max_outcome_val = max_num_actions_1* max_num_actions_2 


	def cartprod(arrays):
	    N = len(arrays)
	    return transpose(meshgrid(*arrays, indexing='ij'),
	                     roll(arange(N + 1), -1)).reshape(-1, N)


	# function to generate all permutations of a list
	def perm(xs) :
	    if xs == [] :
	        yield []
	    for x in xs :
	        ys = [y for y in xs if not y==x]
	        for p in perm(ys) :
	            yield ([x] + p)
	 
	#function to generate all possible assignments of <item> balls into <ks> bins
	def combinations_generator(items, ks):
		if len(ks) == 1:
			for c in itertools.combinations(items, ks[0]):
				yield (c)

		else:
			for c_first in itertools.combinations(items, ks[0]):
				items_remaining= set(items) - set(c_first)
				for c_other in combinations_generator(items_remaining, ks[1:]):
					yield (c_first) + c_other



	def generate_random_tree(nodelist=[], idx=0, parent=None, fixed_children=True, depth=0, max_children=2, max_depth=2):
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
	        	gen_leaves = generate_random_tree(nodelist, len(nodelist), idx + i, depth + 1, max_children, max_depth)
	        	
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


	def gen_random_dict(max_num_actions, max_depth, avoid_ties=True, curr_depth=0, leaf_threshold=0.2):
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
	            act_val, subgame = gen_random_dict(max_num_actions,max_depth, avoid_ties=avoid_ties, curr_depth=curr_depth+1)
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
	            act_val, subgame = gen_random_dict(max_num_actions,max_depth, avoid_ties=avoid_ties, curr_depth=curr_depth+1)
	            sumval += act_val
	            new_dict["c"].append(subgame)


	        return (sumval,new_dict)
	   
	def modify_leaves(d, action_set, avoid_ties=False):
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
	            d[i] = modify_leaves(d[i], action_set, avoid_ties=avoid_ties)
	        return d
	    elif isinstance(d, dict):
	        for k, v in d.items():
	            d[k] = modify_leaves(v, action_set, avoid_ties=avoid_ties)
	        return d
	    else:
	        return d
	    


	def gen_dict_content(num_actions, depth, action_set, avoid_ties=True, curr_depth=0, curr_player=0):
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
	            new_dict["c"].append(gen_dict_content(num_actions,depth, action_set, avoid_ties=avoid_ties, curr_depth=curr_depth+1, curr_player=new_player))
	        
	        return new_dict


	    return None

	def sample_extensive_form_random_dup(max_num_actions, max_depth):
	    num_players = 2
	    total_num_actions = max_num_actions ** max_depth
	    action_set = [list(range(0,total_num_actions)) for act in range(0,num_players)]
	    dictval = []
	    while dictval == []:
	        num_leaves, dictval =  gen_random_dict(max_num_actions,max_depth, avoid_ties=False, curr_depth=0)
	    action_set = [list(range(0,num_leaves)) for act in range(0, num_players)]

	    #traverse dict and put outcomes
	    modify_leaves(dictval, action_set, avoid_ties=False)

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
	        num_leaves, dictval =  gen_random_dict(max_num_actions,max_depth, avoid_ties=True, curr_depth=0)
	    action_set = [list(range(0,num_leaves)) for act in range(0, num_players)]

	    #traverse dict and put outcomes 
	    modify_leaves(dictval, action_set, avoid_ties=True)
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
	    ret_dict['y'] = gen_dict_content(num_actions,depth,action_set, avoid_ties=False, curr_depth=0, curr_player=0)
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
	    ret_dict['y'] = gen_dict_content(num_actions,depth,action_set, avoid_ties=True, curr_depth=0, curr_player=0)
	    return ret_dict

	def annotate_nodes(root, id_generator):
	    if isinstance(root, list):
	        return

	    root["id"] = next(id_generator)
	   
	    for child in root['c']:
	        annotate_nodes(child, id_generator)
	    return

	def find_pure_strategies(root, dictval, num_actions_dict):
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
	        find_pure_strategies(child, dictval, num_actions_dict)
	    return

	def cartprod_pure_strategies(dict_strategies):
	    list_of_lists = []
	   
	    for k,v in dict_strategies.items():
	        # v contains a list of dict, expand it
	        for  k2, v2 in v.items():
	            list_of_lists.append(v2)
	    
	    return  cartprod(list_of_lists)


	def find_action(strategy, currid):
	    for elem in strategy:
	        if elem["id"] == currid:
	            return elem["action"]
	    return -1

	def get_nf_list(root, ix_strategies, num_actions):
	    nf_list = []
	    id_map = []
	    starting_player = root['p']
	    for strategy in ix_strategies:
	        itertree = root
	        while not (isinstance(itertree, dict) and "idleaf" in itertree):
	            currid = itertree["id"]
	            action = find_action(strategy, currid)
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

	def get_indices(root, genval):
	    if isinstance(root, list):
	        return
	    ix = 0
	    for ix in range(0, len(root['c'])):
	        if isinstance(root['c'][ix], list):
	            root['c'][ix] = {"idleaf": next(genval), "data": root['c'][ix]}
	        else:
	            get_indices(root['c'][ix], genval)

	def convert_two_player_ef_to_nf(game):
	    generator = id_generator()
	    game_content = copy.deepcopy(game["y"])
	    annotate_nodes(game_content, generator)
	    get_indices(game_content, id_generator())
	   
	    dict_val = dict()
	    num_actions_dict = dict()
	   
	    find_pure_strategies(game_content, dict_val, num_actions_dict)
	   
	    list_strategies = cartprod_pure_strategies(dict_val)
	   
	    game_matrix, id_map = get_nf_list(game_content, list_strategies, num_actions_dict)

	    res_dict = dict()
	    res_dict["x"] = game["x"]
	    res_dict["z"] = 2
	    res_dict["y"] = game_matrix
	    res_dict["idmap"] = id_map
	    res_dict["annotated"] = game_content
	    return res_dict


	def find_path(root, idval, path):
	    if isinstance(root, dict) and "idleaf" in root:
	       
	        if root['idleaf'] == idval:
	            return True
	        else:
	            return False

	    children = root['c']
	    for ix_child in range(0, len(children)):
	        path.append(ix_child)
	        if find_path(root['c'][ix_child], idval, path):
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
	        find_path(tree_node, idval, path)
	        res_list.append(copy.deepcopy(path))
	    return [tuple(x) for x in res_list]

	def nf_equilibria_to_ef(nf_ix, id_map, annotated_tree,name,  return_dict):
	    list_res = translate_nf_ix_ef(nf_ix, id_map, annotated_tree)
	    return_dict[name] = list(set(list_res))
	    return return_dict

	def get_ix_currpos(player_list, pos, ixval):
	    posmatch = 0
	    for currIx in player_list:
	        if currIx[pos] == ixval:
	            return posmatch
	        posmatch += 1
	    return posmatch

	def annotate_nf_data(game):
	    nf_game = convert_two_player_ef_to_nf(game)
	    print(np.asarray(nf_game['y']).shape)
	    pte(nf_game)
	    nash(nf_game)
	    minimax(nf_game)
	    individual(nf_game)
	    rationalizability(nf_game)
	    game = nf_equilibria_to_ef(nf_game['N'], nf_game['idmap'], nf_game['annotated'], "NNF", game)
	    game = nf_equilibria_to_ef(nf_game['P'], nf_game['idmap'], nf_game['annotated'], "PNF", game)
	    game = nf_equilibria_to_ef(nf_game['M'], nf_game['idmap'], nf_game['annotated'], "MNF", game)
	    game = nf_equilibria_to_ef(nf_game['I'], nf_game['idmap'], nf_game['annotated'], "INF", game)
	    game = nf_equilibria_to_ef(nf_game['R'], nf_game['idmap'], nf_game['annotated'], "RNF", game)
	    return game
	 

	def get_outcome(tuple_ix, game_dict, pos=0):
	    if pos >= len(tuple_ix):
	        return tuple(game_dict)

	    return get_outcome(tuple_ix, game_dict['c'][tuple_ix[pos]], pos+1)


	def pareto_dominates(game_dict, ix_first, ix_second):
	    list_ix_first = game_dict[ix_first]
	    list_ix_second = game_dict[ix_second]
	    ix_to_outcome_first = [get_outcome(ix, game_dict['y']) for ix in list_ix_first]
	    ix_to_outcome_second = [get_outcome(ix, game_dict['y']) for ix in list_ix_second]
	    for tuple_first in ix_to_outcome_first:
	        isDominant = True
	        for tuple_second in ix_to_outcome_second:
	            if tuple_first[0]  < tuple_second[0] or tuple_first[1] < tuple_second[1]:
	                isDominant = False
	                break
	        if isDominant:
	            return True
	    return False