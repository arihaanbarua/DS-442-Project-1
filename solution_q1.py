
# Actions: every boat load of passengers in the format: (missionaries, cannibals) with 1 or 2 people
ACTIONS = [(0, 1), (0, 2), (1, 1), (1, 0), (2, 0)]
 
 
#Format: M_left, C_left, M_right, C_right, Boat
#where M_left and C_left are the number of missionaries and cannibals on the left bank,
#M_right and C_right are the number of missionaries and cannibals on the right bank,
#and Boat is either L (left) or R (right)
def read_initial_state(filename="input.txt"):
    with open(filename) as input_file:
        input_values = [value.strip() for value in input_file.readline().split(",")]
 
    m_left, c_left, m_right, c_right = [int(count) for count in input_values[:4]]
    boat_side = input_values[4].upper()
 
    return (m_left, c_left, m_right, c_right, boat_side)
 
 
# Checking if the state is legal according to specified constraints
def check_legal_state(state):
    m_left, c_left, m_right, c_right, boat_side = state
  
    # passengers on boat cannot be negative and must sum to 3
    if min(m_left, c_left, m_right, c_right) < 0:
        return False
    if m_left + m_right != 3 or c_left + c_right != 3:
        return False
      
    # making sure cannibals aren't the majority on any bank with missionnaries 
    if m_left > 0 and c_left > m_left:
        return False
    if m_right > 0 and c_right > m_right:
        return False
 
    return boat_side in ("L", "R")
 
 
# Successor function finds all reachable legal states
def get_successors(state):
    m_left, c_left, m_right, c_right, boat_side = state
 
    #-1 if people leave the left bank, +1 for people arrive on the left bank
    left_bank_change = -1 if boat_side == "L" else 1
    new_boat_side = "R" if boat_side == "L" else "L"
 
    successors = []
    for missionaries_moved, cannibals_moved in ACTIONS:
        successor = (
            m_left + left_bank_change * missionaries_moved,
            c_left + left_bank_change * cannibals_moved,
            m_right - left_bank_change * missionaries_moved,
            c_right - left_bank_change * cannibals_moved,
            new_boat_side
        )
        #checking to make sure the state is legal
        if check_legal_state(successor):
            successors.append(successor)
 
    return successors
 
 
# Depth First Search
def depth_first_search(initial_state):
    # Each entry is in the format : (state, path_to_state)
    possible_paths_stack = [(initial_state, [initial_state])]
    expanded_states = set()
    node_expansions = 0
 
    while possible_paths_stack:
        state, current_path = possible_paths_stack.pop()
 
        if state in expanded_states:
            continue
 
        #final state ends in all passengers at the right bank (Missonnaries left, cannibals left, missionaries right, cannibals right)
        if state[:4] == (0, 0, 3, 3):
            return current_path, node_expansions
 
        expanded_states.add(state)
        node_expansions += 1
 
        # Pushing in reverse
        for successor in reversed(get_successors(state)):
            if successor not in expanded_states:
                possible_paths_stack.append((successor, current_path + [successor]))
 
    return None, node_expansions
 
 
# Breadth First Search
def breadth_first_search(initial_state):
    possible_paths_queue = [(initial_state, [initial_state])]
    states_in_tree = {initial_state}
    node_expansions = 0
 
    while possible_paths_queue:
        state, current_path = possible_paths_queue.pop(0)
 
        # testing to make sure everyone is on the right bank
        if state[:4] == (0, 0, 3, 3):
            return current_path, node_expansions
 
        node_expansions += 1
 
        for successor in get_successors(state):
            if successor not in states_in_tree:
                states_in_tree.add(successor)
                possible_paths_queue.append((successor, current_path + [successor]))
 
    return None, node_expansions
 
 
#Printing solutions
def print_solution(question_label, solution_path, node_expansions):
    print(f"The solution of {question_label} is:")
 
    if solution_path is None:
        print("Solution Path: No solution found")
        print("Total cost = N/A")
    else:
        formatted_states = []
        for m_left, c_left, m_right, c_right, boat_side in solution_path:
            formatted_states.append(f"({m_left}, {c_left}, {m_right}, {c_right}, {boat_side})")
 
        total_cost = len(solution_path) - 1  #cost = number of actions
 
        print("Solution Path: " + " -> ".join(formatted_states))
        print(f"Total cost = {total_cost}")
 
    print(f"Number of node expansions = {node_expansions}")
 
 

initial_state = read_initial_state()
 
if not check_legal_state(initial_state):
    print("Error: initial state in input.txt is invalid.")
else:
    dfs_path, dfs_expansions = depth_first_search(initial_state)
    print_solution("Q1.1.a (DFS)", dfs_path, dfs_expansions)
 
    print()
 
    bfs_path, bfs_expansions = breadth_first_search(initial_state)
    print_solution("Q1.1.b (BFS)", bfs_path, bfs_expansions)
 
