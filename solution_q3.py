import heapq
import math
import sys
from itertools import count
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "input.txt"


# Boat passenger combinations
MOVES = [
    (0, 1), # 1 cannibal
    (0, 2), # 2 cannibals
    (1, 1), # 1 missionary & 1 cannibal
    (1, 0), # 1 missionary
    (2, 0)  # 2 missionaries
]

def read_input(filename=INPUT_FILE):

    with open(filename, "r") as file:
        line = file.readline().strip()

    values = [value.strip() for value in line.split(",")]

    if len(values) != 5:
        raise ValueError(
            "Input must contain 5 values: "
            "M_left, C_left, M_right, C_right, Boat"
        )

    state = (
        int(values[0]),
        int(values[1]),
        int(values[2]),
        int(values[3]),
        values[4].upper()
    )

    if not is_valid(state):
        raise ValueError("Initial state is invalid.")

    return state

# Does state satisfy the project rules?
def is_valid(state):

    m_left, c_left, m_right, c_right, boat = state

    # Population count can't be negative
    if min(m_left, c_left, m_right, c_right) < 0:
        return False

    # We need 3 missionaries
    if m_left + m_right != 3:
        return False

    # We need 3 cannibals
    if c_left + c_right != 3:
        return False

    # Cannibals can't outnumber missionaries on the left
    # only if missionaries are present
    if m_left > 0 and c_left > m_left:
        return False

    # Cannibals can't outnumber missionaries on the right
    # only if missionaries are present
    if m_right > 0 and c_right > m_right:
        return False

    # Boat needs to be on a side
    if boat not in ("L", "R"):
        return False

    return True


# Goal test
def is_goal(state):

    m_left, c_left, m_right, c_right, boat = state

    return (
        m_left == 0
        and c_left == 0
        and m_right == 3
        and c_right == 3
    )


# Cost Model A
def cost_model_a(missionaries, cannibals):

    return 2 * missionaries + cannibals


# Generate all valid successor states
# [
#     (new_state, step_cost),
#     ...
# ]
def get_successors(state):
    m_left, c_left, m_right, c_right, boat = state

    successors = []

    for missionaries, cannibals in MOVES:

        # Left to right
        if boat == "L":

            # Can't move passengers who aren't on left
            if missionaries > m_left:
                continue

            if cannibals > c_left:
                continue

            new_state = (
                m_left - missionaries,
                c_left - cannibals,
                m_right + missionaries,
                c_right + cannibals,
                "R"
            )

        # Right to left
        else:

            # Can't move passengers who aren't on right
            if missionaries > m_right:
                continue

            if cannibals > c_right:
                continue

            new_state = (
                m_left + missionaries,
                c_left + cannibals,
                m_right - missionaries,
                c_right - cannibals,
                "L"
            )

        # Only include legal states
        if is_valid(new_state):

            step_cost = cost_model_a(
                missionaries,
                cannibals
            )

            successors.append(
                (new_state, step_cost)
            )

    return successors



# Heuristic 1
# h1(s) = 2*M_left + C_left
def h1(state):

    m_left, c_left, _, _, _ = state

    return 2 * m_left + c_left


# Heuristic 2
# h2(s) = ceiling((2*M_left + C_left) / 3)
def h2(state):

    m_left, c_left, _, _, _ = state

    passenger_weight_remaining = (
        2 * m_left + c_left
    )

    return math.ceil(
        passenger_weight_remaining / 3
    )


# Heuristic 3
# Starts with h1, which accounts for the minimum cost of getting
# everyone currently on the left across at least once.
# Then adds a lower bound for unavoidable return trips.
def h3(state):

    m_left, c_left, _, _, boat = state

    # Number of people still on left
    people_left = m_left + c_left

    # h1 base cost
    base_cost = 2 * m_left + c_left

    # Goal state
    if people_left == 0:
        return 0

    # If boat is on the left:
    # First trip can move at most 2 people.
    # Every additional progress step requires
    # the boat to return.
    if boat == "L":

        minimum_returns = max(
            0,
            people_left - 2
        )

    # If boat is on the right:
    # The boat needs to get to the left first.
    # This creates an even larger lower bound.
    else:

        minimum_returns = people_left

    # A returning passenger costs at least 1 going back,
    # and that passenger must eventually cross right again.
    #
    # Therefore each unavoidable return contributes at least
    # 2 additional cost units.
    return (
        base_cost
        + 2 * minimum_returns
    )


# Reconstruct path once the goal is reached
def reconstruct_path(parent, goal_state):

    path = []

    current = goal_state

    while current is not None:

        path.append(current)

        current = parent[current]

    path.reverse()

    return path


# A* SEARCH
def a_star(start_state, heuristic):

    tie_breaker = count()

    # Priority queue entries:
    # f(n), g(n), tie_breaker, state
    # f(n) = g(n) + h(n)
    queue = []

    start_g = 0
    start_h = heuristic(start_state)
    start_f = start_g + start_h

    heapq.heappush(
        queue,
        (
            start_f,
            start_g,
            next(tie_breaker),
            start_state
        )
    )

    # Cheapest g-cost
    best_g = {
        start_state: 0
    }

    parent = {
        start_state: None
    }

    node_expansions = 0

    while queue:

        (
            current_f,
            current_g,
            _,
            current_state
        ) = heapq.heappop(queue)

        # Ignore old heap entries
        if current_g != best_g.get(current_state):
            continue

        # Goal test
        if is_goal(current_state):

            path = reconstruct_path(
                parent,
                current_state
            )

            return (
                path,
                current_g,
                node_expansions
            )

        node_expansions += 1

        successors = get_successors(
            current_state
        )

        for child_state, step_cost in successors:

            # Actual cost from start to child
            new_g = current_g + step_cost

            # Compare costs
            if new_g < best_g.get(
                child_state,
                float("inf")
            ):

                best_g[child_state] = new_g

                parent[child_state] = current_state

                # Estimate remaining cost
                new_h = heuristic(
                    child_state
                )

                # A* evaluation function
                new_f = new_g + new_h

                heapq.heappush(
                    queue,
                    (
                        new_f,
                        new_g,
                        next(tie_breaker),
                        child_state
                    )
                )

    return None, None, node_expansions


# Printing state
def format_state(state):

    m_left, c_left, m_right, c_right, boat = state

    return (
        f"({m_left}, {c_left}, "
        f"{m_right}, {c_right}, {boat})"
    )

# Complete path
def format_path(path):

    if path is None:
        return "No solution"

    return " -> ".join(
        format_state(state)
        for state in path
    )


# Run A* using a specific heuristic
def run_heuristic(
    start_state,
    heuristic_name,
    heuristic_function
):

    path, total_cost, expansions = a_star(
        start_state,
        heuristic_function
    )

    print(
        f"The solution of Q3.1 "
        f"({heuristic_name}) is:"
    )

    if path is None:

        print(
            "Solution Path: No solution found"
        )

        print(
            "Total cost = N/A"
        )

    else:

        print(
            "Solution Path:",
            format_path(path)
        )

        print(
            f"Total cost = {total_cost}"
        )

    print(
        f"Number of node expansions = "
        f"{expansions}"
    )

    print()


# Solution 3
def main():

    try:

        start_state = read_input()

    except FileNotFoundError:

        print(
            "Error: input.txt was not found."
        )

        return

    except ValueError as error:

        print(
            f"Error reading input.txt: {error}"
        )

        return

    if len(sys.argv) == 1:

        run_heuristic(
            start_state,
            "Heuristic 1",
            h1
        )

        run_heuristic(
            start_state,
            "Heuristic 2",
            h2
        )

        run_heuristic(
            start_state,
            "Heuristic 3",
            h3
        )

    else:

        selected = sys.argv[1].upper()

        if selected in ("H1", "1"):

            run_heuristic(
                start_state,
                "Heuristic 1",
                h1
            )

        elif selected in ("H2", "2"):

            run_heuristic(
                start_state,
                "Heuristic 2",
                h2
            )

        elif selected in ("H3", "3"):

            run_heuristic(
                start_state,
                "Heuristic 3",
                h3
            )

        else:

            print("Usage:")
            print(
                "  python3 solution_q3.py"
            )
            print(
                "  python3 solution_q3.py H1"
            )
            print(
                "  python3 solution_q3.py H2"
            )
            print(
                "  python3 solution_q3.py H3"
            )


main()