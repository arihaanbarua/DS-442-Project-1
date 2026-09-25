import heapq
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

    m_left = int(values[0])
    c_left = int(values[1])
    m_right = int(values[2])
    c_right = int(values[3])
    boat = values[4].upper()

    state = (
        m_left,
        c_left,
        m_right,
        c_right,
        boat
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
# Missionary = 2 units
# Cannibal   = 1 unit
# Cost = 2M + C
def cost_model_a(missionaries, cannibals, direction):
    return 2 * missionaries + cannibals


# Cost Model B
# Left -> Right = 2
# Right -> Left = 1
def cost_model_b(missionaries, cannibals, direction):
    if direction == "L->R":
        return 2
    else:
        return 1


# Generate all valid successor states
# [
#     (new_state, step_cost),
#     ...
# ]
def get_successors(state, cost_function):
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

            direction = "L->R"

        # Right to left
        else:

            # Can't move passengers who aren't on right bank
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

            direction = "R->L"

        # Only include legal states
        if is_valid(new_state):

            step_cost = cost_function(
                missionaries,
                cannibals,
                direction
            )

            successors.append(
                (new_state, step_cost)
            )

    return successors


# Reconstruct path from goal
def reconstruct_path(parent, goal_state):
    path = []

    current = goal_state

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()

    return path


# Uniform Cost Search
def uniform_cost_search(start_state, cost_function):

    tie_breaker = count()

    # Priority queue entries:
    # (g_cost, tie_breaker_number, state)
    queue = []

    heapq.heappush(
        queue,
        (
            0,
            next(tie_breaker),
            start_state
        )
    )

    # Cheapest known cost
    best_g = {
        start_state: 0
    }

    parent = {
        start_state: None
    }

    # Number of states whose successors are actually generated
    node_expansions = 0

    while queue:

        current_cost, _, current_state = heapq.heappop(queue)

        # Ignore old queue entries
        if current_cost != best_g.get(current_state):
            continue

        # Goal test
        if is_goal(current_state):

            path = reconstruct_path(
                parent,
                current_state
            )

            return (
                path,
                current_cost,
                node_expansions
            )

        node_expansions += 1

        successors = get_successors(
            current_state,
            cost_function
        )

        for child_state, step_cost in successors:

            new_cost = current_cost + step_cost

            if new_cost < best_g.get(
                child_state,
                float("inf")
            ):

                best_g[child_state] = new_cost

                parent[child_state] = current_state

                heapq.heappush(
                    queue,
                    (
                        new_cost,
                        next(tie_breaker),
                        child_state
                    )
                )

    return None, None, node_expansions

# Printing states
def format_state(state):
    m_left, c_left, m_right, c_right, boat = state

    return (
        f"({m_left}, {c_left}, "
        f"{m_right}, {c_right}, {boat})"
    )

def format_path(path):

    if path is None:
        return "No solution"

    return " -> ".join(
        format_state(state)
        for state in path
    )


# Run UCS cost model
def run_model(start_state, model_name, cost_function):

    path, total_cost, expansions = uniform_cost_search(
        start_state,
        cost_function
    )

    print(
        f"The solution of Q2.1 "
        f"(UCS, cost model {model_name}) is:"
    )

    if path is None:

        print("Solution Path: No solution found")
        print("Total cost = N/A")

    else:

        print(
            "Solution Path:",
            format_path(path)
        )

        print(
            f"Total cost = {total_cost}"
        )

    print(
        f"Number of node expansions = {expansions}"
    )

    print()


# Solution 2
def main():

    try:
        start_state = read_input("input.txt")

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

        run_model(
            start_state,
            "A",
            cost_model_a
        )

        run_model(
            start_state,
            "B",
            cost_model_b
        )

    else:

        selected_model = sys.argv[1].upper()

        if selected_model == "A":

            run_model(
                start_state,
                "A",
                cost_model_a
            )

        elif selected_model == "B":

            run_model(
                start_state,
                "B",
                cost_model_b
            )

        else:

            print(
                "Usage:"
            )

            print(
                "  python solution_q2.py"
            )

            print(
                "  python solution_q2.py A"
            )

            print(
                "  python solution_q2.py B"
            )


main()