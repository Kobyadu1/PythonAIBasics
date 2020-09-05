from copy import deepcopy
import sys

# This version is the one to turn in so make sure u remember to comment it per class standard

# Variables that I used globally
visited_puzzles = []
queue_list = []
f_score_list = []
n = 3

# Puzzle Object class that represents the state of the puzzle and stores the g-score,
# movement to get to the current state of the puzzle along with the puzzle list and a link to parent puzzle
class PuzzleObj:
    puzzle_list = [[0]*n]*n
    g_score = 0
    parent_puzzle = None
    move = ""

    def __init__(self, puzzle_list, g_score, parent_puzzle):
        self.puzzle_list = puzzle_list
        self.parent_puzzle = parent_puzzle
        self.g_score = int(g_score)


# Converts the puzzle object into a string to be checked if the state has been visited
def puzzle_check(puzzle_obj):
    ret_str = ""
    for r in range(n):
        for c in range(n):
            ret_str += str(puzzle_obj.puzzle_list[r][c])
    return ret_str


# Prints the puzzle in (NxN) 3x3 style in the console
def print_puzzle(puzzle_obj):
    for r in range(n):
        row_string = ""
        for c in range(n):
            row_string += str(puzzle_obj.puzzle_list[r][c])+"  "
        print(row_string)


# Finds the index of an element in the list in the form of a single integer 0,0 = 0, 0,1 = 1, etc
def index_of(puzzle_obj, element):
    for r in range(n):
        for c in range(n):
            if puzzle_obj.puzzle_list[r][c] == element:
                return int((r * n) + c)


# Calculates the total manhattan distance for the puzzle
def dist(puzzle_obj):
    total_dist = 0
    for r in range(n):
        for c in range(n):
            if puzzle_obj.puzzle_list[r][c] == 0:
                continue
            else:
                row_val = int((puzzle_obj.puzzle_list[r][c]-1) / n)
                col_val = int((puzzle_obj.puzzle_list[r][c]-1) % n)
                total_dist += (abs(r-row_val) + abs(c-col_val))
    return total_dist


# Computes the fscore for the possible states and stores them in the respective lists
def compute_f_score(puzzle_obj, zero_index):
    row_val = int(int(zero_index) / n)
    col_val = int(int(zero_index) % n)
    for x in range(4):
        if x == 0 and (row_val-1) in range(n):
            # Performs the switch - in this case -1 row is up
            temp_puzzle = PuzzleObj(deepcopy(puzzle_obj.puzzle_list),puzzle_obj.g_score+1,puzzle_obj)
            temp = temp_puzzle.puzzle_list[row_val-1][col_val]
            temp_puzzle.puzzle_list[row_val - 1][col_val] = temp_puzzle.puzzle_list[row_val][col_val]
            temp_puzzle.puzzle_list[row_val][col_val] = temp
            # Checks if the new puzzle has been already visited
            if puzzle_check(temp_puzzle) in visited_puzzles:
                continue
            # If not computes for the h & f scores and then addes the state along with f score to the lists
            # Lastly adds the move string to the state
            h_score = dist(temp_puzzle)
            f_score = h_score + (puzzle_obj.g_score+1)
            queue_list.append(temp_puzzle)
            f_score_list.append(f_score)
            temp_puzzle.move = "Up"
            # Repeats for all 4 directions, checking first if its valid
        elif x == 1 and (row_val+1) in range(n):
            temp_puzzle = PuzzleObj(deepcopy(puzzle_obj.puzzle_list),puzzle_obj.g_score+1, puzzle_obj)
            temp = temp_puzzle.puzzle_list[row_val + 1][col_val]
            temp_puzzle.puzzle_list[row_val + 1][col_val] = temp_puzzle.puzzle_list[row_val][col_val]
            temp_puzzle.puzzle_list[row_val][col_val] = temp
            if puzzle_check(temp_puzzle) in visited_puzzles:
                continue
            h_score = dist(temp_puzzle)
            f_score = h_score + (puzzle_obj.g_score+1)
            queue_list.append(temp_puzzle)
            f_score_list.append(f_score)
            temp_puzzle.move = "Down"
        elif x == 2 and (col_val-1) in range(n):
            temp_puzzle = PuzzleObj(deepcopy(puzzle_obj.puzzle_list),puzzle_obj.g_score+1, puzzle_obj)
            temp = temp_puzzle.puzzle_list[row_val][col_val-1]
            temp_puzzle.puzzle_list[row_val][col_val-1] = temp_puzzle.puzzle_list[row_val][col_val]
            temp_puzzle.puzzle_list[row_val][col_val] = temp
            if puzzle_check(temp_puzzle) in visited_puzzles:
                continue
            h_score = dist(temp_puzzle)
            f_score = h_score + (puzzle_obj.g_score+1)
            queue_list.append(temp_puzzle)
            f_score_list.append(f_score)
            temp_puzzle.move = "Left"
        elif x == 3 and (col_val+1) in range(n):
            temp_puzzle = PuzzleObj(deepcopy(puzzle_obj.puzzle_list),puzzle_obj.g_score+1, puzzle_obj)
            temp = temp_puzzle.puzzle_list[row_val][col_val+1]
            temp_puzzle.puzzle_list[row_val][col_val+1] = temp_puzzle.puzzle_list[row_val][col_val]
            temp_puzzle.puzzle_list[row_val][col_val] = temp
            if puzzle_check(temp_puzzle) in visited_puzzles:
                continue
            h_score = dist(temp_puzzle)
            f_score = h_score + (puzzle_obj.g_score+1)
            queue_list.append(temp_puzzle)
            f_score_list.append(f_score)
            temp_puzzle.move = "Right"


# Checks if the starting puzzle is able to be solved
def is_solveable(puzzle_obj):
    inversions = 0
    zero_row_index = int(int(index_of(puzzle_obj, 0)) / n)
    for i in range(n*n):
        row_val = int(i / n)
        col_val = int(i % n)
        if puzzle_obj.puzzle_list[row_val][col_val] == 0:
            continue
        for j in range(i+1, n*n):
            r = int(j / n)
            c = int(j % n)
            if puzzle_obj.puzzle_list[r][c] == 0:
                continue
            elif not puzzle_obj.puzzle_list[r][c] > puzzle_obj.puzzle_list[row_val][col_val]:
                inversions += 1
    if n % 2 != 0:
        if inversions % 2 == 0:
            return True
        else:
            return False
    else:
        inversions += zero_row_index
        if inversions % 2 == 0:
            return False
        else:
            return True


# Traces through the parent link in the puzzle objects to build the move list
# then reverses it at the end to make sure its in the correct order
def print_moves(puzzle_obj):
    temp_puzzle_list = list()
    move_list = list()
    while puzzle_obj.parent_puzzle is not None:
        move_list.append(puzzle_obj.move)
        temp_puzzle_list.append(puzzle_obj)
        puzzle_obj = puzzle_obj.parent_puzzle
    move_list.reverse()
    temp_puzzle_list.reverse()
    return move_list, temp_puzzle_list


# User Input
n = int(input("Enter the number of values in a row: "))
print("Welcome to this "+str((n*n)-1)+" Puzzle program.\n"
                                      "Enter the numbers in any order you like and don't forget the 0 as the space!")
puzzle_list = [[int(input("Enter number #" + str((r * n) + c + 1) + ": ")) for c in range(n)] for r in range(n)]
puzzle_obj = PuzzleObj(puzzle_list, 0, None)
puzzle_obj.move = None

print_puzzle(puzzle_obj)

if not is_solveable(puzzle_obj):
    print("There is no solution to this puzzle")
    sys.exit(0)

print("\n")

# Main algorithm
while True:
    if dist(puzzle_obj) == 0:
        print("Found goal state in " + str(len(visited_puzzles)) + " iterations and went through "
              + str(float((len(visited_puzzles)/181440)*100))+"% of total options")
        move_list, move_puzzle_list = print_moves(puzzle_obj)
        print("Found solution in "+str(len(move_list))+" moves.\nMove list: "+str(move_list))
        print("\nMoves in visual form:")
        for temp in move_puzzle_list:
            print_puzzle(temp)
            print("\n")
        sys.exit(0)

    visited_puzzles.append(puzzle_check(puzzle_obj))
    zero_index = index_of(puzzle_obj, 0)
    compute_f_score(puzzle_obj, zero_index)
    current = f_score_list.index(min(f_score_list))
    puzzle_obj = queue_list[current]
    del queue_list[current]
    del f_score_list[current]
    #print_puzzle(puzzle_obj)
    #print("\n")
