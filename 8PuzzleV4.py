from copy import deepcopy
import sys

# This version is an updated V1 (Multiple goal states) but
# uses the more accurate data structures and displays move list at the end

visited_puzzles = []
queue_list = []
f_score_list = []
n = 3
# Puzzle Object class that represents the state of the puzzle and stores the g score and h score of the puzzle
# along with a link to parent puzzle and the move made to get to the state
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

# Prints the puzzle in nxn style in the console
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

# Combined set_targets and dist methods from PuzzleV1
def dist(puzzle_obj,zero_index):
    total_dist = 0
    for r in range(n):
        for c in range(n):
            if puzzle_obj.puzzle_list[r][c] == 0:
                continue
            else:
                if puzzle_obj.puzzle_list[r][c] <= zero_index:
                    target = puzzle_obj.puzzle_list[r][c] - 1
                else:
                    target = puzzle_obj.puzzle_list[r][c]
                row_val = int(target/n)
                col_val = int(target%n)
                total_dist += (abs(r-row_val) + abs(c-col_val))
    return total_dist


# Computes the fscore for the possible states and stores them in the
def compute_f_score(puzzle_obj, zero_index):
    row_val = int(int(zero_index) / n)
    col_val = int(int(zero_index) % n)
    for x in range(4):
        if x == 0 and (row_val-1) in range(n):
            temp_puzzle = PuzzleObj(deepcopy(puzzle_obj.puzzle_list),puzzle_obj.g_score+1,puzzle_obj)
            temp = temp_puzzle.puzzle_list[row_val-1][col_val]
            temp_puzzle.puzzle_list[row_val - 1][col_val] = temp_puzzle.puzzle_list[row_val][col_val]
            temp_puzzle.puzzle_list[row_val][col_val] = temp
            if puzzle_check(temp_puzzle) in visited_puzzles:
                continue
            h_score = dist(temp_puzzle,zero_index)
            f_score = h_score + (puzzle_obj.g_score+1)
            queue_list.append(temp_puzzle)
            f_score_list.append(f_score)
            temp_puzzle.move = "Up"
        elif x == 1 and (row_val+1) in range(n):
            temp_puzzle = PuzzleObj(deepcopy(puzzle_obj.puzzle_list),puzzle_obj.g_score+1, puzzle_obj)
            temp = temp_puzzle.puzzle_list[row_val + 1][col_val]
            temp_puzzle.puzzle_list[row_val + 1][col_val] = temp_puzzle.puzzle_list[row_val][col_val]
            temp_puzzle.puzzle_list[row_val][col_val] = temp
            if puzzle_check(temp_puzzle) in visited_puzzles:
                continue
            h_score = dist(temp_puzzle,zero_index)
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
            h_score = dist(temp_puzzle,zero_index)
            f_score = h_score + (puzzle_obj.g_score+1)
            queue_list.append(temp_puzzle)
            f_score_list.append(f_score)
            temp_puzzle.move = "Left"
        elif x == n and (col_val+1) in range(n):
            temp_puzzle = PuzzleObj(deepcopy(puzzle_obj.puzzle_list),puzzle_obj.g_score+1, puzzle_obj)
            temp = temp_puzzle.puzzle_list[row_val][col_val+1]
            temp_puzzle.puzzle_list[row_val][col_val+1] = temp_puzzle.puzzle_list[row_val][col_val]
            temp_puzzle.puzzle_list[row_val][col_val] = temp
            if puzzle_check(temp_puzzle) in visited_puzzles:
                continue
            h_score = dist(temp_puzzle,zero_index)
            f_score = h_score + (puzzle_obj.g_score+1)
            queue_list.append(temp_puzzle)
            f_score_list.append(f_score)
            temp_puzzle.move = "Right"


def print_moves(puzzle_obj):
    move_list = list()
    while puzzle_obj.parent_puzzle is not None:
        move_list.append(puzzle_obj.move)
        puzzle_obj = puzzle_obj.parent_puzzle
    move_list.reverse()
    return move_list


print("Welcome to this 8 Puzzle program.\nEnter the numbers in any order you like and don't forget the 0 as the space!")
puzzle_list = [[int(input("Enter number #" + str((r * n) + c + 1) + ": ")) for c in range(n)] for r in range(n)]
#puzzle_list = [7,2,4,5,0,6,8,3,1]
puzzle_obj = PuzzleObj(puzzle_list, 0, None)
puzzle_obj.move = None

print_puzzle(puzzle_obj)
print("\n")
#puzzle_list = [[PuzzleNode((r * n) + c) for c in range(n)] for r in range(n)]
zero_index = index_of(puzzle_obj, 0)

while True:
    if dist(puzzle_obj,zero_index) == 0:
        print_puzzle(puzzle_obj)
        print("Found goal state in " + str(len(visited_puzzles)) + " iterations and went through "
              + str(float((len(visited_puzzles)/181440)*100))+"% of total options")
        move_list = print_moves(puzzle_obj)
        print("Found solution in "+str(len(move_list))+" moves.\nMove list: "+str(move_list))
        sys.exit(0)

    visited_puzzles.append(puzzle_check(puzzle_obj))
    zero_index = index_of(puzzle_obj, 0)
    compute_f_score(puzzle_obj, zero_index)
    try:
        current = f_score_list.index(min(f_score_list))
    except ValueError:
        print("No solution found in " + str(len(visited_puzzles)) + " iterations and went through "
              + str(float((len(visited_puzzles) / 181440) * 100)) + "% of total options")
        sys.exit(0)
    puzzle_obj = queue_list[current]
    del queue_list[current]
    del f_score_list[current]
    #print_puzzle(puzzle_obj)
    #print("\n")
