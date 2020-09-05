from copy import deepcopy
import sys

visited_puzzles = []
queue_dict = {}
move_dict = {}
n = 3

# Updated 8 Puzzle after realizing that we are solving for only one goal state
# Same implementation as 8PuzzleV1

class PuzzleNode:
    num = 0
    dist = 0

    def __init__(self, num):
        self.num = int(num)

    def __str__(self):
        return str(self.num)


def puzzle_check(puzzle):
    ret_str = ""
    for r in range(n):
        for c in range(n):
            ret_str += str(puzzle[r][c].num)
    return ret_str


def print_puzzle(puzzle):
    for r in range(n):
        row_string = ""
        for c in range(n):
            row_string += str(puzzle[r][c])+"  "
        print(row_string)


def index_of(puzzle, element):
    for r in range(n):
        for c in range(n):
            if puzzle[r][c].num == element:
                return int((r * n) + c)


def dist(puzzle):
    total_dist = 0
    for r in range(n):
        for c in range(n):
            if puzzle[r][c].num == 0:
                continue
            else:
                row_val = int(puzzle[r][c].num/n)
                col_val = int(puzzle[r][c].num%n)
                distance = abs(r-row_val) + abs(c-col_val)
                #print(str(distance)+" Row: "+str(r)+" Col: "+str(c)+" Target: "+str(puzzle[r][c].target)+" Val: "+str(puzzle[r][c].num))
                puzzle[r][c].dist = distance
                total_dist += distance
    return total_dist


def compute_f_score(puzzle, zero_index , g_score):
    row_val = int(int(zero_index) / n)
    col_val = int(int(zero_index) % n)
    for x in range(4):
        if x == 0 and (row_val-1) in range(n):
            temp_puzzle = deepcopy(puzzle)
            temp = temp_puzzle[row_val-1][col_val]
            temp_puzzle[row_val - 1][col_val] = temp_puzzle[row_val][col_val]
            temp_puzzle[row_val][col_val] = temp
            if puzzle_check(temp_puzzle) in visited_puzzles:
                continue
            f_score = dist(temp_puzzle) + g_score
            queue_dict[f_score] = temp_puzzle
            move_dict[f_score] = "Up"
        elif x == 1 and (row_val+1) in range(n):
            temp_puzzle = deepcopy(puzzle)
            temp = temp_puzzle[row_val + 1][col_val]
            temp_puzzle[row_val + 1][col_val] = temp_puzzle[row_val][col_val]
            temp_puzzle[row_val][col_val] = temp
            if puzzle_check(temp_puzzle) in visited_puzzles:
                continue
            f_score = dist(temp_puzzle) + g_score
            queue_dict[f_score] = temp_puzzle
            move_dict[f_score] = "Down"
        elif x == 2 and (col_val-1) in range(n):
            temp_puzzle = deepcopy(puzzle)
            temp = temp_puzzle[row_val][col_val - 1]
            temp_puzzle[row_val][col_val - 1] = temp_puzzle[row_val][col_val]
            temp_puzzle[row_val][col_val] = temp
            if puzzle_check(temp_puzzle) in visited_puzzles:
                continue
            f_score = dist(temp_puzzle)+g_score
            queue_dict[f_score] = temp_puzzle
            move_dict[f_score] = "Left"
        elif x == 3 and (col_val+1) in range(n):
            temp_puzzle = deepcopy(puzzle)
            temp = temp_puzzle[row_val][col_val + 1]
            temp_puzzle[row_val][col_val + 1] = temp_puzzle[row_val][col_val]
            temp_puzzle[row_val][col_val] = temp
            if puzzle_check(temp_puzzle) in visited_puzzles:
                continue
            f_score = dist(temp_puzzle) + g_score
            queue_dict[f_score] = temp_puzzle
            move_dict[f_score] = "Right"



print("Welcome to this 8 Puzzle program.\nEnter the numbers in any order you like and don't forget the 0 as the space!")
puzzle_list = [[PuzzleNode(input("Enter number #" + str((r * n) + c + 1) + ": ")) for c in range(n)] for r in range(n)]
print_puzzle(puzzle_list)
print("\n")
#puzzle_list = [[PuzzleNode((r * n) + c) for c in range(n)] for r in range(n)]

g_score = 0
move_string = ""

while True:
    if dist(puzzle_list) == 0:
        print_puzzle(puzzle_list)
        print("Found goal state in " + str(g_score) + " iterations and went through "
              + str(float((len(visited_puzzles)/181440)*100))+"% of total options")
        print(move_string)
        sys.exit(0)

    visited_puzzles.append(puzzle_check(puzzle_list))
    g_score += 1
    zero_index = index_of(puzzle_list, 0)

    compute_f_score(puzzle_list, zero_index, g_score)
    try:
        current = min(list(queue_dict.keys()))
    except ValueError:
        print("No solution found")
        sys.exit(0)
    puzzle_list = queue_dict[current]
    del queue_dict[current]
    move_string += str(move_dict[current]) + ", "
    del move_dict[current]
    #print_puzzle(puzzle_list)
    #print("\n")


