from copy import copy, deepcopy
import sys

visited_puzzles = []
queue_dict = {}
n = 3

# Intial Protype of 8 Puzzle Problem, assumed all goal states
# Computed h score based on how far each state was from its current goal state, this was not very accurate
# Used a dictionary so inaccurate


class PuzzleNode:
    num = 0
    target_In = 0
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


def set_targets(puzzle):
    zero_index = index_of(puzzle, 0)
    for r in range(n):
        for c in range(n):
            if puzzle[r][c].num == 0:
                continue
            elif puzzle[r][c].num <= zero_index:
                puzzle[r][c].target = puzzle[r][c].num-1
            else:
                puzzle[r][c].target = puzzle[r][c].num


def dist(puzzle):
    total_dist = 0
    for r in range(n):
        for c in range(n):
            if puzzle[r][c].num == 0:
                continue
            else:
                row_val = int(puzzle[r][c].target/n)
                col_val = int(puzzle[r][c].target%n)
                distance = abs(r-row_val) + abs(c-col_val)
                #print(str(distance)+" Row: "+str(r)+" Col: "+str(c)+" Target: "+str(puzzle[r][c].target)+" Val: "+str(puzzle[r][c].num))
                puzzle[r][c].dist = distance
                total_dist += distance
    return total_dist


def compute_f_score(puzzle,zero_index , g_score):
    row_val = int(int(zero_index)/n)
    col_val = int(int(zero_index)%n)
    for x in range(4):
        if x == 0 and (row_val-1) in range(n):
            temp_puzzle1 = deepcopy(puzzle)
            temp = temp_puzzle1[row_val-1][col_val]
            temp_puzzle1[row_val - 1][col_val] = temp_puzzle1[row_val][col_val]
            temp_puzzle1[row_val][col_val] = temp
            if puzzle_check(temp_puzzle1) in visited_puzzles:
                continue
            set_targets(temp_puzzle1)
            queue_dict[dist(temp_puzzle1)+g_score] = temp_puzzle1
        elif x == 1 and (row_val+1) in range(n):
            temp_puzzle2 = deepcopy(puzzle)
            temp = temp_puzzle2[row_val + 1][col_val]
            temp_puzzle2[row_val + 1][col_val] = temp_puzzle2[row_val][col_val]
            temp_puzzle2[row_val][col_val] = temp
            if puzzle_check(temp_puzzle2) in visited_puzzles:
                continue
            set_targets(temp_puzzle2)
            queue_dict[dist(temp_puzzle2)+g_score] = temp_puzzle2
        elif x == 2 and (col_val-1) in range(n):
            temp_puzzlen = deepcopy(puzzle)
            temp = temp_puzzlen[row_val][col_val - 1]
            temp_puzzlen[row_val][col_val - 1] = temp_puzzlen[row_val][col_val]
            temp_puzzlen[row_val][col_val] = temp
            if puzzle_check(temp_puzzlen) in visited_puzzles:
                continue
            set_targets(temp_puzzlen)
            queue_dict[dist(temp_puzzlen)+g_score] = temp_puzzlen
        elif x == 3 and (col_val+1) in range(n):
            temp_puzzle4 = deepcopy(puzzle)
            temp = temp_puzzle4[row_val][col_val + 1]
            temp_puzzle4[row_val][col_val + 1] = temp_puzzle4[row_val][col_val]
            temp_puzzle4[row_val][col_val] = temp
            if puzzle_check(temp_puzzle4) in visited_puzzles:
                continue
            set_targets(temp_puzzle4)
            queue_dict[dist(temp_puzzle4)+g_score] = temp_puzzle4


print("Welcome to this 8 Puzzle program.\nEnter the numbers in any order you like and don't forget the 0 as the space!")
puzzle_list = [[PuzzleNode(input("Enter number #" + str((r * n) + c + 1) + ": ")) for c in range(n)] for r in range(n)]
print_puzzle(puzzle_list)
print("\n")
#puzzle_list = [[PuzzleNode((r * n) + c) for c in range(n)] for r in range(n)]

g_score = 0
set_targets(puzzle_list)


while True:
    if dist(puzzle_list) == 0:
        print_puzzle(puzzle_list)
        print("Found goal state in " + str(g_score) + " iterations and went through "
              + str(float((len(visited_puzzles) / 181440) * 100)) + "% of total options")
        sys.exit(0)

    visited_puzzles.append(puzzle_check(puzzle_list))
    g_score += 1
    zero_index = index_of(puzzle_list, 0)

    compute_f_score(puzzle_list, zero_index, g_score)

    current = min(list(queue_dict.keys()))
    puzzle_list = queue_dict[current]
    del queue_dict[current]
    #print_puzzle(puzzle_list)
    #print("\n")


