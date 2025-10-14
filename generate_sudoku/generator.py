import random
import argparse
from itertools import islice
from datetime import datetime

try:
    from PIL import Image, ImageDraw, ImageFont
    
except ImportError:
    print("Pillow library not found.")
    Image = None

def generate_sudoku(base=3, difficulty=0.5):
    side = base * base

    # pattern for a baseline valid solution
    def pattern(r, c):
        return (base * (r % base) + r // base + c) % side

    # randomize rows, columns and numbers (of valid base pattern)
    def shuffle(s):
        return random.sample(s, len(s))

    rBase = range(base)
    rows = [g * base + r for g in shuffle(rBase) for r in shuffle(rBase)]
    cols = [g * base + c for g in shuffle(rBase) for c in shuffle(rBase)]
    nums = shuffle(range(1, base * base + 1))

    # produce board using randomized baseline pattern
    solution_board = [[nums[pattern(r, c)] for c in cols] for r in rows]

    # remove some of the numbers to make it difficult
    squares = side * side
    empties = int(squares * (0.6 + 0.2 * difficulty))
    puzzle_board = [row[:] for row in solution_board]
    for p in random.sample(range(squares), empties):
        puzzle_board[p // side][p % side] = 0

    # ensure the puzzle has a unique solution
    while True:
        solved = [*islice(short_sudoku_solve(puzzle_board), 2)]
        if len(solved) == 1:
            break
        
        if len(solved) == 0:
            # This should not happen with the current logic, but as a safeguard
            # if the puzzle is unsolvable, we add a number back.
            empties_list = [i for i, n in enumerate(sum(puzzle_board, [])) if n == 0]
            if not empties_list:
                break # Should not be reached
            p = random.choice(empties_list)
            puzzle_board[p//side][p%side] = solution_board[p//side][p%side]
            continue

        diff_pos = [(r, c) for r in range(side) for c in range(side)
                    if solved[0][r][c] != solved[1][r][c]]
        if not diff_pos:
             # If there are multiple solutions but they are identical, we can break
             break
        r, c = random.choice(diff_pos)
        puzzle_board[r][c] = solution_board[r][c]

    return puzzle_board

def short_sudoku_solve(board):
    side = len(board)
    base = int(side**0.5)
    
    # Flatten the board
    flat_board = [n for row in board for n in row]
    blanks = [i for i, n in enumerate(flat_board) if n == 0]

    # Dancing Links (DLX) inspired solver
    # Create the cover matrix
    cover = {}
    for r in range(side):
        for c in range(side):
            for n in range(1, side + 1):
                p = r * side + c
                # Constraints: cell, row, column, box
                constraints = [
                    ("cell", p),
                    ("row", r * side + (n - 1)),
                    ("col", c * side + (n - 1)),
                    ("box", (r // base * base + c // base) * side + (n - 1))
                ]
                cover[(p, n)] = set(constraints)

    used_constraints = set()
    for r in range(side):
        for c in range(side):
            if board[r][c] != 0:
                p = r * side + c
                n = board[r][c]
                used_constraints.update(cover[(p, n)])

    # Recursive solver
    def solve():
        if not blanks:
            yield [flat_board[i:i + side] for i in range(0, len(flat_board), side)]
            return

        p = blanks[0]
        r, c = divmod(p, side)
        
        # Temporarily remove the blank for the recursive call
        blanks.pop(0)

        for n in range(1, side + 1):
            if not cover[(p, n)] & used_constraints:
                # Try placing number n
                used_constraints.update(cover[(p, n)])
                flat_board[p] = n
                
                yield from solve()

                # Backtrack
                used_constraints.difference_update(cover[(p, n)])
                flat_board[p] = 0
        
        # Add the blank back for backtracking
        blanks.insert(0, p)

    yield from solve()


def print_board(board):
    base = int(len(board)**0.5)
    side = len(board)

    def expandLine(line):
        return line[0] + line[5:9].join([line[1:5] * (base - 1)] * base) + line[9:13]

    line0 = expandLine("╔═══╤═══╦═══╗")
    line1 = expandLine("║ . │ . ║ . ║")
    line2 = expandLine("╟───┼───╫───╢")
    line3 = expandLine("╠═══╪═══╬═══╣")
    line4 = expandLine("╚═══╧═══╩═══╝")

    symbol = " 1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    nums = [[""] + [symbol[n] for n in row] for row in board]
    print(line0)
    for r in range(1, side + 1):
        print("".join(n + s for n, s in zip(nums[r - 1], line1.split("."))))
        if r % side != 0:
            if r % base == 0:
                print(line3)
            else:
                print(line2)
    print(line4)

def save_board_as_image(board, filename="sudoku_puzzle.png"):
    if Image is None:
        print("Cannot save image because Pillow is not installed.")
        return

    cell_size = 50
    margin = 20
    side = len(board)
    base = int(side**0.5)
    img_size = side * cell_size + 2 * margin

    img = Image.new("RGB", (img_size, img_size), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", int(cell_size * 0.8))
    except IOError:
        font = ImageFont.load_default()

    # Draw grid
    for i in range(side + 1):
        line_width = 3 if i % base == 0 else 1
        # Vertical lines
        draw.line([(margin + i * cell_size, margin), (margin + i * cell_size, img_size - margin)], fill="black", width=line_width)
        # Horizontal lines
        draw.line([(margin, margin + i * cell_size), (img_size - margin, margin + i * cell_size)], fill="black", width=line_width)

    # Draw numbers
    for r in range(side):
        for c in range(side):
            num = board[r][c]
            if num != 0:
                text = str(num)
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = margin + c * cell_size + (cell_size - text_width) / 2
                y = margin + r * cell_size + (cell_size - text_height) / 2 - bbox[1]
                draw.text((x, y), text, fill="black", font=font)

    img.save(f"img/{filename}")
    print(f"Sudoku puzzle saved as {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Sudoku puzzle.")
    parser.add_argument("--difficulty", type=float, default=0.5, help="A float between 0 (easy) and 1 (hard).")
    args = parser.parse_args()

    puzzle = generate_sudoku(difficulty=args.difficulty)
    
    timestamp = datetime.now().strftime("%H%M%S")
    filename = f"sudoku_puzzle_{args.difficulty}_{timestamp}.png"
    
    #print_board(puzzle)
    save_board_as_image(puzzle, filename=filename)