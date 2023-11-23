import random

# Constants for the Tic-Tac-Toe board
X = 'X'
O = 'O'
EMPTY = ' '

def print_board(board):
    for row in board:
        print(' | '.join(row))
        print('-' * 5)

def is_winner(board, player):
    # Check rows, columns, and diagonals for a win
    for row in board:
        if all(cell == player for cell in row):
            return True
    for col in range(3):
        if all(board[row][col] == player for row in range(3)):
            return True
    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True
    return False

def is_board_full(board):
    return all(cell != EMPTY for row in board for cell in row)

def get_empty_cells(board):
    return [(row, col) for row in range(3) for col in range(3) if board[row][col] == EMPTY]

def minimax(board, depth, maximizing_player):
    if is_winner(board, X):
        return -1
    elif is_winner(board, O):
        return 1
    elif is_board_full(board):
        return 0

    if maximizing_player:
        max_eval = float('-inf')
        for row, col in get_empty_cells(board):
            board[row][col] = O
            eval = minimax(board, depth + 1, False)
            board[row][col] = EMPTY
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for row, col in get_empty_cells(board):
            board[row][col] = X
            eval = minimax(board, depth + 1, True)
            board[row][col] = EMPTY
            min_eval = min(min_eval, eval)
        return min_eval

def find_best_move(board):
    best_val = float('-inf')
    best_move = None

    for row, col in get_empty_cells(board):
        board[row][col] = O
        move_val = minimax(board, 0, False)
        board[row][col] = EMPTY

        if move_val > best_val:
            best_move = (row, col)
            best_val = move_val

    return best_move

def play_tic_tac_toe():
    board = [[EMPTY] * 3 for _ in range(3)]
    current_player = X if random.choice([True, False]) else O

    while True:
        print_board(board)

        if current_player == X:
            row = int(input("Enter row (0, 1, or 2): "))
            col = int(input("Enter column (0, 1, or 2): "))
            if board[row][col] == EMPTY:
                board[row][col] = X
                current_player = O
        else:
            print("AI player is making a move...")
            move = find_best_move(board)
            row, col = move
            board[row][col] = O
            current_player = X

        if is_winner(board, X):
            print_board(board)
            print("You win! Congratulations!")
            break
        elif is_winner(board, O):
            print_board(board)
            print("AI player wins! Better luck next time.")
            break
        elif is_board_full(board):
            print_board(board)
            print("It's a draw!")
            break

if __name__ == "__main__":
    play_tic_tac_toe()
