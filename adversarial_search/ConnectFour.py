"""
Autorzy: [Błażej Kanczkowski s26836, Adam Rzepa s27424]
Zasady gry Connect Four: https://pl.wikipedia.org/wiki/Czwórki
Instrukcja przygotowania środowiska:
  - Wymagany jest Python w wersji 3.6 lub nowszej.
  - Nie są wymagane żadne dodatkowe biblioteki.
  - Aby uruchomić grę, wykonaj polecenie w terminalu: python nazwa_pliku.py
"""
import math

ROWS = 6
COLS = 7
PLAYER_PIECE = 'X'
AI_PIECE = 'O'
EMPTY = '.'


def create_board():
    """Tworzy i zwraca nową, pustą planszę do gry.

       Returns:
           list[list[str]]: Pusta plansza o wymiarach ROWS x COLS.
       """
    return [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]


def print_board(board):
    """Wyświetla aktualny stan planszy w konsoli.

       Parameters:
           board (list[list[str]]): Aktualny stan planszy gry.
       """
    print("\n  " + " ".join(str(i + 1) for i in range(COLS)))
    for row in board:
        print("  " + " ".join(row))
    print()


def drop_piece(board, col, piece):
    """Upuszcza pionek danego gracza do wybranej kolumny.

        Parameters:
            board (list[list[str]]): Aktualny stan planszy gry.
            col (int): Indeks kolumny (0-6), do której ma wpaść pionek.
            piece (str): Pionek gracza ('X' lub 'O').

        Returns:
            bool: True, jeśli ruch się powiódł, False, jeśli kolumna jest pełna.
        """
    for row in reversed(board):
        if row[col] == EMPTY:
            row[col] = piece
            return True
    return False


def winning_move(board, piece):
    """Sprawdza, czy na planszy istnieje zwycięski układ 4 pionków.

       Funkcja przeszukuje planszę w poziomie, pionie i na obu skosach
       w poszukiwaniu czterech identycznych pionków w jednej linii.

       Parameters:
       board (list[list[str]]): Aktualny stan planszy gry.
       piece (str): Pionek gracza do sprawdzenia ('X' lub 'O').

       Returns:
       bool: True, jeśli znaleziono zwycięski układ, w przeciwnym razie False.
       """
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r][c + i] == piece for i in range(4)):
                return True

    for c in range(COLS):
        for r in range(ROWS - 3):
            if all(board[r + i][c] == piece for i in range(4)):
                return True

    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r + i][c + i] == piece for i in range(4)):
                return True

    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if all(board[r - i][c + i] == piece for i in range(4)):
                return True

    return False


def board_full(board):
    """Sprawdza, czy wszystkie pola na planszy są zajęte.

        Parameters:
            board (list[list[str]]): Aktualny stan planszy gry.

        Returns:
            bool: True, jeśli plansza jest pełna, w przeciwnym razie False.
        """
    return all(cell != EMPTY for row in board for cell in row)


def evaluate_window(window, piece):
    """Ocenia pojedynczy, 4-elementowy fragment (okno) planszy.

        Funkcja pomocnicza dla score_position. Przyznaje punkty za układy
        pionków w danym oknie z perspektywy określonego gracza.

        Parameters:
            window (list[str]): 4-elementowa lista reprezentująca fragment planszy.
            piece (str): Pionek, z perspektywy którego oceniamy okno.

        Returns:
            int: Wynik punktowy dla danego okna.
        """
    score = 0
    my_pieces = window.count(piece)
    opponent_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE
    opponent_pieces = window.count(opponent_piece)
    empty_slots = window.count(EMPTY)

    if my_pieces == 4:
        score += 10000
    elif my_pieces == 3 and empty_slots == 1:
        score += 50
    elif my_pieces == 2 and empty_slots == 2:
        score += 2

    if opponent_pieces == 3 and empty_slots == 1:
        score -= 1000

    return score


def score_position(board, piece):
    """Oblicza heurystyczny wynik dla całego stanu planszy.

        Ocenia całą planszę z perspektywy danego gracza, sumując punkty
        z wszystkich możliwych 4-elementowych okien oraz przyznając bonus
        za kontrolę nad środkową kolumną.

        Parameters:
            board (list[list[str]]): Aktualny stan planszy gry.
            piece (str): Pionek, z perspektywy którego oceniamy planszę.

        Returns:
            int: Całkowity wynik punktowy planszy.
        """
    score = 1
    opponent_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE

    center_array = [row[COLS // 2] for row in board]
    my_center_count = center_array.count(piece)
    opponent_center_count = center_array.count(opponent_piece)
    score += my_center_count * 4
    score -= opponent_center_count * 4

    for r in range(ROWS):
        row_array = board[r]
        for c in range(COLS - 3):
            window = row_array[c:c + 4]
            score += evaluate_window(window, piece)

    for c in range(COLS):
        col_array = [board[r][c] for r in range(ROWS)]
        for r in range(ROWS - 3):
            window = col_array[r:r + 4]
            score += evaluate_window(window, piece)

    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r + i][c + i] for i in range(4)]
            score += evaluate_window(window, piece)

    for r in range(3, ROWS):
        for c in range(COLS - 3):
            window = [board[r - i][c + i] for i in range(4)]
            score += evaluate_window(window, piece)

    return score


def is_terminal_node(board):
    """Sprawdza, czy gra osiągnęła stan końcowy.

       Stan końcowy to wygrana jednego z graczy lub zapełnienie całej planszy.

       Parameters:
           board (list[list[str]]): Aktualny stan planszy gry.

       Returns:
           bool: True, jeśli gra się zakończyła, w przeciwnym razie False.
       """
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or board_full(board)


def get_valid_locations(board):
    """Zwraca listę kolumn, do których można jeszcze wrzucić pionek.

        Parameters:
            board (list[list[str]]): Aktualny stan planszy gry.

        Returns:
            list[int]: Lista indeksów kolumn (0-6), które nie są pełne.
        """
    return [col for col in range(COLS) if board[0][col] == EMPTY]


def min_max(board, depth, alpha, beta, maximizing_player):
    """
    Minimax z przycinaniem alpha–beta dla Connect Four.

    Przeszukuje drzewo gry do zadanej głębokości. W węzłach MAX (AI 'O')
    maksymalizuje wynik, a w węzłach MIN (gracz 'X') – minimalizuje.
    Używa **alpha–beta pruning** do odcinania gałęzi, które nie mogą
    wpłynąć na wynik (gdy alpha >= beta), co zachowuje wynik minimaxa,
    ale znacząco przyspiesza obliczenia.

    Parametry:
        board (list[list[str]]): Aktualny stan planszy.
        depth (int): Pozostała głębokość przeszukiwania.
        alpha (float): Najlepsza (najwyższa) jak dotąd gwarantowana
                       wartość dla MAX na ścieżce (początkowo -inf).
        beta (float):  Najlepsza (najniższa) jak dotąd gwarantowana
                       wartość dla MIN na ścieżce (początkowo +inf).
        maximizing_player (bool): True, jeśli ruch należy do MAX (AI 'O');
                                  False dla MIN (gracz 'X').

    Zwraca:
        tuple[int | None, int]:
            (najlepsza_kolumna, ocena_stanu_z_perspektywy_AI).
            Gdy węzeł terminalny / głębokość 0 – kolumna = None.

        Zastosowano **porządkowanie ruchów** (najpierw kolumny bliżej środka),
        co zwykle zwiększa liczbę odcięć alpha–beta.
    """
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return None, 10000000
            elif winning_move(board, PLAYER_PIECE):
                return None, -10000000
            else:
                return None, 0
        else:
            return None, score_position(board, AI_PIECE)

    if maximizing_player:
        value = -math.inf
        valid_locations = get_valid_locations(board)

        center = COLS / 2
        ordered_moves = sorted(valid_locations, key=lambda c: abs(center - c))

        best_col = ordered_moves[0]

        for col in ordered_moves:
            b_copy = [row[:] for row in board]
            drop_piece(b_copy, col, AI_PIECE)

            new_score = min_max(b_copy, depth - 1, alpha, beta, False)[1]

            if new_score > value:
                value = new_score
                best_col = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_col, value

    else:
        value = math.inf
        valid_locations = get_valid_locations(board)

        center = COLS / 2
        ordered_moves = sorted(valid_locations, key=lambda c: abs(center - c))

        best_col = ordered_moves[0]

        for col in ordered_moves:
            b_copy = [row[:] for row in board]
            drop_piece(b_copy, col, PLAYER_PIECE)

            new_score = min_max(b_copy, depth - 1, alpha, beta, True)[1]

            if new_score < value:
                value = new_score
                best_col = col

            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_col, value


def main():
    """Główna funkcja uruchamiająca grę Connect Four.

        Obsługuje wybór trybu gry, główną pętlę rozgrywki, interakcje
        z użytkownikiem oraz wywołania AI.
        """
    board = create_board()
    game_over = False
    turn = 0

    depth_ai_x = 2
    depth_ai_o = 2

    print("Connect Four")
    game_mode = input("Wybierz tryb gry (1 - Gracz vs Gracz, 2 - Gracz vs AI, 3 - AI vs AI): ")
    while game_mode not in ['1', '2', '3']:
        game_mode = input("Nieprawidłowy wybór. Wpisz 1, 2 lub 3: ")

    print_board(board)

    while not game_over:
        piece = PLAYER_PIECE if turn == 0 else AI_PIECE
        player_name = "Gracz 1" if turn == 0 else "Gracz 2"
        is_human_turn = (game_mode == '1') or (game_mode == '2' and turn == 0)

        if is_human_turn:
            try:
                col = int(input(f"Tura {player_name} ({piece}), wybierz kolumnę (1-7): ")) - 1
                if col not in get_valid_locations(board):
                    print("Nieprawidłowy ruch. Spróbuj ponownie.")
                    continue
            except ValueError:
                print("Podaj poprawną liczbę.")
                continue
        else:
            print(f"Tura AI ({piece})... Myślę...")

            if turn == 0:
                depth = depth_ai_x
            else:
                depth = depth_ai_o

            is_maximizing = turn == 1
            col, minimax_score = min_max(board, depth, -math.inf, math.inf, is_maximizing)
            print(f"AI ({piece}) wybrało kolumnę {col + 1}")

        if drop_piece(board, col, piece):
            if winning_move(board, piece):
                if is_human_turn:
                    print(f"{player_name} ({piece}) wygrał!")
                else:
                    print(f"AI ({piece}) wygrało!")
                game_over = True

            print_board(board)

            if not game_over and board_full(board):
                print("Remis! Plansza jest pełna.")
                game_over = True

            turn = 1 - turn


if __name__ == "__main__":
    main()
