import math

import math

def get_grid_dimensions(N):
    """
    Compute the grid dimensions (rows and cols) such that:
    - The grid is as close to a square as possible.
    - The difference between rows and cols is at most 1.
    - The grid can accommodate all N nodes, possibly with some empty slots.
    """
    sqrt_N = math.sqrt(N)
    # Possible rows: floor and ceil of sqrt(N)
    possible_rows = [int(math.floor(sqrt_N)), int(math.ceil(sqrt_N))]
    possible_rows = list(set(possible_rows))  # Ensure uniqueness

    # Try to find the best grid dimensions
    best_rows, best_cols = None, None
    min_difference = None

    for rows in possible_rows:
        if rows <= 0:
            continue
        cols = int(math.ceil(N / rows))
        if abs(rows - cols) <= 1:
            if min_difference is None or abs(rows - cols) < min_difference:
                min_difference = abs(rows - cols)
                best_rows, best_cols = rows, cols

    # If no acceptable grid found, increment rows until acceptable
    if best_rows is None:
        rows = int(math.floor(sqrt_N))
        while True:
            cols = int(math.ceil(N / rows))
            if abs(rows - cols) <= 1:
                best_rows, best_cols = rows, cols
                break
            rows += 1

    return best_rows, best_cols

class GridTopology:
    def __init__(self, origin_world_size, local_rank):
        self.origin_world_size = origin_world_size
        self.local_rank = local_rank

    def get_down_and_right_neighbors(self):
        N = self.origin_world_size
        local_rank = self.local_rank

        # Step 1: Compute grid dimensions (rows and cols)
        rows, cols = get_grid_dimensions(N)

        # Step 2: Compute the current node's position in the grid
        row = local_rank // cols
        col = local_rank % cols

        # Initialize down and right neighbor ranks
        down_rank = None
        right_rank = None

        # Step 3: Compute the down neighbor with wrap-around
        down_row = (row + 1) % rows
        down_col = col
        while True:
            down_rank_candidate = down_row * cols + down_col
            if down_rank_candidate < N:
                down_rank = down_rank_candidate
                break
            else:
                # Move to the next row (wrap-around)
                down_row = (down_row + 1) % rows
                if down_row == row:
                    # No valid neighbor found after a full loop
                    down_rank = local_rank  # Self-loop if necessary
                    break

        # Step 4: Compute the right neighbor with wrap-around
        right_row = row
        right_col = (col + 1) % cols
        while True:
            right_rank_candidate = right_row * cols + right_col
            if right_rank_candidate < N:
                right_rank = right_rank_candidate
                break
            else:
                # Move to the next column (wrap-around)
                right_col = (right_col + 1) % cols
                if right_col == col:
                    # No valid neighbor found after a full loop
                    right_rank = local_rank  # Self-loop if necessary
                    break

        # Ensure that down_rank and right_rank are not None
        # They should at least be self.local_rank if no other valid neighbor exists
        if down_rank is None:
            down_rank = local_rank
        if right_rank is None:
            right_rank = local_rank

        return down_rank, right_rank

if __name__ == '__main__':
    grid = GridTopology(8, 2)
    down, right = grid.get_down_and_right_neighbors()
    print(down, right)