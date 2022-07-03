########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from rubiks.puzzle.slidingpuzzle import SlidingPuzzle, Slide
from rubiks.solvers.solver import Solver, Solution
########################################################################################################################


class NaiveSlidingSolver(Solver):
    """ Base class for a puzzle solver. How it actually solves its puzzle type is
    left to derived classes implementations by overwriting the  'solve_impl' method
     """

    @classmethod
    def know_to_be_optimal(cls):
        return False

    def solve_impl(self, puzzle, **kw_args) -> Solution:
        return self.reduce(puzzle)

    def reduce(self,
               puzzle,
               max_n=1,
               max_m=1) -> Solution:
        assert isinstance(puzzle, SlidingPuzzle), \
            '%s can only solve %s' % (self.get_name(), SlidingPuzzle.__name__)
        self.log_debug('We need to solve ', puzzle)
        n = puzzle.n
        m = puzzle.m
        solution = Solution(cost=0,
                            path=list(),
                            expanded_nodes=float('nan'),
                            puzzle=puzzle)
        offset_n = 0
        offset_m = 0
        sliding_puzzle = puzzle.clone()
        while n > max(2, max_n) or m > max(2, max_m):
            if n >= m:
                self.log_debug('Let us sort out top row')
                moves = self.solve_top_row(sliding_puzzle)
                for move in moves:
                    move.tile = (move.tile[0] + offset_n,
                                 move.tile[1] + offset_m)
                offset_n += 1
                self.log_debug('We now have ', sliding_puzzle, ' ... ')
                sliding_puzzle = SlidingPuzzle(tiles=sliding_puzzle.tiles[1:])
                self.log_debug(' ... and are left solving ', sliding_puzzle)
            else:
                self.log_debug('Let us sort out left col')
                moves = self.solve_left_col(sliding_puzzle)
                for move in moves:
                    move.tile = (move.tile[0] + offset_n,
                                 move.tile[1] + offset_m)
                offset_m += 1
                self.log_debug('We now have ', sliding_puzzle, ' ... ')
                sliding_puzzle = SlidingPuzzle(tiles=sliding_puzzle.tiles[:, 1:])
                self.log_debug(' ... and are left solving ', sliding_puzzle)
            solution.path.extend(moves)
            solution.cost = len(solution.path)
            n = sliding_puzzle.n
            m = sliding_puzzle.m
        """ Now just rotate the bottom right 4 tiles until done """
        moves = list()
        self.log_debug('offset_n: ', offset_m)
        self.log_debug('offset_m: ', offset_n)
        if max_m < 2 or max_n < 2:
            while not sliding_puzzle.in_order():
                move = self.rotate_clock_wise(sliding_puzzle)
                sliding_puzzle = sliding_puzzle.apply_move(move)
                self.log_debug('rotation move: ', move)
                self.log_debug('sliding_puzzle -> ', sliding_puzzle)
                move.tile = (move.tile[0] + offset_n,
                             move.tile[1] + offset_m)
                self.log_debug('rotation move with offsets: ', move)
                moves.append(move)
            solution.path.extend(moves)
            solution.cost = len(solution.path)
        return solution

    def move_value_to_target(self,
                             sliding_puzzle,
                             value,
                             target_pos,
                             positions_to_avoid,
                             moves):
        current_pos = sliding_puzzle.find_tile(value)
        if current_pos != target_pos and current_pos in positions_to_avoid:
            try:
                positions_to_avoid.remove(current_pos)
            except ValueError:
                pass
        self.log_debug('Let\'s move ', value, ' from ', current_pos, ' to ', target_pos)
        while target_pos != current_pos:
            new_moves = list()
            if current_pos[0] > target_pos[0] and (current_pos[0] - 1, current_pos[1]) not in positions_to_avoid:
                """ move up """
                self.log_debug('move up')
                self.moves_on_top_of(sliding_puzzle,
                                     current_pos,
                                     new_moves,
                                     positions_to_avoid=positions_to_avoid)
            elif current_pos[0] < target_pos[0] and (current_pos[0] + 1, current_pos[1]) not in positions_to_avoid:
                """ move down """
                self.log_debug('move down')
                self.moves_below_of(sliding_puzzle,
                                    current_pos,
                                    new_moves,
                                    positions_to_avoid=positions_to_avoid)
            elif current_pos[1] < target_pos[1] and (current_pos[0], current_pos[1] + 1) not in positions_to_avoid:
                """ move right """
                self.log_debug('move right')
                self.moves_to_right_of(sliding_puzzle,
                                       current_pos,
                                       new_moves,
                                       positions_to_avoid=positions_to_avoid)
            elif current_pos[1] > target_pos[1] and (current_pos[0], current_pos[1] - 1) not in positions_to_avoid:
                """ move left """
                self.log_debug('move left')
                self.moves_to_left_of(sliding_puzzle,
                                      current_pos,
                                      new_moves,
                                      positions_to_avoid=positions_to_avoid)
            else:
                assert False, 'Cannot move anymore without breaking something done already?!'
            new_moves.append(Slide(current_pos[0], current_pos[1]))
            self.log_debug(new_moves)
            sliding_puzzle = sliding_puzzle.apply_moves(new_moves)
            moves.extend(new_moves)
            current_pos = sliding_puzzle.find_tile(value)
            self.log_debug('We applied ', new_moves)
            self.log_debug('And got to ', sliding_puzzle, 'empty: ', sliding_puzzle.empty)
        positions_to_avoid.append(target_pos)
        self.log_debug('positions_to_avoid: ', positions_to_avoid)
        return sliding_puzzle

    def solve_top_row(self, sliding_puzzle):
        assert sliding_puzzle.m >= 3
        self.log_debug('solve_top_row for ', sliding_puzzle)
        desired_top_row = sliding_puzzle.tiles.flatten().sort().values[1:sliding_puzzle.m + 1].tolist()
        moves = list()
        end_state = sliding_puzzle.clone()
        desired_top_row = [desired_top_row[-1]] + desired_top_row[:-2] + \
                          [desired_top_row[-2]] + [desired_top_row[-1]]
        self.log_debug('desired_top_row: ', desired_top_row)
        target_positions = [(sliding_puzzle.n - 1, 0),
                            *[(0, pos) for pos in range(sliding_puzzle.m - 2)],
                            (0, sliding_puzzle.m - 1),
                            (1, sliding_puzzle.m - 1),
                            ]
        self.log_debug('target_positions: ', target_positions)
        if sliding_puzzle.n == 3 and sliding_puzzle.m == 3:
            """ ugly special case """
            joker_value = desired_top_row[-1]
            self.log_debug('special case dim 3 joker_value=', joker_value)
            desired_top_row = desired_top_row[1:-1]
            target_positions = target_positions[1:-1]
            self.log_debug('special case dim 3 desired_top_row -> ', desired_top_row)
            self.log_debug('special case dim 3 target_positions -> ', target_positions)
        assert len(target_positions) == len(desired_top_row), 'WTF?'
        positions_to_avoid = list()
        for value, target_pos in zip(desired_top_row, target_positions):
            end_state = self.move_value_to_target(end_state,
                                                  value,
                                                  target_pos,
                                                  positions_to_avoid,
                                                  moves)
        self.log_debug('end_state after loop: ', end_state)
        if sliding_puzzle.n == 3 and sliding_puzzle.m == 3:
            """ ugly special case """
            assert end_state.tiles[0][0] == desired_top_row[0]
            assert end_state.tiles[0][sliding_puzzle.m - 1] == desired_top_row[1]
            joker_pos = end_state.find_tile(joker_value)
            self.log_debug('%d is in position ' % joker_value, joker_pos)
            empty = end_state.empty
            self.log_debug('empty is in position ', empty)
            if joker_pos == (1, 1) and empty == (0, 1):
                move = Slide(1, 1)
                end_state = end_state.apply(move)
                moves.append(move)
                self.log_debug('after moving empty to go to the only annoying solution: ', end_state)
                joker_pos = end_state.find_tile(joker_value)
                assert joker_pos == (0, 1), 'WTF?'
            """ We can now either solve normally, or we are in configuration 1 3 2 """
            if joker_pos != (0, 1):
                end_state = self.move_value_to_target(end_state,
                                                      joker_value,
                                                      (1, 2),
                                                      positions_to_avoid,
                                                      moves)
                self.log_debug('After special move we are in normal case', end_state)
                """ and then usual rotate of the top-right 4 tiles until done """
            else:
                """ configuration 1 3 2 """
                new_moves = list()
                self.moves_below_of(end_state,
                                    (0, 1),
                                    new_moves,
                                    positions_to_avoid=positions_to_avoid)
                end_state = end_state.apply_moves(new_moves)
                moves.extend(new_moves)
                self.log_debug('Almost there: ', end_state)
                last_moves = [(0, 1), (0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (1, 2), (1, 1), (0, 1)]
                last_moves += [(0, 2), (1, 2), (1, 1), (1, 0), (0, 0), (0, 1)] * 2
                last_moves += [(0, 2), (1, 2)]
                last_moves = [Slide(*move) for move in last_moves]
                end_state = end_state.apply_moves(last_moves)
                moves.extend(last_moves)
                sliding_puzzle.tiles = end_state.tiles
                sliding_puzzle.empty = end_state.empty
                self.log_debug(sliding_puzzle)
                return moves
        target_empty_pos = (0, sliding_puzzle.m - 2)
        new_moves = list()
        self.log_debug('target_empty_pos: ', target_empty_pos)
        while end_state.empty != target_empty_pos:
            self.log_debug('end_state.empty: ', end_state.empty)
            self.log_debug('target_empty_pos: ', target_empty_pos)
            if end_state.empty[1] < target_empty_pos[1]:
                move = end_state.empty_right()
            elif end_state.empty[1] > target_empty_pos[1]:
                move = end_state.empty_left()
            elif end_state.empty[0] > target_empty_pos[0]:
                move = end_state.empty_up()
            new_moves.append(move)
            end_state = end_state.apply(move)
            self.log_debug(end_state)
        moves.extend(new_moves)
        move = Slide(0, sliding_puzzle.m - 1)
        end_state = end_state.apply(move)
        self.log_debug(end_state)
        moves.append(move)
        move = Slide(1, sliding_puzzle.m - 1)
        end_state = end_state.apply(move)
        self.log_debug(end_state)
        moves.append(move)
        sliding_puzzle.tiles = end_state.tiles
        sliding_puzzle.empty = end_state.empty
        self.log_debug(sliding_puzzle)
        return moves

    def solve_left_col(self, sliding_puzzle):
        assert sliding_puzzle.n >= 2
        #self.log_debug('solve_left_col for ', sliding_puzzle)
        desired_left_col = sliding_puzzle.tiles.flatten().sort().values[1::sliding_puzzle.m].tolist()
        moves = list()
        end_state = sliding_puzzle.clone()
        desired_left_col = [desired_left_col[-1]] + desired_left_col[:-2] + \
        [desired_left_col[-2]] + [desired_left_col[-1]]
        target_positions = [(0, sliding_puzzle.m - 1),
                            *[(pos, 0) for pos in range(sliding_puzzle.n - 2)],
                            (sliding_puzzle.n - 1, 0),
                            (sliding_puzzle.n - 1, 1),
                            ]
        self.log_debug('desired_left_col: ', desired_left_col)
        self.log_debug('target_positions: ', target_positions)
        positions_to_avoid = list()
        for value, target_pos in zip(desired_left_col, target_positions):
            end_state = self.move_value_to_target(end_state,
                                                  value,
                                                  target_pos,
                                                  positions_to_avoid,
                                                  moves)
        target_empty_pos = (sliding_puzzle.n - 2, 0)
        new_moves = list()
        #self.log_debug('target_empty_pos: ', target_empty_pos)
        while end_state.empty != target_empty_pos:
            #self.log_debug('end_state.empty: ', end_state.empty)
            #self.log_debug('target_empty_pos: ', target_empty_pos)
            if end_state.empty[0] > target_empty_pos[0]:
                move = end_state.empty_up()
            elif end_state.empty[0] < target_empty_pos[0]:
                move = end_state.empty_up()
            elif end_state.empty[1] < target_empty_pos[1]:
                move = end_state.empty_right()
            elif end_state.empty[1] > target_empty_pos[1]:
                move = end_state.empty_left()
            new_moves.append(move)
            end_state = end_state.apply(move)
            #self.log_debug(end_state)
        moves.extend(new_moves)
        move = Slide(sliding_puzzle.n - 1, 0)
        end_state = end_state.apply(move)
        #self.log_debug(end_state)
        moves.append(move)
        move = Slide(sliding_puzzle.n - 1, 1)
        end_state = end_state.apply(move)
        #self.log_debug(end_state)
        moves.append(move)
        sliding_puzzle.tiles = end_state.tiles
        sliding_puzzle.empty = end_state.empty
        #self.log_debug(sliding_puzzle)
        return moves

    @staticmethod
    def rotate_clock_wise(sliding_puzzle):
        assert 2 == sliding_puzzle.n and 2 == sliding_puzzle.m
        if sliding_puzzle.empty[0] == 0:
            if sliding_puzzle.empty[1] == 0:
                return sliding_puzzle.empty_right()
            else:
                return sliding_puzzle.empty_down()
        else:
            if sliding_puzzle.empty[1] == 0:
                return sliding_puzzle.empty_up()
            else:
                return sliding_puzzle.empty_left()

    @staticmethod
    def rotate_counter_clock_wise(sliding_puzzle):
        assert 2 == sliding_puzzle.n and 2 == sliding_puzzle.m
        if sliding_puzzle.empty[0] == 0:
            if sliding_puzzle.empty[1] == 0:
                return sliding_puzzle.empty_down()
            else:
                return sliding_puzzle.empty_left()
        else:
            if sliding_puzzle.empty[1] == 0:
                return sliding_puzzle.empty_right()
            else:
                return sliding_puzzle.empty_up()

    done = 'done'

    @classmethod
    def obvious_get_in_circle(cls, sliding_puzzle: SlidingPuzzle, of, positions_to_avoid=None):
        if positions_to_avoid is None:
            positions_to_avoid = list()
        e = sliding_puzzle.empty
        if e[0] < of[0] - 1 and (e[0] + 1, e[1]) not in positions_to_avoid:
            return sliding_puzzle.empty_down()
        if e[0] > of[0] + 1 and (e[0] - 1, e[1]) not in positions_to_avoid:
            return sliding_puzzle.empty_up()
        if e[1] < of[1] - 1 and (e[0], e[1] + 1) not in positions_to_avoid:
            return sliding_puzzle.empty_right()
        if e[1] > of[1] + 1 and (e[0], e[1] - 1) not in positions_to_avoid:
            return sliding_puzzle.empty_left()
        return None

    @classmethod
    def empty_below_of(cls, sliding_puzzle: SlidingPuzzle, of, positions_to_avoid=None):
        if positions_to_avoid is None:
            positions_to_avoid = list()
        """ We place the empty above 'of' """
        """ First part if when we are not in the circle around 'of' -> we do the obvious """
        move = cls.obvious_get_in_circle(sliding_puzzle, of, positions_to_avoid)
        if move is not None:
            return move
        """ We are in the circle around of -> we go from right if from_right and we can ..."""
        if sliding_puzzle.empty[1] == of[1]:
            """ same column """
            if sliding_puzzle.empty[0] > of[0]:
                return cls.done
            else:
                from_right = any(pos in positions_to_avoid for pos in cls.left_bank(of))
                if from_right:
                    """ go right unless we cannot """
                    if of[1] < sliding_puzzle.m - 1:
                        return sliding_puzzle.empty_right()
                    else:
                        return sliding_puzzle.empty_left()
                else:
                    """ go left unless we cannot """
                    if 0 < of[1]:
                        return sliding_puzzle.empty_left()
                    else:
                        return sliding_puzzle.empty_right()
        """ bottom row """
        if sliding_puzzle.empty[0] > of[0]:
            if sliding_puzzle.empty[1] > of[1]:
                return sliding_puzzle.empty_left()
            return sliding_puzzle.empty_right()
        """ mid row """
        if sliding_puzzle.empty[0] == of[0]:
            if sliding_puzzle.empty[1] < of[1]:
                """ left """
                from_right = (of[0] + 1, of[1] + 1) in positions_to_avoid
                if from_right:
                    if of[1] < sliding_puzzle.m - 1 and of[0] > 0:
                        return sliding_puzzle.empty_up()
                    else:
                        return sliding_puzzle.empty_down()
                else:
                    return sliding_puzzle.empty_down()
            else:
                """ right """
                from_right = (of[0] + 1, of[1] + 1) not in positions_to_avoid
                if from_right:
                    return sliding_puzzle.empty_down()
                else:
                    if of[0] > 0 and of[1] > 0:
                        return sliding_puzzle.empty_up()
                    else:
                        return sliding_puzzle.empty_down()
        """ only remains top row """
        if sliding_puzzle.empty[1] < of[1]:
            """ left """
            from_right = any(pos in positions_to_avoid for pos in cls.left_bank(of))
            if from_right:
                if of[1] < sliding_puzzle.m - 1:
                    return sliding_puzzle.empty_right()
                else:
                    return sliding_puzzle.empty_down()
            else:
                return sliding_puzzle.empty_down()
        else:
            """ right """
            from_right = not any(pos in positions_to_avoid for pos in cls.right_bank(of))
            if from_right:
                return sliding_puzzle.empty_down()
            else:
                if of[1] > 0:
                    return sliding_puzzle.empty_left()
                else:
                    return sliding_puzzle.empty_down()

    @classmethod
    def empty_on_top_of(cls, sliding_puzzle: SlidingPuzzle, of, positions_to_avoid=None):
        if positions_to_avoid is None:
            positions_to_avoid = list()
        """ We place the empty above 'of' """
        """ First part if when we are not in the circle around 'of' -> we do the obvious """
        move = cls.obvious_get_in_circle(sliding_puzzle, of, positions_to_avoid)
        if move is not None:
            return move
        """ We are in the circle around of -> we go from right if from_right and we can ..."""
        if sliding_puzzle.empty[1] == of[1]:
            """ same column """
            if sliding_puzzle.empty[0] < of[0]:
                return cls.done
            else:
                from_right = any(pos in positions_to_avoid for pos in cls.left_bank(of))
                if from_right:
                    """ go right unless we cannot """
                    if of[1] < sliding_puzzle.m - 1:
                        return sliding_puzzle.empty_right()
                    else:
                        return sliding_puzzle.empty_left()
                else:
                    """ go left unless we cannot """
                    if 0 < of[1]:
                        return sliding_puzzle.empty_left()
                    else:
                        return sliding_puzzle.empty_right()
        """ top row """
        if sliding_puzzle.empty[0] < of[0]:
            if sliding_puzzle.empty[1] > of[1]:
                return sliding_puzzle.empty_left()
            return sliding_puzzle.empty_right()
        """ mid row """
        if sliding_puzzle.empty[0] == of[0]:
            if sliding_puzzle.empty[1] < of[1]:
                """ left """
                from_right = (of[0] - 1, of[1] - 1) in positions_to_avoid
                if from_right:
                    if of[1] < sliding_puzzle.m - 1 and of[0] < sliding_puzzle.n - 1:
                        return sliding_puzzle.empty_down()
                    else:
                        return sliding_puzzle.empty_up()
                else:
                    return sliding_puzzle.empty_up()
            else:
                """ right """
                from_right = (of[0] - 1, of[1] + 1) not in positions_to_avoid
                if from_right:
                    return sliding_puzzle.empty_up()
                else:
                    if of[0] < sliding_puzzle.n - 1 and of[1] > 0:
                        return sliding_puzzle.empty_down()
                    else:
                        return sliding_puzzle.empty_up()
        """ only remains bottom row """
        if sliding_puzzle.empty[1] < of[1]:
            """ left """
            from_right = any(pos in positions_to_avoid for pos in cls.left_bank(of))
            if from_right:
                if of[1] < sliding_puzzle.m - 1:
                    return sliding_puzzle.empty_right()
                else:
                    return sliding_puzzle.empty_up()
            else:
                return sliding_puzzle.empty_up()
        else:
            """ right """
            from_right = not any(pos in positions_to_avoid for pos in cls.right_bank(of))
            if from_right:
                return sliding_puzzle.empty_up()
            else:
                if of[1] > 0:
                    return sliding_puzzle.empty_left()
                else:
                    return sliding_puzzle.empty_up()

    @classmethod
    def upper_bank(cls, pos):
        return [(pos[0] - 1, pos[1] - 1),
                (pos[0] - 1, pos[1]),
                (pos[0] - 1, pos[1] + 1)]

    @classmethod
    def lower_bank(cls, pos):
        return [(pos[0] + 1, pos[1] - 1),
                (pos[0] + 1, pos[1]),
                (pos[0] + 1, pos[1] + 1)]

    @classmethod
    def left_bank(cls, pos):
        return [(pos[0] - 1, pos[1] - 1),
                (pos[0], pos[1] - 1),
                (pos[0] + 1, pos[1] - 1)]

    @classmethod
    def right_bank(cls, pos):
        return [(pos[0] - 1, pos[1] + 1),
                (pos[0], pos[1] + 1),
                (pos[0] + 1, pos[1] + 1)]

    @classmethod
    def empty_to_right_of(cls, sliding_puzzle: SlidingPuzzle, of, positions_to_avoid=None):
        if positions_to_avoid is None:
            positions_to_avoid = list()
        """ We place the empty to the right of 'of' """
        """ First part if when we are not in the circle around 'of' -> we do the obvious """
        move = cls.obvious_get_in_circle(sliding_puzzle, of, positions_to_avoid)
        if move is not None:
            return move
        """ We are in the circle around of """
        if sliding_puzzle.empty[0] == of[0]:
            """ same row """
            if sliding_puzzle.empty[1] > of[1]:
                return cls.done
            else:
                """ we're on the left """
                from_below = any(pos in positions_to_avoid for pos in cls.upper_bank(of))
                if from_below:
                    """ go down unless we cannot """
                    if of[0] < sliding_puzzle.n - 1:
                        return sliding_puzzle.empty_down()
                    else:
                        return sliding_puzzle.empty_up()
                else:
                    """ go up unless we cannot """
                    if 0 < of[0]:
                        return sliding_puzzle.empty_up()
                    else:
                        return sliding_puzzle.empty_down()
        """ right column """
        if sliding_puzzle.empty[1] > of[1]:
            if sliding_puzzle.empty[0] < of[0]:
                return sliding_puzzle.empty_down()
            return sliding_puzzle.empty_up()
        """ mid column """
        if sliding_puzzle.empty[1] == of[1]:
            if sliding_puzzle.empty[0] < of[0]:
                """ above """
                from_below = (of[0] - 1, of[1] + 1) in positions_to_avoid
                if from_below:
                    if of[1] > 0 and of[0] < sliding_puzzle.n - 1:
                        return sliding_puzzle.empty_left()
                    else:
                        return sliding_puzzle.empty_right()
                else:
                    return sliding_puzzle.empty_right()
            else:
                """ below """
                from_below = (of[0] + 1, of[1] + 1) not in positions_to_avoid
                if from_below:
                    return sliding_puzzle.empty_right()
                else:
                    if of[0] > 0 and of[1] > 0:
                        return sliding_puzzle.empty_left()
                    else:
                        return sliding_puzzle.empty_right()
        """ only remains left column """
        if sliding_puzzle.empty[0] < of[0]:
            """ above """
            from_below = any(pos in positions_to_avoid for pos in cls.upper_bank(of))
            if from_below:
                if of[0] < sliding_puzzle.n - 1:
                    return sliding_puzzle.empty_down()
                else:
                    return sliding_puzzle.empty_right()
            else:
                return sliding_puzzle.empty_right()
        else:
            """ below """
            from_below = not any(pos in positions_to_avoid for pos in cls.lower_bank(of))
            if from_below:
                return sliding_puzzle.empty_right()
            else:
                if of[0] > 0:
                    return sliding_puzzle.empty_up()
                else:
                    return sliding_puzzle.empty_right()

    @classmethod
    def empty_to_left_of(cls, sliding_puzzle: SlidingPuzzle, of, positions_to_avoid=None):
        if positions_to_avoid is None:
            positions_to_avoid = list()
        """ We place the empty to the right of 'of' """
        move = cls.obvious_get_in_circle(sliding_puzzle, of, positions_to_avoid)
        if move is not None:
            return move
        """ We are in the circle around of """
        if sliding_puzzle.empty[0] == of[0]:
            """ same row """
            if sliding_puzzle.empty[1] < of[1]:
                return cls.done
            else:
                """ we're on the right """
                from_below = any(pos in positions_to_avoid for pos in cls.upper_bank(of))
                if from_below:
                    """ go down unless we cannot """
                    if of[0] < sliding_puzzle.n - 1:
                        return sliding_puzzle.empty_down()
                    else:
                        return sliding_puzzle.empty_up()
                else:
                    """ go up unless we cannot """
                    if 0 < of[0]:
                        return sliding_puzzle.empty_up()
                    else:
                        return sliding_puzzle.empty_down()
        """ left column """
        if sliding_puzzle.empty[1] < of[1]:
            if sliding_puzzle.empty[0] < of[0]:
                return sliding_puzzle.empty_down()
            return sliding_puzzle.empty_up()
        """ mid column """
        if sliding_puzzle.empty[1] == of[1]:
            if sliding_puzzle.empty[0] < of[0]:
                """ above """
                from_below = (of[0] - 1, of[1] - 1) in positions_to_avoid
                if from_below:
                    if of[1] < sliding_puzzle.m - 1 and of[0] < sliding_puzzle.n - 1:
                        return sliding_puzzle.empty_right()
                    else:
                        return sliding_puzzle.empty_left()
                else:
                    return sliding_puzzle.empty_left()
            else:
                """ below """
                from_below = (of[0] + 1, of[1] - 1) not in positions_to_avoid
                if from_below:
                    return sliding_puzzle.empty_left()
                else:
                    if of[1] < sliding_puzzle.m - 1 and of[0] > 0:
                        return sliding_puzzle.empty_right()
                    else:
                        return sliding_puzzle.empty_left()
        """ only remains right column """
        if sliding_puzzle.empty[0] < of[0]:
            """ above """
            from_below = any(pos in positions_to_avoid for pos in cls.upper_bank(of))
            if from_below:
                if of[0] < sliding_puzzle.n - 1:
                    return sliding_puzzle.empty_down()
                else:
                    return sliding_puzzle.empty_left()
            else:
                return sliding_puzzle.empty_left()
        else:
            """ below """
            from_below = not any(pos in positions_to_avoid for pos in cls.lower_bank(of))
            if from_below:
                return sliding_puzzle.empty_left()
            else:
                if of[0] > 0:
                    return sliding_puzzle.empty_up()
                else:
                    return sliding_puzzle.empty_left()

    @classmethod
    def moves_on_top_of(cls, sliding_puzzle: SlidingPuzzle, of, moves, positions_to_avoid=None):
        move = cls.empty_on_top_of(sliding_puzzle, of, positions_to_avoid=positions_to_avoid)
        if isinstance(move, str) and move == cls.done:
            return
        moves.append(move)
        return cls.moves_on_top_of(sliding_puzzle.apply(move),
                                   of,
                                   moves,
                                   positions_to_avoid=positions_to_avoid)

    @classmethod
    def moves_below_of(cls, sliding_puzzle: SlidingPuzzle, of, moves, positions_to_avoid=None):
        move = cls.empty_below_of(sliding_puzzle, of, positions_to_avoid=positions_to_avoid)
        if isinstance(move, str) and move == cls.done:
            return
        moves.append(move)
        return cls.moves_below_of(sliding_puzzle.apply(move),
                                  of,
                                  moves,
                                  positions_to_avoid=positions_to_avoid)

    @classmethod
    def moves_to_right_of(cls, sliding_puzzle: SlidingPuzzle, of, moves, positions_to_avoid=None):
        move = cls.empty_to_right_of(sliding_puzzle, of, positions_to_avoid=positions_to_avoid)
        if isinstance(move, str) and move == cls.done:
            return
        moves.append(move)
        return cls.moves_to_right_of(sliding_puzzle.apply(move),
                                     of,
                                     moves,
                                     positions_to_avoid=positions_to_avoid)

    @classmethod
    def moves_to_left_of(cls, sliding_puzzle: SlidingPuzzle, of, moves, positions_to_avoid=None):
        move = cls.empty_to_left_of(sliding_puzzle, of, positions_to_avoid=positions_to_avoid)
        if isinstance(move, str) and move == cls.done:
            return
        moves.append(move)
        return cls.moves_to_left_of(sliding_puzzle.apply(move),
                                    of,
                                    moves,
                                    positions_to_avoid=positions_to_avoid)

########################################################################################################################
