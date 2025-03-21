import numpy as np
import math
class HungarianOptimizer:
    NONE = 1
    STAR = 2
    PRIME = 3
    def __init__(self):
        self.kHungarianOptimizerRowNotFound = -1
        self.kHungarianOptimizerColNotFound = -2
    def __init2(self, costs):
        self.height = costs.shape[0]
        self.width = costs.shape[1]
        self.mat_size = max(self.height, self.width)
        self.costs = np.zeros([self.mat_size, self.mat_size])
        self.costs[:self.height, :self.width] += costs
        self.__marks = np.ones_like(self.costs, np.int8)
        self.__rows_covered = np.full(self.mat_size, False)
        self.__cols_covered = np.full(self.mat_size, False)
        self.__stars_in_col = np.full(self.mat_size, 0)
        self.__assignments = np.zeros([self.mat_size*2, 2], np.int8)
    def maximize(self, costs):
        __max_cost = np.amax(costs) if costs.size > 0 else 0
        costs = __max_cost - costs
        return self.minimize(costs)
    def minimize(self, costs):
        self.__init2(costs)
        self.__do_munkres()
        return self.__find_assignments()
    def __do_munkres(self):
        max_num_iter = 1000
        num_iter = 0
        self.__fn_state = self.__reduce_rows
        while self.__fn_state != None and num_iter < max_num_iter:
            self.__fn_state()
            num_iter += 1
        if num_iter >= max_num_iter:
            self.__check_star()
    def __find_assignments(self):
        """
        Convert the final costs matrix into a set of assignments of agents to tasks.
        Return an array of pairs of integers.
        """
        assignments = []
        for row in range(self.height):
            for col in range(self.width):
                if self.__is_starred(row, col):
                    assignments.append([row, col])
                    break
        return np.array(assignments).reshape([-1, 2])
    def __reduce_rows(self):
        """
        Step 1. 
        For each row of the matrix, find the smallest element and substract it 
        from every element in its row. then, go to Step 2, __star_zeros
        """
        for row in range(self.mat_size):
            min_cost = self.costs[row, 0]
            for col in range(1, self.mat_size):
                min_cost = min(min_cost, self.costs[row, col])
            for col in range(self.mat_size):
                self.costs[row, col] -= min_cost
        self.__fn_state = self.__star_zeros
    def __star_zeros(self):
        """
        Step 2. 
        Find a zero Z in the matrix. If there is no starred zero in its row 
        or column, star Z. Repeat for every element in the matrix. Then, go to 
        Step 3. __cover_starred_zeros.
        """
        # since no rows or cols are covered on entry to this step, we use the 
        # covers as a quick way of making which rows & cols have stars in them.
        for row in range(self.mat_size):
            if self.__rows_covered[row]:
                continue
            for col in range(self.mat_size):
                if self.__cols_covered[col]:
                    continue
                if math.isclose(self.costs[row, col], 0.0):
                    self.__star(row, col)
                    self.__rows_covered[row] = True
                    self.__cols_covered[col] = True
                    break
        self.__clear_covers()
        self.__fn_state = self.__cover_starred_zeros
    def __cover_starred_zeros(self):
        """
        Step 3. 
        Cover each column containing a starred zero. If all columns are covered, 
        the starred zeros describe a complete set of unique assignments. In this 
        case, terminate the algorithm. Otherwise, go to step 4, __prime_zeros.
        """
        num_covered = 0
        for col in range(self.mat_size):
            if self.__stars_in_col[col] > 0:
                self.__cols_covered[col] = True
                num_covered += 1
        if num_covered >= self.mat_size:
            self.__fn_state = None
            return
        self.__fn_state = self.__prime_zeros
    def __prime_zeros(self):
        """
        Step 4. 
        Find a noncovered zero and prime it. If there is no starred zero in the 
        row containing this primed zero, go to step 5, __make_augmenting_path. Otherwise, cover this 
        row and uncover the column containing the starred zero. Continue in this manner 
        until there are no uncovered zeros left, then go to step 6, __augment_path.
        """
        # this loop is guaranteed to terminate in at most mat_size iters
        # as __find_zero return a location only if there is at least one uncovered
        # zero in the matrix. Each iteration, either one row is covered or the 
        # loop terminates. Since there are mat_size rows, after that many 
        # iterations there are uncovered cells and hence no uncovered zeros,
        # so the loop terminates.
        while True:
            zero_row, zero_col = self.__find_zero()
            # no uncovered zeros
            if zero_row == None:
                # No uncovered zeroes
                self.__fn_state = self.__augment_path
                return
            self.__marks[zero_row, zero_col] = HungarianOptimizer.PRIME
            star_col = self.__find_x_in_row(zero_row, HungarianOptimizer.STAR)
            if star_col != self.kHungarianOptimizerColNotFound:
                self.__rows_covered[zero_row] = True
                self.__cols_covered[star_col] = False
            else:
                self.__assignments[0][0] = zero_row
                self.__assignments[0][1] = zero_col
                self.__fn_state = self.__make_augmenting_path
                return
    def __make_augmenting_path(self):
        """
        Step 5.
        Construct a series of alternating primed and starred zeros as follows.
        Let Z0 represent the uncovered primed zero found in Step 4.
        Let Z1 denote the starred zero in the col of Z0(if any).
        Let Z2 denote the primed zero in the row of Z1 (there will always be one).
        Continue until the series terminates at a primed zero that has no starred zero in its col. 
        Unstar each starred zero of the series, star each primed zero of the series,
        erase all primes and uncover every line in the matrix.
        Return to Step 3. 
        """
        done = False
        count = 0
        while not done:
            row = self.__find_x_in_col(self.__assignments[count, 1], HungarianOptimizer.STAR)
            if row != self.kHungarianOptimizerRowNotFound:
                count += 1
                self.__assignments[count, 0] = row
                self.__assignments[count, 1] = self.__assignments[count-1, 1]
            else:
                done = True

            if not done:
                col = self.__find_x_in_row(self.__assignments[count, 0], HungarianOptimizer.PRIME)
                count += 1
                self.__assignments[count, 0] = self.__assignments[count-1, 0]
                self.__assignments[count, 1] = col

        # then, modify it
        for i in range(count+1):
            row = self.__assignments[i, 0]
            col = self.__assignments[i, 1]
            if self.__is_starred(row, col):
                self.__unstar(row, col)
            else:
                self.__star(row, col)
        self.__clear_covers()
        self.__clear_primes()
        self.__fn_state = self.__cover_starred_zeros
    def __augment_path(self):
        """
        Step 6.
        Add the smallest uncovered value in the matrix to every element of each
        covered row, and subtract it from every element of each uncovered col.
        Return to Step 4 without altering any stars, primes or covered lines
        """
        minval = self.__find_smallest_uncovered()
        self.costs[self.__rows_covered] += minval
        self.costs[:, np.logical_not(self.__cols_covered)] -= minval
        self.__fn_state = self.__prime_zeros
    def __find_smallest_uncovered(self):
        """Find the smallest uncovered cell in the matrix."""
        minval = math.inf
        uncov_col = []
        uncov_row = []
        for i in range(self.mat_size):
            if not self.__rows_covered[i]:
                uncov_row.append(i)
            if not self.__cols_covered[i]:
                uncov_col.append(i)
        for row in uncov_row:
            for col in uncov_col:
                minval = min(minval, self.costs[row, col])
        return minval
    def __find_x_in_row(self, row, x):
        star_idxs, = np.where(self.__marks[row] == x)
        if len(star_idxs) != 0:
            return star_idxs[0]
        return self.kHungarianOptimizerColNotFound
    def __find_x_in_col(self, col, x):
        star_idxs, = np.where(self.__marks[:, col].squeeze() == x)
        if len(star_idxs) != 0:
            return star_idxs[0]
        return self.kHungarianOptimizerRowNotFound

    def __find_zero(self):
        uncov_col = []
        uncov_row = []
        for i in range(self.mat_size):
            if not self.__rows_covered[i]:
                uncov_row.append(i)
            if not self.__cols_covered[i]:
                uncov_col.append(i)
        for row in uncov_row:
            for col in uncov_col:
                if math.isclose(self.costs[row, col], 0.0):
                    return row, col
        return None, None

    def __clear_covers(self):
        for i in range(self.mat_size):
            self.__rows_covered[i] = False
            self.__cols_covered[i] = False
    def __clear_primes(self):
        for row in range(self.mat_size):
            for col in range(self.mat_size):
                if self.__marks[row, col] == HungarianOptimizer.PRIME:
                    self.__marks[row, col] = HungarianOptimizer.NONE
    def print_matrix(self):
        for row in range(self.mat_size):
            for col in range(self.mat_size):
                print(f"{self.costs[row, col]} ", end="")
                if self.__is_starred(row, col):
                    print("*", end="")
                if self.__marks[row, col] == HungarianOptimizer.PRIME:
                    print("'", end="")
            print('\n')
    def __star(self, row, col):
        self.__marks[row, col] = HungarianOptimizer.STAR
        self.__stars_in_col[col] += 1
    def __unstar(self, row, col):
        self.__marks[row, col] = HungarianOptimizer.NONE
        self.__stars_in_col[col] -= 1
    def __is_starred(self, row, col):
        return self.__marks[row, col] == HungarianOptimizer.STAR
    def __check_star(self):
        for row in range(self.height):
            star_col = -1
            is_single = True
            for col in range(self.width):
                if self.__is_starred(row, col):
                    if star_col == -1:
                        star_col = col
                    else:
                        is_single = False
                        break
            if not is_single:
                for col in range(self.width):
                    self.__unstar(row, col)