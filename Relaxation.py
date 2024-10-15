import numpy as np
import time


class Relaxation2D:
    '''
    Equation:
    P(x, y) * d^2(U)/d(x^2) + Q(x, y) * d^2(U)/d(y^2) + S(x, y) * U = f(x, y)

    Parameters:
    -----------
    1) x_deriv2 and y_deriv2 : dict
        - 'func' - P(x, y) or Q(x, y) function
        - 'bounds' - (min, max) point
        - 'bound_cond' - (left/top, right/bottom) boundary conditions,
                         each bound condition is (alpha0, alpha1, func)
                         func is callable analog of 1D A, B 
        - 'points_num' - number of point in lattice's axis_i
    2) S : callable
          Function S(x, y)
    3) f : callable
          Function f(x, y)
    4) inner_conds : dict
        - 'bounds' - [(x_min, x_max), (y_min, y_max)] inner condition area
        - 'u_value' - inner condition value
    '''
    def  __init__(self, x_deriv2, y_deriv2, S, f, inner_conds=[]):
        self.x_min, self.x_max = x_deriv2['bounds']
        self.y_min, self.y_max = y_deriv2['bounds']
        self.hx = (self.x_max - self.x_min) / (x_deriv2['points_num'] - 1) 
        self.hy = (self.y_max - self.y_min) / (y_deriv2['points_num'] - 1)
        self.lattice_size = (y_deriv2['points_num'], x_deriv2['points_num'])
        self.history = []

        n_rows, n_cols = self.lattice_size
        self.x = self.x_min + self.hx * np.arange(n_cols)
        self.y = self.y_min + self.hy * np.arange(n_rows)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.P = x_deriv2['Func']
        self.Q = y_deriv2['Func']
        self.S = S
        self.f = f
        self.inner_conds = inner_conds
        
        self.__init_def_coeffs()
        self.__init_boundary_conditions(x_deriv2['bound_cond'], y_deriv2['bound_cond'])
        self.__init_inner_conds()

    def solve(self, max_iter_num, criterion = 1e-7, w = 1.5):
        n, m = self.lattice_size
        U = np.random.normal(size=self.lattice_size)
        U_next = U.copy()
    
        work_time = time.time()
        for k in range(max_iter_num):
            for i in range(n):
                for j in range(m):
                    U_left = U_next[i, j - 1] if j > 0 else 0
                    U_right = U_next[i, j + 1] if j < m - 1 else 0
                    U_top = U_next[i - 1, j] if i > 0 else 0
                    U_bottom = U_next[i + 1, j] if i < n - 1 else 0
                    U_next[i, j] = ((1 - w) * U[i, j] + (w / self.B[i, j]) * (self.D[i, j] - 
                                    self.AX[i, j] * U_left -
                                    self.CX[i, j] * U_right -
                                    self.AY[i, j] * U_top -
                                    self.CY[i, j] * U_bottom))
            # check accuracy
            iter_deltaU = np.mean(np.abs(U - U_next) / np.abs(U_next + 1e-12))
            if iter_deltaU < criterion:
                return U
            if time.time() - work_time > 0.5:
                self.history.append(iter_deltaU)
                work_time = time.time()
            U = U_next.copy()

        return U_next

    def __init_def_coeffs(self):
        self.AX = self.P(self.X, self.Y) / self.hx**2
        self.CX = self.P(self.X, self.Y) / self.hx**2
        self.AY = self.Q(self.X, self.Y) / self.hy**2
        self.CY = self.Q(self.X, self.Y) / self.hy**2
        self.B = self.S(self.X, self.Y) - 2 * self.AX - 2 * self.AY
        self.D = self.f(self.X, self.Y)
    
    def __init_boundary_conditions(self, x_cond, y_cond):
        self.left_cond = np.array([(x_cond[0][0], x_cond[0][1], x_cond[0][2](y_i)) for y_i in self.y]).reshape(-1, 3)
        self.right_cond = np.array([(x_cond[1][0], x_cond[1][1], x_cond[1][2](y_i)) for y_i in self.y]).reshape(-1, 3)
        self.top_cond = np.array([(y_cond[0][0], y_cond[0][1], y_cond[0][2](x_j)) for x_j in self.x]).reshape(-1, 3)
        self.bottom_cond = np.array([(y_cond[1][0], y_cond[1][1], y_cond[1][2](x_j)) for x_j in self.x]).reshape(-1, 3)

        n, m = self.lattice_size 

        # left boundary
        a_left0, a_left1, f_left = self.left_cond[:, 0], self.left_cond[:, 1], self.left_cond[:, 2]
        self.AX[:, 0] = np.zeros(n)
        self.B[:, 0] = a_left0 - a_left1 / self.hx
        self.CX[:, 0] = a_left1 / self.hx
        self.D[:, 0] = f_left

        self.AY[:, 0] = np.zeros(n)
        self.CY[:, 0] = np.zeros(n)

        # right boundary
        b_right0, b_right1, f_right = self.right_cond[:, 0], self.right_cond[:, 1], self.right_cond[:, 2]
        self.AX[:, m - 1] = -b_right1 / self.hx
        self.B[:, m - 1] = b_right0 + b_right1 / self.hx
        self.CX[:, m - 1] = np.zeros(n)
        self.D[:, m - 1] = f_right

        self.AY[:, m - 1] = np.zeros(n)
        self.CY[:, m - 1] = np.zeros(n)

        # top boundary
        a_top0, a_top1, f_top = self.top_cond[:, 0], self.top_cond[:, 1], self.top_cond[:, 2]
        self.AY[0, :] = np.zeros(m)
        self.B[0, :] = a_top0 - a_top1 / self.hy
        self.CY[0, :] = a_top1 / self.hy
        self.D[0, :] = f_top

        self.AX[0, :] = np.zeros(n)
        self.CX[0, :] = np.zeros(n)

        # bottom boundary
        b_bottom0, b_bottom1, f_bottom = self.bottom_cond[:, 0], self.bottom_cond[:, 1], self.bottom_cond[:, 2]
        self.AY[n - 1, :] = -b_bottom1 / self.hy
        self.B[n - 1, :] = b_bottom0 + b_bottom1 / self.hy
        self.CY[n - 1, :] = np.zeros(m)
        self.D[n - 1, :] = f_bottom

        self.AX[n - 1, :] = np.zeros(n)
        self.CX[n - 1, :] = np.zeros(n)

    def __init_inner_conds(self):
        for inner_cond in self.inner_conds:
            (x1, x2), (y1, y2) = inner_cond['bounds']
            temp_mask = ((self.X >= x1) & (self.X <= x2) & (self.Y >= y1) & (self.Y <= y2))

            self.AX[temp_mask] = 0
            self.CX[temp_mask] = 0
            self.AY[temp_mask] = 0
            self.CY[temp_mask] = 0

            self.B[temp_mask] = 1
            self.D[temp_mask] = inner_cond['u_value']
