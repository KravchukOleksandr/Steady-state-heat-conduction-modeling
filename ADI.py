import numpy as np
import time


class Thomas1D:
    '''
    lb_cond/rb_cond - left/right boundary condition
    lb_cond = (alpha0, alpha1, A)
    rb_cond = (beta0, beta1, B)
    '''
    def __init__(self, a, b, c, d, lb_cond, rb_cond, h):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        self.y = np.zeros(a.shape)
        self.p = np.zeros(a.shape)
        self.q = np.zeros(a.shape)

        self.lb_cond = lb_cond
        self.rb_cond = rb_cond
        self.h = h
    
    def solve(self):
        self.__forward()
        self.__backward()
        return self.y

    def __forward(self):
        n = self.y.shape[0] - 1
        h = self.h
        alpha0, alpha1, A = self.lb_cond

        self.p[0] = alpha1 / (alpha1 - alpha0 * h)
        self.q[0] = -A * h / (alpha1 - alpha0 * h)
        for i in range(1, n):
            eps = 1e-12
            denominator = self.b[i] - self.a[i] * self.p[i - 1] + eps
            if denominator > 1.05 * eps:
                self.p[i] = self.c[i] / denominator
                self.q[i] = (self.a[i] * self.q[i - 1] - self.d[i]) / denominator
            else:
                self.p[i] = 0
                self.q[i] = self.d[i]
        
    def __backward(self):
        n = self.y.shape[0] - 1
        h = self.h
        alpha0, alpha1, A = self.lb_cond
        beta0, beta1, B = self.rb_cond

        self.y[n] = (h * B + beta1 * self.q[n - 1]) / (h * beta0 - beta1 * self.p[n - 1] + beta1)
        for i in range(n - 1, 0, -1):
            self.y[i] = self.p[i] * self.y[i + 1] + self.q[i]
        self.y[0] = (A * h - alpha1 * self.y[1]) / (alpha0 * h - alpha1)


class ADI2D:
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
        self.__init_coeffs()
        self.__init_boundary_conditions(x_deriv2['bound_cond'], y_deriv2['bound_cond'])
        self.__init_inner_conds()
    
    def solve(self, max_iter_num, criterion = 1e-7):
        U = np.zeros(self.lattice_size)
        U_next = U.copy()
        
        work_time = time.time()
        for k in range(max_iter_num):
            # columns
            for j in range(1, self.lattice_size[0] - 1):
                a = self.AY[:, j]
                c = self.CY[:, j]
                b = self.B[:, j]
                d = self.D[:, j] - self.AX[:, j] * U_next[:, j - 1] - self.CX[:, j] * U_next[:, j + 1]
                lb_cond = self.top_cond[j]
                rb_cond = self.bottom_cond[j]

                column_model = Thomas1D(a, b, c, d, lb_cond, rb_cond, self.hy)
                U_next[:, j] = column_model.solve()

            # rows
            for i in range(1, self.lattice_size[1] - 1):
                a = self.AX[i, :]
                c = self.CX[i, :]
                b = self.B[i, :]
                d = self.D[i, :] - self.AY[i, :] * U_next[i - 1, :] - self.CY[i, :] * U_next[i + 1, :]
                lb_cond = self.left_cond[i]
                rb_cond = self.right_cond[i]

                row_model = Thomas1D(a, b, c, d, lb_cond, rb_cond, self.hx)
                U_next[i, :] = row_model.solve()
            
            # check accuracy
            iter_deltaU = np.mean(np.abs(U - U_next) / np.abs(U_next + 1e-12))
            if iter_deltaU < criterion:
                return U
            if time.time() - work_time > 0.5:
                self.history.append(iter_deltaU)
                work_time = time.time()
            U = U_next.copy()
        
        return U

    def __init_coeffs(self):
        self.AX = self.P(self.X, self.Y) / self.hx**2
        self.CX = self.P(self.X, self.Y) / self.hx**2
        self.AY = self.Q(self.X, self.Y) / self.hy**2
        self.CY = self.Q(self.X, self.Y) / self.hy**2
        self.B = 2 * self.AX + 2 * self.AY - self.S(self.X, self.Y)
        self.D = self.f(self.X, self.Y)
    
    def __init_boundary_conditions(self, x_cond, y_cond):
        self.left_cond = [(x_cond[0][0], x_cond[0][1], x_cond[0][2](y_i)) for y_i in self.y]
        self.right_cond = [(x_cond[1][0], x_cond[1][1], x_cond[1][2](y_i)) for y_i in self.y]
        self.top_cond = [(y_cond[0][0], y_cond[0][1], y_cond[0][2](x_j)) for x_j in self.x]
        self.bottom_cond = [(y_cond[1][0], y_cond[1][1], y_cond[1][2](x_j)) for x_j in self.x]
    
    def __init_inner_conds(self):
        for inner_cond in self.inner_conds:
            (x1, x2), (y1, y2) = inner_cond['bounds']
            temp_mask = ((self.X >= x1) & (self.X <= x2) & (self.Y >= y1) & (self.Y <= y2))

            self.AX[temp_mask] = 0
            self.CX[temp_mask] = 0
            self.AY[temp_mask] = 0
            self.CY[temp_mask] = 0

            self.B[temp_mask] = -1
            self.D[temp_mask] = inner_cond['u_value']
