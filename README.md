# Steady-state-heat-conduction-modeling
ADI, Matrix Thomas and Relaxation methods for solving 2D steady-state heat equation  

Equation form:  
$` P(x, y) \frac{\partial^2 u}{\partial x^2} + Q(x, y) \frac{\partial^2 u}{\partial y^2} + S(x, y) = f(x, y) `$, where:  
$` P(x, y), Q(x, y)`$ - material properties,  
$` S(x, y)`$ - the internal heat generation (or absorption) within the material,  
$` f(x, y)`$ - external heat input.  

Boundary conditions:  
$` \alpha_0 U(x_0, y) + \alpha_1 \frac{\partial U}{\partial x} \Big|_{(x_0, y)} = g_{\alpha}(y)`$,  
$` \beta_0 \cdot U(x_n, y) + \beta_1 \frac{\partial U}{\partial x} \Big|_{(x_n, y)} = g_{\beta}(y)`$,  
$` \gamma_0 \cdot U(x, y_0) + \gamma_1 \frac{\partial U}{\partial y} \Big|_{(x, y_0)} = g_{\gamma}(x)`$,  
$` \delta_0 \cdot U(x, y_n) + \delta_1 \frac{\partial U}{\partial y} \Big|_{(x, y_n)} = g_{\delta}(x)`$.  

Internal conditions:  
$` U = \text{const}`$ on $`G_i, i = \overline{1,k}`$  
$` G_i : [x_1, x_2] \times [y_1, y_2] `$
