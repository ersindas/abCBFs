# Supplementary Material: Robust Backup Controller and Backup Set for the Uncertain Planar Quadrotor Model

This material was part of the paper in an earlier version, but it was removed from the final journal version because of the page limit. The paper now points here instead of to a supplementary material section.

This gives the proof that the backup controller $k_{\rm b}$ in eq. (38) of the paper keeps the backup set $\mathcal{C}_{\rm B}$ in eq. (39) forward invariant for every possible parameter $\theta \in \Theta$, i.e., that Assumption 2 holds for the quadrotor model in eq. (36). The corresponding numerical check is implemented in `quadrotor/control.py`, in the function `checkValidBackup`.

## Notation

We use $\lambda_{\min}(S)$ and $\lambda_{\max}(S)$ for the smallest and largest eigenvalues of a symmetric matrix $S$, and define

$$
a_m(\theta) \triangleq \frac{1}{m_0} + \delta_m, \qquad a_J(\theta) \triangleq \frac{1}{J_0} + \delta_J, \qquad e_1 \triangleq \begin{bmatrix}1 & 0\end{bmatrix}^\top,
$$

$$
K_\vartheta \triangleq \begin{bmatrix}k_\vartheta & k_\omega\end{bmatrix}, \qquad b_\vartheta \triangleq \begin{bmatrix}0 & 1\end{bmatrix}^\top, \qquad A_\vartheta(a) \triangleq \begin{bmatrix}0 & 1 \\ -a k_\vartheta & -a k_\omega\end{bmatrix},
$$

$$
\Delta_\vartheta \triangleq \sqrt{\varrho_\vartheta\, e_1^\top P_\vartheta^{-1} e_1}, \qquad \Delta_M \triangleq \sqrt{\varrho_\vartheta\, K_\vartheta P_\vartheta^{-1} K_\vartheta^\top}, \qquad \bar\vartheta \triangleq \vartheta_{\rm r} + \Delta_\vartheta.
$$

The parameter bounds in Assumption 1 give known constants such that, for all $\theta \in \Theta$,

$$
\delta_g^- \le \delta_g \le \delta_g^+, \qquad \delta_\ell^- \le \delta_\ell \le \delta_\ell^+, \qquad 0 < a_J^- \le a_J(\theta) \le a_J^+, \qquad 0 < a_m^- \le a_m(\theta) \le a_m^+.
$$

## Proposition 1

Consider the uncertain quadrotor model (36), the safety function (37) with the safe set $\mathcal{C}_{\rm S}$, the backup controller (38), and the backup set $\mathcal{C}_{\rm B}$ in (39). If the controller gains $k_\vartheta > 0$, $k_\omega > 0$ and the design parameters $P_\vartheta = P_\vartheta^\top \succ 0$, $Q_\vartheta = Q_\vartheta^\top \succ 0$, $\varrho_\vartheta > 0$, $\vartheta_{\rm r} \in (0, \vartheta_{\max})$, $r_x, r_z > 0$, and $r_\vartheta \in (0, \vartheta_{\max}^2)$, with $\bar\vartheta < \pi/2$, are chosen such that

1. $A_\vartheta(a_J^-)^\top P_\vartheta + P_\vartheta A_\vartheta(a_J^-) \preceq -Q_\vartheta$
2. $A_\vartheta(a_J^+)^\top P_\vartheta + P_\vartheta A_\vartheta(a_J^+) \preceq -Q_\vartheta$
3. $\Delta_M \le M_{\max}$
4. $\vartheta_{\rm r} - \Delta_\vartheta \ge 0$
5. $\vartheta_{\rm r} + \Delta_\vartheta \le \sqrt{\vartheta_{\max}^2 - r_\vartheta}$
6. $\varrho_\vartheta \ge \lambda_{\max}(P_\vartheta) \left( \dfrac{2 \lVert P_\vartheta b_\vartheta \rVert F_{\max} \max\{|\delta_\ell^-|, |\delta_\ell^+|\}}{\lambda_{\min}(Q_\vartheta)} \right)^2$
7. $a_m^- F_{\max} \cos(\bar\vartheta) - g_0 - \delta_g^+ \ge 0$
8. $e^{-\kappa r_x} + e^{-\kappa r_z} + e^{-\kappa r_\vartheta} \le 1$

then the moment saturation in (38) is inactive on $\mathcal{C}_{\rm B}$, $\mathcal{C}_{\rm B} \subseteq \mathcal{C}_{\rm S}$ holds, and the backup controller (38) renders $\mathcal{C}_{\rm B}$ robustly forward invariant for (36) for all $\theta \in \Theta$, i.e., Assumption 2 is satisfied.

## Proof

We first consider the pitch dynamics. On the set $\mathcal{C}_{\rm B}$, $h_{\vartheta,\rm b}(x) \ge 0$ implies $y_\vartheta^\top P_\vartheta y_\vartheta \le \varrho_\vartheta$, and hence

$$
|\vartheta - \vartheta_{\rm r}| \le \Delta_\vartheta, \qquad |K_\vartheta y_\vartheta| \le \Delta_M.
$$

Thus, by conditions 4 and 5, the pitch angle satisfies $0 \le \vartheta \le \bar\vartheta \le \sqrt{\vartheta_{\max}^2 - r_\vartheta}$ on $\mathcal{C}_{\rm B}$. Moreover, condition 3 gives $|K_\vartheta y_\vartheta| \le M_{\max}$, so the moment saturation in (38) is inactive on $\mathcal{C}_{\rm B}$.

We next check the robust invariance of the pitch dynamics under the backup controller, which satisfy

$$
\dot y_\vartheta = A_\vartheta(a_J(\theta)) y_\vartheta + b_\vartheta \delta_\ell F_{\max}.
$$

Since $A_\vartheta(a)$ depends affinely on $a$, conditions 1 and 2 imply

$$
A_\vartheta(a)^\top P_\vartheta + P_\vartheta A_\vartheta(a) \preceq -Q_\vartheta, \quad \forall a \in [a_J^-, a_J^+].
$$

Therefore, with $V_\vartheta \triangleq y_\vartheta^\top P_\vartheta y_\vartheta$, we have

$$
\dot V_\vartheta = y_\vartheta^\top \left( A_\vartheta(a_J(\theta))^\top P_\vartheta + P_\vartheta A_\vartheta(a_J(\theta)) \right) y_\vartheta + 2 y_\vartheta^\top P_\vartheta b_\vartheta \delta_\ell F_{\max}
$$
$$
\le -\lambda_{\min}(Q_\vartheta) \lVert y_\vartheta \rVert^2 + 2 F_{\max} \lVert P_\vartheta b_\vartheta \rVert \max\{|\delta_\ell^-|, |\delta_\ell^+|\} \lVert y_\vartheta \rVert.
$$

On the boundary $h_{\vartheta,\rm b}(x) = 0$ we have $V_\vartheta = \varrho_\vartheta$, and hence $\lVert y_\vartheta \rVert \ge \sqrt{\varrho_\vartheta / \lambda_{\max}(P_\vartheta)} > 0$. Therefore, by condition 6, $\dot V_\vartheta \le 0$ on this boundary. Equivalently, $\dot h_{\vartheta,\rm b}(x) \ge 0$ there, so the pitch component of $\mathcal{C}_{\rm B}$ is robustly forward invariant.

We now check the horizontal component of the motion. Since $0 \le \vartheta \le \bar\vartheta < \pi/2$ on $\mathcal{C}_{\rm B}$, the horizontal dynamics of (36) under (38) satisfy

$$
\dot v_x = a_m(\theta) F_{\max} \sin(\vartheta) - c_x v_x.
$$

On the boundary $h_{v_x,\rm b}(x) = 0$ we have $v_x = 0$, and therefore

$$
\dot h_{v_x,\rm b}(x) = a_m(\theta) F_{\max} \sin(\vartheta) \ge 0,
$$

using $a_m(\theta) > 0$ and $\vartheta \ge 0$. Hence the $v_x$ component of $\mathcal{C}_{\rm B}$ is robustly forward invariant. Moreover, on $\mathcal{C}_{\rm B}$, $v_x \ge 0$, so $\dot h_{x,\rm b}(x) = v_x \ge 0$, and the surface clearance component of $\mathcal{C}_{\rm B}$ is robustly forward invariant too.

We next check the altitude component of the motion. On the boundary $h_{v_z,\rm b}(x) = 0$ we have $v_z = 0$, and the vertical dynamics of (36) satisfy

$$
\dot h_{v_z,\rm b}(x) = \dot v_z = -g_0 - \delta_g + a_m(\theta) F_{\max} \cos(\vartheta) \ge -g_0 - \delta_g^+ + a_m^- F_{\max} \cos(\bar\vartheta) \ge 0,
$$

using $0 \le \vartheta \le \bar\vartheta < \pi/2$ and condition 7. Therefore the $v_z$ component of $\mathcal{C}_{\rm B}$ is robustly forward invariant. Furthermore, on $\mathcal{C}_{\rm B}$, $v_z \ge 0$, so $\dot h_{z,\rm b}(x) = v_z \ge 0$, and the altitude clearance component of $\mathcal{C}_{\rm B}$ is robustly forward invariant as well.

Finally, we show $\mathcal{C}_{\rm B} \subseteq \mathcal{C}_{\rm S}$. Since $h_{x,\rm b}(x) \ge 0$ and $h_{z,\rm b}(x) \ge 0$, we have $h_x(x) \ge r_x$ and $h_z(x) \ge r_z$. Moreover, by condition 5, $h_{\vartheta,\rm b}(x) \ge 0$ implies $h_\vartheta(x) = \vartheta_{\max}^2 - \vartheta^2 \ge r_\vartheta$. Therefore, by condition 8, the smooth minimum safety function (37) satisfies $h(x) \ge 0$ for all $x \in \mathcal{C}_{\rm B}$, i.e., $\mathcal{C}_{\rm B} \subseteq \mathcal{C}_{\rm S}$.

Combining the pitch, horizontal, and altitude invariance results with the input admissibility and $\mathcal{C}_{\rm B} \subseteq \mathcal{C}_{\rm S}$, the backup controller (38) renders the backup set (39) robustly forward invariant for (36) for all $\theta \in \Theta$. $\blacksquare$
