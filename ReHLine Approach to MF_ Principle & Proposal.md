## **Principle**
### **1. Target**
Given $Y$ and $r$, train $P \& Q$ such that $\hat{Y} = PQ^T+A+B+\mu J_{nm}$ can be used to fill NA values(or $0$) in $Y$, where:


- $
Y = \begin{pmatrix}
y_{11}& y_{12} & y_{13} & \cdots &y_{1m} \\
y_{21}& y_{22} & y_{23} & \cdots &y_{2m} \\
\vdots & \vdots & \vdots & & \vdots \\
y_{n1}& y_{n2} & y_{n3} & \cdots &y_{nm}
\end{pmatrix}$ is input matrix 
<br>


-  $
P = \begin{pmatrix}
p_{11}& p_{12} & \cdots &p_{1r} \\
p_{21}& p_{22} & \cdots &p_{2r} \\
\vdots & \vdots & \vdots &  \vdots \\
p_{n1}& p_{n2} & \cdots &p_{nr} \\
\end{pmatrix}$ and $
Q = \begin{pmatrix}
q_{11}& q_{12} & \cdots &q_{1r} \\
q_{21}& q_{22} & \cdots &q_{2r} \\
\vdots & \vdots & \vdots &  \vdots \\
q_{m1}& q_{m2} & \cdots &q_{mr} \\
\end{pmatrix}$ are factor matrices
<br>

- $
A = \begin{pmatrix}
a_{1}& a_{1} & a_{1} & \cdots &a_{1} \\
a_{2}& a_{2} & a_{2} & \cdots &a_{2}  \\
\vdots & \vdots & \vdots & & \vdots \\
a_{n}& a_{n} & a_{n} & \cdots &a_{n} 
\end{pmatrix}$ and $B = \begin{pmatrix}
b_{1}& b_{2} & b_{3} & \cdots &b_{m} \\
b_{1}& b_{2} & b_{3} & \cdots &b_{m}  \\
\vdots & \vdots & \vdots & & \vdots \\
b_{1}& b_{2} & b_{3} & \cdots &b_{m} 
\end{pmatrix}$ are bias matrices
<br>

-  $\mu$ is global bias
<br>

-  $r$ is rank of matrix factorization
<br>



### **2. Training Algorithm**
####  $P \text- Q$ Circle: Blockwise Coordinate Descent
- Fix $Q$ to update $P$:
  For each $k$ from $1$ to $n$:
  -  Objective function is:
    $$
     Obj( P_{[k,:]}) = L[\ (Y_{[k,:]})^T\ \ , \ \  Q  (P_{[k,:]})^T\ ] + A + B + \mu \cdot J_{nm}
    $$

  - Use `ReHLine` to update:
    $$
    P^*_{[k,:]} = [\argmin_{p_k\in\mathbb{R}^r} Obj(p_k)]^T
    $$
	i.e. the k-th row of $P$
<br>

- Fix $P$ to update $Q$:

  For each $h$ from $1$ to $m$:
  -  Objective function is:
    $$
     Obj( Q_{[h,:]}) = L[\ Y_{[:,h]}\ \ , \ \  P  (Q_{[h,:]})^T\ ] + A + B + \mu \cdot J_{nm}
    $$
  - Use `ReHLine` to update:
    $$
    Q^*_{[h,:]} = [\argmin_{q_h\in\mathbb{R}^r} Obj(q_h)]^T
    $$
	i.e. the h-th row of $Q$

---

## **Proposal**
### **1. Class Specification**
```python
class reline.mfL(loss, constrain=[], C=1.0,
				 U=np.empty(shape=(0, 0)), V=np.empty(shape=(0, 0)),
				 Tau=np.empty(shape=(0, 0)), S=np.empty(shape=(0, 0)), T=np.empty(shape=(0, 0)),
				 A=np.empty(shape=(0, 0)), b=np.empty(shape=0),
				 max_iter=1000, tol=0.0001, shrink=1, warm_start=0, verbose=0, trace_freq=100) )
```
Bases: `rehline._base._BaseReHLine`, `sklearn.base.BaseEstimator`

#### Parameters:
- **loss : dict**
  A dictionary specifying the loss function parameters, including:
  - Squared Loss `square`: $L(y, \hat{y})=(y-\hat{y})^2,\ y\in R$
  - Absolute Loss `absolute`: $L(y, \hat{y})=|y-\hat{y}|,\  y\in R$
  - Huber Loss `huber`: $L(y, \hat{y})=max(0,1-y\hat{y}),\  y\in {0,1}$

- **U, V : array of shape (L, n_samples), default=np.empty(shape=(0, 0))**
The parameters pertaining to the ReLU part in the loss function.

- **Tau, S, T : array of shape (H, n_samples), default=np.empty(shape=(0, 0))**
The parameters pertaining to the ReHU part in the loss function.

- **verbose : int, default=0**
Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in liblinear that, if enabled, may not work properly in a multithreaded context.

- **max_iter : int, default=1000**
The maximum number of iterations to be run.


<span style="color:darkgreen;">

- **(future) C : float, default=1.0**
  Regularization parameter. The strength of the regularization is inversely proportional to $C$. Must be strictly positive. $C$ will be absorbed by the ReHLine parameters when `self.make_ReLHLoss` is conducted.
  
- **(future) constraint : list of dict**
  A list of dictionaries, where each dictionary represents a constraint. Each dictionary must contain a 'name' key, which specifies the type of constraint.

- **(future) A : array of shape (K, n_features), default=np.empty(shape=(0, 0))**
The coefficient matrix in the linear constraint.

- **(future) b : array of shape (K, ), default=np.empty(shape=0)**
The intercept vector in the linear constraint.


</span>

<br>

#### Attributes:
- **coef_array : array-like**
  The optimized laten vector (one row of $P \text{ or } Q$).

- **n_iter : int**
  The number of iterations performed by the `ReHLine_solver`.

- **opt_result : object**
  The optimization result object.

- **primal_obj : array-like**
  The primal objective function values.

- **other attributes sopported by** `_BaseReHLine`

<br>


####  Methods
- **fit(X, y)**
  Fit the model based on the given training data. Here $X=P, y = Y_{column} \text{ or } X=Q, y=Y_{row}$ and should be applIed
  
<span style="color:darkgreen;">
	
- **(future) decision_function(X)**
  The decision function evaluated on the given dataset. Here $X$ is the input data for evaluation.
  
</span>
<br>

### **2. Workflow**
- **Loss -> ReHLine Parameters**
  - [x] Absolute Loss `absolute`
  $\text{Let } S=\text{length of }y$
	$U = \begin{pmatrix}
		1&1 &1 & \cdots &1 \\
		-1&-1 & -1 & \cdots &-1 \\
	\end{pmatrix} 
	\in R^{2\times S}$
	$V = \begin{pmatrix}
		-y^T \\
		y^T \\
	\end{pmatrix} 
	\in R^{2\times S}$

  - [ ] Squared Loss `square`

  - [ ] Huber Loss `huber`

<br>

- **Implementation**
  - [ ] P-Q Circle via `ReHLine`
  - [ ] Methods for Updating $A,B,\mu$
  - [ ] Parallel Computing

<br>

- **Encapsulation**
So far `reline.mfL()` is designed to solve the sub-problem(P-Q Circle) of MF. Ultimately, through structural design, this class can be encapsulated to achieve:
  - [ ] One-Click Solution for Matrix Factorization:
        Input $Y$ then output $P,Q$ results.

  - [ ] Recommender System Problem:
		 Transform *[userID, itemID, rating]* dataset into one matrix $Y$ then conduct MF to it.
		


