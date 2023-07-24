# Function optimization
This repository implements representative methods for numerically computing extrema of multivariate functions. All of these are iterative solutions in which the function searches for a direction in which the function increases or decreases from an initial position given to the domain of the function, moves in that direction, and repeats this until convergence.

<br></br>

## Gradient method(1 variable case)
The function $f(x)$ can be computed numerically as follows, knowing that there is only one x with $f\prime(x)=0$ in the domain under consideration.
First, a point $x_0$, which is considered to be close to the maximum value, is given as an initial value. If $f\prime(x)=0$, take the maximum value there. If $f\prime(x)=0$ is positive, go right on the x-axis. If $f\prime(x)=0$ is negative, go left on the x-axis.

### Algorighm
#### 1. Give the initial value of $x$, let $h \leftarrow h_0$

#### 2. Define as follows

$$
h=sgn(f\prime(x))\lVert h \rVert, X\leftarrow x, X\prime \leftarrow x+h
$$

$sgn$ is the sign function.

#### 3. If $f(X) < f (X\prime)$, do the following calculations
- Repeat following calculation until $f(X)\geqq f(X\prime)$

$$
h\leftarrow2h, X\leftarrow X\prime, X\prime \leftarrow X+h
$$

- Update following variables

$$
x\leftarrow X, h\leftarrow h/2
$$

#### 4. If not $f(X) < f (X\prime)$, do the following calculations

- Repeat following calculation until $f(X)\leqq f(X\prime)$

$$
h\leftarrow h/2, X\prime \leftarrow X\prime-h
$$

- Update following variables

$$
x\leftarrow X\prime, h\leftarrow 2h
$$

#### 5. Go back to step2 and repeat them until $\lVert f\prime(x) \rVert \leqq \epsilon$

#### 6. Return the resulting x

You can try gradient method by runngin below command. In this example, we want to find the maximum value of $y=-x^2$.

```bash
python3 gradient_method.py
```

<img src='images/gradient_method.png' width='500'>

<br></br>

## Gradient method(Multivariable case)
The gradient method for finding the maximum value of the bivariate function $f(x,y)$ is as follows. First, a point $(x_0,y_0)$ that is considered to be close to the point of maximum value is given as an initial value. Since the direction in which the function value increases the most is given by the gradient $\triangledown f$, we proceed to the point where the function value reaches its maximum on a straight line in that direction. At that point, the gradient $\triangledown f$ is calculated again, and the same process is repeated. This is done until convergence is reached.

### Algorithm
#### 1. Give initial value of $x$

#### 2. Linear search with $F\prime$ for function $F(t)=f(x+t\triangledown f(x))$
Putting $x(t)=x_0+t\triangledown f_0$, differentiating $F(t)=f(x(t))$ with $t$ yields

$$
\frac{dF}{dt}=\sum_{i=1}^n\frac{\partial f}{\partial x_i} \frac{dx_i}{dt}=(\triangledown f, \triangledown f_0)
$$

#### 3. Let $\triangle x \leftarrow t\triangledown f(x),x \leftarrow x+\triangle x$ using $t$ obtained in step 2

#### 4. Go back to step2 and repeat this loop until $\|\triangle x\|<\delta$

#### 5. Return x

At a point determined by a gradient method linear search for the function $f(x_1,...,x_n)$, the isosurface of $f(x_1,...,x_n)$ passing through that point is tangent to the search line. Therefore, the direction of the next linear search is orthogonal to the previous search direction.

You can try gradient method in the multivariable case by runngin below command. In this example, we want to find the maximum value of $z=-x^2-y^2$.

```bash
python3 hill_climbing.py
```

<img src='images/multi_1.png' width='500'>

<img src='images/multi_2.png' width='500'>

<br></br>

## Reference
- [Optimization Mathematics That You Can Understand: From Fundamental Principles to Calculation Methods](https://www.amazon.co.jp/-/en/%E9%87%91%E8%B0%B7-%E5%81%A5%E4%B8%80/dp/4320017862/ref=sr_1_1?adgrpid=52832566945&hvadid=658804283256&hvdev=c&hvlocphy=9163303&hvnetw=g&hvqmt=e&hvrand=13823473628811259621&hvtargid=kwd-333784071069&hydadcr=27705_14678557&jp-ad-ap=0&keywords=%E3%81%93%E3%82%8C%E3%81%AA%E3%82%89%E5%88%86%E3%81%8B%E3%82%8B%E6%9C%80%E9%81%A9%E5%8C%96%E6%95%B0%E5%AD%A6&qid=1690020873&s=books&sr=1-1)
