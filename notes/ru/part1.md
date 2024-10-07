# Определения
## Кубит
Кубит — квантовый бит — это фундаментальная единица квантовой информации.
В любой конкретный момент времени он находится в состоянии суперпозиции, представленном линейной комбинацией:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$
$$\alpha^2 + \beta^2 = 1$$
где $\alpha$ и $\beta$ амплитуды вероятности измерить состояние кубита $|0\rangle$ или $|1\rangle$.

В том числе можно представить как:
$$|\psi\rangle = r_1|0\rangle + r_2*e^{\phi i}|1\rangle$$
$$r_1^2 + r_2^2 = 1$$
$$r_1 = \cos(\frac{\theta}{2}); r_2=\sin(\frac{\theta}{2}) $$
$$|\psi\rangle = \cos(\frac{\theta}{2})|0\rangle + \sin(\frac{\theta}{2})e^{\phi i}|1\rangle $$

# Гейты
## Гейт Адамара
### Описание
Гейт Адамара представляет собой матрицу:
$$H_1 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} $$
Матрица Адамара расширяется для N-кубитов следующим образом:
$$
H^{\otimes n}
$$
где $\otimes n$ означает тензорное произведение (произведение кронекера) $n$ раз.

### Примеры использования
$$ H \times |0\rangle = |+\rangle $$
$$ H \times |1\rangle = |-\rangle $$
$$ H \times |+\rangle = |0\rangle $$
$$ H \times |-\rangle = |1\rangle $$



###  Пример гейта
<img src="../circuits/qrng.svg" alt="SVG image" />

## Гейт X
### Описание
В классических дискретных вычислениях оператор NOT – это оператор инверсии. В соответствии с этим
определением классического оператора NOT, квантовый гейт X может быть определен по аналогии:
$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle \rightarrow X|\psi\rangle = \alpha|1\rangle + \beta|0\rangle$

Гейт X представлен следующей матрицей:
$$X = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} $$

### Примеры использования
$$ X \times |0\rangle = |1\rangle $$
$$ X \times |1\rangle = |0\rangle $$

## Гейт Rx 
### Описание
Гейт поворота на угол $\theta$ вокруг оси $x$:
$$ Rx(\theta) = e^{-i\frac{\theta}{2}X} = \begin{bmatrix} \cos(\frac{\theta}{2}) & -i*\sin(\frac{\theta}{2}) \\ -i*\sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})  \end{bmatrix}$$ 

## Гейт Ry 
Гейт поворота на угол $\theta$ вокруг оси $y$:
$$ Ry(\theta) = e^{-i\frac{\theta}{2}Y} = \begin{bmatrix} \cos(\frac{\theta}{2}) & -\sin(\frac{\theta}{2}) \\ \sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})  \end{bmatrix}$$ 

## Гейт Rz
Гейт поворота на угол $\theta$ вокруг оси $z$:
$$ Rz(\theta) = e^{-i\frac{\theta}{2}Z} = \begin{bmatrix} e^{-i\frac{\theta}{2}} & 0 \\ 0 & e^{-i\frac{\theta}{2}}  \end{bmatrix}$$ 


# Алгоритмы
## Quantum Random Generator
Quantum Random Generator - алгоритм получения случайного бита с помощью кубита.

<img src="../circuits/qrng.svg" alt="SVG image" />

QRNG - выставляет кубит $|0\rangle$ в состояние $|+\rangle$ и при измерении
можно равновероятно получить как логически 0 так и логически 1.

### Вероятность измерить состояние
Распределение вероятностей при использовании гейта Адамара будет слеудющим:

| State | P   |
|--|-----|
| 0| 0.5 |
| 1 | 0.5|

Если применить другой поворот, например Rx(pi/4) получим следующее распределение вероятностей (примерно):

| State | P    |
|--|------|
| 0| 0.85 |
| 1 | 0.15 |

Следующая схема:

<img src="../circuits/qrng_with_rx.svg" alt="SVG image" />

Сфера блоха для q0:
<img src="../bloch/qubit_qrng_rx.png" alt="PNG image" />


## Quantum key distribution <TBD>

### BB84 Explanation <TBD>

## Deutsch algorithm <TBD>

## Deutsch-Jozsa algorithm <TBD>

## Bernstein-Vazirani algorithm <TBD>

## Simon problem <TBD>