# Eigen  https://eigen.tuxfamily.org/dox/index.html

## 动态大小空间Dynamic

编译时大小未知，

typedef Matrix<double, Dynamic, Dynamic> MatrixXd;

typedef Matrix<int, Dynamic, 1> VectorXi
Matrix<float, 3, Dynamic>

MatrixXd :is a dynamic-size matrix whose size is currently 0-by-0, and whose array of coefficients hasn't yet been allocated at all.

Matrix3f a;

MatrixXf b;

`a` is a 3-by-3 matrix, with a plain float[9] array of uninitialized coefficients, 

`b` is a dynamic-size matrix whose size is currently 0-by-0, and whose array of coefficients hasn't yet been allocated at all.

Constructors taking sizes are also available. For matrices, the number  of rows is always passed first. For vectors, just pass the vector size.  They allocate the array of coefficients with the given size, but don't  initialize the coefficients themselves: 

MatrixXf a(10,15);

VectorXf b(30);

`a` is a 10x15 dynamic-size matrix, with allocated but currently uninitialized coefficients. 

`b` is a dynamic-size vector of size 30, with allocated but currently uninitialized coefficients.

## Resizing

rows(),cols(),size(),resize(),conservativeResize()

## Assignment and resizing

MatrixXf a(2,2);

std::cout << "a is of size " << a.rows() << "x" << a.cols() << std::endl;  a is of size 2x2

MatrixXf b(3,3);

a = b;

std::cout << "a is now of size " << a.rows() << "x" << a.cols() << std::endl;  a is now of size 3x3

## Fixed vs. Dynamic size

The simple answer is: use fixed sizes for very small sizes where you  can, and use dynamic sizes for larger sizes or where you have to. For  small sizes, especially for sizes smaller than (roughly) 16, using fixed sizes is hugely beneficial to performance, as it allows Eigen to avoid dynamic memory allocation and to unroll loops. 

## Optional template parameters

Matrix<typename Scalar,

​       int RowsAtCompileTime,

​       int ColsAtCompileTime,

​       int Options = 0,

​       int MaxRowsAtCompileTime = RowsAtCompileTime,

​       int MaxColsAtCompileTime = ColsAtCompileTime>

RowMajor: It specifies that the matrices of this type use row-major storage order; by default, the storage order is column-major.

Matrix<float, 3, 3, RowMajor>

 MaxRowsAtCompileTime和MaxColsAtCompleTime在您想要指定的情况下很有用，即使在编译时矩阵的确切大小未知，但在编译时固定的上限是已知的。您可能想要这样做的最大原因是避免动态内存分配。例如，下面的矩阵类型使用12个浮点的普通数组，没有动态内存分配 

Matrix<float, Dynamic, Dynamic, 0, 3, 4> // 上限为row = 3 ,col = 4

## Convenience typedefs

MatrixNt for Matrix<type, N, N>. For example, MatrixXi for Matrix<int, Dynamic, Dynamic>. 

MatrixXNt for Matrix<type, Dynamic, N>. For example, MatrixX3i for Matrix<int, Dynamic, 3>. 

MatrixNXt for Matrix<type, N, Dynamic>. For example, Matrix4Xd for Matrix<d, 4, Dynamic>. 

VectorNt for Matrix<type, N, 1>. For example, Vector2f for Matrix<float, 2, 1>. 

RowVectorNt for Matrix<type, 1, N>. For example, RowVector3d for Matrix<double, 1, 3>.

## Matrix and vector arithmetic

矩阵/向量的运算

binary operator + as in `a+b` 

binary operator - as in `a-b` 

unary operator - as in `-a` 

compound operator += as in `a+=b` 

compound operator -= as in `a-=b` 

binary operator * as in `matrix*scalar` 

binary operator * as in `scalar*matrix` 

binary operator / as in `matrix/scalar` 

compound operator *= as in `matrix*=scalar` 

compound operator /= as in `matrix/=scalar` 

## Transposition and conjugation

transpose(),conjugate(),adjoint(),transposeInplace()

MatrixXf a(2,3); a << 1, 2, 3, 4, 5, 6;

cout << "Here is the initial matrix a:\n" << a << endl;

a.transposeInPlace();

cout << "and after being transposed:\n" << a << endl;

```c++
Here is the initial matrix a:
1 2 3
4 5 6
and after being transposed:
1 4
2 5
3 6
```

## Matrix-matrix and matrix-vector multiplication

注意维度

## Dot product and cross product

Eigen::Vector3d v(1,2,3)

Eigen::Vector3d w(0,1,2)

v.dot(w) //点乘

v.cross(w) //差乘

## The Array class and coefficient-wise operations(理解成行列式)？？？

Array类提供通用数组，而Matrix类用于线性代数。此外，Array类提供了一种简单的方法来执行按系数的运算，这种运算可能没有线性代数的含义。

```c++
Array<float,Dynamic,1>          ArrayXf
Array<float,3,1>                Array3f
Array<double,Dynamic,Dynamic>   ArrayXXd
Array<double,3,3>               Array33d
```

## Addition and subtraction

```c++
#include <Eigen/Dense>
#include <iostream>
 
int main()
{
  Eigen::ArrayXXf a(3,3);
  Eigen::ArrayXXf b(3,3);
  a << 1,2,3,
       4,5,6,
       7,8,9;
  b << 1,2,3,
       1,2,3,
       1,2,3;
       
  // Adding two arrays
  std::cout << "a + b = " << std::endl << a + b << std::endl << std::endl;
 
  // Subtracting a scalar from an array
  std::cout << "a - 2 = " << std::endl << a - 2 << std::endl;
    
a + b = 
 2  4  6
 5  7  9
 8 10 12

a - 2 = 
-1  0  1
 2  3  4
 5  6  7
```

## Array multiplication

First of all, of course you can multiply an array by a scalar, this  works in the same way as matrices. Where arrays are fundamentally  different from matrices, is when you multiply two together. Matrices  interpret multiplication as matrix product and arrays interpret  multiplication as coefficient-wise product. Thus, two arrays can be  multiplied if and only if they have the same dimensions.

```c++
#include <Eigen/Dense>
#include <iostream>
 
int main()
{
  Eigen::ArrayXXf a(2,2);
  Eigen::ArrayXXf b(2,2);
  a << 1,2,
       3,4;
  b << 5,6,
       7,8;
  std::cout << "a * b = " << std::endl << a * b << std::endl;
}
a * b = 
 5 12
21 32
```

## Converting between array and matrix expressions

cannot apply [Matrix](https://eigen.tuxfamily.org/dox/classEigen_1_1Matrix.html) operations on arrays, or [Array](https://eigen.tuxfamily.org/dox/classEigen_1_1Array.html) operations on matrices. Thus, if you need to do linear algebraic  operations such as matrix multiplication, then you should use matrices;  if you need to do coefficient-wise operations, then you should use  arrays. 



## Block operations

Block of size (p,q),starting at (i,j)=>matrix.block(i,j,p,q)=>matrix.block<p,q>(i,j)

As always in Eigen, indices start at 0.

## Columns and rows

ith row matrix.row(i)

jth col matrix.col(j)

 ## Corner-related operations

https://eigen.tuxfamily.org/dox/group__TutorialBlockOperations.html

## Slicing and Indexing

不常用

## Advanced initialization

不常用

## Reductions, visitors and broadcasting

不常用

## Reshape

不常用

## STL iterators and algorithms

```c++
VectorXi v = VectorXi::Random(4);
cout << "Here is the vector v:\n";
for(auto x : v) cout << x << " "; //迭代器
cout << "\n";
```

```c++
Array4i v = Array4i::Random().abs();
cout << "Here is the initial vector v:\n" << v.transpose() << "\n";
std::sort(v.begin(), v.end());// 可以使用std::sort
cout << "Here is the sorted vector v:\n" << v.transpose() << "\n";

```

## Interfacing with raw buffers: the Map class

很重要，在后端优化的时候常用

**数字数组转Eigen**：数字数组，希望在Eigen中用作向量或矩阵。虽然一种选择是制作数据的副本，但最常见的情况是，您可能希望将此内存重新用作特征类型。幸运的是，这对于Map类来说非常容易。

```c++
Map<Matrix<typename Scalar,int RowsAtCompileTime,int ColsAtCompileTime>>
```

要构造Map变量，您需要另外两条信息：指向定义系数数组的内存区域的指针，以及所需的矩阵或向量形状。例如，要定义在编译时确定大小的浮点矩阵，可以执行以下操作：

```c++
Map<MatrixXf> mf(pf,rows,cols);
```

`pf` is a `float` `*` pointing to the array of memory. A fixed-size read-only vector of integers might be declared as

```c++
Map<const Vector4i> mi(pi);
Map<const Matrix<int,4,1>> mi(pi);
Map<Matrix<typedef Scalar,int RowsAtCompileTime,int ColsAtCompileTime>> m(array)；// 默认列优先
Map<Matrix<typedef Scalar,int RowsAtCompileTime,int ColsAtCompileTime,ColMajor>> m(array);
Map<Matrix<typedef Scalar,int RowsAtCompileTime,int ColsAtCompileTime,RowMajor>> m(array);// 行优先
```

```c++
int data[] = {1,2,3,4,5,6,7,8,9};
Map<RowVectorXi> v(data,4);
cout << "The mapped vector v is: " << v << "\n";//1 2 3 4
new (&v) Map<RowVectorXi>(data+4,5);
cout << "Now v is: " << v << "\n";// 5 6 7 8 9
```

## Reference  

https://eigen.tuxfamily.org/dox/group__QuickRefPage.html

# Dense linear problems and decompositions

**很重要**：要开始解方程了 
$$
Ax=b
$$
Where *A* and *b* are matrices (*b* could be a vector, as a special case). You want to find a solution *x*.

若A为mxn（列为Rm，行为Rn）的矩阵，它的各列为a1,……an.若x为Rn中的向量，则A与x的积（Ax）就是A的各列以x中对应的元素为权的线性组合。

Ax=[a1,a2,……,an]*[x1,x2……,xn].transpose()=x1 * a1 + x2 * a2 + …… + xn * an;

**解集相同**

若A是mxn矩阵，它的各列为a1,……an，而b属于Rm,Ax = b

x1 * a1 + x2 * a2 + …… + xn * an = b;

增广矩阵为：

[a1,a2,……,an,b]

方程Ax=b 有解当且仅当，b是A的各列的线性组合；

Ax 的行向量规则：

若乘积Ax有定义，则Ax中的第ℹ个元素是A的第ℹ行元素与x的相应元素乘积之和；

**线性方程组**

Ax=0

若A为mxn（列为Rm，行为Rn）的矩阵，0是Rm中的零向量。这样方程组至少有一个解x=0（Rn中的零向量）；

其次方程Ax=0有平凡解，当且仅当方程至少有一个自由变量。

## Linear algebra and decompositions



