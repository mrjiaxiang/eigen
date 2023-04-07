#include <Eigen/Dense>

#include <iostream>

using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {
    // MatrixXf a(10, 15);
    // VectorXf b(30);
    // cout << "a = " << a.size() << endl;
    // cout << "b = " << b.size() << endl;

    // Eigen::MatrixXd m(2, 5);
    // cout << "a = " << m.size() << endl;
    // m.resize(2, 5);
    // // m.conservativeResize(4, 3);
    // std::cout << "The matrix m is of size " << m.rows() << "x" << m.cols()
    //           << std::endl;
    // std::cout << "It has " << m.size() << " coefficients" << std::endl;
    // Eigen::VectorXd v(2);
    // cout << "v = " << v.size() << endl;
    // v.resize(5);
    // std::cout << "The vector v is of size " << v.size() << std::endl;
    // std::cout << "As a matrix, v is of size " << v.rows() << "x" << v.cols()
    //           << std::endl;

    // MatrixXcf a = Matrix4cf::Random(2, 2);
    // Matrix4f a = Matrix4f::Random(4, 4);
    // cout << "Here is the matrix a\n" << a << endl;

    // cout << "Here is the matrix a^T\n" << a.transpose() << endl;

    // cout << "Here is the conjugate of a\n" << a.conjugate() << endl;

    // cout << "Here is the matrix a^*\n" << a.adjoint() << endl;

    // Eigen::Vector3d v(1, 2, 3);
    // Eigen::Vector3d w(0, 1, 2);

    // std::cout << "Dot product: " << v.dot(w) << std::endl;
    // double dp = v.adjoint() *
    //             w; // automatic conversion of the inner product to a scalar
    // std::cout << "Dot product via a matrix product: " << dp << std::endl;
    // std::cout << "Cross product:\n" << v.cross(w) << std::endl;

    // Eigen::ArrayXXf m(2, 2);
    // m(0, 0) = 1.0;
    // m(0, 1) = 2.0;
    // m(1, 0) = 3.0;
    // m(1, 1) = m(0, 1) + m(1, 0);
    // std::cout << m << std::endl << std::endl;

    // m << 1.0, 2.0, 3.0, 4.0;

    // // print values to standard output
    // std::cout << m << std::endl;

    // Eigen::Matrix4f m;
    // m << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;
    // cout << m << endl << endl;

    // cout << m.reshaped(2, 8) << endl;

    // cout << m.block<2, 2>(1, 1) << endl;

    // Eigen::Array22f m;
    // m << 1, 2, 3, 4;
    // Eigen::Array44f a = Eigen::Array44f::Constant(0.6);
    // cout << a << endl;

    // a.block<2, 2>(1, 1) = m;

    // cout << a << endl;

    // a.block<2, 3>(0, 0) = a.block(2, 1, 2, 3);
    // cout << a << endl;

    // int array[8];
    // for (int i = 0; i < 8; i++) {
    //     array[i] = i;
    // }

    // cout << Map<Matrix<int, 2, 4>>(array) << endl << endl;

    // cout << "col :" << Map<Matrix<int, 2, 4, ColMajor>>(array) << endl <<
    // endl;

    // cout << "row :" << Map<Matrix<int, 2, 4, RowMajor>>(array) << endl <<
    // endl;

    typedef Matrix<float, 1, Dynamic> MatrixType;
    typedef Map<MatrixType> MapType;
    typedef Map<const MatrixType> MapTypeConst;

    const int n_dims = 5;

    MatrixType m1(n_dims), m2(n_dims);

    m1.setRandom();
    m2.setRandom();

    cout << "m1 : " << m1 << endl;

    cout << endl;

    cout << "m2 : " << m2 << endl;

    cout << endl;

    float *p = &m2(0);

    MapType m2map(p, m2.size());

    cout << "m2map : " << m2map << endl;

    cout << endl;

    MapTypeConst m2mapconst(p, m2.size());

    cout << "m2map : " << m2map << endl;

    cout << endl;

    return 0;
}