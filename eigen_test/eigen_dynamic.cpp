#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <iostream>

using namespace std;
using namespace Eigen;

class Demo {
  public:
    Demo(const int buff_size) { name_ = new char[buff_size]; }
    ~Demo() { delete[] name_; }
    void extendBuff(int new_size) {
        if (name_ != nullptr) {
            std::cout << 111111111111 << std::endl;
            delete[] name_;
        }
        name_ = new char[new_size];
    }
    char *name_;
};

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

    // typedef Matrix<float, 1, Dynamic> MatrixType;
    // typedef Map<MatrixType> MapType;
    // typedef Map<const MatrixType> MapTypeConst;

    // const int n_dims = 5;

    // MatrixType m1(n_dims), m2(n_dims);

    // m1.setRandom();
    // m2.setRandom();

    // cout << "m1 : " << m1 << endl;

    // cout << endl;

    // cout << "m2 : " << m2 << endl;

    // cout << endl;

    // float *p = &m2(0);

    // MapType m2map(p, m2.size());

    // cout << "m2map : " << m2map << endl;

    // cout << endl;

    // MapTypeConst m2mapconst(p,m2.size());

    // Demo *demo;
    // demo = new Demo(17);
    // demo->extendBuff(20);
    // const char a[] = "111111111111111111111111111111111111111111111111111";
    // // std::cout << sizeof(a) << std::endl;

    // strcpy(demo->name_, a);

    // std::cout << demo->name_ << std::endl;

    // Eigen::Matrix2f A, b;
    // A << 2, -1, -1, 3;
    // b << 1, 2, 3, 1;
    // std::cout << "Here is the matrix A:\n" << A << std::endl;
    // std::cout << "Here is the right hand side b:\n" << b << std::endl;
    // // Eigen::Matrix2f x = A.partialPivLu().solve(b);
    // // Eigen::Matrix2f x = A.fullPivLu().solve(b);
    // Eigen::Matrix2f x = A.completeOrthogonalDecomposition().solve(b);
    // std::cout << "The solution is:\n" << x << std::endl;

    // 旋转矩阵 Eigen::Matrix3d Eigen::Matrix3f rotation_matrix
    // 旋转向量 Eigen::AngleAxisd rotation_vector(M_PI/4,Eigen::Vector3d(0,0,1))
    // 绕z轴旋转45度 欧拉角 Eigen::Vector3d euler_angles =
    // rotation_matrix.eulerAngles(2,1,0) ZYX顺序，即rpy 四元数
    // 四元数 Eigen::Quaterniond

    // Eigen/Geometry 模块提供了各种旋转和平移的表示
    // 3D 旋转矩阵直接使用 Matrix3d 或 Matrix3f
    Matrix3d rotation_matrix = Matrix3d::Identity();
    // 旋转向量使用 AngleAxis,
    // 它底层不直接是Matrix，但运算可以当作矩阵（因为重载了运算符）
    AngleAxisd rotation_vector(M_PI / 4, Vector3d(0, 0, 1)); // 沿 Z 轴旋转 45
                                                             // 度
    cout.precision(3);
    cout << "rotation matrix =\n"
         << rotation_vector.matrix() << endl; // 用matrix()转换成矩阵
    // 也可以直接赋值
    rotation_matrix = rotation_vector.toRotationMatrix();
    // 用 AngleAxis 可以进行坐标变换
    Vector3d v(1, 0, 0);
    Vector3d v_rotated = rotation_vector * v;
    cout << "(1,0,0) after rotation (by angle axis) = " << v_rotated.transpose()
         << endl;
    // 或者用旋转矩阵
    v_rotated = rotation_matrix * v;
    cout << "(1,0,0) after rotation (by matrix) = " << v_rotated.transpose()
         << endl;

    // 欧拉角: 可以将旋转矩阵直接转换成欧拉角
    Vector3d euler_angles =
        rotation_matrix.eulerAngles(2, 1, 0); // ZYX顺序，即yaw-pitch-roll顺序
    cout << "yaw pitch roll = " << euler_angles.transpose() << endl;

    // 欧氏变换矩阵使用 Eigen::Isometry
    Isometry3d T = Isometry3d::Identity(); // 虽然称为3d，实质上是4＊4的矩阵
    T.rotate(rotation_vector);             // 按照rotation_vector进行旋转
    T.pretranslate(Vector3d(1, 3, 4)); // 把平移向量设成(1,3,4)
    cout << "Transform matrix = \n" << T.matrix() << endl;

    // 用变换矩阵进行坐标变换
    Vector3d v_transformed = T * v; // 相当于R*v+t
    cout << "v tranformed = " << v_transformed.transpose() << endl;

    // 对于仿射和射影变换，使用 Eigen::Affine3d 和 Eigen::Projective3d 即可，略

    // 四元数
    // 可以直接把AngleAxis赋值给四元数，反之亦然
    Quaterniond q = Quaterniond(rotation_vector);
    cout << "quaternion from rotation vector = " << q.coeffs().transpose()
         << endl; // 请注意coeffs的顺序是(x,y,z,w),w为实部，前三者为虚部
    // 也可以把旋转矩阵赋给它
    q = Quaterniond(rotation_matrix);
    cout << "quaternion from rotation matrix = " << q.coeffs().transpose()
         << endl;
    // 使用四元数旋转一个向量，使用重载的乘法即可
    v_rotated = q * v; // 注意数学上是qvq^{-1}
    cout << "(1,0,0) after rotation = " << v_rotated.transpose() << endl;
    // 用常规向量乘法表示，则应该如下计算
    cout << "should be equal to "
         << (q * Quaterniond(0, 1, 0, 0) * q.inverse()).coeffs().transpose()
         << endl;

    return 0;
}