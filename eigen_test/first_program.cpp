#include <Eigen/Dense>
#include <iostream>

int main() {
    Eigen::MatrixXd m(2, 2);
    m(0, 0) = 1;
    m(1, 0) = 1.2;
    m(0, 1) = 1.4;
    m(1, 1) = 1.5;
    std::cout << "matrix = " << m << std::endl;
    return 0;
}
