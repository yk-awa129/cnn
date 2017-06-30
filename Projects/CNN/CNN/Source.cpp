/* Project Properties -> c/c++ -> language -> Openmp:no->yes 
                                  Code generation -> Enable Enhanced instruction:No->AVE2*/


#include <iostream>
#include "../Library/Eigen/Core"
using namespace std;
using namespace Eigen;


class CNN
{
private:
	Matrix<double, 5,5,5> kernel; /* kernelN=5 WxH=5x5 | 28*28 -> 2880 */
	
	Matrix<double, 500, 2880> fcLayer1; /* 2880 -> 500 */
	Matrix<double, 100, 500> fcLayer2; /* 500 -> 10 */

public:
	void show() {
		cout << fcLayer1 << endl;
	};
	
};


int main() {
	Matrix <double, 3, 2> m;

	m << 1, 2, 3, 4, 5, 6;

	Vector2d a;

	a(0) = 2;
	a(1) = 3;

	cout << m << endl << a << endl << m*a;
	
}
