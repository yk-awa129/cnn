/* Project Properties -> c/c++ -> language -> Openmp:no->yes
Code generation -> Enable Enhanced instruction:No->AVE2*/


#include <iostream>
#include <random>
#include "../Library/Eigen/Core"
using namespace std;
using namespace Eigen;


class CNN
{
private:
	Matrix<double, Dynamic, Dynamic> input;
	Matrix<double, Dynamic, Dynamic> kernel[5]; /* kernelN=5 WxH=5x5 | 28x28 -> 24x24x5=2880 */
	Matrix<double, Dynamic, Dynamic> kernelB;
	/* pooling 2880 -> 12x12x5=720*/
	Matrix<double, Dynamic, Dynamic> fcLayer1; /* 720 ->100 */
	Matrix<double, Dynamic, Dynamic> fcLayer1B;
	Matrix<double, Dynamic, Dynamic> fcLayer2; /* 100 -> 10 */
	Matrix<double, Dynamic, Dynamic> fcLayer2B;

	Matrix<double, Dynamic, Dynamic> outk[5];
	Matrix<double, Dynamic, Dynamic> outb;
	Matrix<double, Dynamic, Dynamic> outf1;
	Matrix<double, Dynamic, Dynamic> outf2;


public:
	CNN() {

		random_device rd;
		mt19937 mt(rd());
		uniform_real_distribution<double> score(-0.001, 0.001);

		input.resize(28, 28);

		for (int i = 0; i < 5; i++) {
			kernel[i].resize(5, 5);
			for (int n = 0; n < 5; n++) {
				for (int m = 0; m < 5; m++) {
					kernel[i](n, m) = score(mt);
				}
			}
		}
		kernelB.resize(1, 5);
		for (int i = 0; i < 5; i++) { kernelB(i) = score(mt); }

		fcLayer1.resize(100, 720);
		for (int i = 0; i < 100; i++) {
			for (int j = 0; j < 720; j++) {
				fcLayer1(i, j) = score(mt);
			}
		}
		fcLayer1B.resize(720, 1);
		for (int i = 0; i < 720; i++) {
			fcLayer1B(i) = score(mt);
		}

		fcLayer2.resize(10, 100);
		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < 100; j++) {
				fcLayer2(i, j) = score(mt);
			}
		}

		fcLayer2B.resize(100, 1);
		for (int i = 0; i < 100; i++) {
			fcLayer1B(i) = score(mt);
		}

		for (int i = 0; i < 5; i++) {
			outk[i].setZero(24, 24);
		}
		outb.setZero(720, 1);
		outf1.setZero(100, 1);
		outf2.setZero(10, 1);
	}
	void inputImage() {
		input.setOnes();
	}
	void forwardProp() {
		/* conv */
		for (int n = 0; n < 5; n++) {
			for (int i = 0; i < 24; i++) {
				for (int j = 0; j < 24; j++) {
					outk[n](i, j) = kernelB(0, n);
					for (int s = 0; s < 5; s++) {
						for (int t = 0; t < 5; t++) {
							outk[n](i, j) += input(i + s, j + t)*kernel[n](s, t);
						}
					}
					
				}
			}
		}

		/* conv */

		/* pooling */
		double max;
		for (int n = 0; n < 5; n++) {
			for (int i = 0; i < 12; i++) {
				for (int j = 0; j < 12; j++) {
					max = 0;
					for (int s = 0; s < 2; s++) {
						for (int t = 0; t < 2; t++) {
							if (outk[n](i * 2 + s, j * 2 + t) > max) {
								max = outk[n](i + s, j + t);
							}
						}
					}
					outb(i + 12 * j + 144 * n) = max;
				}
			}
		}

		/* pooling */

		/* fullconnect1 */
		outf1 = fcLayer1*outb;
		outf1 -= fcLayer1B;
		/* fullconnect1 */

		/* fullconnect2 */
		outf2 = fcLayer2*outf1;
		outf2 -= fcLayer2B;
		/* fullconnect2 */

	}
	void backProp() {

	}
	void show() {
		cout << kernel[0] << endl;
	}

};


int main() {

	CNN test;
	test.inputImage();
	

	return 0;
}
