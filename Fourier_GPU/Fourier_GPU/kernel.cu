// Нахождение теплового сопротивления (резистора, диода, транзистора)
// с помощью быстрого дискретного преобразования Фурье реализованного 
// на видеокарте Nvidia Geforce 840M т.к. метод обладает большим 
// ресурсом параллелизма.
// Джейсон Сандерс и Эдвард Кэндрот
// Технология CUDA в примерах. Введение в программирование графических процессоров.

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <complex>
#include <omp.h>
#include <windows.h> // WinExec
#include <math.h> // pow, sin, cos

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Демонстрирует возможности языка cuda.
int demo_add()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;

}

const float M_PI = 3.14159265f;

// for cuda C
// Преобразует индексы трёхмерного массива в индекс одномерного вектора. 
int index(int i, int j, int k, int m, int n, int l)
{
	return ((i) + (j) * (m + 2) + (k) * (m + 2) * (n + 2));
}

// Выделяет оперативную память под вектор u.

void alloc_u3D(float*& u, int m, int n, int l) {

	//u = new float[(m+2)*(n+2)*(l+2)];

	u = (float*)malloc((m + 2) * (n + 2) * (l + 2) * sizeof(float));

    /*u = new float * *[m + 2];
    for (int i = 0; i <= m + 1; i++) {
        u[i] = new float * [n + 2];
        for (int j = 0; j <= n + 1; j++) {
            u[i][j] = new float[l + 2];
        }
    }*/
}

// Освобождает оперативную память из под вектора u.

void free_u3D(float*& u, int m, int n, int l) {

	//delete[] u;

	free(u);

    /*for (int i = 0; i <= m + 1; i++) {
        for (int j = 0; j <= n + 1; j++) {
            delete[] u[i][j];
        }
    }
    for (int i = 0; i <= m + 1; i++) {
        delete[] u[i];
    }
    delete[] u;*/
}

// Инициализирует массив u значением initVal.

void init_u3D(float*& u, int m, int n, int l, float initVal) {

/*#pragma omp parallel for
    for (int i = 0; i < m + 2; i++) {
        for (int j = 0; j < n + 2; j++) {
            for (int k = 0; k < l + 2; k++) {
                u[i][j][k] = initVal;
            }
        }
    }*/
	const int size = (m + 2) * (n + 2) * (l + 2);
#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		u[i] = initVal;
	}
}


__global__ void initKernel(float* u, int m, int n, int l, float initVal)
{
	int i = threadIdx.x+blockIdx.x*blockDim.x;
	const int size = (m + 2) * (n + 2) * (l + 2);
	if (i < size) {
		u[i] = initVal;
		i += blockDim.x * gridDim.x;
	}
	
}

// Использует выделения и уничтожения памяти внутри fft.

void fft_non_optimaze_memory(std::complex<float>*& a, int n, bool invert) {

	if (n == 1) return;

	std::complex<float>* a0 = new std::complex<float>[n / 2];
	std::complex<float>* a1 = new std::complex<float>[n / 2];

	for (int i = 0, j = 0; i < n; i += 2, ++j) {
		a0[j] = a[i];
		a1[j] = a[i + 1];
	}
	fft_non_optimaze_memory(a0, n / 2, invert);
	fft_non_optimaze_memory(a1, n / 2, invert);

	std::complex<float> w(1, 0);
	float ang = 2.0 * M_PI / n * (invert ? -1 : 1);
	std::complex<float> wn(cos(ang), sin(ang));
	for (int i = 0; i < n / 2; ++i) {
		a[i] = a0[i] + w * a1[i];
		a[i + n / 2] = a0[i] - w * a1[i];

		if (invert) {
			a[i] /= 2;
			a[i + n / 2] /= 2;
		}

		w *= wn;
	}

	delete[] a0;
	delete[] a1;
}


// На месте без выделенния и уничтожения оперативной памяти.

void fft(std::complex<float>*& a, int n, bool invert) {

	if (n == 1) return;



	for (int i = 1, j = 0; i < n; ++i) {
		int bit = n >> 1;
		for (; j >= bit; bit >>= 1)
			j -= bit;
		j += bit;
		if (i < j)
			swap(a[i], a[j]);
	}

	/*for (int i = 0; i < n; i++) {
		//if (fabs(a[i]._Val[1]) > 1.0e-28)
		{
			printf("%d %e\n", i, a[i]._Val[1]);
			getchar();
		}
	}*/

	for (int len = 2; len <= n; len <<= 1) {
		float ang = 2 * M_PI / len * (invert ? -1 : 1);
		std::complex<float> wlen(cos(ang), sin(ang));
		for (int i = 0; i < n; i += len) {
			std::complex<float> w(1);
			for (int j = 0; j < len / 2; ++j) {
				std::complex<float> u = a[i + j], v = a[i + j + len / 2] * w;
				a[i + j] = u + v;
				a[i + j + len / 2] = u - v;
				w *= wlen;
			}
		}
	}
	if (invert)
		for (int i = 0; i < n; ++i)
			a[i] /= n;


}

const int MAXN = 262145;
// Максимальное число потоков в процессоре.
const int NUMBER_THREADS = 256;

int rev[MAXN];

std::complex<float>** a1_gl = nullptr;


std::complex<float>** wlen_pw_gl = nullptr;

// Вспомогательная процедура для быстрого преобразования Фурье на месте.
void calc_rev(int n) {

	int lg_n = 0;
	while ((1 << lg_n) < n)  ++lg_n;

	for (int i = 0; i < n; ++i) {
		rev[i] = 0;
		for (int j = 0; j < lg_n; ++j)
			if (i & (1 << j))
				rev[i] |= 1 << (lg_n - 1 - j);
	}
	/*rev[0] = 0;
	for (int i = 1, j = 0; i < n; ++i) {
		int bit = n >> 1;
		for (; j >= bit; bit >>= 1)
			j -= bit;
		j += bit;
		rev[i] = j;
	}*/
}

// Быстрое преобразование Фурье.

void fft1(std::complex<float>*& a, int n, bool invert, std::complex<float>*& wlen_pw) {

	for (int i = 0; i < n; ++i)
		if (i < rev[i])
			swap(a[i], a[rev[i]]);

	for (int len = 2; len <= n; len <<= 1) {
		float ang = 2 * M_PI / len * (invert ? -1 : +1);
		int len2 = len >> 1;

		std::complex<float> wlen(cos(ang), sin(ang));
		wlen_pw[0] = std::complex<float>(1, 0);
		for (int i = 1; i < len2; ++i)
			wlen_pw[i] = wlen_pw[i - 1] * wlen;

		for (int i = 0; i < n; i += len) {
			std::complex<float> t,
				* pu = a + i,
				* pv = a + i + len2,
				* pu_end = a + i + len2,
				* pw = wlen_pw;
			for (; pu != pu_end; ++pu, ++pv, ++pw) {
				t = *pv * *pw;
				*pv = *pu - t;
				*pu += t;
			}
		}
	}

	if (invert)
		for (int i = 0; i < n; ++i) 
			a[i] /= n;

}

// Вызывает быстрое преобразование Фурье.

void DFTx(float*& u, float*& a, int m, int n, int l) {

	int Nx = m + 1; // < M+2
	int Ny = n + 1; // < N+2
	int Nz = l + 1;

	// Находим искомую функцию.

	calc_rev(Nx - 2);

#pragma omp parallel for
	for (int k = 1; k < Nz - 1; k++) {

		int id = omp_get_thread_num();
		std::complex<float>* a1 = a1_gl[id];
		std::complex<float>* wlen_pw = wlen_pw_gl[id];

		for (int j = 0; j <= Ny - 1; j++) {

			a1[0] = (0.0, 0.0);
			for (int i = 1; i < Nx - 1; i++) {
				a1[i - 1] = (0.0, a[index(i,j,k,m,n,l)]);
			}
			fft1(a1, Nx - 2, false, wlen_pw);
			for (int i = 1; i < Nx - 1; i++) {
				u[index(i, j, k, m, n, l)] = -a1[i - 1]._Val[0];
			}
		}

		a1 = nullptr;
		wlen_pw = nullptr;
		//delete[] a1;
		//delete[] wlen_pw;
	}

}


__global__ void FFTxKernel(float* u, const float* a, float* a1Re, float* a1Im, int m, int n, int l)
{

	int Nx = m + 1; // < M+2
	int Ny = n + 1; // < N+2
	int Nz = l + 1;

	int im1 = Nx - 2;

	const float M_PI = 3.14159265f;

	int k = threadIdx.x + blockIdx.x * blockDim.x;
	//const int size = (m + 2) * (n + 2) * (l + 2);
	while (k < Nz - 1) {
		if ((k > 0) && (k < Nz - 1)) {

			// a1 размером [l*m];	

			for (int j = 0; j <= Ny - 1; j++) {

				a1Re[(k - 1) * (im1) + 0] = 0.0;
				a1Im[(k - 1) * (im1) + 0] = 0.0;
				for (int i = 1; i < Nx - 1; i++) {
					a1Im[(k - 1) * (im1) + i - 1] = 0.0;
					a1Re[(k - 1) * (im1) + i - 1] = a[((i)+(j) * (m + 2) + (k) * (m + 2) * (n + 2))];
				}
				//fft(a1, Nx - 2, false);

				int n_ = Nx - 2;
				bool invert = false;

				if (n_ == 1) return;



				for (int i = 1, j_ = 0; i < n_; ++i) {
					int bit = n_ >> 1;
					for (; j_ >= bit; bit >>= 1)
						j_ -= bit;
					j_ += bit;
					if (i < j_) {
						//swap(a1[(k - 1) * m + i], a1[(k - 1) * m + j_]);
						float tmp = a1Re[(k - 1) * (im1)+ i];
						a1Re[(k - 1) * (im1)+ i] = a1Re[(k - 1) * (im1)+ j_];
						a1Re[(k - 1) * (im1)+ j_] = tmp;

						tmp = a1Im[(k - 1) * (im1)+ i];
						a1Im[(k - 1) * (im1)+ i] = a1Im[(k - 1) * (im1)+ j_];
						a1Im[(k - 1) * (im1)+ j_] = tmp;
					}
				}

				for (int len = 2; len <= n_; len <<= 1) {
					float ang = 2 * M_PI / len * (invert ? -1 : 1);
					//std::complex<float> wlen((float)(cos(ang)),(float)(sin(ang)));
					float wlenRe = (float)(cosf(ang));
					float wlenIm = (float)(sinf(ang));

					for (int i = 0; i < n_; i += len) {
						//std::complex<float> w((float)(1.0), (float)(0.0));
						float wRe = 1.0;
						float wIm = 0.0;

						for (int j_ = 0; j_ < len / 2; ++j_) {
							//std::complex<float> u_ = a1[(k - 1) * (im1) + i + j_], v = a1[(k - 1) * (im1) + i + j_ + len / 2] * w;
							float u_Re = a1Re[(k - 1) * (im1)+ i + j_];
							float u_Im = a1Im[(k - 1) * (im1)+ i + j_];

							float vRe = (a1Re[(k - 1) * (im1)+ i + j_ + len / 2] * wRe - a1Im[(k - 1) * (im1)+ i + j_ + len / 2] * wIm);
							float vIm = (a1Re[(k - 1) * (im1)+ i + j_ + len / 2] * wIm + a1Im[(k - 1) * (im1)+ i + j_ + len / 2] * wRe);

							a1Re[(k - 1) * (im1)+ i + j_] = u_Re + vRe;
							a1Im[(k - 1) * (im1)+ i + j_] = u_Im + vIm;

							a1Re[(k - 1) * (im1)+ i + j_ + len / 2] = u_Re - vRe;
							a1Im[(k - 1) * (im1)+ i + j_ + len / 2] = u_Im - vIm;

							//w *= wlen;
							float tmpwRe = (wRe * wlenRe - wIm * wlenIm);
							wIm = (wRe * wlenIm + wIm * wlenRe);
							wRe = tmpwRe;
						}
					}
				}
				if (invert)
					for (int i = 0; i < n_; ++i) {
						a1Re[(k - 1) * (im1)+ i] /= n_;
						//a1Im[(k - 1) * m + i] /= n_;
					}
				// end fft

				for (int i = 1; i < Nx - 1; i++) {
					u[((i)+(j) * (m + 2) + (k) * (m + 2) * (n + 2))] = -a1Re[(k - 1) * (im1)+ i - 1];
				}
			}

		}

		k += blockDim.x * gridDim.x;
	}
}


// Вызывает быстрое преобразование Фурье.

void DFTy(float*& u, float*& a, int m, int n, int l) {

	int Nx = m + 1; // < M+2
	int Ny = n + 1; // < N+2
	int Nz = l + 1;

	calc_rev(Ny - 2);

	// Находим искомую функцию.
#pragma omp parallel for
	for (int k = 1; k < Nz - 1; k++) {

		int id = omp_get_thread_num();
		std::complex<float>* a1 = a1_gl[id];
		std::complex<float>* wlen_pw = wlen_pw_gl[id];

		for (int i = 0; i <= Nx - 1; i++) {

			for (int j = 1; j < Ny - 1; j++) {
				a1[j - 1] = (0.0, a[index(i, j, k, m, n, l)]);
			}
			fft1(a1, Ny - 2, false, wlen_pw);
			for (int j = 1; j < Ny - 1; j++) {
				u[index(i, j, k, m, n, l)] = -a1[j - 1]._Val[0];
			}
		}

		a1 = nullptr;
		wlen_pw = nullptr;
	}
}



__global__ void FFTyKernel(float* u, const float* a, float* a1Re, float* a1Im, int m, int n, int l)
{
	int Nx = m + 1; // < M+2
	int Ny = n + 1; // < N+2
	int Nz = l + 1;

	int im1 = Ny - 2;

	const float M_PI = 3.14159265f;

	int k = threadIdx.x + blockIdx.x * blockDim.x;
	const int size = (m + 2) * (n + 2) * (l + 2);
	while (k < Nz - 1) 
	{
		if ((k > 0) && (k < Nz - 1)) {

			// a1 размером [(Nz-2)*(Ny-2)]


			for (int i = 0; i <= Nx - 1; i++) {

				for (int j = 1; j < Ny - 1; j++) {
					a1Im[(k - 1) * (im1) + j - 1] = 0.0;
					a1Re[(k - 1) * (im1)+ j - 1] = a[((i)+(j) * (m + 2) + (k) * (m + 2) * (n + 2))];
				}
				//fft(a1, Ny - 2, false);
				int n_ = Ny - 2;
				bool invert = false;

				if (n_ == 1) return;

				for (int i_ = 1, j_ = 0; i_ < n_; ++i_) {
					int bit = n_ >> 1;
					for (; j_ >= bit; bit >>= 1)
						j_ -= bit;
					j_ += bit;
					if (i_ < j_) {
						//swap(a1[(k - 1) * n + i_], a1[(k - 1) * n + j_]);
						float tmp = a1Re[(k - 1) * (im1)+ i_];
						a1Re[(k - 1) * (im1)+ i_] = a1Re[(k - 1) * (im1)+ j_];
						a1Re[(k - 1) * (im1)+ j_] = tmp;

						tmp = a1Im[(k - 1) * (im1)+ i_];
						a1Im[(k - 1) * (im1)+ i_] = a1Im[(k - 1) * (im1)+ j_];
						a1Im[(k - 1) * (im1)+ j_] = tmp;
					}
				}

				for (int len = 2; len <= n_; len <<= 1) {
					float ang = 2 * M_PI / len * (invert ? -1 : 1);
					//std::complex<float> wlen((float)(cos(ang)), (float)(sin(ang)));
					float wlenRe = (float)(cosf(ang));
					float wlenIm = (float)(sinf(ang));

					for (int i_ = 0; i_ < n_; i_ += len) {
						//std::complex<float> w((float)(1));
						float wRe = 1.0;
						float wIm = 0.0;

						for (int j_ = 0; j_ < len / 2; ++j_) {
							//std::complex<float> u_ = a1[(k - 1) * n + i_ + j_], v = a1[(k - 1) * n + i_ + j_ + len / 2] * w;
							float u_Re = a1Re[(k - 1) * (im1)+ i_ + j_];
							float u_Im = a1Im[(k - 1) * (im1)+ i_ + j_];

							float vRe = (a1Re[(k - 1) * (im1)+ i_ + j_ + len / 2] * wRe - a1Im[(k - 1) * (im1)+ i_ + j_ + len / 2] * wIm);
							float vIm = (a1Re[(k - 1) * (im1)+ i_ + j_ + len / 2] * wIm + a1Im[(k - 1) * (im1)+ i_ + j_ + len / 2] * wRe);

							//a1[(k - 1) * (im1) + i_ + j_] = u_ + v;
							a1Re[(k - 1) * (im1)+ i_ + j_] = u_Re + vRe;
							a1Im[(k - 1) * (im1)+ i_ + j_] = u_Im + vIm;

							//a1[(k - 1) * (im1) + i_ + j_ + len / 2] = u_ - v;
							a1Re[(k - 1) * (im1)+ i_ + j_ + len / 2] = u_Re - vRe;
							a1Im[(k - 1) * (im1)+ i_ + j_ + len / 2] = u_Im - vIm;


							//w *= wlen;
							float tmpwRe = (wRe * wlenRe - wIm * wlenIm);
							wIm = (wRe * wlenIm + wIm * wlenRe);
							wRe = tmpwRe;

						}
					}
				}
				if (invert)
					for (int i_ = 0; i_ < n_; ++i_) {
						a1Re[(k - 1) * (im1)+ i_] /= n_;
						//a1Im[(k - 1) * n + i_] /= n_;
					}
				// end fft

				for (int j = 1; j < Ny - 1; j++) {
					u[((i)+(j) * (m + 2) + (k) * (m + 2) * (n + 2))] = -a1Re[(k - 1) * (im1)+ j - 1];
				}
			}


		}
	
		k += blockDim.x * gridDim.x;
	}
}

// Вызывает быстрое преобразование Фурье.

void IDFTx(float*& u, float*& f, int m, int n, int l) {

	int Nx = m + 1; // < M+2
	int Ny = n + 1; // < N+2
	int Nz = l + 1;

	calc_rev(Nx - 2);

	// Преобразование правой части.
#pragma omp parallel for
	for (int k = 1; k < Nz - 1; k++) {

		int id = omp_get_thread_num();
		std::complex<float>* a1 = a1_gl[id];
		std::complex<float>* wlen_pw = wlen_pw_gl[id];


		for (int j = 0; j <= Ny - 1; j++) {

			for (int i = 1; i < Nx - 1; i++) {

				a1[i - 1] = (0.0, f[index(i, j, k, m, n, l)]);
			}

			fft1(a1, Nx - 2, true, wlen_pw);
			for (int i = 1; i < Nx - 1; i++) {

				u[index(i, j, k, m, n, l)] = a1[i - 1]._Val[0];
			}
		}

		a1 = nullptr;
		wlen_pw = nullptr;
	}
}



__global__ void IFFTxKernel(float* u, const float* f, float* a1Re, float* a1Im, int m, int n, int l)
{

	int Nx = m + 1; // < M+2
	int Ny = n + 1; // < N+2
	int Nz = l + 1;

	int im1 = Nx - 2;

	const float M_PI = 3.14159265f;

	int k = threadIdx.x + blockIdx.x * blockDim.x;

	//int k = threadIdx.x;
	//const int size = (m + 2) * (n + 2) * (l + 2);
	while (k < Nz - 1) {
		if ((k > 0) && (k < Nz - 1)) {

			// a1 размером [(Nz-2)*(Nx-2)]

			for (int j = 0; j <= Ny - 1; j++) {

				for (int i = 1; i < Nx - 1; i++) {

					a1Im[(k - 1) * (im1) + i - 1] = 0.0;
					a1Re[(k - 1) * (im1) + i - 1] = f[((i)+(j) * (m + 2) + (k) * (m + 2) * (n + 2))];
				}

				//fft(a1, Nx - 2, true);

				int nsizex = Nx - 2;
				bool invert = true;

				if (nsizex == 1) return;



				for (int i = 1, j_ = 0; i < nsizex; ++i) {
					int bit = nsizex >> 1;
					for (; j_ >= bit; bit >>= 1)
						j_ -= bit;
					j_ += bit;
					if (i < j_)
					{
						//swap(a1[(k - 1) * m + i], a1[(k - 1) * m + j_]);
						float tmp = a1Re[(k - 1) * (im1) + i];
						a1Re[(k - 1) * (im1) + i] = a1Re[(k - 1) * (im1) + j_];
						a1Re[(k - 1) * (im1) + j_] = tmp;

						tmp = a1Im[(k - 1) * (im1) + i];
						a1Im[(k - 1) * (im1) + i] = a1Im[(k - 1) * (im1) + j_];
						a1Im[(k - 1) * (im1) + j_] = tmp;

					}
				}

				for (int len = 2; len <= nsizex; len <<= 1) {
					float ang = 2 * M_PI / len * (invert ? -1 : 1);
					//std::complex<float> wlen((float)(cos(ang)), (float)(sin(ang)));
					float wlenRe = (float)(cosf(ang));
					float wlenIm = (float)(sinf(ang));

					for (int i = 0; i < nsizex; i += len) {
						//std::complex<float> w((float)(1));
						float wRe = 1.0;
						float wIm = 0.0;

						for (int j_ = 0; j_ < len / 2; ++j_) {
							//std::complex<float> u_ = a1[(k - 1) * m + i + j_], v = a1[(k - 1) * m + i + j_ + len / 2] * w;
							float u_Re = a1Re[(k - 1) * (im1) + i + j_];
							float u_Im = a1Im[(k - 1) * (im1) + i + j_];

							float vRe = (a1Re[(k - 1) * (im1) + i + j_ + len / 2] * wRe - a1Im[(k - 1) * (im1) + i + j_ + len / 2] * wIm);
							float vIm = (a1Re[(k - 1) * (im1) + i + j_ + len / 2] * wIm + a1Im[(k - 1) * (im1) + i + j_ + len / 2] * wRe);

							//a1[(k - 1) * m + i + j_] = u_ + v;
							a1Re[(k - 1) * (im1) + i + j_] = u_Re + vRe;
							a1Im[(k - 1) * (im1) + i + j_] = u_Im + vIm;

							//a1[(k - 1) * m + i + j_ + len / 2] = u_ - v;
							a1Re[(k - 1) * (im1) + i + j_ + len / 2] = u_Re - vRe;
							a1Im[(k - 1) * (im1) + i + j_ + len / 2] = u_Im - vIm;

							//w *= wlen;
							float tmpwRe = (wRe * wlenRe - wIm * wlenIm);
							wIm = (wRe * wlenIm + wIm * wlenRe);
							wRe = tmpwRe;
						}
					}
				}
				if (invert)
					for (int i = 0; i < nsizex; ++i)
						a1Re[(k - 1) * (im1) + i] /= nsizex;
				// end fft


				for (int i = 1; i < Nx - 1; i++) {

					u[((i)+(j) * (m + 2) + (k) * (m + 2) * (n + 2))] = a1Re[(k - 1) * (im1) + i - 1];
				}
			}

		}
	
		k += blockDim.x * gridDim.x;
		//k += 32;
	}
}


void IFFTxKerneldebug(float* u, float* f, float* a1Re, float* a1Im, int m, int n, int l)
{

	int Nx = m + 1; // < M+2
	int Ny = n + 1; // < N+2
	int Nz = l + 1;

	int im1 = Nx - 2;

	const float M_PI = 3.1415926535f;

	//int k = 0;
	//const int size = (m + 2) * (n + 2) * (l + 2);
	for (int k = 0; k < Nz - 1; k++)
	{
		if ((k > 0) && (k < Nz - 1)) {

			// a1 размером [(Nz-2)*(Nx-2)];

			for (int j = 0; j <= Ny - 1; j++) {


				if (0) {

					std::complex<float>* a1 = new std::complex<float>[Nx-2];

					for (int i = 1; i < Nx - 1; i++) {

						a1[i - 1] = (0.0, f[((i)+(j) * (m + 2) + (k) * (m + 2) * (n + 2))]);

						/*if (fabs(f[((i)+(j) * (m + 2) + (k) * (m + 2) * (n + 2))] > 1.0e-28)) {
							printf("f=%e Re=%e Im=%e\n", f[((i)+(j) * (m + 2) + (k) * (m + 2) * (n + 2))], a1[i - 1]._Val[0], a1[i - 1]._Val[1]);
							getchar();
						}*/

					}

					fft(a1, Nx - 2, true);

					for (int i = 1; i < Nx - 1; i++) {

						u[((i)+(j) * (m + 2) + (k) * (m + 2) * (n + 2))] = a1[i - 1]._Val[0];
					}

					delete[] a1;
				}
				else {

					for (int i = 1; i < Nx - 1; i++) {

						a1Im[(k - 1) * (im1)+i - 1] = 0.0;
						a1Re[(k - 1) * (im1)+i - 1] = f[((i)+(j) * (m + 2) + (k) * (m + 2) * (n + 2))];
					}


					int nsizex = Nx - 2;
					bool invert = true;

					if (nsizex == 1) return;



					for (int i = 1, jloc1 = 0; i < nsizex; ++i) {
						int bit = nsizex >> 1;
						for (; jloc1 >= bit; bit >>= 1)
							jloc1 -= bit;
						jloc1 += bit;
						if (i < jloc1)
						{
							//swap(a1[i], a1[jloc1]);
							float tmp = a1Re[(k - 1) * (im1)+i];
							a1Re[(k - 1) * (im1)+i] = a1Re[(k - 1) * (im1)+jloc1];
							a1Re[(k - 1) * (im1)+jloc1] = tmp;

							tmp = a1Im[(k - 1) * (im1)+i];
							a1Im[(k - 1) * (im1)+i] = a1Im[(k - 1) * (im1)+jloc1];
							a1Im[(k - 1) * (im1)+jloc1] = tmp;

						}
					}

					

					for (int len = 2; len <= nsizex; len <<= 1) {
						float ang = 2 * M_PI / len * (invert ? -1 : 1);
						//std::complex<float> wlen((float)(cos(ang)), (float)(sin(ang)));
						float wlenRe = (float)(cosf(ang));
						float wlenIm = (float)(sinf(ang));

						for (int i = 0; i < nsizex; i += len) {
							//std::complex<float> w((float)(1));
							float wRe = 1.0;
							float wIm = 0.0;

							for (int jloc1 = 0; jloc1 < len / 2; ++jloc1) {
								//std::complex<float> u_ = a1[(k - 1) * m + i + jloc1], v = a1[(k - 1) * m + i + jloc1 + len / 2] * w;
								float u_Re = a1Re[(k - 1) * (im1)+i + jloc1];
								float u_Im = a1Im[(k - 1) * (im1)+i + jloc1];

								float vRe = (a1Re[(k - 1) * (im1)+i + jloc1 + len / 2] * wRe - a1Im[(k - 1) * (im1)+i + jloc1 + len / 2] * wIm);
								float vIm = (a1Re[(k - 1) * (im1)+i + jloc1 + len / 2] * wIm + a1Im[(k - 1) * (im1)+i + jloc1 + len / 2] * wRe);

								//a1[(k - 1) * m + i + jloc1] = u_ + v;
								a1Re[(k - 1) * (im1)+i + jloc1] = u_Re + vRe;
								a1Im[(k - 1) * (im1)+i + jloc1] = u_Im + vIm;

								//a1[(k - 1) * m + i + jloc1 + len / 2] = u_ - v;
								a1Re[(k - 1) * (im1)+i + jloc1 + len / 2] = u_Re - vRe;
								a1Im[(k - 1) * (im1)+i + jloc1 + len / 2] = u_Im - vIm;

								//w *= wlen;
								float tmpwRe = (wRe * wlenRe - wIm * wlenIm);
								wIm = (wRe * wlenIm + wIm * wlenRe);
								wRe = tmpwRe;
							}
						}
					}
					if (invert)
						for (int i = 0; i < nsizex; ++i) {
							a1Re[(k - 1) * (im1)+i] /= nsizex;
							//a1Re[(k - 1) * (im1)+i] = (a1Re[(k - 1) * (im1)+i]*nsizex) / (nsizex*nsizex);
						}
					// end fft


					for (int i = 1; i < Nx - 1; i++) {

						u[((i)+(j) * (m + 2) + (k) * (m + 2) * (n + 2))] = a1Re[(k - 1) * (im1)+i - 1];
					}
					
				}
			}

		}

		//k += 1;
	}
}

// Вызывает быстрое преобразование Фурье.

void IDFTy(float*& u, float*& f, int m, int n, int l) {

	int Nx = m + 1; // < M+2
	int Ny = n + 1; // < N+2
	int Nz = l + 1;

	calc_rev(Ny - 2);

	// Преобразование правой части.
#pragma omp parallel for
	for (int k = 1; k < Nz - 1; k++) {


		int id = omp_get_thread_num();
		std::complex<float>* a1 = a1_gl[id];
		std::complex<float>* wlen_pw = wlen_pw_gl[id];

		for (int i = 0; i <= Nx - 1; i++) {

			for (int j = 1; j < Ny - 1; j++) {
				a1[j - 1] = (0.0, f[index(i, j, k, m, n, l)]);
			}
			fft1(a1, Ny - 2, true, wlen_pw);
			for (int j = 1; j < Ny - 1; j++) {
				u[index(i, j, k, m, n, l)] = a1[j - 1]._Val[0];
			}
		}

		a1 = nullptr;
		wlen_pw = nullptr;
	}
}


__global__ void IFFTyKernel(float* u, const float* f, float* a1Re, float* a1Im, int m, int n, int l)
{

	int Nx = m + 1; // < M+2
	int Ny = n + 1; // < N+2
	int Nz = l + 1;

	int im1 = Ny - 2;

	const float M_PI = 3.14159265f;

	int k = threadIdx.x + blockIdx.x * blockDim.x;

	//int k = threadIdx.x;
	const int size = (m + 2) * (n + 2) * (l + 2);
	while (k < Nz - 1) {
		if ((k > 0) && (k < Nz - 1)) {

			// a1 размером [l*n]

			for (int i = 0; i <= Nx - 1; i++) {

				for (int j = 1; j < Ny - 1; j++) {
					a1Im[(k - 1) * im1 + j - 1] = 0.0;
					a1Re[(k - 1) * im1 + j - 1] = f[((i)+(j) * (m + 2) + (k) * (m + 2) * (n + 2))];
				}
				//fft(a1, Ny - 2, true);

				int n_ = Ny - 2;
				bool invert = true;

				if (n_ == 1) return;

				for (int i_ = 1, j_ = 0; i_ < n_; ++i_) {
					int bit = n_ >> 1;
					for (; j_ >= bit; bit >>= 1)
						j_ -= bit;
					j_ += bit;
					if (i_ < j_)
					{
						//swap(a1[(k - 1) * n + i_], a1[(k - 1) * n + j_]);
						float tmp = a1Re[(k - 1) * im1 + i_];
						a1Re[(k - 1) * im1 + i_] = a1Re[(k - 1) * im1 + j_];
						a1Re[(k - 1) * im1 + j_] = tmp;

						tmp = a1Im[(k - 1) * im1 + i_];
						a1Im[(k - 1) * im1 + i_] = a1Im[(k - 1) * im1 + j_];
						a1Im[(k - 1) * im1 + j_] = tmp;
					}
				}

				for (int len = 2; len <= n_; len <<= 1) {
					float ang = 2 * M_PI / len * (invert ? -1 : 1);
					//std::complex<float> wlen((float)(cos(ang)), (float)(sin(ang)));
					float wlenRe = (float)(cosf(ang));
					float wlenIm = (float)(sinf(ang));

					for (int i_ = 0; i_ < n_; i_ += len) {
						//std::complex<float> w((float)(1));
						float wRe = 1.0;
						float wIm = 0.0;

						for (int j_ = 0; j_ < len / 2; ++j_) {
							//std::complex<float> u_ = a1[(k - 1) * n + i_ + j_], v = a1[(k - 1) * n + i_ + j_ + len / 2] * w;
							float u_Re = a1Re[(k - 1) * im1 + i_ + j_];
							float u_Im = a1Im[(k - 1) * im1 + i_ + j_];

							float vRe = (a1Re[(k - 1) * im1 + i_ + j_ + len / 2] * wRe - a1Im[(k - 1) * im1 + i_ + j_ + len / 2] * wIm);
							float vIm = (a1Re[(k - 1) * im1 + i_ + j_ + len / 2] * wIm + a1Im[(k - 1) * im1 + i_ + j_ + len / 2] * wRe);


							//a1[(k - 1) * n + i_ + j_] = u_ + v;
							a1Re[(k - 1) * im1 + i_ + j_] = u_Re + vRe;
							a1Im[(k - 1) * im1 + i_ + j_] = u_Im + vIm;


							//a1[(k - 1) * n + i_ + j_ + len / 2] = u_ - v;
							a1Re[(k - 1) * im1 + i_ + j_ + len / 2] = u_Re - vRe;
							a1Im[(k - 1) * im1 + i_ + j_ + len / 2] = u_Im - vIm;

							//w *= wlen;
							float tmpwRe = (wRe * wlenRe - wIm * wlenIm);
							wIm = (wRe * wlenIm + wIm * wlenRe);
							wRe = tmpwRe;
						}
					}
				}
				if (invert)
					for (int i_ = 0; i_ < n_; ++i_) {
						a1Re[(k - 1) * im1 + i_] /= n_;
					}
				// end fft


				for (int j = 1; j < Ny - 1; j++) {
					u[((i)+(j) * (m + 2) + (k) * (m + 2) * (n + 2))] = a1Re[(k - 1) * im1 + j - 1];
				}
			}

		}
	
		k += blockDim.x * gridDim.x;
		//k += 32;
	}
}


// В массив source копирует массив b со знаком плюс или минус.

void copy3D(float*& source, float*& b, int m, int n, int l,
	bool plus, bool scal, float val) {

	//int Nx = m + 1; // < M+2
	//int Ny = n + 1; // < N+2
	//int Nz = l + 1;

	if (scal) {
		// добавляем скаляр.
		/*
#pragma omp parallel for
		for (int i = 0; i <= Nx; i++) {
			for (int j = 0; j <= Ny; j++) {
				for (int k = 0; k <= Nz; k++) {

					source[i][j][k] += val;
				}
			}
		}*/

		const int size = (m + 2) * (n + 2) * (l+2);

#pragma omp parallel for
		for (int i = 0; i < size; i++) {
			source[i] += val;
		}

	}
	else {

		if (plus) {

/*#pragma omp parallel for
			for (int i = 0; i <= Nx; i++) {
				for (int j = 0; j <= Ny; j++) {
					for (int k = 0; k <= Nz; k++) {

						source[i][j][k] = b[i][j][k];
					}
				}
			}*/

			const int size = (m + 2) * (n + 2) * (l + 2);

#pragma omp parallel for
			for (int i = 0; i < size; i++) {
				source[i] = b[i];
			}

		}
		else {

/*#pragma omp parallel for
			for (int i = 0; i <= Nx; i++) {
				for (int j = 0; j <= Ny; j++) {
					for (int k = 0; k <= Nz; k++) {

						source[i][j][k] = -b[i][j][k];
					}
				}
			}*/

			const int size = (m + 2) * (n + 2) * (l + 2);

#pragma omp parallel for
			for (int i = 0; i < size; i++) {
				source[i] = -b[i];
			}

		}
	}
} // copy3D


__global__ void addKernel2(float* c, const float val, int m, int n, int l)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int size = (m + 2) * (n + 2) * (l + 2);

	while (i < size)
	{
		c[i] += val;

		i += blockDim.x * gridDim.x;
	}
}


__global__ void copyKernel(float* source, const float* b, int m, int n, int l,
	bool plus, bool scal, float val)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	//int i = threadIdx.x;
	const int size = (m + 2) * (n + 2) * (l + 2);
	//int size = 128;

    while (i < size)
	{

		if (scal) {
			if ((i >= 0) && (i < size)) {
				source[i] += val;
			}
		}
		else {

			if (plus) {
				if ((i >= 0) && (i < size)) {
					source[i] = b[i];
				}
			}
			else
			{
				if ((i >= 0) && (i < size)) {
					source[i] = -b[i];
				}
			}
		}

		/*if (i < size)
			source[i] = b[i];*/

		i += blockDim.x * gridDim.x;
		//i += 32;
	}
}


// экспорт 3D полевой величины u в программу tecplot 360.

void exporttecplot3D(float*& u, float*& x, float*& y, float*& z, int m, int n, int l) {
	FILE* fp;
	errno_t err;
	// создание файла для записи.
	if ((err = fopen_s(&fp, "temperature_xyz_plot.PLT", "w")) != 0) {
		printf("Create File Error\n");
	}
	else {

		// запись имён переменных
		fprintf(fp, "VARIABLES = x y z Temperature\n");
		fprintf(fp, "zone\n");
		fprintf(fp, "I=%d, J=%d, K=%d, F=POINT\n", m + 2, n + 2, l + 2);
		for (int k = 0; k < l + 2; k++) for (int j = 0; j < n + 2; j++) for (int i = 0; i < m + 2; i++)   fprintf(fp, "%e %e %e %e\n", x[i], y[j], z[k], u[index(i, j, k, m, n, l)]);
		fclose(fp);

		WinExec("C:\\Program Files (x86)\\Tecplot\\Tec360 2008\\bin\\tec360.exe temperature_xyz_plot.PLT", SW_NORMAL);
	}

	//getchar();

} // exporttecplot3D

// экспорт 3D полевой величины u в программу tecplot 360.

void exporttecplot3D_fft(float*& u, float*& x, float*& y, float*& z, int m, int n, int l) {
	FILE* fp;
	errno_t err;
	// создание файла для записи.
	if ((err = fopen_s(&fp, "temperature_xyz_plot.PLT", "w")) != 0) {
		printf("Create File Error\n");
	}
	else {

		// запись имён переменных
		fprintf(fp, "VARIABLES = x y z Temperature\n");
		fprintf(fp, "zone\n");
		fprintf(fp, "I=%d, J=%d, K=%d, F=POINT\n", m - 1, n - 1, l + 2);
		for (int k = 0; k < l + 2; k++) for (int j = 1; j < n; j++) for (int i = 1; i < m; i++)   fprintf(fp, "%e %e %e %e\n", x[i], y[j], z[k], u[index(i, j, k, m, n, l)]);
		fclose(fp);

		WinExec("C:\\Program Files (x86)\\Tecplot\\Tec360 2008\\bin\\tec360.exe temperature_xyz_plot.PLT", SW_NORMAL);
	}

	//getchar();

} // exporttecplot3D_fft


// экспорт 3D полевой величины u в программу tecplot 360.

void exporttecplot3D22D(float*& u, float*& x, float*& y, float*& z, int m, int n, int l,int k) {
	FILE* fp;
	errno_t err;
	// создание файла для записи.
	if ((err = fopen_s(&fp, "temperature_xy_plot.PLT", "w")) != 0) {
		printf("Create File Error\n");
	}
	else {

		// запись имён переменных
		fprintf(fp, "VARIABLES = x y Temperature\n");
		fprintf(fp, "zone\n");
		fprintf(fp, "I=%d, J=%d, K=1, F=POINT\n", m + 2, n + 2);
		for (int j = 0; j < n + 2; j++) for (int i = 0; i < m + 2; i++)   fprintf(fp, "%e %e %e\n", x[i], y[j], u[index(i, j, k+1, m, n, l)]);
		fclose(fp);

		WinExec("C:\\Program Files (x86)\\Tecplot\\Tec360 2008\\bin\\tec360.exe temperature_xy_plot.PLT", SW_NORMAL);
	}

	//getchar();

} // exporttecplot3D2_2D

// экспорт 3D полевой величины u в программу tecplot 360.

void exporttecplot3D21D(float*& u, float*& x, float*& y, float*& z, int m, int n, int l, int k) {
	FILE* fp;
	errno_t err;
	// создание файла для записи.
	if ((err = fopen_s(&fp, "temperature_x_plot.PLT", "w")) != 0) {
		printf("Create File Error\n");
	}
	else {

		// запись имён переменных
		fprintf(fp, "VARIABLES = x Temperature\n");

		int j0 = (n + 1) / 2;
		for (int i = 1; i < m + 1; i++)   fprintf(fp, "%e %e\n", x[i], u[index(i, j0, k + 1, m, n, l)]);
		fclose(fp);

		WinExec("C:\\Program Files (x86)\\Tecplot\\Tec360 2008\\bin\\tec360.exe temperature_x_plot.PLT", SW_NORMAL);
	}

	//getchar();

} // exporttecplot3D2_1D

// Структура для описания одного слоя подложки.

struct STACK_LAYERS {
	float chikness; // Толщина в м.
	float lambda; // теплопроводность, Вт/(м*К).
	int idiv_layer; // Количество ячеек в слое по толщине.
	float mplane, mortho; // Ортотропность коэффициента теплопроводности 24.05.2021.
	// Для расчёта вклада каждого слоя в Rt.
	float Tmax;
	float alpha; // Температурная зависимость теплопроводности.
	//std::string name; // имя материала.
	char* name;
};

// Процедура mesh_size
// подбирает размер равномерной сетки в плоскости Oxy в 
// целях экономии ресурсов памяти и увеличения быстродействия,
// подбирает сетку так чтобы рассчитываемая топология в плоскости Oxy 
// была разрешена расчётной сеткойи эта сетка не была избыточной.

void mesh_size(int& N, int& M,
	int n_xgr, int n_y, float& lengthx, float& lengthy,
	float& multiplyer, float size_x, float size_y, float distance_x,
	float distance_y, float h1, float h2)
{
	/*if (n_xgr <= 2048)*/ {

		M = 33;
		lengthx = (M + 1) * h1;

		if (multiplyer * (n_xgr)*distance_x + 3.0 * size_x < lengthx) {
			// Ok
		}
		else {
			M = 65;
			lengthx = (M + 1) * h1;

			if (multiplyer * (n_xgr)*distance_x + 3.0 * size_x < lengthx) {
				// Ok
			}
			else {
				M = 129;
				lengthx = (M + 1) * h1;

				if (multiplyer * (n_xgr)*distance_x < lengthx) {
					// Ok
				}
				else {
					M = 257;
					lengthx = (M + 1) * h1;

					if (multiplyer * (n_xgr)*distance_x < lengthx) {
						// Ok
					}
					else {
						M = 513;
						lengthx = (M + 1) * h1;

						if (multiplyer * (n_xgr)*distance_x < lengthx) {
							// Ok
						}
						else {
							M = 1025;
							lengthx = (M + 1) * h1;

							if (multiplyer * (n_xgr)*distance_x < lengthx) {
								// Ok
							}
							else {
								M = 2049;
								lengthx = (M + 1) * h1;

								if (multiplyer * (n_xgr)*distance_x < lengthx) {
									// Ok
								}
								else {
									M = 4097;
									lengthx = (M + 1) * h1;

									if (multiplyer * (n_xgr)*distance_x < lengthx) {
										// Ok
									}
									else {
										M = 8193;
										lengthx = (M + 1) * h1;

										if (multiplyer * (n_xgr)*distance_x < lengthx) {
											// Ok
										}
										else {
											M = 16385;
											lengthx = (M + 1) * h1;

											if (multiplyer * (n_xgr)*distance_x < lengthx) {
												// Ok
											}
											else {
												M = 32769;
												lengthx = (M + 1) * h1;

												if (multiplyer * (n_xgr)*distance_x < lengthx) {
													// Ok
												}
												else {
													M = 65537;
													lengthx = (M + 1) * h1;

													if (multiplyer * (n_xgr)*distance_x < lengthx) {
														// Ok
													}
													else {
														M = 131073;
														lengthx = (M + 1) * h1;

														if (multiplyer * (n_xgr)*distance_x < lengthx) {
															// Ok
														}
														else {
															M = 262145;
															lengthx = (M + 1) * h1;

															if (multiplyer * (n_xgr)*distance_x < lengthx) {
																// Ok
															}
															else {
																// СБОЙ
															}
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}

	if (n_y == 1) {

		N = 129;
		lengthy = (N + 1) * h2;

		//N = 33;
		//lengthy = (N + 1) * h2;
	}
	else {
		/*if (n_y <= 512)*/ {

			N = 33;
			lengthx = (N + 1) * h2;

			if (multiplyer * (n_y)*distance_y < lengthy) {
				// Ok
			}
			else {
				N = 65;
				lengthy = (N + 1) * h2;

				if (multiplyer * (n_y)*distance_y < lengthy) {
					// Ok
				}
				else {

					N = 129;
					lengthy = (N + 1) * h1;

					if (multiplyer * (n_y)*distance_y < lengthy) {
						// Ok
					}
					else {
						N = 257;
						lengthy = (N + 1) * h2;

						if (multiplyer * (n_y)*distance_y < lengthy) {
							// Ok
						}
						else {
							N = 513;
							lengthy = (N + 1) * h2;

							if (multiplyer * (n_y)*distance_y < lengthy) {
								// Ok
							}
							else {
								N = 1025;
								lengthy = (N + 1) * h2;

								if (multiplyer * (n_y)*distance_y < lengthy) {
									// Ok
								}
								else {
									N = 2049;
									lengthy = (N + 1) * h2;

									if (multiplyer * (n_y)*distance_y < lengthy) {
										// Ok
									}
									else {
										N = 4097;
										lengthy = (N + 1) * h2;

										if (multiplyer * (n_y)*distance_y < lengthy) {
											// Ok
										}
										else {
											N = 8193;
											lengthy = (N + 1) * h2;

											if (multiplyer * (n_y)*distance_y < lengthy) {
												// Ok
											}
											else {
												N = 16385;
												lengthy = (N + 1) * h2;

												if (multiplyer * (n_y)*distance_y < lengthy) {
													// Ok
												}
												else {
													// СБОЙ
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
} // mesh_size


// Ядро солвера.
// Реализует метод прогонки с трёхдиагональной матрицей для решения 
// вдоль вертикального направления перпендикулярно слоям подложки.
void progonka(int Nx, int Ny, int Nz,
	float*& lambda, float*& lambda_multiplyer_normal, float*& lambda_multiplyer_plane,
	float*& hz, float h1, float h2, float*& b, float*& a)
{

	int m = Nx - 1;
	int n = Ny - 1;
	int l = Nz - 1;

	// Ядро солвера.
//#pragma omp parallel for
	for (int i = 1; i < Nx - 1; i++) {
		for (int j = 1; j < Ny - 1; j++) {

			//float* P = new float[Nz + 1];
			//float* Q = new float[Nz + 1];
			float* P = (float*)malloc((Nz+1)*sizeof(float));
			float* Q = (float*)malloc((Nz + 1) * sizeof(float));

			float Rj1 = lambda_multiplyer_normal[1] * (2.0 * lambda[0] * lambda[1] / (lambda[0] + lambda[1])) *
				(2.0 / (hz[1] * (hz[1] + hz[0])));
			float Rjm1 = lambda_multiplyer_normal[1] * (2.0 * lambda[0] * lambda[1] / (lambda[0] + lambda[1])) *
				(2.0 / (hz[0] * (hz[1] + hz[0])));

			// Для фиксированного k решение системы  с трёх диагональной матрицей:
			float b1 = Rj1;
			float a1 = (2.0 / (h1 * h1) + 2.0 / (h2 * h2)) * lambda[1] * lambda_multiplyer_plane[1] +
				(Rj1 + Rjm1) - 2.0 * cos(2.0 * M_PI * i / Nx) * lambda[1] * lambda_multiplyer_plane[1] / (h1 * h1) -
				2.0 * cos(2.0 * M_PI * j / Ny) * lambda[1] * lambda_multiplyer_plane[1] / (h2 * h2);
			P[1] = b1 / a1;
			float d1 = -b[index(i, j, 1, m, n, l)];// -b[i][j][1];
			Q[1] = d1 / a1;

			for (int k = 2; k <= Nz; k++) {

				float lmax = lambda[k];
				if (k != Nz) {
					lmax = lambda[k + 1];
				}
				float Rj = lambda_multiplyer_normal[k] * (2.0 * lambda[k] * lmax / (lambda[k] + lmax)) *
					(2.0 / (hz[k] * (hz[k] + hz[k - 1])));
				float Rjm1 = lambda_multiplyer_normal[k] * (2.0 * lambda[k - 1] * lambda[k] / (lambda[k - 1] + lambda[k])) *
					(2.0 / (hz[k - 1] * (hz[k] + hz[k - 1])));

				// bk -> k+1
				// ck -> k-1
				float bk = Rj;
				float ck = Rjm1;
				float ak = (2.0 / (h1 * h1) + 2.0 / (h2 * h2)) * lambda[k] * lambda_multiplyer_plane[k] + (Rj + Rjm1) -
					2.0 * cos(2.0 * M_PI * i / Nx) * lambda[k] * lambda_multiplyer_plane[k] / (h1 * h1) -
					2.0 * cos(2.0 * M_PI * j / Ny) * lambda[k] * lambda_multiplyer_plane[k] / (h2 * h2);
				float dk = -b[index(i, j, k, m, n, l)];// b[i][j][k];

				P[k] = bk / (ak - ck * P[k - 1]);
				Q[k] = (dk + ck * Q[k - 1]) / (ak - ck * P[k - 1]);
			}
			//a[i][j][Nz] 
			a[index(i, j, Nz, m, n, l)] = Q[Nz];

			for (int k = Nz - 1; k >= 1; k--) {
				//a[i][j][k] = P[k] * a[i][j][k + 1] + Q[k];
				a[index(i, j, k, m, n, l)] = P[k] * a[index(i, j, k+1, m, n, l)] + Q[k];
			}

			free(P);
			free(Q);
		}
	}
} // progonka

// Ядро солвера.
// Реализует метод прогонки с трёхдиагональной матрицей для решения 
// вдоль вертикального направления перпендикулярно слоям подложки.
__global__  void progonkaKernel(int Nx, int Ny, int Nz,
	const float* lambda, const float* lambda_multiplyer_normal, const float* lambda_multiplyer_plane,
	const float* hz, float h1, float h2, float* b, float* a, float* P, float* Q)
{

	int m = Nx - 1;
	int n = Ny - 1;
	int l = Nz - 1;

	const float M_PI = 3.14159265f;

	int im1 = Nz + 1;

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	// Ядро солвера.
//#pragma omp parallel for
	//for (int i = 1; i < Nx - 1; i++) 
	while (i < Nx - 1)
	{
		if ((i > 0) && (i < Nx - 1)) {
			for (int j = 1; j < Ny - 1; j++) {

				

				float Rj1 = lambda_multiplyer_normal[1] * (2.0 * lambda[0] * lambda[1] / (lambda[0] + lambda[1])) *
					(2.0 / (hz[1] * (hz[1] + hz[0])));
				float Rjm1 = lambda_multiplyer_normal[1] * (2.0 * lambda[0] * lambda[1] / (lambda[0] + lambda[1])) *
					(2.0 / (hz[0] * (hz[1] + hz[0])));

				// Для фиксированного k решение системы  с трёх диагональной матрицей:
				float b1 = Rj1;
				float a1 = (2.0 / (h1 * h1) + 2.0 / (h2 * h2)) * lambda[1] * lambda_multiplyer_plane[1] +
					(Rj1 + Rjm1) - 2.0 * cos(2.0 * M_PI * i / Nx) * lambda[1] * lambda_multiplyer_plane[1] / (h1 * h1) -
					2.0 * cos(2.0 * M_PI * j / Ny) * lambda[1] * lambda_multiplyer_plane[1] / (h2 * h2);
				P[(i-1)*im1+1] = b1 / a1;
				float d1 = -b[((i)+(j) * (m + 2) + (1) * (m + 2) * (n + 2))];// -b[i][j][1];
				Q[(i - 1) * im1 + 1] = d1 / a1;

				for (int k = 2; k <= Nz; k++) {

					float lmax = lambda[k];
					if (k != Nz) {
						lmax = lambda[k + 1];
					}
					float Rj = lambda_multiplyer_normal[k] * (2.0 * lambda[k] * lmax / (lambda[k] + lmax)) *
						(2.0 / (hz[k] * (hz[k] + hz[k - 1])));
					float Rjm1 = lambda_multiplyer_normal[k] * (2.0 * lambda[k - 1] * lambda[k] / (lambda[k - 1] + lambda[k])) *
						(2.0 / (hz[k - 1] * (hz[k] + hz[k - 1])));

					// bk -> k+1
					// ck -> k-1
					float bk = Rj;
					float ck = Rjm1;
					float ak = (2.0 / (h1 * h1) + 2.0 / (h2 * h2)) * lambda[k] * lambda_multiplyer_plane[k] + (Rj + Rjm1) -
						2.0 * cos(2.0 * M_PI * i / Nx) * lambda[k] * lambda_multiplyer_plane[k] / (h1 * h1) -
						2.0 * cos(2.0 * M_PI * j / Ny) * lambda[k] * lambda_multiplyer_plane[k] / (h2 * h2);
					float dk = -b[((i)+(j) * (m + 2) + (k) * (m + 2) * (n + 2))];// b[i][j][k];

					P[(i - 1) * im1 + k] = bk / (ak - ck * P[(i - 1) * im1 + k - 1]);
					Q[(i - 1) * im1 + k] = (dk + ck * Q[(i - 1) * im1 + k - 1]) / (ak - ck * P[(i - 1) * im1 + k - 1]);
				}
				//a[i][j][Nz] 
				a[((i)+(j) * (m + 2) + (Nz) * (m + 2) * (n + 2))] = Q[(i - 1) * im1 + Nz];

				for (int k = Nz - 1; k >= 1; k--) {
					//a[i][j][k] = P[(i-1)*im1+k] * a[i][j][k + 1] + Q[(i-1)*im1+k];
					a[((i)+(j) * (m + 2) + (k) * (m + 2) * (n + 2))] = P[(i - 1) * im1 + k] * a[((i)+(j) * (m + 2) + (k+1) * (m + 2) * (n + 2))] + Q[(i - 1) * im1 + k];
				}

				
			}
		}

		i += blockDim.x * gridDim.x;
	}
} // progonkaKernel



// Начало 18.05.2021 - окончание 31.05.2021.
// Находит статическое тепловое сопротивление для постоянных
// коэффициентов теплопроводности не зависящих от температуры.
// Теперь использует видеокарту nvidia GeForce 840M 5.06.2021.
void FFTsolver3Dqlinear(float& thermal_resistance,
	float size_x, float size_y, float distance_x, float distance_y,
	float size_gx, int n_x, int n_y, int n_gx,
	bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7, bool b8, bool b9,
	float d1, float d2, float d3,
	float d4, float d5, float d6,
	float d7, float d8, float d9,
	float k1, float k2, float k3,
	float k4, float k5, float k6,
	float k7, float k8, float k9,
	float mplane1, float mplane2, float mplane3,
	float mplane4, float mplane5, float mplane6,
	float mplane7, float mplane8, float mplane9,
	float mortogonal1, float mortogonal2, float mortogonal3,
	float mortogonal4, float mortogonal5, float mortogonal6,
	float mortogonal7, float mortogonal8, float mortogonal9,
	int& time, float Tamb, float Pdiss, bool export3D, bool exportxy2D, bool exportx1D, int idGPU) {

	// Замер времени.
	unsigned int calculation_start_time = 0; // начало счёта мс.
	unsigned int calculation_end_time = 0; // окончание счёта мс.
	unsigned int calculation_seach_time = 0; // время выполнения участка кода в мс.

	calculation_start_time = clock();

	float* u = nullptr, * source = nullptr; // рассчитываемый потенциал и тепловая мощность.

	float* a = nullptr, * b = nullptr; // Коэффициенты разложения в дискретный ряд Фурье.

	float* dev_a = nullptr, * dev_b = nullptr;

	float Pdiss_sum = 0.0;

	float lengthx = 4098.0e-6;// 2050.0e-6;
	float lengthy = 1625.0e-6;

	int M = 4097; // 2049;//511;
	int N = 129;// 511;



	/*lengthx = lengthy = 1.0;
	M = 32;
	N = 159;*/

	//float Tamb = 22.0; // - температура корпуса задаётся пользователем. 

	//Tamb = 0.0;

	// постоянные шаги сетки h1 по оси x и h2 по оси y.
	float h1 = (float)(lengthx / (M + 1));
	float h2 = (float)(lengthy / (N + 1));

	h1 = 0.5 * size_x;
	if (n_y == 1) {
		h2 = 0.1 * size_y;
	}
	else {
		h2 = 0.5 * size_y;
	}


	lengthx = (M + 1) * h1;
	lengthy = (N + 1) * h2;

	float multiplyer = 1.2;

	float n_xgr = n_x * n_gx;
	if (fabs(distance_x) > 1.0e-30) {
		n_xgr += ((n_gx - 1) * (size_gx / distance_x));
	}
	

	mesh_size(N, M, n_xgr, n_y, lengthx, lengthy, multiplyer, size_x, size_y, distance_x, distance_y, h1, h2);


	//float* xf = new float[M + 2];
	float* xf = (float*)malloc((M + 2) * sizeof(float));
	for (int i = 0; i < M + 2; i++) {
		xf[i] = (float)(i * h1);
	}
	//float* yf = new float[N + 2];
	float* yf = (float*)malloc((N + 2) * sizeof(float));
	for (int i = 0; i < N + 2; i++) {
		yf[i] = (float)(i * h2);
	}

	int ilayer_count = 8;
	STACK_LAYERS* stack_layer = nullptr;
	//float chikness_min = 1.0e30;

	//char airname[4] = "air";
	//std::string airname = "air";

	if (b9) {
		ilayer_count = 10;
	}
	else if (b8) {
		ilayer_count = 9;
	}
	else if (b7) {
		ilayer_count = 8;
	}
	else if (b6) {
		ilayer_count = 7;
	}
	else if (b5) {
		ilayer_count = 6;
	}
	else if (b4) {
		ilayer_count = 5;
	}
	else if (b3) {
		ilayer_count = 4;
	}
	else if (b2) {
		ilayer_count = 3;
	}
	else {
		ilayer_count = 2;
	}

	stack_layer = new STACK_LAYERS[ilayer_count];

	if (b1) {
		//const int idiv_layer = 4;		
		stack_layer[0].chikness = d1;// 1.6e-3;
		stack_layer[0].lambda = k1; // МД40
		stack_layer[0].idiv_layer = 14;
		stack_layer[0].mplane = mplane1;
		stack_layer[0].mortho = mortogonal1;
		stack_layer[0].alpha = 0.0;// alpha1;
		//stack_layer[0].name = s1;

		if (b2) {
			stack_layer[1].chikness = d2;
			stack_layer[1].lambda = k2; // AuSn
			stack_layer[1].idiv_layer = 14;
			stack_layer[1].mplane = mplane2;
			stack_layer[1].mortho = mortogonal2;
			stack_layer[1].alpha = 0.0;// alpha2;
			//stack_layer[1].name = s2;
			if (b3) {
				stack_layer[2].chikness = d3;
				stack_layer[2].lambda = k3; // Cu
				stack_layer[2].idiv_layer = 14;
				stack_layer[2].mplane = mplane3;
				stack_layer[2].mortho = mortogonal3;
				stack_layer[2].alpha = 0.0;// alpha3;
				//stack_layer[2].name = s3;

				if (b4) {
					stack_layer[3].chikness = d4;
					stack_layer[3].lambda = k4; // AuSn
					stack_layer[3].idiv_layer = 14;
					stack_layer[3].mplane = mplane4;
					stack_layer[3].mortho = mortogonal4;
					stack_layer[3].alpha = 0.0;// alpha4;
					//stack_layer[3].name = s4;

					if (b5) {
						stack_layer[4].chikness = d5;
						stack_layer[4].lambda = k5; // SiC
						stack_layer[4].idiv_layer = 14;
						stack_layer[4].mplane = mplane5;
						stack_layer[4].mortho = mortogonal5;
						stack_layer[4].alpha = 0.0;// alpha5;
						//stack_layer[4].name = s5;

						if (b6) {
							stack_layer[5].chikness = d6;
							stack_layer[5].lambda = k6;// 130.0; // GaN
							stack_layer[5].idiv_layer = 14;
							stack_layer[5].mplane = mplane6;
							stack_layer[5].mortho = mortogonal6;
							stack_layer[5].alpha = 0.0;// alpha6;
							//stack_layer[5].name = s6;

							if (b7) {
								stack_layer[6].chikness = d7;
								stack_layer[6].lambda = k7;// 317.0; // gold
								stack_layer[6].idiv_layer = 14;
								stack_layer[6].mplane = mplane7;
								stack_layer[6].mortho = mortogonal7;
								stack_layer[6].alpha = 0.0;// alpha7;
								//stack_layer[6].name = s7;
								if (b8) {
									stack_layer[7].chikness = d8;
									stack_layer[7].lambda = k8;// 317.0; // gold
									stack_layer[7].idiv_layer = 14;
									stack_layer[7].mplane = mplane8;
									stack_layer[7].mortho = mortogonal8;
									stack_layer[7].alpha = 0.0;// alpha8;
									//stack_layer[7].name = s8;
									if (b9) {
										stack_layer[8].chikness = d9;
										stack_layer[8].lambda = k9;// 317.0; // gold
										stack_layer[8].idiv_layer = 14;
										stack_layer[8].mplane = mplane9;
										stack_layer[8].mortho = mortogonal9;
										stack_layer[8].alpha = 0.0;// alpha9;
										//stack_layer[8].name = s9;
										stack_layer[9].chikness = d9;//0.11 * (d1 + d2 + d3 + d4 + d5 + d6 + d7+d8+d9);
										stack_layer[9].lambda = 0.026;// 1.0;// 0.026;// 0.026; // air
										stack_layer[9].idiv_layer = 14;
										stack_layer[9].mplane = 1.0;
										stack_layer[9].mortho = 1.0;
										stack_layer[9].alpha = 0.0;
										//stack_layer[9].name = airname;
									}
									else {
										stack_layer[8].chikness = d8;//0.11 * (d1 + d2 + d3 + d4 + d5 + d6 + d7+d8+d9);
										stack_layer[8].lambda = 0.026;// 1.0;// 0.026;// 0.026; // air
										stack_layer[8].idiv_layer = 14;
										stack_layer[8].mplane = 1.0;
										stack_layer[8].mortho = 1.0;
										stack_layer[8].alpha = 0.0;
										//stack_layer[8].name = airname;
									}
								}
								else {
									stack_layer[7].chikness = d7;//0.11 * (d1 + d2 + d3 + d4 + d5 + d6 + d7+d8+d9);
									stack_layer[7].lambda = 0.026;// 1.0;// 0.026;// 0.026; // air
									stack_layer[7].idiv_layer = 14;
									stack_layer[7].mplane = 1.0;
									stack_layer[7].mortho = 1.0;
									stack_layer[7].alpha = 0.0;
									//stack_layer[7].name = airname;
								}
							}
							else {
								stack_layer[6].chikness = d6;//0.11 * (d1 + d2 + d3 + d4 + d5 + d6 + d7+d8+d9);
								stack_layer[6].lambda = 0.026;// 1.0;// 0.026;// 0.026; // air
								stack_layer[6].idiv_layer = 14;
								stack_layer[6].mplane = 1.0;
								stack_layer[6].mortho = 1.0;
								stack_layer[6].alpha = 0.0;
								//stack_layer[6].name = airname;
							}
						}
						else {
							stack_layer[5].chikness = d5;//0.11 * (d1 + d2 + d3 + d4 + d5 + d6 + d7+d8+d9);
							stack_layer[5].lambda = 0.026;// 1.0;// 0.026;// 0.026; // air
							stack_layer[5].idiv_layer = 14;
							stack_layer[5].mplane = 1.0;
							stack_layer[5].mortho = 1.0;
							stack_layer[5].alpha = 0.0;
							//stack_layer[5].name = airname;
						}
					}
					else {
						stack_layer[4].chikness = d4;//0.11 * (d1 + d2 + d3 + d4 + d5 + d6 + d7+d8+d9);
						stack_layer[4].lambda = 0.026;// 1.0;// 0.026;// 0.026; // air
						stack_layer[4].idiv_layer = 14;
						stack_layer[4].mplane = 1.0;
						stack_layer[4].mortho = 1.0;
						stack_layer[4].alpha = 0.0;
						//stack_layer[4].name = airname;
					}
				}
				else {
					stack_layer[3].chikness = d3;//0.11 * (d1 + d2 + d3 + d4 + d5 + d6 + d7+d8+d9);
					stack_layer[3].lambda = 0.026;// 1.0;// 0.026;// 0.026; // air
					stack_layer[3].idiv_layer = 14;
					stack_layer[3].mplane = 1.0;
					stack_layer[3].mortho = 1.0;
					stack_layer[3].alpha = 0.0;
					//stack_layer[3].name = airname;
				}
			}
			else {
				stack_layer[2].chikness = d2;//0.11 * (d1 + d2 + d3 + d4 + d5 + d6 + d7+d8+d9);
				stack_layer[2].lambda = 0.026;// 1.0;// 0.026;// 0.026; // air
				stack_layer[2].idiv_layer = 14;
				stack_layer[2].mplane = 1.0;
				stack_layer[2].mortho = 1.0;
				stack_layer[2].alpha = 0.0;
				//stack_layer[2].name = airname;
			}
		}
		else {
			stack_layer[1].chikness = d1;//0.11 * (d1 + d2 + d3 + d4 + d5 + d6 + d7+d8+d9);
			stack_layer[1].lambda = 0.026;// 1.0;// 0.026;// 0.026; // air
			stack_layer[1].idiv_layer = 14;
			stack_layer[1].mplane = 1.0;
			stack_layer[1].mortho = 1.0;
			stack_layer[1].alpha = 0.0;
			//stack_layer[1].name = airname;
		}

	}

	float lengthz = 0.0;
	int inodes = 0;
	//lengthz = 1.0;
	for (int i = 0; i < ilayer_count; i++) {
		lengthz += stack_layer[i].chikness;
		inodes += stack_layer[i].idiv_layer;
	}
	int izstop = inodes - stack_layer[ilayer_count - 1].idiv_layer + 1;
	int L = inodes + 1;


	//int izstop = 63;
	//int L = izstop=63;
	//float hzc = (float)(lengthz / (L + 1));
	//float* hz = new float[L + 2];
	float* hz = (float*)malloc((L + 2) * sizeof(float));
	//float* zf = new float[L + 2];
	float* zf = (float*)malloc((L + 2) * sizeof(float));
	//float* lambda = new float[L + 2];// Теплопроводность.
	float* lambda = (float*)malloc((L + 2) * sizeof(float));
	//float* lambda_multiplyer_plane = new float[L + 2];// Теплопроводность.
	float* lambda_multiplyer_plane = (float*)malloc((L + 2) * sizeof(float));
	//float* lambda_multiplyer_normal = new float[L + 2];// Теплопроводность.
	float* lambda_multiplyer_normal = (float*)malloc((L + 2) * sizeof(float));
	
	int* layer_id = new int[L + 2];


	float* dev_lambda, * dev_lambda_multiplyer_normal, * dev_lambda_multiplyer_plane, * dev_hz;

	/*for (int i = 0; i < L + 2; i++) {
		zf[i] = (float)(i * hzc);
		hz[i] = hzc;
		if (i > izstop) {
			lambda[i] = 1.0; // air
		}
		else {
			lambda[i] = 1.0; // SiC
		}
	}
	*/
	// 19.05.2021
	zf[0] = 0.0;
	float q = 1.2;
	float qinv = 1 / q;
	hz[0] = stack_layer[0].chikness * (qinv - 1.0) / (pow(qinv, 1.0*stack_layer[0].idiv_layer) - 1.0);
	// Постоянный шаг по оси z в каждом интервале.
	//hz[0] = stack_layer[0].chikness / stack_layer[0].idiv_layer;
	float qsum = 0.0;
	int ic = 1;
	layer_id[0] = 0;
	lambda[0] = stack_layer[0].lambda;
	lambda_multiplyer_plane[0] = stack_layer[0].mplane;
	lambda_multiplyer_normal[0] = stack_layer[0].mortho;
	for (int i = 0; i < ilayer_count; i++) {
		for (int j = 0; j < stack_layer[i].idiv_layer; j++) {
			if (i < ilayer_count - 1) {

				hz[ic] = stack_layer[i].chikness * (qinv - 1.0) / (pow(qinv, 1.0*stack_layer[i].idiv_layer) - 1.0);
				for (int i7 = 1; i7 <= j; i7++) hz[ic] *= qinv;
			}
			else {
				hz[ic] = stack_layer[i].chikness * (qinv - 1.0) / (pow(qinv, 1.0*stack_layer[i].idiv_layer) - 1.0);
				for (int i7 = 0; i7 <= stack_layer[i].idiv_layer - 1 - j; i7++) hz[ic] *= qinv;
				qsum += hz[ic];
			}
			// Постоянный шаг по оси z в каждом интервале.
			//hz[ic] = stack_layer[i].chikness / stack_layer[i].idiv_layer;
			layer_id[ic] = i;
			lambda[ic] = stack_layer[i].lambda;
			lambda_multiplyer_plane[ic] = stack_layer[i].mplane;
			lambda_multiplyer_normal[ic] = stack_layer[i].mortho;
			zf[ic] = zf[ic-1] + hz[ic];
			ic++;
		}
	}
	//zf[1 + ic] = zf[ic] + stack_layer[ilayer_count - 1].chikness / stack_layer[ilayer_count - 1].idiv_layer;
	//hz[ic] = stack_layer[ilayer_count - 1].chikness / stack_layer[ilayer_count - 1].idiv_layer;

	//hz[ic + 1] = stack_layer[ilayer_count - 1].chikness / stack_layer[ilayer_count - 1].idiv_layer;


	hz[ic] = hz[ic + 1] = (stack_layer[ilayer_count - 1].chikness - qsum) / 2.0;

	zf[ic] = zf[ic - 1] + hz[ic];


	zf[1 + ic] = 0.0;
	for (int i7 = 0; i7 < ilayer_count; i7++) {
		zf[1 + ic] += stack_layer[i7].chikness;
	}

	layer_id[ic] = ilayer_count - 1;
	layer_id[ic + 1] = ilayer_count - 1;
	lambda[ic] = stack_layer[ilayer_count - 1].lambda;
	lambda[ic + 1] = stack_layer[ilayer_count - 1].lambda;
	lambda_multiplyer_plane[ic] = stack_layer[ilayer_count - 1].mplane;
	lambda_multiplyer_normal[ic] = stack_layer[ilayer_count - 1].mortho;
	lambda_multiplyer_plane[ic + 1] = stack_layer[ilayer_count - 1].mplane;
	lambda_multiplyer_normal[ic + 1] = stack_layer[ilayer_count - 1].mortho;


	int Nx = M + 1; // < M+2
	int Ny = N + 1; // < N+2
	int Nz = L + 1;

	for (int k = 1; k <= Nz; k++) {

		//printf("%d %d lam=%e %e\n", k, izstop - 1, lambda[k], hz[k]);
	}
	//getchar();

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(idGPU);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error1;
	}

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("%s\n", prop.name);
	// GeForce 840M 384потока 1ГГц каждый март 2014 года. 28нм.

	float* dev_a1Re = nullptr, * dev_a1Im = nullptr;
	float* dev_a2Re = nullptr, * dev_a2Im = nullptr;

	float* dev_P, * dev_Q;

	int size = (M + 2) * (N + 2) * (L + 2);


	alloc_u3D(source, M, N, L);
	alloc_u3D(b, M, N, L);

	// инициализация.
	init_u3D(b, M, N, L, 0.0);
	init_u3D(source, M, N, L, 0.0);

	const float power_1_polosok = Pdiss / (n_y * (n_x + 1) * n_gx);//0.675;

	int in_x = (int)(distance_x / h1);
	int in_y = (int)(distance_y / h2);

	// Задаём тепловую мощность тепловыделения.
	//for (int i = M / 2 - 117; i < M / 2 + 124; i += 26) {// ПТБШ 1,25мм 2s 204Mb
	//for (int i = M / 2 - 234; i < M / 2 + 240; i += 26) {// ПТБШ 2,5мм 2s 204Mb
	//for (int i = M / 2 - 468; i < M / 2 + 480; i += 26) { // ПТБШ 5мм 2s 204Mb
		//for (int i = M / 2 - 936; i < M / 2 + 946; i += 26) { // ПТБШ 10мм 4s 420Mb
	int istart0 = M / 2 - (int)(0.5 * (in_x * n_xgr));
	int iend0 = istart0 + ((int)((in_x * n_x))) + (in_x == 0 ? 3 : 1);

	for (int igr = 0; igr < n_gx; igr++)
	{

		if (igr > 0) {
			istart0 += (int)(size_gx / h1) + ((int)((in_x * n_x))) + (in_x == 0 ? 3 : 1);
			iend0 += (int)(size_gx / h1) + ((int)((in_x * n_x))) + (in_x == 0 ? 3 : 1);
		}

		if (n_x == 0) {
			// Один источник по оси oX
			int i = M / 2;
			for (int j = (n_y == 1 ? N / 2 - 5 : N / 2 - (int)(0.5 * (in_y * (n_y)))); j < (n_y == 1 ? N / 2 + 5 : N / 2 + (int)(0.5 * (in_y * (n_y)) + (in_y == 0 ? 3 : 1))); j += (n_y == 1 ? 1 : in_y)) {
				//if ((i - (M / 2 - 117)) % 26 == 0)
				{


					//source[i][j][izstop] = 0.675 / (5*h1*h2*hz[izstop-1]);
					float Ssource = size_x * size_y;
					float Vol_source = Ssource * hz[izstop - 1];
					if (n_y == 1) {
						//source[i + 1][j][izstop - 1] += power_1_polosok / (Vol_source);
						//source[i][j][izstop - 1] += power_1_polosok / (Vol_source);

						source[index(i+1, j, izstop - 1, M, N, L)] += power_1_polosok / (Vol_source);
						source[index(i, j, izstop - 1, M, N, L)] += power_1_polosok / (Vol_source);
					}
					else {
						//source[i + 1][j][izstop - 1] += power_1_polosok / (Vol_source);
						//source[i][j][izstop - 1] += power_1_polosok / (Vol_source);
						//source[i + 1][j + 1][izstop - 1] += power_1_polosok / (Vol_source);
						//source[i][j + 1][izstop - 1] += power_1_polosok / (Vol_source);

						source[index(i+1, j, izstop - 1, M, N, L)] += power_1_polosok / (Vol_source);
						source[index(i, j, izstop - 1, M, N, L)] += power_1_polosok / (Vol_source);
						source[index(i + 1, j+1, izstop - 1, M, N, L)] += power_1_polosok / (Vol_source);
						source[index(i, j+1, izstop - 1, M, N, L)] += power_1_polosok / (Vol_source);


						Pdiss_sum += power_1_polosok;
					}


				}
			}
			if (n_y == 1) {
				Pdiss_sum += power_1_polosok;
			}
		}
		else {

			for (int i = istart0; i < iend0; i += in_x) {
				//for (int i = M / 2 - (int)(0.5*(in_x*n_x)); i < M / 2 + (int)(0.5 * (in_x * n_x)+(in_x == 0 ? 3 : 1)); i += in_x) { // ПТБШ 20мм 4s 420Mb
					//for (int j = N / 2 - 2; j < N / 2 + 3; j++) {
				for (int j = (n_y == 1 ? N / 2 - 5 : N / 2 - (int)(0.5 * (in_y * (n_y)))); j < (n_y == 1 ? N / 2 + 5 : N / 2 + (int)(0.5 * (in_y * (n_y)) + (in_y == 0 ? 3 : 1))); j += (n_y == 1 ? 1 : in_y)) {
					//if ((i - (M / 2 - 117)) % 26 == 0)
					{


						//source[i][j][izstop] = 0.675 / (5*h1*h2*hz[izstop-1]);
						float Ssource = size_x * size_y;
						float Vol_source = Ssource * hz[izstop - 1];
						if (n_y == 1) {
							//source[i + 1][j][izstop - 1] += power_1_polosok / (Vol_source);
							//source[i][j][izstop - 1] += power_1_polosok / (Vol_source);

							source[index(i + 1, j, izstop - 1, M, N, L)] += power_1_polosok / (Vol_source);
							source[index(i, j, izstop - 1, M, N, L)] += power_1_polosok / (Vol_source);
						}
						else {
							//source[i + 1][j][izstop - 1] += power_1_polosok / (Vol_source);
							//source[i][j][izstop - 1] += power_1_polosok / (Vol_source);
							//source[i + 1][j + 1][izstop - 1] += power_1_polosok / (Vol_source);
							//source[i][j + 1][izstop - 1] += power_1_polosok / (Vol_source);

							source[index(i + 1, j, izstop - 1, M, N, L)] += power_1_polosok / (Vol_source);
							source[index(i, j, izstop - 1, M, N, L)] += power_1_polosok / (Vol_source);
							source[index(i + 1, j + 1, izstop - 1, M, N, L)] += power_1_polosok / (Vol_source);
							source[index(i, j + 1, izstop - 1, M, N, L)] += power_1_polosok / (Vol_source);

							Pdiss_sum += power_1_polosok;
						}


					}
				}
				if (n_y == 1) {
					Pdiss_sum += power_1_polosok;
				}
			}
		}
	}


	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_a, (size) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_a failed!");
		goto Error1;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_b, (size) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_b failed!");
		goto Error1;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_lambda, (L + 2) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_lambda failed!");
		goto Error1;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_lambda_multiplyer_normal, (L + 2) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_lambda_multiplyer_normal failed!");
		goto Error1;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_lambda_multiplyer_plane, (L + 2) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_lambda_multiplyer_plane failed!");
		goto Error1;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_hz, (L + 2) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_hz failed!");
		goto Error1;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_P, ((Nx - 2) * (Nz + 1)) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_P failed!");
		goto Error1;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_Q, ((Nx - 2) * (Nz + 1)) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_Q failed!");
		goto Error1;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, b, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy dev_a, b HostToDevice failed!");
		goto Error1;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_b, source, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy dev_b, source HostToDevice failed!");
		goto Error1;
	}


	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_a1Re, ((L - 1) * (M - 1)) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_a1Re failed!");
		goto Error1;
	}
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_a1Im, ((L - 1) * (M - 1)) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_a1Im failed!");
		goto Error1;
	}

	//IDFTx(b, source, M, N, L);

		/*
		float* a1Re = (float*)malloc((L - 1) * (M - 1) * sizeof(float));
		float* a1Im = (float*)malloc((L - 1) * (M - 1) * sizeof(float));
		IFFTxKerneldebug(b, source,  a1Re, a1Im, M, N, L);

		free(a1Re);
		free(a1Im);

		float bmax1 = -1.0e30;
		for (int i87 = 0; i87 < size; i87++) {
			if (bmax1 < b[i87]) {
				bmax1 = b[i87];
			}
		}
		printf("bmax1=%e\n", bmax1);
		*/

	IFFTxKernel << <128, 128 >> > (dev_a, dev_b, dev_a1Re, dev_a1Im, M, N, L);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "IFFTxKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error1;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching IFFTxKernel!\n", cudaStatus);
		//	goto Error1;
	}


	cudaFree(dev_a1Re);
	cudaFree(dev_a1Im);

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_a2Re, ((L - 1) * (N - 1)) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_a2Re failed!");
		goto Error1;
	}
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_a2Im, ((L - 1) * (N - 1)) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_a2Im failed!");
		goto Error1;
	}


	//copy3D(source, b, M, N, L, true, false, 0.0);

		/*
		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_a, b, size * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy dev_a, b HostToDevice failed!");
			goto Error1;
		}
		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_b, source, size * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy dev_b, source HostToDevice failed!");
			goto Error1;
		}*/
		//printf("size=%d \n", size);
	copyKernel << <128, 128 >> > (dev_b, dev_a, M, N, L, true, false, 0.0f);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "copyKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error1;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching copyKernel!\n", cudaStatus);
		//goto Error1;
	}

	//IDFTy(b, source, M, N, L);

	IFFTyKernel << <128, 128 >> > (dev_a, dev_b, dev_a2Re, dev_a2Im, M, N, L);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "IFFTyKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error1;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching IFFTyKernel!\n", cudaStatus);
		//goto Error1;
	}


	cudaFree(dev_a2Re);
	cudaFree(dev_a2Im);


	/*
	cudaStatus = cudaMemcpy(b, dev_a, size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy b, dev_a DeviceToHost failed!");
		goto Error1;
	}*/


	/* float bmax = -1.0e30;
	for (int i87 = 0; i87 < size; i87++) {
		if (bmax < b[i87]) {
			bmax = b[i87];
		}
	}
	printf("bmax2=%e\n", bmax);
	*/

	//std::cout << "1" << std::endl;

	//free_u3D(source, M, N, L);
	//alloc_u3D(a, M, N, L);

	a = source;
	source = nullptr;

	init_u3D(a, M, N, L, 0.0);

	// Ядро солвера
	//progonka(Nx, Ny, Nz, lambda, lambda_multiplyer_normal, lambda_multiplyer_plane, hz, h1, h2, b, a);

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_lambda, lambda, (L + 2) * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy dev_lambda, lambda HostToDevice failed!");
		goto Error1;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_lambda_multiplyer_normal, lambda_multiplyer_normal, (L + 2) * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy dev_lambda_multiplyer_normal, lambda_multiplyer_normal HostToDevice failed!");
		goto Error1;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_lambda_multiplyer_plane, lambda_multiplyer_plane, (L + 2) * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy dev_lambda_multiplyer_plane, lambda_multiplyer_plane HostToDevice failed!");
		goto Error1;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_hz, hz, (L + 2) * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy dev_hz, hz HostToDevice failed!");
		goto Error1;
	}

	// Ядро солвера.
	progonkaKernel << <128, 128 >> > (Nx, Ny, Nz, dev_lambda, dev_lambda_multiplyer_normal, dev_lambda_multiplyer_plane,
		dev_hz, h1, h2, dev_a, dev_b, dev_P, dev_Q);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "progonkaKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error1;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching progonkaKernel!\n", cudaStatus);
		//goto Error1;
	}

	/*float amax = -1.0e30;
	for (int i87 = 0; i87 < size; i87++) {
		if (amax < fabs(a[i87])) {
			amax = fabs(a[i87]);
		}
	}
	printf("amax=%e\n", amax);*/

	//std::cout << "2" << std::endl;

	//free_u3D(b, M, N, L);
	//alloc_u3D(u, M, N, L);

	u = b;
	b = nullptr;


	//init_u3D(u, M, N, L, 0.0);

	/*// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_a, u, size * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy dev_a, b HostToDevice failed!");
			goto Error1;
		}
		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_b, a, size * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy dev_b, source HostToDevice failed!");
			goto Error1;
		}*/

		// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_a1Re, ((L - 1) * (M - 1)) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_a1Re failed!");
		goto Error1;
	}
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_a1Im, ((L - 1) * (M - 1)) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_a1Im failed!");
		goto Error1;
	}

	//DFTx(u, a, M, N, L);

	FFTxKernel << <128, 128 >> > (dev_a, dev_b, dev_a1Re, dev_a1Im, M, N, L);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "FFTxKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error1;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching FFTxKernel!\n", cudaStatus);
		//goto Error1;
	}


	cudaFree(dev_a1Re);
	cudaFree(dev_a1Im);


	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_a2Re, ((L - 1) * (N - 1)) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_a2Re failed!");
		goto Error1;
	}
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_a2Im, ((L - 1) * (N - 1)) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_a2Im failed!");
		goto Error1;
	}


	//copy3D(a, u, M, N, L, false, false, 0.0);

	copyKernel << <128, 128 >> > (dev_b, dev_a, M, N, L, false, false, 0.0f);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "copyKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error1;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching copyKernel!\n", cudaStatus);
		//goto Error1;
	}

	//DFTy(u, a, M, N, L);

	FFTyKernel << <128, 128 >> > (dev_a, dev_b, dev_a2Re, dev_a2Im, M, N, L);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "FFTyKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error1;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching FFTyKernel!\n", cudaStatus);
		//goto Error1;
	}


	cudaFree(dev_a2Re);
	cudaFree(dev_a2Im);

	//copy3D(u, u, M, N, L, true, true, Tamb);
	addKernel2<<<128, 128>>>(dev_a, Tamb, M, N, L);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel2 launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error1;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel2!\n", cudaStatus);
		//goto Error1;
	}

	cudaStatus = cudaMemcpy(u, dev_a, size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy b, dev_a DeviceToHost failed!");
		goto Error1;
	}

	

	//std::cout << "3" << std::endl;

	free_u3D(a, M, N, L);

	float tmax1 = -1.0e30;

	for (int i = 0; i < M + 2; i++) {
		for (int j = 0; j < N + 2; j++) {
			//if (u[i][j][izstop - 1] > tmax1) 
			if (u[index(i,j,izstop-1,M,N,L)] > tmax1)
			{
				//tmax1 = u[i][j][izstop - 1];
				tmax1 = u[index(i,j,izstop-1,M,N,L)];
			}
		}
	}

	for (int i7 = 0; i7 < ilayer_count; i7++) {
		stack_layer[i7].Tmax = -1.0e30;
	}

	for (int k = 0; k < L + 2; k++) {
		for (int i = 0; i < M + 2; i++) {
			for (int j = 0; j < N + 2; j++) {
				//if (u[i][j][k] > stack_layer[layer_id[k]].Tmax) {
					//stack_layer[layer_id[k]].Tmax = u[i][j][k];
				//}
				if (u[index(i, j, k, M, N, L)] > stack_layer[layer_id[k]].Tmax) {
					stack_layer[layer_id[k]].Tmax = u[index(i, j, k, M, N, L)];
				}
			}
		}
	}

	//printf("ambient temperature %2.1f C\n", Tamb);
	//printf("maximum temperature %2.1f C\n", tmax);
	//printf("thermal resistance %2.2f C/W\n", (tmax - Tamb) / Pdiss_sum);

	thermal_resistance = (tmax1 - Tamb) / Pdiss_sum;

	FILE* fp10;
	errno_t err10;
	// создание файла для записи.
	if ((err10 = fopen_s(&fp10, "report_Table.txt", "w")) != 0) {
		printf("Create File Error\n");
	}
	else {

		// запись имён переменных
		fprintf(fp10, "Delta T in layer, K; Thermal resistance in Layer, K/W - in %%\n");
		for (int i7 = ilayer_count - 2; i7 >= 0; i7--) {

			if (i7 > 0) {
				fprintf(fp10, "%1.1f %1.2f %2.0f%%\n", /*stack_layer[i7].name,*/ stack_layer[i7].Tmax - stack_layer[i7 - 1].Tmax, (stack_layer[i7].Tmax - stack_layer[i7 - 1].Tmax) / Pdiss_sum, 100.0 * ((stack_layer[i7].Tmax - stack_layer[i7 - 1].Tmax) / Pdiss_sum) / thermal_resistance);
			}
			else {
				fprintf(fp10, "%1.1f %1.2f %2.0f%%\n", /*stack_layer[i7].name,*/ stack_layer[i7].Tmax - Tamb, (stack_layer[i7].Tmax - Tamb) / Pdiss_sum, 100.0 * ((stack_layer[i7].Tmax - Tamb) / Pdiss_sum) / thermal_resistance);
			}
		}
		fprintf(fp10, "Result Delta T,K; Result thermal resistance, K/W - 100%%.\n");
		fprintf(fp10, "%1.1f %1.2f\n", (tmax1 - Tamb), (tmax1 - Tamb) / Pdiss_sum);

		fclose(fp10);
	}





	// экспорт 3D полевой величины u в программу tecplot 360.
	//exporttecplot3D(u, xf, yf, zf, M, N, izstop - 1);

	if (export3D) {
		exporttecplot3D_fft(u, xf, yf, zf, M, N, izstop - 1);
	}
	if (exportxy2D) {
		exporttecplot3D22D(u, xf, yf, zf, M, N, L, izstop - 1);
	}
	if (exportx1D) {
		exporttecplot3D21D(u, xf, yf, zf, M, N, L, izstop - 1);
	}


	Error1 :

	free_u3D(u, M, N, L);

	delete[] stack_layer;

	free(xf);
	free(yf);
	free(zf);
	free(hz);
	free(lambda);
	free(lambda_multiplyer_plane);
	free(lambda_multiplyer_normal);
	free(layer_id);

	cudaFree(dev_a);
	cudaFree(dev_b);

	cudaFree(dev_P);
	cudaFree(dev_Q);

	cudaFree(dev_lambda);
	cudaFree(dev_lambda_multiplyer_normal);
	cudaFree(dev_lambda_multiplyer_plane);
	cudaFree(dev_hz);

	//cudaFree(dev_a1Re);
	//cudaFree(dev_a1Im);
	//cudaFree(dev_a2Re);
	//cudaFree(dev_a2Im);

	calculation_end_time = clock();
	calculation_seach_time = calculation_end_time - calculation_start_time;
	int im = 0, is = 0, ims = 0;
	im = (int)(calculation_seach_time / 60000); // минуты
	is = (int)((calculation_seach_time - 60000 * im) / 1000); // секунды
	ims = (int)((calculation_seach_time - 60000 * im - 1000 * is)); // /10 миллисекунды делённые на 10
	//printf(" %1d:%2d:%3d \n", im, is, ims);

	time = (int)(calculation_seach_time);

	//printf("calculation complete...\n");
	//printf("please, press any key to continue...\n");

} // DFTsolver3Dqlinear();



// Начало 18.05.2021 - окончание 31.05.2021.
// Находит статическое тепловое сопротивление для теплопроводностей
// зависящих от температуры.
// Теперь использует видеокарту nvidia GeForce 840M 5.06.2021.
void FFTsolver3Dqnonlinear(float& thermal_resistance,
	float size_x, float size_y, float distance_x, float distance_y,
	float size_gx, int n_x, int n_y, int n_gx,
	bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7, bool b8, bool b9,
	float d1, float d2, float d3,
	float d4, float d5, float d6,
	float d7, float d8, float d9,
	float k1, float k2, float k3,
	float k4, float k5, float k6,
	float k7, float k8, float k9,
	float mplane1, float mplane2, float mplane3,
	float mplane4, float mplane5, float mplane6,
	float mplane7, float mplane8, float mplane9,
	float mortogonal1, float mortogonal2, float mortogonal3,
	float mortogonal4, float mortogonal5, float mortogonal6,
	float mortogonal7, float mortogonal8, float mortogonal9,
	float alpha1, float alpha2, float alpha3,
	float alpha4, float alpha5, float alpha6,
	float alpha7, float alpha8, float alpha9,
	int& time, float Tamb, float Pdiss, bool export3D, bool exportxy2D, bool exportx1D, int idGPU) {

	// Замер времени.
	unsigned int calculation_start_time = 0; // начало счёта мс.
	unsigned int calculation_end_time = 0; // окончание счёта мс.
	unsigned int calculation_seach_time = 0; // время выполнения участка кода в мс.

	calculation_start_time = clock();

	float* u = nullptr, *source = nullptr; // рассчитываемый потенциал и тепловая мощность.

	float* a = nullptr, *b = nullptr; // Коэффициенты разложения в дискретный ряд Фурье.

	float* dev_a = nullptr, * dev_b = nullptr;

	float Pdiss_sum = 0.0;

	float lengthx = 4098.0e-6;// 2050.0e-6;
	float lengthy = 1625.0e-6;

	int M = 4097; // 2049;//511;
	int N = 129;// 511;



	/*lengthx = lengthy = 1.0;
	M = 32;
	N = 159;*/

	//float Tamb = 22.0; // - температура корпуса задаётся пользователем. 

	//Tamb = 0.0;

	// постоянные шаги сетки h1 по оси x и h2 по оси y.
	float h1 = (float)(lengthx / (M + 1));
	float h2 = (float)(lengthy / (N + 1));

	h1 = 0.5 * size_x;
	if (n_y == 1) {
		h2 = 0.1 * size_y;
	}
	else {
		h2 = 0.5 * size_y;
	}

	lengthx = (M + 1) * h1;
	lengthy = (N + 1) * h2;

	float multiplyer = 1.2;

	float n_xgr = n_x * n_gx;
	if (fabs(distance_x) > 1.0e-30) {
		n_xgr += ((n_gx - 1) * (size_gx / distance_x));
	}

	mesh_size(N, M, n_xgr, n_y, lengthx, lengthy, multiplyer, size_x, size_y, distance_x, distance_y, h1, h2);


	//float* xf = new float[M + 2];
	float* xf = (float*)malloc((M+2)*sizeof(float));
	for (int i = 0; i < M + 2; i++) {
		xf[i] = (float)(i * h1);
	}
	//float* yf = new float[N + 2];
	float* yf = (float*)malloc((N + 2) * sizeof(float));
	for (int i = 0; i < N + 2; i++) {
		yf[i] = (float)(i * h2);
	}

	int ilayer_count = 8;
	STACK_LAYERS* stack_layer = nullptr;
	//float chikness_min = 1.0e30;

	//char airname[4] = "air";
	//std::string airname = "air";

	if (b9) {
		ilayer_count = 10;
	}
	else if (b8) {
		ilayer_count = 9;
	}
	else if (b7) {
		ilayer_count = 8;
	}
	else if (b6) {
		ilayer_count = 7;
	}
	else if (b5) {
		ilayer_count = 6;
	}
	else if (b4) {
		ilayer_count = 5;
	}
	else if (b3) {
		ilayer_count = 4;
	}
	else if (b2) {
		ilayer_count = 3;
	}
	else {
		ilayer_count = 2;
	}



	if (b1) {
		//const int idiv_layer = 4;
		stack_layer = new STACK_LAYERS[ilayer_count];
		stack_layer[0].chikness = d1;// 1.6e-3;
		stack_layer[0].lambda = k1; // МД40
		stack_layer[0].idiv_layer = 14;
		stack_layer[0].mplane = mplane1;
		stack_layer[0].mortho = mortogonal1;
		stack_layer[0].alpha = alpha1;
		//stack_layer[0].name = s1;

		if (b2) {
			stack_layer[1].chikness = d2;
			stack_layer[1].lambda = k2; // AuSn
			stack_layer[1].idiv_layer = 14;
			stack_layer[1].mplane = mplane2;
			stack_layer[1].mortho = mortogonal2;
			stack_layer[1].alpha = alpha2;
			//stack_layer[1].name = s2;
			if (b3) {
				stack_layer[2].chikness = d3;
				stack_layer[2].lambda = k3; // Cu
				stack_layer[2].idiv_layer = 14;
				stack_layer[2].mplane = mplane3;
				stack_layer[2].mortho = mortogonal3;
				stack_layer[2].alpha = alpha3;
				//stack_layer[2].name = s3;

				if (b4) {
					stack_layer[3].chikness = d4;
					stack_layer[3].lambda = k4; // AuSn
					stack_layer[3].idiv_layer = 14;
					stack_layer[3].mplane = mplane4;
					stack_layer[3].mortho = mortogonal4;
					stack_layer[3].alpha = alpha4;
					//stack_layer[3].name = s4;

					if (b5) {
						stack_layer[4].chikness = d5;
						stack_layer[4].lambda = k5; // SiC
						stack_layer[4].idiv_layer = 14;
						stack_layer[4].mplane = mplane5;
						stack_layer[4].mortho = mortogonal5;
						stack_layer[4].alpha = alpha5;
						//stack_layer[4].name = s5;

						if (b6) {
							stack_layer[5].chikness = d6;
							stack_layer[5].lambda = k6;// 130.0; // GaN
							stack_layer[5].idiv_layer = 14;
							stack_layer[5].mplane = mplane6;
							stack_layer[5].mortho = mortogonal6;
							stack_layer[5].alpha = alpha6;
							//stack_layer[5].name = s6;

							if (b7) {
								stack_layer[6].chikness = d7;
								stack_layer[6].lambda = k7;// 317.0; // gold
								stack_layer[6].idiv_layer = 14;
								stack_layer[6].mplane = mplane7;
								stack_layer[6].mortho = mortogonal7;
								stack_layer[6].alpha = alpha7;
								//stack_layer[6].name = s7;
								if (b8) {
									stack_layer[7].chikness = d8;
									stack_layer[7].lambda = k8;// 317.0; // gold
									stack_layer[7].idiv_layer = 14;
									stack_layer[7].mplane = mplane8;
									stack_layer[7].mortho = mortogonal8;
									stack_layer[7].alpha = alpha8;
									//stack_layer[7].name = s8;
									if (b9) {
										stack_layer[8].chikness = d9;
										stack_layer[8].lambda = k9;// 317.0; // gold
										stack_layer[8].idiv_layer = 14;
										stack_layer[8].mplane = mplane9;
										stack_layer[8].mortho = mortogonal9;
										stack_layer[8].alpha = alpha9;
										//stack_layer[8].name = s9;
										stack_layer[9].chikness = d9;//0.11 * (d1 + d2 + d3 + d4 + d5 + d6 + d7+d8+d9);
										stack_layer[9].lambda = 0.026;// 1.0;// 0.026;// 0.026; // air
										stack_layer[9].idiv_layer = 14;
										stack_layer[9].mplane = 1.0;
										stack_layer[9].mortho = 1.0;
										stack_layer[9].alpha = 0.0;
										//stack_layer[9].name = airname;
									}
									else {
										stack_layer[8].chikness = d8;//0.11 * (d1 + d2 + d3 + d4 + d5 + d6 + d7+d8+d9);
										stack_layer[8].lambda = 0.026;// 1.0;// 0.026;// 0.026; // air
										stack_layer[8].idiv_layer = 14;
										stack_layer[8].mplane = 1.0;
										stack_layer[8].mortho = 1.0;
										stack_layer[8].alpha = 0.0;
										//stack_layer[8].name = airname;
									}
								}
								else {
									stack_layer[7].chikness = d7;//0.11 * (d1 + d2 + d3 + d4 + d5 + d6 + d7+d8+d9);
									stack_layer[7].lambda = 0.026;// 1.0;// 0.026;// 0.026; // air
									stack_layer[7].idiv_layer = 14;
									stack_layer[7].mplane = 1.0;
									stack_layer[7].mortho = 1.0;
									stack_layer[7].alpha = 0.0;
									//stack_layer[7].name = airname;
								}
							}
							else {
								stack_layer[6].chikness = d6;//0.11 * (d1 + d2 + d3 + d4 + d5 + d6 + d7+d8+d9);
								stack_layer[6].lambda = 0.026;// 1.0;// 0.026;// 0.026; // air
								stack_layer[6].idiv_layer = 14;
								stack_layer[6].mplane = 1.0;
								stack_layer[6].mortho = 1.0;
								stack_layer[6].alpha = 0.0;
								//stack_layer[6].name = airname;
							}
						}
						else {
							stack_layer[5].chikness = d5;//0.11 * (d1 + d2 + d3 + d4 + d5 + d6 + d7+d8+d9);
							stack_layer[5].lambda = 0.026;// 1.0;// 0.026;// 0.026; // air
							stack_layer[5].idiv_layer = 14;
							stack_layer[5].mplane = 1.0;
							stack_layer[5].mortho = 1.0;
							stack_layer[5].alpha = 0.0;
							//stack_layer[5].name = airname;
						}
					}
					else {
						stack_layer[4].chikness = d4;//0.11 * (d1 + d2 + d3 + d4 + d5 + d6 + d7+d8+d9);
						stack_layer[4].lambda = 0.026;// 1.0;// 0.026;// 0.026; // air
						stack_layer[4].idiv_layer = 14;
						stack_layer[4].mplane = 1.0;
						stack_layer[4].mortho = 1.0;
						stack_layer[4].alpha = 0.0;
						//stack_layer[4].name = airname;
					}
				}
				else {
					stack_layer[3].chikness = d3;//0.11 * (d1 + d2 + d3 + d4 + d5 + d6 + d7+d8+d9);
					stack_layer[3].lambda = 0.026;// 1.0;// 0.026;// 0.026; // air
					stack_layer[3].idiv_layer = 14;
					stack_layer[3].mplane = 1.0;
					stack_layer[3].mortho = 1.0;
					stack_layer[3].alpha = 0.0;
					//stack_layer[3].name = airname;
				}
			}
			else {
				stack_layer[2].chikness = d2;//0.11 * (d1 + d2 + d3 + d4 + d5 + d6 + d7+d8+d9);
				stack_layer[2].lambda = 0.026;// 1.0;// 0.026;// 0.026; // air
				stack_layer[2].idiv_layer = 14;
				stack_layer[2].mplane = 1.0;
				stack_layer[2].mortho = 1.0;
				stack_layer[2].alpha = 0.0;
				//stack_layer[2].name = airname;
			}
		}
		else {
			stack_layer[1].chikness = d1;//0.11 * (d1 + d2 + d3 + d4 + d5 + d6 + d7+d8+d9);
			stack_layer[1].lambda = 0.026;// 1.0;// 0.026;// 0.026; // air
			stack_layer[1].idiv_layer = 14;
			stack_layer[1].mplane = 1.0;
			stack_layer[1].mortho = 1.0;
			stack_layer[1].alpha = 0.0;
			//stack_layer[1].name = airname;
		}

	}

	float lengthz = 0.0;
	int inodes = 0;
	//lengthz = 1.0;
	for (int i = 0; i < ilayer_count; i++) {
		lengthz += stack_layer[i].chikness;
		inodes += stack_layer[i].idiv_layer;
	}
	int izstop = inodes - stack_layer[ilayer_count - 1].idiv_layer + 1;
	int L = inodes + 1;


	//int izstop = 63;
	//int L = izstop=63;
	//float hzc = (float)(lengthz / (L + 1));
	//float* hz = new float[L + 2];
	float* hz = (float*)malloc((L + 2) * sizeof(float));
	//float* zf = new float[L + 2];
	float* zf = (float*)malloc((L + 2) * sizeof(float));
	//float* lambda = new float[L + 2];// Теплопроводность.
	float* lambda = (float*)malloc((L + 2) * sizeof(float));
	//float* lambda_multiplyer_plane = new float[L + 2];// Теплопроводность.
	float* lambda_multiplyer_plane = (float*)malloc((L + 2) * sizeof(float));
	//float* lambda_multiplyer_normal = new float[L + 2];// Теплопроводность.
	float* lambda_multiplyer_normal = (float*)malloc((L + 2) * sizeof(float));
	//int* layer_id = new int[L + 2];
	int* layer_id = (int*)malloc((L + 2) * sizeof(int));
	//float* temp_maxz = new float[L + 2];
	float* temp_maxz = (float*)malloc((L + 2) * sizeof(float));

	float* dev_lambda, * dev_lambda_multiplyer_normal, * dev_lambda_multiplyer_plane, * dev_hz;

	/*for (int i = 0; i < L + 2; i++) {
		zf[i] = (float)(i * hzc);
		hz[i] = hzc;
		if (i > izstop) {
			lambda[i] = 1.0; // air
		}
		else {
			lambda[i] = 1.0; // SiC
		}
	}
	*/
	// 19.05.2021
	zf[0] = 0.0;
	float q = 1.2;
	float qinv = 1 / q;
	hz[0] = stack_layer[0].chikness * (qinv - 1.0) / (pow(qinv, 1.0*stack_layer[0].idiv_layer) - 1.0);
	// Постоянный шаг по оси z в каждом интервале.
	//hz[0] = stack_layer[0].chikness / stack_layer[0].idiv_layer;
	float qsum = 0.0;
	int ic = 1;
	layer_id[0] = 0;
	temp_maxz[0] = Tamb;
	lambda[0] = stack_layer[0].lambda;
	lambda_multiplyer_plane[0] = stack_layer[0].mplane;
	lambda_multiplyer_normal[0] = stack_layer[0].mortho;
	for (int i = 0; i < ilayer_count; i++) {
		for (int j = 0; j < stack_layer[i].idiv_layer; j++) {
			if (i < ilayer_count - 1) {

				hz[ic] = stack_layer[i].chikness * (qinv - 1.0) / (pow(qinv, 1.0*stack_layer[i].idiv_layer) - 1.0);
				for (int i7 = 1; i7 <= j; i7++) hz[ic] *= qinv;
			}
			else {
				hz[ic] = stack_layer[i].chikness * (qinv - 1.0) / (pow(qinv, 1.0*stack_layer[i].idiv_layer) - 1.0);
				for (int i7 = 0; i7 <= stack_layer[i].idiv_layer - 1 - j; i7++) hz[ic] *= qinv;
				qsum += hz[ic];
			}
			// Постоянный шаг по оси z в каждом интервале.
			//hz[ic] = stack_layer[i].chikness / stack_layer[i].idiv_layer;
			layer_id[ic] = i;
			lambda[ic] = stack_layer[i].lambda;
			temp_maxz[ic] = Tamb;
			lambda_multiplyer_plane[ic] = stack_layer[i].mplane;
			lambda_multiplyer_normal[ic] = stack_layer[i].mortho;
			zf[ic] = zf[ic-1] + hz[ic];
			ic++;
		}
	}
	//zf[1 + ic] = zf[ic] + stack_layer[ilayer_count - 1].chikness / stack_layer[ilayer_count - 1].idiv_layer;
	//hz[ic] = stack_layer[ilayer_count - 1].chikness / stack_layer[ilayer_count - 1].idiv_layer;

	//hz[ic + 1] = stack_layer[ilayer_count - 1].chikness / stack_layer[ilayer_count - 1].idiv_layer;


	hz[ic] = hz[ic + 1] = (stack_layer[ilayer_count - 1].chikness - qsum) / 2.0;
	zf[ic] = zf[ic-1] + hz[ic];


	zf[1 + ic] = 0.0;
	for (int i7 = 0; i7 < ilayer_count; i7++) {
		zf[1 + ic] += stack_layer[i7].chikness;
	}
	layer_id[ic] = ilayer_count - 1;
	layer_id[ic + 1] = ilayer_count - 1;
	lambda[ic] = stack_layer[ilayer_count - 1].lambda;
	lambda[ic + 1] = stack_layer[ilayer_count - 1].lambda;
	temp_maxz[ic] = Tamb;
	temp_maxz[ic + 1] = Tamb;
	lambda_multiplyer_plane[ic] = stack_layer[ilayer_count - 1].mplane;
	lambda_multiplyer_normal[ic] = stack_layer[ilayer_count - 1].mortho;
	lambda_multiplyer_plane[ic + 1] = stack_layer[ilayer_count - 1].mplane;
	lambda_multiplyer_normal[ic + 1] = stack_layer[ilayer_count - 1].mortho;


	int Nx = M + 1; // < M+2
	int Ny = N + 1; // < N+2
	int Nz = L + 1;

	for (int k = 1; k <= Nz; k++) {

		//printf("%d %d lam=%e %e\n", k, izstop - 1, lambda[k], hz[k]);
	}
	//getchar();

	float tmax1 = -1.0e30;

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(idGPU);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error1;
	}

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("%s\n",prop.name);
	// GeForce 840M 384потока 1ГГц каждый март 2014 года. 28нм.

	

	float* dev_a1Re=nullptr, * dev_a1Im = nullptr;
	float* dev_a2Re = nullptr, * dev_a2Im = nullptr;

	float* dev_P, * dev_Q;

	int size = (M + 2) * (N + 2) * (L + 2);

	for (int inl_pass = 0; inl_pass < 3; inl_pass++) {

		Pdiss_sum = 0.0;

		if (inl_pass == 0) {
			alloc_u3D(source, M, N, L);
			alloc_u3D(b, M, N, L);

			
			// Allocate GPU buffers for three vectors (two input, one output)    .
			cudaStatus = cudaMalloc((void**)&dev_a, (size) * sizeof(float));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc dev_a failed!");
				goto Error1;
			}

			// Allocate GPU buffers for three vectors (two input, one output)    .
			cudaStatus = cudaMalloc((void**)&dev_b, (size) * sizeof(float));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc dev_b failed!");
				goto Error1;
			}
			
			// Allocate GPU buffers for three vectors (two input, one output)    .
			cudaStatus = cudaMalloc((void**)&dev_lambda, (L+2) * sizeof(float));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc dev_lambda failed!");
				goto Error1;
			}

			// Allocate GPU buffers for three vectors (two input, one output)    .
			cudaStatus = cudaMalloc((void**)&dev_lambda_multiplyer_normal, (L+2) * sizeof(float));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc dev_lambda_multiplyer_normal failed!");
				goto Error1;
			}

			// Allocate GPU buffers for three vectors (two input, one output)    .
			cudaStatus = cudaMalloc((void**)&dev_lambda_multiplyer_plane, (L+2) * sizeof(float));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc dev_lambda_multiplyer_plane failed!");
				goto Error1;
			}

			// Allocate GPU buffers for three vectors (two input, one output)    .
			cudaStatus = cudaMalloc((void**)&dev_hz, (L+2) * sizeof(float));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc dev_hz failed!");
				goto Error1;
			}		

			// Allocate GPU buffers for three vectors (two input, one output)    .
			cudaStatus = cudaMalloc((void**)&dev_P, ((Nx-2)*(Nz+1)) * sizeof(float));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc dev_P failed!");
				goto Error1;
			}

			// Allocate GPU buffers for three vectors (two input, one output)    .
			cudaStatus = cudaMalloc((void**)&dev_Q, ((Nx - 2) * (Nz + 1)) * sizeof(float));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc dev_Q failed!");
				goto Error1;
			}
		}
		else {
			source = u;
			u = nullptr;
			b = a;
			a = nullptr;
		}

		// инициализация.
		init_u3D(b, M, N, L, 0.0);
		init_u3D(source, M, N, L, 0.0);

		const float power_1_polosok = Pdiss / (n_y * (n_x + 1) * n_gx);//0.675;

		int in_x = (int)(distance_x / h1);
		int in_y = (int)(distance_y / h2);

		// Задаём тепловую мощность тепловыделения.
		//for (int i = M / 2 - 117; i < M / 2 + 124; i += 26) {// ПТБШ 1,25мм 2s 204Mb
		//for (int i = M / 2 - 234; i < M / 2 + 240; i += 26) {// ПТБШ 2,5мм 2s 204Mb
		//for (int i = M / 2 - 468; i < M / 2 + 480; i += 26) { // ПТБШ 5мм 2s 204Mb
			//for (int i = M / 2 - 936; i < M / 2 + 946; i += 26) { // ПТБШ 10мм 4s 420Mb
		int istart0 = M / 2 - (int)(0.5 * (in_x * n_xgr));
		int iend0 = istart0 + ((int)((in_x * (n_x)))) + (in_x == 0 ? 3 : 1);

		for (int igr = 0; igr < n_gx; igr++)
		{

			if (igr > 0) {
				istart0 += (int)(size_gx / h1) + ((int)((in_x * (n_x)))) + (in_x == 0 ? 3 : 1);
				iend0 += (int)(size_gx / h1) + ((int)((in_x * (n_x)))) + (in_x == 0 ? 3 : 1);
			}

			if (n_x == 0) {
				// Один источник по оси oX
				int i = M / 2;
				for (int j = (n_y == 1 ? N / 2 - 5 : N / 2 - (int)(0.5 * (in_y * (n_y)))); j < (n_y == 1 ? N / 2 + 5 : N / 2 + (int)(0.5 * (in_y * (n_y)) + (in_y == 0 ? 3 : 1))); j += (n_y == 1 ? 1 : in_y)) {
					//if ((i - (M / 2 - 117)) % 26 == 0)
					{


						//source[i][j][izstop] = 0.675 / (5*h1*h2*hz[izstop-1]);
						float Ssource = size_x * size_y;
						float Vol_source = Ssource * hz[izstop - 1];
						if (n_y == 1) {
							//source[i + 1][j][izstop - 1] += power_1_polosok / (Vol_source);
							//source[i][j][izstop - 1] += power_1_polosok / (Vol_source);

							source[index(i + 1, j, izstop - 1, M, N, L)] += power_1_polosok / (Vol_source);
							source[index(i, j, izstop - 1, M, N, L)] += power_1_polosok / (Vol_source);
						}
						else {
							//source[i + 1][j][izstop - 1] += power_1_polosok / (Vol_source);
							//source[i][j][izstop - 1] += power_1_polosok / (Vol_source);
							//source[i + 1][j + 1][izstop - 1] += power_1_polosok / (Vol_source);
							//source[i][j + 1][izstop - 1] += power_1_polosok / (Vol_source);

							source[index(i + 1, j, izstop - 1, M, N, L)] += power_1_polosok / (Vol_source);
							source[index(i, j, izstop - 1, M, N, L)] += power_1_polosok / (Vol_source);
							source[index(i + 1, j + 1, izstop - 1, M, N, L)] += power_1_polosok / (Vol_source);
							source[index(i, j + 1, izstop - 1, M, N, L)] += power_1_polosok / (Vol_source);


							Pdiss_sum += power_1_polosok;
						}


					}
				}
				if (n_y == 1) {
					Pdiss_sum += power_1_polosok;
				}
			}
			else {

				for (int i = istart0; i < iend0; i += in_x) {
					//for (int i = M / 2 - (int)(0.5*(in_x*n_x)); i < M / 2 + (int)(0.5 * (in_x * n_x)+(in_x == 0 ? 3 : 1)); i += in_x) { // ПТБШ 20мм 4s 420Mb
						//for (int j = N / 2 - 2; j < N / 2 + 3; j++) {
					for (int j = (n_y == 1 ? N / 2 - 5 : N / 2 - (int)(0.5 * (in_y * (n_y)))); j < (n_y == 1 ? N / 2 + 5 : N / 2 + (int)(0.5 * (in_y * (n_y)) + (in_y == 0 ? 3 : 1))); j += (n_y == 1 ? 1 : in_y)) {
						//if ((i - (M / 2 - 117)) % 26 == 0)
						{


							//source[i][j][izstop] = 0.675 / (5*h1*h2*hz[izstop-1]);
							float Ssource = size_x * size_y;
							float Vol_source = Ssource * hz[izstop - 1];
							if (n_y == 1) {
								//source[i + 1][j][izstop - 1] += power_1_polosok / (Vol_source);
								//source[i][j][izstop - 1] += power_1_polosok / (Vol_source);

								source[index(i + 1, j, izstop - 1, M, N, L)] += power_1_polosok / (Vol_source);
								source[index(i, j, izstop - 1, M, N, L)] += power_1_polosok / (Vol_source);
							}
							else {
								//source[i + 1][j][izstop - 1] += power_1_polosok / (Vol_source);
								//source[i][j][izstop - 1] += power_1_polosok / (Vol_source);
								//source[i + 1][j + 1][izstop - 1] += power_1_polosok / (Vol_source);
								//source[i][j + 1][izstop - 1] += power_1_polosok / (Vol_source);

								source[index(i + 1, j, izstop - 1, M, N, L)] += power_1_polosok / (Vol_source);
								source[index(i, j, izstop - 1, M, N, L)] += power_1_polosok / (Vol_source);
								source[index(i + 1, j + 1, izstop - 1, M, N, L)] += power_1_polosok / (Vol_source);
								source[index(i, j + 1, izstop - 1, M, N, L)] += power_1_polosok / (Vol_source);

								Pdiss_sum += power_1_polosok;
							}


						}
					}
					if (n_y == 1) {
						Pdiss_sum += power_1_polosok;
					}
				}
			}
		}

		//printf("Pdiss=%e\n", Pdiss_sum);

		
		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_a, b, size * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy dev_a, b HostToDevice failed!");
			goto Error1;
		}
		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_b, source, size * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy dev_b, source HostToDevice failed!");
			goto Error1;
		}
		

		// Allocate GPU buffers for three vectors (two input, one output)    .
		cudaStatus = cudaMalloc((void**)&dev_a1Re, ((L -1) * (M -1)) * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc dev_a1Re failed!");
			goto Error1;
		}
		// Allocate GPU buffers for three vectors (two input, one output)    .
		cudaStatus = cudaMalloc((void**)&dev_a1Im, ((L -1) * (M - 1)) * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc dev_a1Im failed!");
			goto Error1;
		}
		

		//IDFTx(b, source, M, N, L);


		/*
		float* a1Re = (float*)malloc((L - 1) * (M - 1) * sizeof(float));
		float* a1Im = (float*)malloc((L - 1) * (M - 1) * sizeof(float));
		IFFTxKerneldebug(b, source,  a1Re, a1Im, M, N, L);
		
		free(a1Re);
		free(a1Im);
		
		float bmax1 = -1.0e30;
		for (int i87 = 0; i87 < size; i87++) {
			if (bmax1 < b[i87]) {
				bmax1 = b[i87];
			}
		}
		printf("bmax1=%e\n", bmax1);
		*/

		//printf("ok\n");
		//getchar();
		
		IFFTxKernel<<<128,128>>>(dev_a, dev_b, dev_a1Re, dev_a1Im, M, N, L);
		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "IFFTxKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error1;
		}
		
		// cudaDeviceSynchronize waits for the kernel to finish, and returns
	    // any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching IFFTxKernel!\n", cudaStatus);
		//	goto Error1;
		}


		cudaFree(dev_a1Re);
		cudaFree(dev_a1Im);

		// Allocate GPU buffers for three vectors (two input, one output)    .
		cudaStatus = cudaMalloc((void**)&dev_a2Re, ((L -1) * (N -1)) * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc dev_a2Re failed!");
			goto Error1;
		}
		// Allocate GPU buffers for three vectors (two input, one output)    .
		cudaStatus = cudaMalloc((void**)&dev_a2Im, ((L -1) * (N -1)) * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc dev_a2Im failed!");
			goto Error1;
		}
		
		//copy3D(source, b, M, N, L, true, false, 0.0);

		/*
		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_a, b, size * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy dev_a, b HostToDevice failed!");
			goto Error1;
		}
		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_b, source, size * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy dev_b, source HostToDevice failed!");
			goto Error1;
		}*/
		//printf("size=%d \n", size);
		copyKernel<<<128,128>>>(dev_b, dev_a, M, N, L, true, false, 0.0f);
		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "copyKernel post IFFTxKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error1;
		}
		
		// cudaDeviceSynchronize waits for the kernel to finish, and returns
	    // any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching copyKernel!\n", cudaStatus);
			//goto Error1;
		}
	

		//IDFTy(b, source, M, N, L);

		
		
		IFFTyKernel<<<128,128>>>(dev_a, dev_b, dev_a2Re, dev_a2Im, M, N, L);
		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "IFFTyKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error1;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
	    // any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching IFFTyKernel!\n", cudaStatus);
			//goto Error1;
		}
		

		cudaFree(dev_a2Re);
		cudaFree(dev_a2Im);

		
		/*
		cudaStatus = cudaMemcpy(b, dev_a, size * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy b, dev_a DeviceToHost failed!");
			goto Error1;
		}*/
		

		/* float bmax = -1.0e30;
		for (int i87 = 0; i87 < size; i87++) {
			if (bmax < b[i87]) {
				bmax = b[i87];
			}
		}
		printf("bmax2=%e\n", bmax);
		*/

		//std::cout << "1" << std::endl;

		//free_u3D(source, M, N, L);
		//alloc_u3D(a, M, N, L);

		a = source;
		source = nullptr;

		//init_u3D(a, M, N, L, 0.0);

		// Ядро солвера.
		//progonka(Nx, Ny, Nz, lambda, lambda_multiplyer_normal, lambda_multiplyer_plane, hz, h1, h2, b, a);

		

		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_lambda, lambda, (L+2) * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy dev_lambda, lambda HostToDevice failed!");
			goto Error1;
		}

		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_lambda_multiplyer_normal, lambda_multiplyer_normal, (L+2) * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy dev_lambda_multiplyer_normal, lambda_multiplyer_normal HostToDevice failed!");
			goto Error1;
		}

		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_lambda_multiplyer_plane, lambda_multiplyer_plane, (L+2) * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy dev_lambda_multiplyer_plane, lambda_multiplyer_plane HostToDevice failed!");
			goto Error1;
		}

		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_hz, hz, (L+2) * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy dev_hz, hz HostToDevice failed!");
			goto Error1;
		}

		// Ядро солвера.
		progonkaKernel<<<128,128>>>(Nx, Ny, Nz, dev_lambda, dev_lambda_multiplyer_normal, dev_lambda_multiplyer_plane,
			dev_hz, h1, h2, dev_a, dev_b, dev_P, dev_Q);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "progonkaKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error1;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching progonkaKernel!\n", cudaStatus);
			//goto Error1;
		}

		/*float amax = -1.0e30;
		for (int i87 = 0; i87 < size; i87++) {
			if (amax < fabs(a[i87])) {
				amax = fabs(a[i87]);
			}
		}
		printf("amax=%e\n", amax);*/

		//std::cout << "2" << std::endl;

		//free_u3D(b, M, N, L);
		//alloc_u3D(u, M, N, L);

		u = b;
		b = nullptr;

		//init_u3D(u, M, N, L, 0.0);

		/*// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_a, u, size * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy dev_a, b HostToDevice failed!");
			goto Error1;
		}
		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_b, a, size * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy dev_b, source HostToDevice failed!");
			goto Error1;
		}*/

		// Allocate GPU buffers for three vectors (two input, one output)    .
		cudaStatus = cudaMalloc((void**)&dev_a1Re, ((L - 1) * (M - 1)) * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc dev_a1Re failed!");
			goto Error1;
		}
		// Allocate GPU buffers for three vectors (two input, one output)    .
		cudaStatus = cudaMalloc((void**)&dev_a1Im, ((L - 1) * (M - 1)) * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc dev_a1Im failed!");
			goto Error1;
		}

		//DFTx(u, a, M, N, L);
		FFTxKernel<<<128,128>>>(dev_a, dev_b, dev_a1Re, dev_a1Im, M, N, L);
		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "FFTxKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error1;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching FFTxKernel!\n", cudaStatus);
			//goto Error1;
		}


		cudaFree(dev_a1Re);
		cudaFree(dev_a1Im);


		// Allocate GPU buffers for three vectors (two input, one output)    .
		cudaStatus = cudaMalloc((void**)&dev_a2Re, ((L - 1) * (N - 1)) * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc dev_a2Re failed!");
			goto Error1;
		}
		// Allocate GPU buffers for three vectors (two input, one output)    .
		cudaStatus = cudaMalloc((void**)&dev_a2Im, ((L - 1) * (N - 1)) * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc dev_a2Im failed!");
			goto Error1;
		}

		//copy3D(a, u, M, N, L, false, false, 0.0);
		copyKernel<<<128,128>>>(dev_b, dev_a, M, N, L, false, false, 0.0f);
		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "copyKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error1;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching copyKernel!\n", cudaStatus);
			//goto Error1;
		}

		//DFTy(u, a, M, N, L);
		FFTyKernel<<<128,128>>>(dev_a, dev_b, dev_a2Re, dev_a2Im, M, N, L);
		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "FFTyKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error1;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching FFTyKernel!\n", cudaStatus);
			//goto Error1;
		}


		cudaFree(dev_a2Re);
		cudaFree(dev_a2Im);

		//copy3D(u, u, M, N, L, true, true, Tamb);
		addKernel2<<<128,128>>>(dev_a, Tamb, M, N, L);
		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel2 launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error1;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel2!\n", cudaStatus);
			//goto Error1;
		}

		cudaStatus = cudaMemcpy(u, dev_a, size * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy b, dev_a DeviceToHost failed!");
			goto Error1;
		}

		//std::cout << "3" << std::endl;

		//free_u3D(a, M, N, L);

		/*tmax1 = -1.0e30;

		for (int i = 0; i < M + 2; i++) {
			for (int j = 0; j < N + 2; j++) {
				if (u[i][j][izstop - 1] > tmax1) {
					tmax1 = u[i][j][izstop - 1];
				}
			}
		}*/

#pragma omp parallel for
		for (int i7 = 0; i7 < ilayer_count; i7++) {
			stack_layer[i7].Tmax = -1.0e30;
		}

#pragma omp parallel for
		for (int k = 0; k < L + 2; k++) {
			temp_maxz[k] = -1.0e30;
		}

		for (int k = 0; k < L + 2; k++) {
			for (int i = 0; i < M + 2; i++) {
				for (int j = 0; j < N + 2; j++) {
					//if (u[i][j][k] > stack_layer[layer_id[k]].Tmax) {
						//stack_layer[layer_id[k]].Tmax = u[i][j][k];
					//}
					if (u[index(i, j, k, M, N, L)] > stack_layer[layer_id[k]].Tmax) {
						stack_layer[layer_id[k]].Tmax = u[index(i, j, k, M, N, L)];
					}
				}
			}
		}

#pragma omp parallel for
		for (int k = 0; k < L + 2; k++) {
			for (int i = 0; i < M + 2; i++) {
				for (int j = 0; j < N + 2; j++) {
					//if (u[i][j][k] > temp_maxz[k]) 
					if (u[index(i,j,k,M,N,L)] > temp_maxz[k])
					{
						//temp_maxz[k] = u[i][j][k];
						temp_maxz[k] = u[index(i, j, k, M, N, L)];
					}
				}
			}
		}

		tmax1 = temp_maxz[izstop - 1];
		printf("%e\n", tmax1);
		//getchar();

#pragma omp parallel for
		for (int k = 0; k < L + 2; k++) {
			lambda[k] = stack_layer[layer_id[k]].lambda * pow(((273.15 + temp_maxz[k]) / 300.0), stack_layer[layer_id[k]].alpha);
		}

	}

	free_u3D(a, M, N, L);

	//printf("ambient temperature %2.1f C\n", Tamb);
	//printf("maximum temperature %2.1f C\n", tmax);
	//printf("thermal resistance %2.2f C/W\n", (tmax - Tamb) / Pdiss_sum);

	thermal_resistance = (tmax1 - Tamb) / Pdiss_sum;

	FILE* fp10;
	errno_t err10;
	// создание файла для записи.
	if ((err10 = fopen_s(&fp10, "report_Table.txt", "w")) != 0) {
		printf("Create File Error\n");
	}
	else {

		// запись имён переменных
		fprintf(fp10, "Pdiss=%e\n", Pdiss_sum);
		fprintf(fp10, "Delta T in layer, K; Thermal resistance in Layer, K/W - in %%\n");
		for (int i7 = ilayer_count - 2; i7 >= 0; i7--) {

			if (i7 > 0) {
				fprintf(fp10, "%1.1f %1.2f %2.0f%%\n", /*stack_layer[i7].name,*/ stack_layer[i7].Tmax - stack_layer[i7 - 1].Tmax, (stack_layer[i7].Tmax - stack_layer[i7 - 1].Tmax) / Pdiss_sum, 100.0 * ((stack_layer[i7].Tmax - stack_layer[i7 - 1].Tmax) / Pdiss_sum) / thermal_resistance);
			}
			else {
				fprintf(fp10, "%1.1f %1.2f %2.0f%%\n", /*stack_layer[i7].name,*/ stack_layer[i7].Tmax - Tamb, (stack_layer[i7].Tmax - Tamb) / Pdiss_sum, 100.0 * ((stack_layer[i7].Tmax - Tamb) / Pdiss_sum) / thermal_resistance);
			}
		}
		fprintf(fp10, "Result Delta T,K; Result thermal resistance, K/W - 100%%.\n");
		fprintf(fp10, "%1.1f %1.2f\n", (tmax1 - Tamb), (tmax1 - Tamb) / Pdiss_sum);

		fclose(fp10);
	}


	// экспорт 3D полевой величины u в программу tecplot 360.
	//exporttecplot3D(u, xf, yf, zf, M, N, izstop - 1);

	if (export3D) {
		exporttecplot3D_fft(u, xf, yf, zf, M, N, izstop - 1);
	}
	if (exportxy2D) {
		exporttecplot3D22D(u, xf, yf, zf, M, N, L, izstop - 1);
	}
	if (exportx1D) {
		exporttecplot3D21D(u, xf, yf, zf, M, N, L, izstop - 1);
	}


	free_u3D(u, M, N, L);


	Error1 : 

	delete[] stack_layer;

	free(xf);
	free(yf);
	free(zf);
	free(hz);
	free(lambda);
	free(lambda_multiplyer_plane);
	free(lambda_multiplyer_normal);
	free(layer_id);
	free(temp_maxz);

	cudaFree(dev_a);
	cudaFree(dev_b);

	cudaFree(dev_P);
	cudaFree(dev_Q);

	cudaFree(dev_lambda);
	cudaFree(dev_lambda_multiplyer_normal);
	cudaFree(dev_lambda_multiplyer_plane);
	cudaFree(dev_hz);

	//cudaFree(dev_a1Re);
	//cudaFree(dev_a1Im);
	//cudaFree(dev_a2Re);
	//cudaFree(dev_a2Im);

	calculation_end_time = clock();
	calculation_seach_time = calculation_end_time - calculation_start_time;
	int im = 0, is = 0, ims = 0;
	im = (int)(calculation_seach_time / 60000); // минуты
	is = (int)((calculation_seach_time - 60000 * im) / 1000); // секунды
	ims = (int)((calculation_seach_time - 60000 * im - 1000 * is)); // /10 миллисекунды делённые на 10
	//printf(" %1d:%2d:%3d \n", im, is, ims);

	time = (int)(calculation_seach_time);

	//printf("calculation complete...\n");
	//printf("please, press any key to continue...\n");

} // DFTsolver3Dqnonlinear();

// Выделяет и освобождает общую оперативную память и вызывает решатели 
// FFTsolver3Dqlinear - для постоянной теплопроводности,
// FFTsolver3Dqnonlinear - для теплопроводности зависящей от температуры.
void fourier_solve(
	float& thermal_resistance, float size_x, float size_y, float distance_x, float distance_y,
	float size_gx, int n_x, int n_y, int n_gx,
	bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7, bool b8, bool b9,
	float d1, float d2, float d3, float d4, float d5, float d6, float d7, float d8, float d9,
	float k1, float k2, float k3, float k4, float k5, float k6, float k7, float k8, float k9,
	float rhoCp1, float rhoCp2, float rhoCp3, float rhoCp4, float rhoCp5, float rhoCp6,
	float rhoCp7, float rhoCp8, float rhoCp9,
	float mplane1, float mplane2, float mplane3,
	float mplane4, float mplane5, float mplane6,
	float mplane7, float mplane8, float mplane9,
	float mortogonal1, float mortogonal2, float mortogonal3,
	float mortogonal4, float mortogonal5, float mortogonal6,
	float mortogonal7, float mortogonal8, float mortogonal9,
	float alpha1, float alpha2, float alpha3,
	float alpha4, float alpha5, float alpha6,
	float alpha7, float alpha8, float alpha9,
	int& time, float Tamb, float Pdiss,
	bool export3D, bool exportxy2D, bool exportx1D,
	bool bfloat, int idGPU)
{

	float eps = -1.0e-10;

	if (bfloat) {
		// Тип float.

		/*a1_gl = new std::complex<float> *[NUMBER_THREADS];
		wlen_pw_gl = new std::complex<float> *[NUMBER_THREADS];
		for (int i = 0; i < NUMBER_THREADS; i++)
		{
			a1_gl[i] = new std::complex<float>[MAXN];
			wlen_pw_gl[i] = new std::complex<float>[MAXN];
		}*/

		float thermal_resistancef = (float)(thermal_resistance);

		if (((alpha1 > eps) || (!b1)) && ((alpha2 > eps) || (!b2)) && ((alpha3 > eps) || (!b3))
			&& ((alpha4 > eps) || (!b4)) && ((alpha5 > eps) || (!b5)) && ((alpha6 > eps) || (!b6)) &&
			((alpha7 > eps) || (!b7)) && ((alpha8 > eps) || (!b8)) && ((alpha9 > eps) || (!b9)))
		{

			//thermal_resistance = 1.0;
			FFTsolver3Dqlinear(thermal_resistancef,
				size_x, size_y, distance_x, distance_y,
				size_gx, n_x - 1, n_y, n_gx,
				b1, b2, b3, b4, b5, b6, b7, b8, b9,
				d1, d2, d3, d4, d5, d6, d7, d8, d9,
				k1, k2, k3, k4, k5, k6, k7, k8, k9,
				mplane1, mplane2, mplane3,
				mplane4, mplane5, mplane6,
				mplane7, mplane8, mplane9,
				mortogonal1, mortogonal2, mortogonal3,
				mortogonal4, mortogonal5, mortogonal6,
				mortogonal7, mortogonal8, mortogonal9,
				//s1, s2, s3, s4, s5, s6, s7, s8, s9,
				time, Tamb, Pdiss,
				export3D, exportxy2D, exportx1D, idGPU);
		}
		else {
			//thermal_resistance = 1.0;
			FFTsolver3Dqnonlinear(thermal_resistancef,
				size_x, size_y, distance_x, distance_y,
				size_gx, n_x - 1, n_y, n_gx,
				b1, b2, b3, b4, b5, b6, b7, b8, b9,
				d1, d2, d3, d4, d5, d6, d7, d8, d9,
				k1, k2, k3, k4, k5, k6, k7, k8, k9,
				mplane1, mplane2, mplane3,
				mplane4, mplane5, mplane6,
				mplane7, mplane8, mplane9,
				mortogonal1, mortogonal2, mortogonal3,
				mortogonal4, mortogonal5, mortogonal6,
				mortogonal7, mortogonal8, mortogonal9,
				alpha1, alpha2, alpha3,
				alpha4, alpha5, alpha6,
				alpha7, alpha8, alpha9,
				//s1, s2, s3, s4, s5, s6, s7, s8, s9,
				time, Tamb, Pdiss,
				export3D, exportxy2D, exportx1D, idGPU);
		}

		thermal_resistance = (float)(thermal_resistancef);

		/*for (int i = 0; i < NUMBER_THREADS; i++)
		{
			delete[] a1_gl[i];
			delete[] wlen_pw_gl[i];
		}
		delete[] a1_gl;
		delete[] wlen_pw_gl;
		a1_gl = nullptr;
		wlen_pw_gl = nullptr;
		*/

	}
	else {
		// Тип float.

		/*a1_gl = new std::complex<float> *[NUMBER_THREADS];
		wlen_pw_gl = new std::complex<float> *[NUMBER_THREADS];
		for (int i = 0; i < NUMBER_THREADS; i++)
		{
			a1_gl[i] = new std::complex<float>[MAXN];
			wlen_pw_gl[i] = new std::complex<float>[MAXN];
		}*/

		float thermal_resistancef = (float)(thermal_resistance);

		if (((alpha1 > eps) || (!b1)) && ((alpha2 > eps) || (!b2)) && ((alpha3 > eps) || (!b3))
			&& ((alpha4 > eps) || (!b4)) && ((alpha5 > eps) || (!b5)) && ((alpha6 > eps) || (!b6)) &&
			((alpha7 > eps) || (!b7)) && ((alpha8 > eps) || (!b8)) && ((alpha9 > eps) || (!b9)))
		{

			//thermal_resistance = 1.0;
			FFTsolver3Dqlinear(thermal_resistancef,
				size_x, size_y, distance_x, distance_y,
				size_gx, n_x - 1, n_y, n_gx,
				b1, b2, b3, b4, b5, b6, b7, b8, b9,
				d1, d2, d3, d4, d5, d6, d7, d8, d9,
				k1, k2, k3, k4, k5, k6, k7, k8, k9,
				mplane1, mplane2, mplane3,
				mplane4, mplane5, mplane6,
				mplane7, mplane8, mplane9,
				mortogonal1, mortogonal2, mortogonal3,
				mortogonal4, mortogonal5, mortogonal6,
				mortogonal7, mortogonal8, mortogonal9,
				//s1, s2, s3, s4, s5, s6, s7, s8, s9,
				time, Tamb, Pdiss,
				export3D, exportxy2D, exportx1D, idGPU);
		}
		else {
			//thermal_resistance = 1.0;
			FFTsolver3Dqnonlinear(thermal_resistancef,
				size_x, size_y, distance_x, distance_y,
				size_gx, n_x - 1, n_y, n_gx,
				b1, b2, b3, b4, b5, b6, b7, b8, b9,
				d1, d2, d3, d4, d5, d6, d7, d8, d9,
				k1, k2, k3, k4, k5, k6, k7, k8, k9,
				mplane1, mplane2, mplane3,
				mplane4, mplane5, mplane6,
				mplane7, mplane8, mplane9,
				mortogonal1, mortogonal2, mortogonal3,
				mortogonal4, mortogonal5, mortogonal6,
				mortogonal7, mortogonal8, mortogonal9,
				alpha1, alpha2, alpha3,
				alpha4, alpha5, alpha6,
				alpha7, alpha8, alpha9,
				//s1, s2, s3, s4, s5, s6, s7, s8, s9,
				time, Tamb, Pdiss,
				export3D, exportxy2D, exportx1D, idGPU);
		}

		thermal_resistance = (float)(thermal_resistancef);

		/*for (int i = 0; i < NUMBER_THREADS; i++)
		{
			delete[] a1_gl[i];
			delete[] wlen_pw_gl[i];
		}
		delete[] a1_gl;
		delete[] wlen_pw_gl;
		a1_gl = nullptr;
		wlen_pw_gl = nullptr;*/
	}
}


int main()
{

	float thermal_resistance=0.0;
	float size_x=0.5e-6;
	float size_y=125.0e-6;
	float distance_x=26.0e-6;
	float distance_y=0.0;
	float size_gx=0.0;
	int n_x=10;
	int n_y=1;
	int n_gx=1;
	bool b1=true; 
	bool b2=true;
	bool b3=true;
	bool b4=true;
	bool b5=true;
	bool b6=true;
	bool b7=false;
	bool b8=false; 
	bool b9=false;
	float d1=1600e-6;
	float d2=25.0e-6;
	float d3=250.0e-6;
	float d4=25.0e-6;
	float d5=100.0e-6;
	float d6=3.0e-6;
	float d7=0.0;
	float d8=0.0;
	float d9=0.0;
	float k1= 210.0;
	float k2 = 57.0;
	float k3 = 390.0;
	float k4 = 57.0;
	float k5 = 370.0;
	float k6 = 130.0; 
	float k7=0.0;
	float k8=0.0;
	float k9=0.0;
	float rhoCp1 = 0.0;
	float rhoCp2 = 0.0;
	float rhoCp3 = 0.0;
	float rhoCp4 = 0.0;
	float rhoCp5 = 0.0;
	float rhoCp6 = 0.0;
	float rhoCp7 = 0.0;
	float rhoCp8 = 0.0;
	float rhoCp9 = 0.0;
	float mplane1 = 1.0;
	float mplane2 = 1.0;
	float mplane3 = 1.0;
	float mplane4 = 1.0;
	float mplane5 = 1.27;
	float mplane6 = 1.0;
	float mplane7 = 1.0;
	float mplane8 = 1.0;
	float mplane9 = 1.0;
	float mortogonal1 = 1.0;
	float mortogonal2 = 1.0;
	float mortogonal3 = 1.0;
	float mortogonal4 = 1.0;
	float mortogonal5 = 1.0;
	float mortogonal6 = 1.0;
	float mortogonal7 = 1.0;
	float mortogonal8 = 1.0;
	float mortogonal9 = 1.0;
	float alpha1 = 0.0;
	float alpha2 = 0.0;
	float alpha3 = 0.0;
	float alpha4 = 0.0;
	float alpha5 = -1.5;
	float alpha6 = -0.43;
	float alpha7 = 0.0;
	float alpha8 = 0.0;
	float alpha9 = 0.0;
	int time=0;
	float Tamb=86.0;
	float Pdiss=6.875;
	bool export3D=false;
	bool exportxy2D=false;
	bool exportx1D=true;
	bool bfloat=true;
	int idGPU = 0;

	FILE* fp;

#ifdef MINGW_COMPILLER
	int err = 0;
	fp = fopen64("source.txt", "r");
	if (fp != nullptr) {
		err = 0;
	}
	else {
		err = 1; // ошибка открытия.
	}
#else
	errno_t err;
	err = fopen_s(&fp, "source.txt", "r");
#endif
	

	
	if (err != 0) {
		printf("Error!!! No input File source.txt.\n");
		printf("You must use the graphical user interface\n");
		printf("FourierHEAT3D.exe in Delphi, which will \n");
		printf("prepare the model and write the source.txt\n");
		printf("file in the required format.\n");
		//system("PAUSE");
		system("pause");
		// Если файла source.txt нет то мы не можем ничего обрабатывать
		// т.к. элементарно отсутствуют входные данные. Поэтому мы выходим из 
		// приложения.
		exit(1);
	}
	else
	{
		if (fp != nullptr) {
			float fin = 0.0;
			int din = 0;

			fscanf_s(fp, "%f", &thermal_resistance);
			fscanf_s(fp, "%f", &size_x);
			fscanf_s(fp, "%f", &size_y);
			fscanf_s(fp, "%f", &distance_x);
			fscanf_s(fp, "%f", &distance_y);

			fscanf_s(fp, "%f", &size_gx);
			fscanf_s(fp, "%d", &n_x);
			fscanf_s(fp, "%d", &n_y);
			fscanf_s(fp, "%d", &n_gx);

			fscanf_s(fp, "%d", &din);
			if (din > 0) {
				b1 = true;
			}
			else {
				b1 = false;
			}

			fscanf_s(fp, "%d", &din);
			if (din > 0) {
				b2 = true;
			}
			else {
				b2 = false;
			}

			fscanf_s(fp, "%d", &din);
			if (din > 0) {
				b3 = true;
			}
			else {
				b3 = false;
			}

			fscanf_s(fp, "%d", &din);
			if (din > 0) {
				b4 = true;
			}
			else {
				b4 = false;
			}

			fscanf_s(fp, "%d", &din);
			if (din > 0) {
				b5 = true;
			}
			else {
				b5 = false;
			}

			fscanf_s(fp, "%d", &din);
			if (din > 0) {
				b6 = true;
			}
			else {
				b6 = false;
			}

			fscanf_s(fp, "%d", &din);
			if (din > 0) {
				b7 = true;
			}
			else {
				b7 = false;
			}

			fscanf_s(fp, "%d", &din);
			if (din > 0) {
				b8 = true;
			}
			else {
				b8 = false;
			}

			fscanf_s(fp, "%d", &din);
			if (din > 0) {
				b9 = true;
			}
			else {
				b9 = false;
			}

			fscanf_s(fp, "%f", &d1);
			fscanf_s(fp, "%f", &d2);
			fscanf_s(fp, "%f", &d3);
			fscanf_s(fp, "%f", &d4);
			fscanf_s(fp, "%f", &d5);
			fscanf_s(fp, "%f", &d6);
			fscanf_s(fp, "%f", &d7);
			fscanf_s(fp, "%f", &d8);
			fscanf_s(fp, "%f", &d9);

			fscanf_s(fp, "%f", &k1);
			fscanf_s(fp, "%f", &k2);
			fscanf_s(fp, "%f", &k3);
			fscanf_s(fp, "%f", &k4);
			fscanf_s(fp, "%f", &k5);
			fscanf_s(fp, "%f", &k6);
			fscanf_s(fp, "%f", &k7);
			fscanf_s(fp, "%f", &k8);
			fscanf_s(fp, "%f", &k9);

			fscanf_s(fp, "%f", &rhoCp1);
			fscanf_s(fp, "%f", &rhoCp2);
			fscanf_s(fp, "%f", &rhoCp3);
			fscanf_s(fp, "%f", &rhoCp4);
			fscanf_s(fp, "%f", &rhoCp5);
			fscanf_s(fp, "%f", &rhoCp6);
			fscanf_s(fp, "%f", &rhoCp7);
			fscanf_s(fp, "%f", &rhoCp8);
			fscanf_s(fp, "%f", &rhoCp9);

			fscanf_s(fp, "%f", &mplane1);
			fscanf_s(fp, "%f", &mplane2);
			fscanf_s(fp, "%f", &mplane3);
			fscanf_s(fp, "%f", &mplane4);
			fscanf_s(fp, "%f", &mplane5);
			fscanf_s(fp, "%f", &mplane6);
			fscanf_s(fp, "%f", &mplane7);
			fscanf_s(fp, "%f", &mplane8);
			fscanf_s(fp, "%f", &mplane9);

			fscanf_s(fp, "%f", &mortogonal1);
			fscanf_s(fp, "%f", &mortogonal2);
			fscanf_s(fp, "%f", &mortogonal3);
			fscanf_s(fp, "%f", &mortogonal4);
			fscanf_s(fp, "%f", &mortogonal5);
			fscanf_s(fp, "%f", &mortogonal6);
			fscanf_s(fp, "%f", &mortogonal7);
			fscanf_s(fp, "%f", &mortogonal8);
			fscanf_s(fp, "%f", &mortogonal9);

			fscanf_s(fp, "%f", &alpha1);
			fscanf_s(fp, "%f", &alpha2);
			fscanf_s(fp, "%f", &alpha3);
			fscanf_s(fp, "%f", &alpha4);
			fscanf_s(fp, "%f", &alpha5);
			fscanf_s(fp, "%f", &alpha6);
			fscanf_s(fp, "%f", &alpha7);
			fscanf_s(fp, "%f", &alpha8);
			fscanf_s(fp, "%f", &alpha9);

			fscanf_s(fp, "%d", &time);
			fscanf_s(fp, "%f", &Tamb);
			fscanf_s(fp, "%f", &Pdiss);

			fscanf_s(fp, "%d", &din);
			if (din > 0) {
				export3D = true;
			}
			else {
				export3D = false;
			}

			fscanf_s(fp, "%d", &din);
			if (din > 0) {
				exportxy2D = true;
			}
			else {
				exportxy2D = false;
			}

			fscanf_s(fp, "%d", &din);
			if (din > 0) {
				exportx1D = true;
			}
			else {
				exportx1D = false;
			}

			fscanf_s(fp, "%d", &din);
			if (din > 0) {
				bfloat = true;
			}
			else {
				bfloat = false;
			}

			fscanf_s(fp, "%d", &idGPU);

			fclose(fp);

			fourier_solve(
				thermal_resistance, size_x, size_y, distance_x, distance_y,
				size_gx, n_x, n_y, n_gx,
				b1, b2, b3, b4, b5, b6, b7, b8, b9,
				d1, d2, d3, d4, d5, d6, d7, d8, d9,
				k1, k2, k3, k4, k5, k6, k7, k8, k9,
				rhoCp1, rhoCp2, rhoCp3, rhoCp4, rhoCp5, rhoCp6,
				rhoCp7, rhoCp8, rhoCp9,
				mplane1, mplane2, mplane3,
				mplane4, mplane5, mplane6,
				mplane7, mplane8, mplane9,
				mortogonal1, mortogonal2, mortogonal3,
				mortogonal4, mortogonal5, mortogonal6,
				mortogonal7, mortogonal8, mortogonal9,
				alpha1, alpha2, alpha3,
				alpha4, alpha5, alpha6,
				alpha7, alpha8, alpha9,
				time, Tamb, Pdiss, export3D, exportxy2D, exportx1D, bfloat,
				idGPU);


			int im = 0, is = 0, ims = 0;
			im = (int)(time / 60000); // минуты
			is = (int)((time - 60000 * im) / 1000); // секунды
			ims = (int)((time - 60000 * im - 1000 * is)); // /10 миллисекунды делённые на 10
			printf(" %1d:%2d:%3d \n", im, is, ims);

			printf("thermal resistance = %1.2f \n", thermal_resistance);

		}
	}

	getchar();
	// 1min 9s. 

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
