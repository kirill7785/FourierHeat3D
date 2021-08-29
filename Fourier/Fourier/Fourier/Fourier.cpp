// Fourier.cpp : Defines the exported functions for the DLL.
// Рабочая версия от 18.06.2021

#include "pch.h" // use stdafx.h in Visual Studio 2017 and earlier
#include <utility>
#include <limits.h>
//#include <iostream>
//#include <string.h>
#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <omp.h>

// https://e-maxx.ru/algo/fft_multiply
#include "Fourier.h"

//typedef float Real;
const double M_PI = 3.14159265;

// Выделяет оперативную память под вектор u.
template <typename doublerealT>
void alloc_u3D(doublerealT***& u, int m, int n, int l) {
	u = new doublerealT * *[m + 2];
	for (int i = 0; i <= m + 1; ++i) {
		u[i] = new doublerealT * [n + 2];
		for (int j = 0; j <= n + 1; ++j) {
			u[i][j] = new doublerealT[l + 2];
		}
	}
}

// Освобождает оперативную память из под вектора u.
template <typename doublerealT>
void free_u3D(doublerealT***& u, int m, int n, int l) {
	for (int i = 0; i <= m + 1; ++i) {
		for (int j = 0; j <= n + 1; ++j) {
			delete[] u[i][j];
		}
	}
	for (int i = 0; i <= m + 1; ++i) {
		delete[] u[i];
	}
	delete[] u;
}

// Инициализирует массив u значением initVal.
template <typename doublerealT>
void init_u3D(doublerealT***& u, int m, int n, int l, doublerealT initVal) {

#pragma omp parallel for
	for (int i = 0; i < m + 2; ++i) {
		for (int j = 0; j < n + 2; ++j) {
			for (int k = 0; k < l + 2; ++k) {
				u[i][j][k] = initVal;
			}
		}
	}
}

//typedef std::complex<Real> base;

// Использует выделения и уничтожения памяти внутри fft.
template <typename doublerealT>
void fft_non_optimaze_memory(std::complex<doublerealT>*& a, int n, bool invert) {

	if (n == 1) return;

	std::complex<doublerealT>* a0 = new std::complex<doublerealT>[n / 2];
	std::complex<doublerealT>* a1 = new std::complex<doublerealT>[n / 2];

	for (int i = 0, j = 0; i < n; i += 2, ++j) {
		a0[j] = a[i];
		a1[j] = a[i + 1];
	}
	fft_non_optimaze_memory(a0, n / 2, invert);
	fft_non_optimaze_memory(a1, n / 2, invert);

	std::complex<doublerealT> w(1, 0);
	doublerealT ang = 2.0 * M_PI / n * (invert ? -1 : 1);
	std::complex<doublerealT> wn(cos(ang), sin(ang));
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
template <typename doublerealT>
void fft0(std::complex<doublerealT>*& a, int n, bool invert) {

	if (n == 1) return;

	

	for (int i = 1, j = 0; i < n; ++i) {
		int bit = n >> 1;
		for (; j >= bit; bit >>= 1)
			j -= bit;
		j += bit;
		if (i < j)
			swap(a[i], a[j]);
	}
	
	for (int len = 2; len <= n; len <<= 1) {
		doublerealT ang = 2 * M_PI / len * (invert ? -1 : 1);
		std::complex<doublerealT> wlen(cos(ang), sin(ang));
		for (int i = 0; i < n; i += len) {
			std::complex<doublerealT> w(1);
			for (int j = 0; j < len / 2; ++j) {
				std::complex<doublerealT> u = a[i + j], v = a[i + j + len / 2] * w;
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


int MAXN = 262145;
// Максимальное число потоков в процессоре.
const int NUMBER_THREADS = omp_get_num_procs(); //256;

const int MAXN_rev = 262145;
int rev[MAXN_rev];
template <typename doublerealT>
std::complex<doublerealT>** a1_gl = nullptr;

template <typename doublerealT>
std::complex<doublerealT>** wlen_pw_gl = nullptr;

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
template <typename doublerealT>
void fft(std::complex<doublerealT>* &a, int n, bool invert, std::complex<doublerealT>* &wlen_pw) {

	for (int i = 0; i < n; ++i)
		if (i < rev[i])
			swap(a[i], a[rev[i]]);

	for (int len = 2; len <= n; len <<= 1) {
		doublerealT ang = 2 * M_PI / len * (invert ? -1 : +1);
		int len2 = len >> 1;

		std::complex<doublerealT> wlen(cos(ang), sin(ang));
		wlen_pw[0] = std::complex<doublerealT>(1, 0);
		for (int i = 1; i < len2; ++i)
			wlen_pw[i] = wlen_pw[i - 1] * wlen;

		for (int i = 0; i < n; i += len) {
			std::complex<doublerealT> t,
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
template <typename doublerealT>
void DFTx(doublerealT***& u, doublerealT***& a, int m, int n, int l) {

	int Nx = m + 1; // < M+2
	int Ny = n + 1; // < N+2
	int Nz = l + 1;

	// Находим искомую функцию.

	calc_rev(Nx - 2);

#pragma omp parallel for
	for (int k = 1; k < Nz - 1; ++k) {

		int id = omp_get_thread_num();
		std::complex<doublerealT>* a1 = a1_gl<doublerealT>[id];
		std::complex<doublerealT>* wlen_pw = wlen_pw_gl<doublerealT>[id];

		for (int j = 0; j <= Ny - 1; ++j) {

			a1[0] = (0.0, 0.0);
			for (int i = 1; i < Nx - 1; ++i) {
				a1[i - 1] = (0.0, a[i][j][k]);
			}
			fft(a1, Nx - 2, false, wlen_pw);
			for (int i = 1; i < Nx - 1; ++i) {
				u[i][j][k] = -a1[i - 1]._Val[0];
			}
		}

		a1 = nullptr;
		wlen_pw = nullptr;
		//delete[] a1;
		//delete[] wlen_pw;
	}

}

// Вызывает быстрое преобразование Фурье.
template <typename doublerealT>
void DFTy(doublerealT***& u, doublerealT***& a, int m, int n, int l) {

	int Nx = m + 1; // < M+2
	int Ny = n + 1; // < N+2
	int Nz = l + 1;

	calc_rev(Ny - 2);

	// Находим искомую функцию.
#pragma omp parallel for
	for (int k = 1; k < Nz - 1; ++k) {

		int id = omp_get_thread_num();
		std::complex<doublerealT>* a1 = a1_gl<doublerealT>[id];
		std::complex<doublerealT>* wlen_pw = wlen_pw_gl<doublerealT>[id];

		for (int i = 0; i <= Nx - 1; ++i) {

			for (int j = 1; j < Ny - 1; ++j) {
				a1[j - 1] = (0.0, a[i][j][k]);
			}
			fft(a1, Ny - 2, false, wlen_pw);
			for (int j = 1; j < Ny - 1; ++j) {
				u[i][j][k] = -a1[j - 1]._Val[0];
			}
		}

		a1 = nullptr;
		wlen_pw = nullptr;
	}
}

// Вызывает быстрое преобразование Фурье.
template <typename doublerealT>
void IDFTx(doublerealT***& u, doublerealT***& f, int m, int n, int l) {

	int Nx = m + 1; // < M+2
	int Ny = n + 1; // < N+2
	int Nz = l + 1;

	calc_rev(Nx - 2);

	// Преобразование правой части.
#pragma omp parallel for
	for (int k = 1; k < Nz - 1; ++k) {

		int id = omp_get_thread_num();
		std::complex<doublerealT>* a1 = a1_gl<doublerealT>[id];
		std::complex<doublerealT>* wlen_pw = wlen_pw_gl<doublerealT>[id];
		

		for (int j = 0; j <= Ny - 1; ++j) {

			for (int i = 1; i < Nx - 1; ++i) {

				a1[i - 1] = (0.0, f[i][j][k]);
			}

			fft(a1, Nx - 2, true, wlen_pw);
			for (int i = 1; i < Nx - 1; ++i) {

				u[i][j][k] = a1[i - 1]._Val[0];
			}
		}

		a1 = nullptr;
		wlen_pw = nullptr;
	}
}

// Вызывает быстрое преобразование Фурье.
template <typename doublerealT>
void IDFTy(doublerealT***& u, doublerealT***& f, int m, int n, int l) {

	int Nx = m + 1; // < M+2
	int Ny = n + 1; // < N+2
	int Nz = l + 1;

	calc_rev(Ny - 2);

	// Преобразование правой части.
#pragma omp parallel for
	for (int k = 1; k < Nz - 1; ++k) {


		int id = omp_get_thread_num();
		std::complex<doublerealT>* a1 = a1_gl<doublerealT>[id];
		std::complex<doublerealT>* wlen_pw = wlen_pw_gl<doublerealT>[id];

		for (int i = 0; i <= Nx - 1; ++i) {

			for (int j = 1; j < Ny - 1; ++j) {
				a1[j - 1] = (0.0, f[i][j][k]);
			}
			fft(a1, Ny - 2, true, wlen_pw);
			for (int j = 1; j < Ny - 1; ++j) {
				u[i][j][k] = a1[j - 1]._Val[0];
			}
		}

		a1 = nullptr;
		wlen_pw = nullptr;
	}
}

// В массив source копирует массив b со знаком плюс или минус.
template <typename doublerealT>
void copy3D(doublerealT***& source, doublerealT***& b, int m, int n, int l,
	bool plus, bool scal, doublerealT val) {

	int Nx = m + 1; // < M+2
	int Ny = n + 1; // < N+2
	int Nz = l + 1;

	if (scal) {
		// добавляем скаляр.
#pragma omp parallel for
		for (int i = 0; i <= Nx; ++i) {
			for (int j = 0; j <= Ny; ++j) {
				for (int k = 0; k <= Nz; ++k) {

					source[i][j][k] += val;
				}
			}
		}
	}
	else {

		if (plus) {

#pragma omp parallel for
			for (int i = 0; i <= Nx; ++i) {
				for (int j = 0; j <= Ny; ++j) {
					for (int k = 0; k <= Nz; ++k) {

						source[i][j][k] = b[i][j][k];
					}
				}
			}

		}
		else {

#pragma omp parallel for
			for (int i = 0; i <= Nx; ++i) {
				for (int j = 0; j <= Ny; ++j) {
					for (int k = 0; k <= Nz; ++k) {

						source[i][j][k] = -b[i][j][k];
					}
				}
			}
		}
	}
} // copy3D


// экспорт 3D полевой величины u в программу tecplot 360.
template <typename doublerealT>
void exporttecplot3D(doublerealT***& u, doublerealT*& x, doublerealT*& y, doublerealT*& z, int m, int n, int l) {
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
		for (int k = 0; k < l + 2; ++k) for (int j = 0; j < n + 2; ++j) for (int i = 0; i < m + 2; ++i)   fprintf(fp, "%e %e %e %e\n", x[i], y[j], z[k], u[i][j][k]);
		fclose(fp);

		WinExec("C:\\Program Files (x86)\\Tecplot\\Tec360 2008\\bin\\tec360.exe temperature_xyz_plot.PLT", SW_NORMAL);
	}

	//getchar();

} // exporttecplot3D

// экспорт 3D полевой величины u в программу tecplot 360.
template <typename doublerealT>
void exporttecplot3D_fft(doublerealT***& u, doublerealT*& x, doublerealT*& y, doublerealT*& z, int m, int n, int l) {
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
		for (int k = 0; k < l + 2; ++k) for (int j = 1; j < n; ++j) for (int i = 1; i < m; ++i)   fprintf(fp, "%e %e %e %e\n", x[i], y[j], z[k], u[i][j][k]);
		fclose(fp);

		WinExec("C:\\Program Files (x86)\\Tecplot\\Tec360 2008\\bin\\tec360.exe temperature_xyz_plot.PLT", SW_NORMAL);
	}

	//getchar();

} // exporttecplot3D_fft


// экспорт 3D полевой величины u в программу tecplot 360.
template <typename doublerealT>
void exporttecplot3D22D(doublerealT***& u, doublerealT*& x, doublerealT*& y, doublerealT*& z, int m, int n, int l) {
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
		for (int j = 0; j < n + 2; ++j) for (int i = 0; i < m + 2; ++i)   fprintf(fp, "%e %e %e\n", x[i], y[j], u[i][j][l + 1]);
		fclose(fp);

		WinExec("C:\\Program Files (x86)\\Tecplot\\Tec360 2008\\bin\\tec360.exe temperature_xy_plot.PLT", SW_NORMAL);
	}

	//getchar();

} // exporttecplot3D2_2D

// экспорт 3D полевой величины u в программу tecplot 360.
template <typename doublerealT>
void exporttecplot3D21D(doublerealT***& u, doublerealT*& x, doublerealT*& y, doublerealT*& z, int m, int n, int l) {
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
		for (int i = 1; i < m + 1; ++i)   fprintf(fp, "%e %e\n", x[i], u[i][j0][l + 1]);
		fclose(fp);

		WinExec("C:\\Program Files (x86)\\Tecplot\\Tec360 2008\\bin\\tec360.exe temperature_x_plot.PLT", SW_NORMAL);
	}

	//getchar();

} // exporttecplot3D2_1D

// Структура для описания одного слоя подложки.
template <typename doublerealT>
struct STACK_LAYERS {
	doublerealT chikness; // Толщина в м.
	doublerealT lambda; // теплопроводность, Вт/(м*К).
	int idiv_layer; // Количество ячеек в слое по толщине.
	doublerealT mplane, mortho; // Ортотропность коэффициента теплопроводности 24.05.2021.
	// Для расчёта вклада каждого слоя в Rt.
	doublerealT Tmax;
	doublerealT alpha; // Температурная зависимость теплопроводности.
	//std::string name; // имя материала.
	char* name;
};

// Процедура mesh_size
// подбирает размер равномерной сетки в плоскости Oxy в 
// целях экономии ресурсов памяти и увеличения быстродействия,
// подбирает сетку так чтобы рассчитываемая топология в плоскости Oxy 
// была разрешена расчётной сеткойи эта сетка не была избыточной.
template <typename doublerealT>
void mesh_size(int &N, int &M,
	int n_xgr, int n_y, doublerealT&lengthx, doublerealT&lengthy,
	doublerealT&multiplyer, doublerealT size_x, doublerealT size_y, doublerealT distance_x,
	doublerealT distance_y, doublerealT h1, doublerealT h2)
{
	/*if (n_xgr <= 2048)*/ {

		M = 33;
		lengthx = (M + 1) * h1;

		if (multiplyer * (n_xgr)*distance_x + 30.0 * size_x < lengthx) {
			// Ok
		}
		else {
			M = 65;
			lengthx = (M + 1) * h1;

			if (multiplyer * (n_xgr)*distance_x + 30.0 * size_x < lengthx) {
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
		lengthy = (N + 1) * h1;
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
template <typename doublerealT>
void progonka(int Nx, int Ny, int Nz,
	doublerealT* &lambda, doublerealT* &lambda_multiplyer_normal, doublerealT* & lambda_multiplyer_plane,
	doublerealT* &hz, doublerealT h1, doublerealT h2, doublerealT*** &b, doublerealT*** &a)
{
	// Ядро солвера.
#pragma omp parallel for
	for (int i = 1; i < Nx - 1; ++i) {
		for (int j = 1; j < Ny - 1; ++j) {

			doublerealT* P = new doublerealT[Nz + 1];
			doublerealT* Q = new doublerealT[Nz + 1];

			doublerealT Rj1 = lambda_multiplyer_normal[1] * (2.0 * lambda[0] * lambda[1] / (lambda[0] + lambda[1])) *
				(2.0 / (hz[1] * (hz[1] + hz[0])));
			doublerealT Rjm1 = lambda_multiplyer_normal[1] * (2.0 * lambda[0] * lambda[1] / (lambda[0] + lambda[1])) *
				(2.0 / (hz[0] * (hz[1] + hz[0])));

			// Для фиксированного k решение системы  с трёх диагональной матрицей:
			doublerealT b1 = Rj1;
			doublerealT a1 = (2.0 / (h1 * h1) + 2.0 / (h2 * h2)) * lambda[1] * lambda_multiplyer_plane[1] + 
				(Rj1 + Rjm1) - 2.0 * cos(2.0 * M_PI * i / Nx) * lambda[1] * lambda_multiplyer_plane[1] / (h1 * h1) -
				2.0 * cos(2.0 * M_PI * j / Ny) * lambda[1] * lambda_multiplyer_plane[1] / (h2 * h2);
			P[1] = b1 / a1;
			doublerealT d1 = -b[i][j][1];
			Q[1] = d1 / a1;

			for (int k = 2; k <= Nz; ++k) {

				doublerealT lmax = lambda[k];
				if (k != Nz) {
					lmax = lambda[k + 1];
				}
				doublerealT Rj = lambda_multiplyer_normal[k] * (2.0 * lambda[k] * lmax / (lambda[k] + lmax)) *
					(2.0 / (hz[k] * (hz[k] + hz[k - 1])));
				doublerealT Rjm1 = lambda_multiplyer_normal[k] * (2.0 * lambda[k - 1] * lambda[k] / (lambda[k - 1] + lambda[k])) *
					(2.0 / (hz[k - 1] * (hz[k] + hz[k - 1])));

				// bk -> k+1
				// ck -> k-1
				doublerealT bk = Rj;
				doublerealT ck = Rjm1;
				doublerealT ak = (2.0 / (h1 * h1) + 2.0 / (h2 * h2)) * lambda[k] * lambda_multiplyer_plane[k] + (Rj + Rjm1) -
					2.0 * cos(2.0 * M_PI * i / Nx) * lambda[k] * lambda_multiplyer_plane[k] / (h1 * h1) -
					2.0 * cos(2.0 * M_PI * j / Ny) * lambda[k] * lambda_multiplyer_plane[k] / (h2 * h2);
				doublerealT dk = -b[i][j][k];

				P[k] = bk / (ak - ck * P[k - 1]);
				Q[k] = (dk + ck * Q[k - 1]) / (ak - ck * P[k - 1]);
			}
			a[i][j][Nz] = Q[Nz];

			for (int k = Nz - 1; k >= 1; k--) {
				a[i][j][k] = P[k] * a[i][j][k + 1] + Q[k];
			}

			delete[] P;
			delete[] Q;
		}
	}
} // progonka

// Начало 18.05.2021 - окончание 31.05.2021.
// Находит статическое тепловое сопротивление для постоянных
// коэффициентов теплопроводности не зависящих от температуры.
template <typename doublerealT>
void FFTsolver3Dqlinear(doublerealT& thermal_resistance,
	doublerealT size_x, doublerealT size_y, doublerealT distance_x, doublerealT distance_y,
	doublerealT size_gx, int n_x, int n_y, int n_gx,
	bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7, bool b8, bool b9,
	doublerealT d1, doublerealT d2, doublerealT d3, 
	doublerealT d4, doublerealT d5, doublerealT d6,
	doublerealT d7, doublerealT d8, doublerealT d9,
	doublerealT k1, doublerealT k2, doublerealT k3,
	doublerealT k4, doublerealT k5, doublerealT k6,
	doublerealT k7, doublerealT k8, doublerealT k9,
	doublerealT mplane1, doublerealT mplane2, doublerealT mplane3,
	doublerealT mplane4, doublerealT mplane5, doublerealT mplane6,
	doublerealT mplane7, doublerealT mplane8, doublerealT mplane9,
	doublerealT mortogonal1, doublerealT mortogonal2, doublerealT mortogonal3,
	doublerealT mortogonal4, doublerealT mortogonal5, doublerealT mortogonal6,
	doublerealT mortogonal7, doublerealT mortogonal8, doublerealT mortogonal9,
	int &time, doublerealT Tamb, doublerealT Pdiss, bool export3D, bool exportxy2D, bool exportx1D) {

	// Замер времени.
	unsigned int calculation_start_time = 0; // начало счёта мс.
	unsigned int calculation_end_time = 0; // окончание счёта мс.
	unsigned int calculation_seach_time = 0; // время выполнения участка кода в мс.

	calculation_start_time = clock();

	doublerealT*** u = nullptr, *** source = nullptr; // рассчитываемый потенциал и тепловая мощность.

	doublerealT*** a = nullptr, *** b = nullptr; // Коэффициенты разложения в дискретный ряд Фурье.

	doublerealT Pdiss_sum = 0.0;

	doublerealT lengthx = 4098.0e-6;// 2050.0e-6;
	doublerealT lengthy = 1625.0e-6;

	int M = 4097; // 2049;//511;
	int N = 129;// 511;

	

	/*lengthx = lengthy = 1.0;
	M = 32;
	N = 159;*/

	//doublerealT Tamb = 22.0; // - температура корпуса задаётся пользователем. 

	//Tamb = 0.0;

	// постоянные шаги сетки h1 по оси x и h2 по оси y.
	doublerealT h1 = (doublerealT)(lengthx / (M + 1));
	doublerealT h2 = (doublerealT)(lengthy / (N + 1));

	h1 = 0.5 * size_x;
	if (n_y == 1) {
		h2 = 0.1 * size_y;
	}
	else {
		h2 = 0.5 * size_y;
	}	


	lengthx = (M + 1) * h1;
	lengthy = (N + 1) * h2;

	doublerealT multiplyer = 1.2;

	doublerealT n_xgr = n_x * n_gx;
	if (fabs(distance_x) > 1.0e-30) {
		n_xgr += ((n_gx - 1) * (size_gx / distance_x));
	}

	mesh_size(N, M, n_xgr, n_y, lengthx, lengthy, multiplyer, size_x, size_y, distance_x, distance_y, h1, h2);


	int MAXN = (M > N ? M + 2 : N + 2);
	a1_gl<doublerealT> = new std::complex<doublerealT> *[NUMBER_THREADS];
	wlen_pw_gl<doublerealT> = new std::complex<doublerealT> *[NUMBER_THREADS];
	for (int i = 0; i < NUMBER_THREADS; ++i)
	{
		a1_gl<doublerealT>[i] = new std::complex<doublerealT>[MAXN];
		wlen_pw_gl<doublerealT>[i] = new std::complex<doublerealT>[MAXN];
	}

	doublerealT* xf = new doublerealT[M + 2];
	for (int i = 0; i < M + 2; ++i) {
		xf[i] = (doublerealT)(i * h1);
	}
	doublerealT* yf = new doublerealT[N + 2];
	for (int i = 0; i < N + 2; ++i) {
		yf[i] = (doublerealT)(i * h2);
	}

	int ilayer_count = 8;
	STACK_LAYERS<doublerealT>* stack_layer = nullptr;
	//doublerealT chikness_min = 1.0e30;

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

	stack_layer = new STACK_LAYERS<doublerealT>[ilayer_count];

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

	doublerealT lengthz = 0.0;
	int inodes = 0;
	//lengthz = 1.0;
	for (int i = 0; i < ilayer_count; ++i) {
		lengthz += stack_layer[i].chikness;
		inodes += stack_layer[i].idiv_layer;
	}
	int izstop = inodes - stack_layer[ilayer_count - 1].idiv_layer + 1;
	int L = inodes + 1;


	//int izstop = 63;
	//int L = izstop=63;
	//doublerealT hzc = (doublerealT)(lengthz / (L + 1));
	doublerealT* hz = new doublerealT[L + 2];
	doublerealT* zf = new doublerealT[L + 2];
	doublerealT* lambda = new doublerealT[L + 2];// Теплопроводность.
	doublerealT* lambda_multiplyer_plane = new doublerealT[L + 2];// Теплопроводность.
	doublerealT* lambda_multiplyer_normal = new doublerealT[L + 2];// Теплопроводность.
	int* layer_id = new int[L + 2];
	/*for (int i = 0; i < L + 2; ++i) {
		zf[i] = (doublerealT)(i * hzc);
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
	doublerealT q = 1.2;
	doublerealT qinv = 1 / q;
	hz[0] = stack_layer[0].chikness * (qinv - 1.0) / (pow(qinv, stack_layer[0].idiv_layer)-1.0);
	// Постоянный шаг по оси z в каждом интервале.
	//hz[0] = stack_layer[0].chikness / stack_layer[0].idiv_layer;
	doublerealT qsum = 0.0;
	int ic = 1;
	layer_id[0] = 0;
	lambda[0] = stack_layer[0].lambda;
	lambda_multiplyer_plane[0] = stack_layer[0].mplane;
	lambda_multiplyer_normal[0] = stack_layer[0].mortho;
	for (int i = 0; i < ilayer_count; ++i) {
		for (int j = 0; j < stack_layer[i].idiv_layer; ++j) {
			if (i < ilayer_count - 1) {

				hz[ic] = stack_layer[i].chikness * (qinv - 1.0) / (pow(qinv, stack_layer[i].idiv_layer) - 1.0);
				for (int i7 = 1; i7 <= j; ++i7) hz[ic] *= qinv;
			}
			else {
				hz[ic] = stack_layer[i].chikness * (qinv - 1.0) / (pow(qinv, stack_layer[i].idiv_layer) - 1.0);
				for (int i7 = 0; i7 <= stack_layer[i].idiv_layer-1-j; ++i7) hz[ic] *= qinv;
				qsum += hz[ic];
			}
			// Постоянный шаг по оси z в каждом интервале.
			//hz[ic] = stack_layer[i].chikness / stack_layer[i].idiv_layer;
			layer_id[ic] = i;
			lambda[ic] = stack_layer[i].lambda;
			lambda_multiplyer_plane[ic] = stack_layer[i].mplane;
			lambda_multiplyer_normal[ic] = stack_layer[i].mortho;
			zf[1 + ic] = zf[ic] + hz[ic];
			++ic;
		}
	}
	//zf[1 + ic] = zf[ic] + stack_layer[ilayer_count - 1].chikness / stack_layer[ilayer_count - 1].idiv_layer;
	//hz[ic] = stack_layer[ilayer_count - 1].chikness / stack_layer[ilayer_count - 1].idiv_layer;
	
	//hz[ic + 1] = stack_layer[ilayer_count - 1].chikness / stack_layer[ilayer_count - 1].idiv_layer;

	
	hz[ic] = hz[ic+1]=(stack_layer[ilayer_count - 1].chikness - qsum) / 2.0;

	zf[1 + ic] = 0.0;
	for (int i7 = 0; i7 < ilayer_count; ++i7) {
		zf[1 + ic] += stack_layer[i7].chikness;
	}
	layer_id[ic] = ilayer_count - 1;
	layer_id[ic+1] = ilayer_count - 1;
	lambda[ic] = stack_layer[ilayer_count - 1].lambda;
	lambda[ic + 1] = stack_layer[ilayer_count - 1].lambda;
	lambda_multiplyer_plane[ic] = stack_layer[ilayer_count - 1].mplane;
	lambda_multiplyer_normal[ic] = stack_layer[ilayer_count - 1].mortho;
	lambda_multiplyer_plane[ic+1] = stack_layer[ilayer_count - 1].mplane;
	lambda_multiplyer_normal[ic+1] = stack_layer[ilayer_count - 1].mortho;


	int Nx = M + 1; // < M+2
	int Ny = N + 1; // < N+2
	int Nz = L + 1;

	for (int k = 1; k <= Nz; ++k) {

		//printf("%d %d lam=%e %e\n", k, izstop - 1, lambda[k], hz[k]);
	}
	//getchar();

	alloc_u3D(source, M, N, L);
	alloc_u3D(b, M, N, L);

	// инициализация.
	init_u3D<doublerealT>(b, M, N, L, 0.0);
	init_u3D<doublerealT>(source, M, N, L, 0.0);

	const doublerealT power_1_polosok = Pdiss / (n_y*(n_x+1)*n_gx);//0.675;

	int in_x = (int)(distance_x / h1);
	int in_y = (int)(distance_y / h2);

	// Задаём тепловую мощность тепловыделения.
	//for (int i = M / 2 - 117; i < M / 2 + 124; i += 26) {// ПТБШ 1,25мм 2s 204Mb
	//for (int i = M / 2 - 234; i < M / 2 + 240; i += 26) {// ПТБШ 2,5мм 2s 204Mb
	//for (int i = M / 2 - 468; i < M / 2 + 480; i += 26) { // ПТБШ 5мм 2s 204Mb
		//for (int i = M / 2 - 936; i < M / 2 + 946; i += 26) { // ПТБШ 10мм 4s 420Mb
	int istart0 = M / 2 - (int)(0.5 * (in_x*n_xgr));
	int iend0 = istart0 + ((int)((in_x * n_x))) + (in_x == 0 ? 3 : 1);

	for (int igr = 0; igr < n_gx; ++igr) 
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
					doublerealT Ssource = size_x * size_y;
					doublerealT Vol_source = Ssource * hz[izstop - 1];
					if (n_y == 1) {
						source[i + 1][j][izstop - 1] += power_1_polosok / (Vol_source);
						source[i][j][izstop - 1] += power_1_polosok / (Vol_source);
					}
					else {
						source[i + 1][j][izstop - 1] += power_1_polosok / (Vol_source);
						source[i][j][izstop - 1] += power_1_polosok / (Vol_source);
						source[i + 1][j + 1][izstop - 1] += power_1_polosok / (Vol_source);
						source[i][j + 1][izstop - 1] += power_1_polosok / (Vol_source);

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
					//for (int j = N / 2 - 2; j < N / 2 + 3; ++j) {
				for (int j = (n_y == 1 ? N / 2 - 5 : N / 2 - (int)(0.5 * (in_y * (n_y)))); j < (n_y == 1 ? N / 2 + 5 : N / 2 + (int)(0.5 * (in_y * (n_y)) + (in_y == 0 ? 3 : 1))); j += (n_y == 1 ? 1 : in_y)) {
					//if ((i - (M / 2 - 117)) % 26 == 0)
					{


						//source[i][j][izstop] = 0.675 / (5*h1*h2*hz[izstop-1]);
						doublerealT Ssource = size_x * size_y;
						doublerealT Vol_source = Ssource * hz[izstop - 1];
						if (n_y == 1) {
							source[i + 1][j][izstop - 1] += power_1_polosok / (Vol_source);
							source[i][j][izstop - 1] += power_1_polosok / (Vol_source);
						}
						else {
							source[i + 1][j][izstop - 1] += power_1_polosok / (Vol_source);
							source[i][j][izstop - 1] += power_1_polosok / (Vol_source);
							source[i + 1][j + 1][izstop - 1] += power_1_polosok / (Vol_source);
							source[i][j + 1][izstop - 1] += power_1_polosok / (Vol_source);

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


	IDFTx(b, source, M, N, L);

	copy3D<doublerealT>(source, b, M, N, L, true, false, 0.0);

	IDFTy(b, source, M, N, L);

	//std::cout << "1" << std::endl;

	//free_u3D(source, M, N, L);
	//alloc_u3D(a, M, N, L);

	a = source;
	source = nullptr;

	init_u3D<doublerealT>(a, M, N, L, 0.0);

	// Ядро солвера
	progonka(Nx, Ny, Nz, lambda, lambda_multiplyer_normal, lambda_multiplyer_plane, hz, h1, h2, b, a);

	//std::cout << "2" << std::endl;
	
	//free_u3D(b, M, N, L);
	//alloc_u3D(u, M, N, L);

	u = b;
	b = nullptr;


	init_u3D<doublerealT>(u, M, N, L, 0.0);

	DFTx(u, a, M, N, L);

	copy3D<doublerealT>(a, u, M, N, L, false, false, 0.0);

	DFTy(u, a, M, N, L);

	copy3D(u, u, M, N, L, true, true, Tamb);

	//std::cout << "3" << std::endl;

	free_u3D(a, M, N, L);

	doublerealT tmax1 = -1.0e30;

	for (int i = 0; i < M + 2; ++i) {
		for (int j = 0; j < N + 2; ++j) {
			if (u[i][j][izstop - 1] > tmax1) {
				tmax1 = u[i][j][izstop - 1];
			}
		}
	}

	for (int i7 = 0; i7 < ilayer_count; ++i7) {
		stack_layer[i7].Tmax = -1.0e30;
	}

	for (int k = 0; k < L + 2; ++k) {
		for (int i = 0; i < M + 2; ++i) {
			for (int j = 0; j < N + 2; ++j) {
				if (u[i][j][k] > stack_layer[layer_id[k]].Tmax) {
					stack_layer[layer_id[k]].Tmax = u[i][j][k];
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
				fprintf(fp10, "%1.1f %1.2f %2.0f%%\n", /*stack_layer[i7].name,*/ stack_layer[i7].Tmax - stack_layer[i7 - 1].Tmax, (stack_layer[i7].Tmax - stack_layer[i7 - 1].Tmax)/ Pdiss_sum, 100.0*((stack_layer[i7].Tmax - stack_layer[i7 - 1].Tmax) / Pdiss_sum) / thermal_resistance);
			}
			else {
				fprintf(fp10, "%1.1f %1.2f %2.0f%%\n", /*stack_layer[i7].name,*/ stack_layer[i7].Tmax - Tamb, (stack_layer[i7].Tmax - Tamb) / Pdiss_sum, 100.0*((stack_layer[i7].Tmax - Tamb) / Pdiss_sum)/ thermal_resistance);
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
	    exporttecplot3D22D(u, xf, yf, zf, M, N, izstop - 1);
	}
	if (exportx1D) {
	    exporttecplot3D21D(u, xf, yf, zf, M, N, izstop - 1);
	}


	free_u3D(u, M, N, L);

	for (int i = 0; i < NUMBER_THREADS; ++i)
	{
		delete[] a1_gl<doublerealT>[i];
		delete[] wlen_pw_gl<doublerealT>[i];
	}
	delete[] a1_gl<doublerealT>;
	delete[] wlen_pw_gl<doublerealT>;
	a1_gl<doublerealT> = nullptr;
	wlen_pw_gl<doublerealT> = nullptr;

	delete[] stack_layer;

	delete[] xf;
	delete[] yf;
	delete[] zf;
	delete[] hz;
	delete[] lambda;
	delete[] lambda_multiplyer_plane;
	delete[] lambda_multiplyer_normal;
	delete[] layer_id;

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
template <typename doublerealT>
void FFTsolver3Dqnonlinear(doublerealT& thermal_resistance,
	doublerealT size_x, doublerealT size_y, doublerealT distance_x, doublerealT distance_y,
	doublerealT size_gx, int n_x, int n_y, int n_gx,
	bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7, bool b8, bool b9,
	doublerealT d1, doublerealT d2, doublerealT d3,
	doublerealT d4, doublerealT d5, doublerealT d6,
	doublerealT d7, doublerealT d8, doublerealT d9,
	doublerealT k1, doublerealT k2, doublerealT k3,
	doublerealT k4, doublerealT k5, doublerealT k6,
	doublerealT k7, doublerealT k8, doublerealT k9,
	doublerealT mplane1, doublerealT mplane2, doublerealT mplane3,
	doublerealT mplane4, doublerealT mplane5, doublerealT mplane6,
	doublerealT mplane7, doublerealT mplane8, doublerealT mplane9,
	doublerealT mortogonal1, doublerealT mortogonal2, doublerealT mortogonal3,
	doublerealT mortogonal4, doublerealT mortogonal5, doublerealT mortogonal6,
	doublerealT mortogonal7, doublerealT mortogonal8, doublerealT mortogonal9,
	doublerealT alpha1, doublerealT alpha2, doublerealT alpha3,
	doublerealT alpha4, doublerealT alpha5, doublerealT alpha6,
	doublerealT alpha7, doublerealT alpha8, doublerealT alpha9,
	int& time, doublerealT Tamb, doublerealT Pdiss, bool export3D, bool exportxy2D, bool exportx1D) {

	// Замер времени.
	unsigned int calculation_start_time = 0; // начало счёта мс.
	unsigned int calculation_end_time = 0; // окончание счёта мс.
	unsigned int calculation_seach_time = 0; // время выполнения участка кода в мс.

	calculation_start_time = clock();

	doublerealT*** u=nullptr, *** source=nullptr; // рассчитываемый потенциал и тепловая мощность.

	doublerealT*** a=nullptr, *** b=nullptr; // Коэффициенты разложения в дискретный ряд Фурье.

	doublerealT Pdiss_sum = 0.0;

	doublerealT lengthx = 4098.0e-6;// 2050.0e-6;
	doublerealT lengthy = 1625.0e-6;

	int M = 4097; // 2049;//511;
	int N = 129;// 511;



	/*lengthx = lengthy = 1.0;
	M = 32;
	N = 159;*/

	//doublerealT Tamb = 22.0; // - температура корпуса задаётся пользователем. 

	//Tamb = 0.0;

	// постоянные шаги сетки h1 по оси x и h2 по оси y.
	doublerealT h1 = (doublerealT)(lengthx / (M + 1));
	doublerealT h2 = (doublerealT)(lengthy / (N + 1));

	h1 = 0.5 * size_x;
	if (n_y == 1) {
		h2 = 0.1 * size_y;
	}
	else {
		h2 = 0.5 * size_y;
	}

	lengthx = (M + 1) * h1;
	lengthy = (N + 1) * h2;

	doublerealT multiplyer = 1.2;

	doublerealT n_xgr = 0;
	n_xgr = n_x * n_gx;
	if (fabs(distance_x) > 1.0e-30) {
		n_xgr += ((n_gx - 1) * (size_gx / distance_x));
	}

	mesh_size(N, M, n_xgr, n_y, lengthx, lengthy, multiplyer, size_x, size_y, distance_x, distance_y, h1, h2);

	int MAXN = (M > N ? M + 2 : N + 2);
	a1_gl<doublerealT> = new std::complex<doublerealT> *[NUMBER_THREADS];
	wlen_pw_gl<doublerealT> = new std::complex<doublerealT> *[NUMBER_THREADS];
	for (int i = 0; i < NUMBER_THREADS; ++i)
	{
		a1_gl<doublerealT>[i] = new std::complex<doublerealT>[MAXN];
		wlen_pw_gl<doublerealT>[i] = new std::complex<doublerealT>[MAXN];
	}

	doublerealT* xf = new doublerealT[M + 2];
	for (int i = 0; i < M + 2; ++i) {
		xf[i] = (doublerealT)(i * h1);
	}
	doublerealT* yf = new doublerealT[N + 2];
	for (int i = 0; i < N + 2; ++i) {
		yf[i] = (doublerealT)(i * h2);
	}

	int ilayer_count = 8;
	STACK_LAYERS<doublerealT>* stack_layer = nullptr;
	//doublerealT chikness_min = 1.0e30;

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
		stack_layer = new STACK_LAYERS<doublerealT>[ilayer_count];
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

	doublerealT lengthz = 0.0;
	int inodes = 0;
	//lengthz = 1.0;
	for (int i = 0; i < ilayer_count; ++i) {
		lengthz += stack_layer[i].chikness;
		inodes += stack_layer[i].idiv_layer;
	}
	int izstop = inodes - stack_layer[ilayer_count - 1].idiv_layer + 1;
	int L = inodes + 1;


	//int izstop = 63;
	//int L = izstop=63;
	//doublerealT hzc = (doublerealT)(lengthz / (L + 1));
	doublerealT* hz = new doublerealT[L + 2];
	doublerealT* zf = new doublerealT[L + 2];
	doublerealT* lambda = new doublerealT[L + 2];// Теплопроводность.
	doublerealT* lambda_multiplyer_plane = new doublerealT[L + 2];// Теплопроводность.
	doublerealT* lambda_multiplyer_normal = new doublerealT[L + 2];// Теплопроводность.
	int* layer_id = new int[L + 2];
	doublerealT* temp_maxz=new doublerealT[L + 2];
	/*for (int i = 0; i < L + 2; ++i) {
		zf[i] = (doublerealT)(i * hzc);
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
	doublerealT q = 1.2;
	doublerealT qinv = 1 / q;
	hz[0] = stack_layer[0].chikness * (qinv - 1.0) / (pow(qinv, stack_layer[0].idiv_layer) - 1.0);
	// Постоянный шаг по оси z в каждом интервале.
	//hz[0] = stack_layer[0].chikness / stack_layer[0].idiv_layer;
	doublerealT qsum = 0.0;
	int ic = 1;
	layer_id[0] = 0;
	temp_maxz[0] = Tamb;
	lambda[0] = stack_layer[0].lambda;
	lambda_multiplyer_plane[0] = stack_layer[0].mplane;
	lambda_multiplyer_normal[0] = stack_layer[0].mortho;
	for (int i = 0; i < ilayer_count; ++i) {
		for (int j = 0; j < stack_layer[i].idiv_layer; ++j) {
			if (i < ilayer_count - 1) {

				hz[ic] = stack_layer[i].chikness * (qinv - 1.0) / (pow(qinv, stack_layer[i].idiv_layer) - 1.0);
				for (int i7 = 1; i7 <= j; ++i7) hz[ic] *= qinv;
			}
			else {
				hz[ic] = stack_layer[i].chikness * (qinv - 1.0) / (pow(qinv, stack_layer[i].idiv_layer) - 1.0);
				for (int i7 = 0; i7 <= stack_layer[i].idiv_layer - 1 - j; ++i7) hz[ic] *= qinv;
				qsum += hz[ic];
			}
			// Постоянный шаг по оси z в каждом интервале.
			//hz[ic] = stack_layer[i].chikness / stack_layer[i].idiv_layer;
			layer_id[ic] = i;
			lambda[ic] = stack_layer[i].lambda;
			temp_maxz[ic] = Tamb;
			lambda_multiplyer_plane[ic] = stack_layer[i].mplane;
			lambda_multiplyer_normal[ic] = stack_layer[i].mortho;
			zf[1 + ic] = zf[ic] + hz[ic];
			++ic;
		}
	}
	//zf[1 + ic] = zf[ic] + stack_layer[ilayer_count - 1].chikness / stack_layer[ilayer_count - 1].idiv_layer;
	//hz[ic] = stack_layer[ilayer_count - 1].chikness / stack_layer[ilayer_count - 1].idiv_layer;

	//hz[ic + 1] = stack_layer[ilayer_count - 1].chikness / stack_layer[ilayer_count - 1].idiv_layer;


	hz[ic] = hz[ic + 1] = (stack_layer[ilayer_count - 1].chikness - qsum) / 2.0;

	zf[1 + ic] = 0.0;
	for (int i7 = 0; i7 < ilayer_count; ++i7) {
		zf[1 + ic] += stack_layer[i7].chikness;
	}
	layer_id[ic] = ilayer_count - 1;
	layer_id[ic + 1] = ilayer_count - 1;
	lambda[ic] = stack_layer[ilayer_count - 1].lambda;
	lambda[ic + 1] = stack_layer[ilayer_count - 1].lambda;
	temp_maxz[ic] = Tamb;
	temp_maxz[ic+1] = Tamb;
	lambda_multiplyer_plane[ic] = stack_layer[ilayer_count - 1].mplane;
	lambda_multiplyer_normal[ic] = stack_layer[ilayer_count - 1].mortho;
	lambda_multiplyer_plane[ic + 1] = stack_layer[ilayer_count - 1].mplane;
	lambda_multiplyer_normal[ic + 1] = stack_layer[ilayer_count - 1].mortho;


	int Nx = M + 1; // < M+2
	int Ny = N + 1; // < N+2
	int Nz = L + 1;

	for (int k = 1; k <= Nz; ++k) {

		//printf("%d %d lam=%e %e\n", k, izstop - 1, lambda[k], hz[k]);
	}
	//getchar();

	doublerealT tmax1 = -1.0e30;

	for (int inl_pass = 0; inl_pass < 3; ++inl_pass) {

		Pdiss_sum = 0.0;

		if (inl_pass == 0) {
			alloc_u3D(source, M, N, L);
			alloc_u3D(b, M, N, L);
		}
		else {
			source = u;
			u = nullptr;
			b = a;
			a = nullptr;
		}

		// инициализация.
		init_u3D<doublerealT>(b, M, N, L, 0.0);
		init_u3D<doublerealT>(source, M, N, L, 0.0);

		const doublerealT power_1_polosok = Pdiss / (n_y * (n_x+1) * n_gx);//0.675;

		int in_x = (int)(distance_x / h1);
		int in_y = (int)(distance_y / h2);

		// Задаём тепловую мощность тепловыделения.
		//for (int i = M / 2 - 117; i < M / 2 + 124; i += 26) {// ПТБШ 1,25мм 2s 204Mb
		//for (int i = M / 2 - 234; i < M / 2 + 240; i += 26) {// ПТБШ 2,5мм 2s 204Mb
		//for (int i = M / 2 - 468; i < M / 2 + 480; i += 26) { // ПТБШ 5мм 2s 204Mb
			//for (int i = M / 2 - 936; i < M / 2 + 946; i += 26) { // ПТБШ 10мм 4s 420Mb
		int istart0 = M / 2 - (int)(0.5 * (in_x * n_xgr));
		int iend0 = istart0 + ((int)((in_x * (n_x)))) + (in_x == 0 ? 3 : 1);

		for (int igr = 0; igr < n_gx; ++igr)
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
						doublerealT Ssource = size_x * size_y;
						doublerealT Vol_source = Ssource * hz[izstop - 1];
						if (n_y == 1) {
							source[i + 1][j][izstop - 1] += power_1_polosok / (Vol_source);
								source[i][j][izstop - 1] += power_1_polosok / (Vol_source);
						}
						else {
							source[i + 1][j][izstop - 1] += power_1_polosok / (Vol_source);
							source[i][j][izstop - 1] += power_1_polosok / (Vol_source);
							source[i + 1][j + 1][izstop - 1] += power_1_polosok / (Vol_source);
							source[i][j + 1][izstop - 1] += power_1_polosok / (Vol_source);

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
						//for (int j = N / 2 - 2; j < N / 2 + 3; ++j) {
					for (int j = (n_y == 1 ? N / 2 - 5 : N / 2 - (int)(0.5 * (in_y * (n_y)))); j < (n_y == 1 ? N / 2 + 5 : N / 2 + (int)(0.5 * (in_y * (n_y)) + (in_y == 0 ? 3 : 1))); j += (n_y == 1 ? 1 : in_y)) {
						//if ((i - (M / 2 - 117)) % 26 == 0)
						{


							//source[i][j][izstop] = 0.675 / (5*h1*h2*hz[izstop-1]);
							doublerealT Ssource = size_x * size_y;
							doublerealT Vol_source = Ssource * hz[izstop - 1];
							if (n_y == 1) {
								source[i + 1][j][izstop - 1] += power_1_polosok / (Vol_source);
								source[i][j][izstop - 1] += power_1_polosok / (Vol_source);
							}
							else {
								source[i + 1][j][izstop - 1] += power_1_polosok / (Vol_source);
								source[i][j][izstop - 1] += power_1_polosok / (Vol_source);
								source[i + 1][j + 1][izstop - 1] += power_1_polosok / (Vol_source);
								source[i][j + 1][izstop - 1] += power_1_polosok / (Vol_source);

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

		IDFTx(b, source, M, N, L);

		copy3D<doublerealT>(source, b, M, N, L, true, false, 0.0);

		IDFTy(b, source, M, N, L);

		//std::cout << "1" << std::endl;

		//free_u3D(source, M, N, L);
		//alloc_u3D(a, M, N, L);

		a = source;
		source = nullptr;

		init_u3D<doublerealT>(a, M, N, L, 0.0);

		// Ядро солвера.
		progonka(Nx, Ny, Nz, lambda, lambda_multiplyer_normal, lambda_multiplyer_plane, hz, h1, h2, b, a);

		//std::cout << "2" << std::endl;

		//free_u3D(b, M, N, L);
		//alloc_u3D(u, M, N, L);

		u = b;
		b = nullptr;


		init_u3D<doublerealT>(u, M, N, L, 0.0);

		DFTx(u, a, M, N, L);

		copy3D<doublerealT>(a, u, M, N, L, false, false, 0.0);

		DFTy(u, a, M, N, L);

		copy3D(u, u, M, N, L, true, true, Tamb);

		//std::cout << "3" << std::endl;

		//free_u3D(a, M, N, L);

		/*tmax1 = -1.0e30;

		for (int i = 0; i < M + 2; ++i) {
			for (int j = 0; j < N + 2; ++j) {
				if (u[i][j][izstop - 1] > tmax1) {
					tmax1 = u[i][j][izstop - 1];
				}
			}
		}*/

#pragma omp parallel for
		for (int i7 = 0; i7 < ilayer_count; ++i7) {
			stack_layer[i7].Tmax = -1.0e30;
		}

#pragma omp parallel for
		for (int k = 0; k < L + 2; ++k) {
			temp_maxz[k] = -1.0e30;
		}

		for (int k = 0; k < L + 2; ++k) {
			for (int i = 0; i < M + 2; ++i) {
				for (int j = 0; j < N + 2; ++j) {
					if (u[i][j][k] > stack_layer[layer_id[k]].Tmax) {
						stack_layer[layer_id[k]].Tmax = u[i][j][k];
					}
				}
			}
		}

#pragma omp parallel for
		for (int k = 0; k < L + 2; ++k) {
			for (int i = 0; i < M + 2; ++i) {
				for (int j = 0; j < N + 2; ++j) {
					if (u[i][j][k] > temp_maxz[k]) {
						temp_maxz[k] = u[i][j][k];
					}
				}
			}
		}

		tmax1 = temp_maxz[izstop - 1];

#pragma omp parallel for
		for (int k = 0; k < L + 2; ++k) {
			lambda[k] = stack_layer[layer_id[k]].lambda * pow(((273.15+ temp_maxz[k])/300.0), stack_layer[layer_id[k]].alpha);
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
		exporttecplot3D22D(u, xf, yf, zf, M, N, izstop - 1);
	}
	if (exportx1D) {
		exporttecplot3D21D(u, xf, yf, zf, M, N, izstop - 1);
	}


	free_u3D(u, M, N, L);

	for (int i = 0; i < NUMBER_THREADS; ++i)
	{
		delete[] a1_gl<doublerealT>[i];
		delete[] wlen_pw_gl<doublerealT>[i];
	}
	delete[] a1_gl<doublerealT>;
	delete[] wlen_pw_gl<doublerealT>;
	a1_gl<doublerealT> = nullptr;
	wlen_pw_gl<doublerealT> = nullptr;

	delete[] stack_layer;

	delete[] xf;
	delete[] yf;
	delete[] zf;
	delete[] hz;
	delete[] lambda;
	delete[] lambda_multiplyer_plane;
	delete[] lambda_multiplyer_normal;
	delete[] layer_id;
	delete[] temp_maxz;

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
    double& thermal_resistance, double size_x, double size_y, double distance_x, double distance_y,
	double size_gx, int n_x, int n_y, int n_gx,
	bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7, bool b8, bool b9,
	double d1, double d2, double d3, double d4, double d5, double d6, double d7, double d8, double d9,
	double k1, double k2, double k3, double k4, double k5, double k6, double k7, double k8, double k9, 
	double rhoCp1, double rhoCp2, double rhoCp3, double rhoCp4, double rhoCp5, double rhoCp6,
	double rhoCp7, double rhoCp8, double rhoCp9,	
	double mplane1, double mplane2, double mplane3,
	double mplane4, double mplane5, double mplane6,
	double mplane7, double mplane8, double mplane9,
	double mortogonal1, double mortogonal2, double mortogonal3,
	double mortogonal4, double mortogonal5, double mortogonal6,
	double mortogonal7, double mortogonal8, double mortogonal9,
	double alpha1, double alpha2, double alpha3,
	double alpha4, double alpha5, double alpha6,
	double alpha7, double alpha8, double alpha9,
	int &time, double Tamb, double Pdiss, 
	bool export3D, bool exportxy2D, bool exportx1D,
	bool bfloat)
{

	double eps = -1.0e-10;

	if (bfloat) {
		// Тип float.

		float thermal_resistancef = (float)(thermal_resistance);

		if (((alpha1 > eps) || (!b1)) && ((alpha2 > eps) || (!b2)) && ((alpha3 > eps) || (!b3))
			&& ((alpha4 > eps) || (!b4)) && ((alpha5 > eps) || (!b5)) && ((alpha6 > eps) || (!b6)) &&
			((alpha7 > eps) || (!b7)) && ((alpha8 > eps) || (!b8)) && ((alpha9 > eps) || (!b9)))
		{

			//thermal_resistance = 1.0;
			FFTsolver3Dqlinear<float>(thermal_resistancef,
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
				export3D, exportxy2D, exportx1D);
		}
		else {
			//thermal_resistance = 1.0;
			FFTsolver3Dqnonlinear<float>(thermal_resistancef,
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
				export3D, exportxy2D, exportx1D);
		}

		thermal_resistance = (double)(thermal_resistancef);

	}
	else {
		// Тип double.

		

		if (((alpha1 > eps) || (!b1)) && ((alpha2 > eps) || (!b2)) && ((alpha3 > eps) || (!b3))
			&& ((alpha4 > eps) || (!b4)) && ((alpha5 > eps) || (!b5)) && ((alpha6 > eps) || (!b6)) &&
			((alpha7 > eps) || (!b7)) && ((alpha8 > eps) || (!b8)) && ((alpha9 > eps) || (!b9)))
		{

			//thermal_resistance = 1.0;
			FFTsolver3Dqlinear<double>(thermal_resistance,
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
				export3D, exportxy2D, exportx1D);
		}
		else {
			//thermal_resistance = 1.0;
			FFTsolver3Dqnonlinear<double>(thermal_resistance,
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
				export3D, exportxy2D, exportx1D);
		}

		
	}	

	

}