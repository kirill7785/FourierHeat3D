
// Fourier.h - Contains declarations of math functions
#pragma once

#ifdef FOURIER_EXPORTS
#define FOURIER_API __declspec(dllexport)
#else
#define FOURIER_API __declspec(dllimport)
#endif

// Методом быстрого преобразование Фурье решает уравнение Пуассона
// в прямоугольном параллелепипеде для нахождения теплового сопротивления
// планарных структур.
extern "C" FOURIER_API void fourier_solve(
    double &thermal_resistance, double size_x, double size_y, double distance_x, double distance_y,
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
	int &time,
	double Tamb, double Pdiss, bool export3D, bool exportxy2D, bool exportx1D,
	bool bfloat);