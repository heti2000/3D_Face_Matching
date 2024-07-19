#ifndef DATA_H
#define DATA_H

#include <memory> 
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <utility>

#include "hdf5.h"
#include "H5Cpp.h"
#include <random>
#include <time.h>
#include <cassert>
#include "../lib/Eigen.h"

using namespace std;
using namespace H5;
using namespace Eigen;


namespace bfm_utils
{

	template<typename T, typename E>
	void Raw2Mat(Eigen::MatrixBase<E>& matData, const std::unique_ptr<T[]>& aData) {
		const unsigned int rows = matData.rows();
		const unsigned int cols = matData.cols();
		const unsigned int stride = cols;

		for (unsigned int i = 0; i < rows; ++i) {
			for (unsigned int j = 0; j < cols; ++j) {
				matData(i, j) = aData[i * stride + j];
			}
		}
	}

	template<typename T, typename E>
	void LoadH5Model(hid_t file, const std::string& strPath, std::unique_ptr<T[]>& aData, Eigen::MatrixBase<E>& matData, hid_t predType) {
		hid_t dataSet = H5Dopen(file, strPath.c_str(), H5P_DEFAULT);
		herr_t status = H5Dread(dataSet, predType, H5S_ALL, H5S_ALL, H5P_DEFAULT, aData.get());
		Raw2Mat(matData, aData);
		
	}

	inline double *randn(int nArray, double dScale)
	{
		assert(dScale >= 0.0);

		std::random_device randomDevice;
		std::mt19937 generator(randomDevice());
		double *dResArray = new double[nArray];
		std::normal_distribution<double> dis(0, dScale);

		for (int i = 0; i < nArray; i++)
			dResArray[i] = dis(generator);
		
		return dResArray;
	}


} // NAMESPACE BFM_UTILS

#endif 

