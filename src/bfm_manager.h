#ifndef BFM_MANAGER_H
#define BFM_MANAGER_H

#include "../lib/Eigen.h"
#include "constants.hpp"
#include <hdf5.h>
#include <H5Cpp.h>

#include "data.h"

#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <memory>

using Eigen::Matrix;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::Dynamic;


class BFM_Manager{
    public:

        // constructor functions
        BFM_Manager();
        ~BFM_Manager();

        BFM_Manager(const std::string& strBFM_Path, const std::string& landmark_path, const int& BFM_Version);

        /////////////////////////////////////////////////////
        // Set and Get  
        /////////////////////////////////////////////////////

        // get shape pca basis
        inline Eigen::MatrixXd& Get_ShapePc()  {return m_matShapePc;}
        // get shape variance
        inline Eigen::VectorXd& Get_ShapeEv()  {return m_vecShapeEv;}
        // get shape mean
        inline Eigen::VectorXd& Get_ShapeMu()  {return m_vecShapeMu;}
        // get expr pca basis
        inline Eigen::MatrixXd& Get_ExprPc()  {return m_matExprPc;}
        // get expr variance
        inline Eigen::VectorXd& Get_ExprEv()  {return m_vecExprEv;}
        // get expr mean
        inline Eigen::VectorXd& Get_ExprMu()  {return m_vecExprMu;}
        // get tex pca basis
        inline Eigen::MatrixXd& Get_TexPc()  {return m_matTexPc;}
        // get tex variance
        inline Eigen::VectorXd& Get_TexEv()  {return m_vecTexEv;}
        // get tex mean
        inline Eigen::VectorXd& Get_TexMu()  {return m_vecTexMu;}

        // get current face triangles
        inline Eigen::VectorXi& Get_CurrentFaces()  {return m_vecTriangleList;}
        
        // get current landmarks idx
        inline unsigned int Get_LandmarkIdx(int i)  {return m_vecLandmarkIndices[i];}
        

        // get size of shape
        inline unsigned int Get_nVertices()  {return m_nVertices;}
        // get size of expr
        inline unsigned int Get_nExprPcs()  {return m_nExprPcs;}
        // get size of id points
        inline unsigned int Get_nIdPcs()  {return m_nIdPcs;}
        // get size of faces
        inline unsigned int Get_nFaces()  {return m_nFaces;}
        // get size of landmarks
        inline unsigned int Get_nLandmarks()  {return m_vecLandmarkIndices.size();}

    private:

        //////////////////////////////////////////////////////////////
        // Variables declaration
        //////////////////////////////////////////////////////////////
        // file path
        std::string m_model_path = R"(../models/BFM/model2017-1_face12_nomouth.h5)";
        std::string m_landmark_path = R"(../models/BFM/Landmarks68_BFM.anl)";
        int m_BFM_Version = 2017;
        H5::H5File h5_file;


        // H5 dataset path
        // Mu =Mean, Ev = Variance, Pc = pca basis
        // Mu size = 3* nVertices = 85764
        // Ev size = nIdPcs = 199  for color and shape
        // Ev size = nExprPcs = 100 for expression
        // Pc size = 3* nVertices * nIdPcs = 85764 * 199
        std::string m_ShapeMuH5Path = R"(/shape/model/mean)";
        std::string m_ShapeEvH5Path = R"(shape/model/pcaVariance)";
        std::string m_ShapePcH5Path = R"(shape/model/pcaBasis)";
        std::string m_TexMuH5Path =   R"(color/model/mean)";
        std::string m_TexEvH5Path =   R"(color/model/pcaVariance)";
        std::string m_TexPcH5Path =   R"(color/model/pcaBasis)";
        std::string m_ExprMuH5Path =  R"(expression/model/mean)";
        std::string m_ExprEvH5Path =  R"(expression/model/pcaVariance)";
        std::string m_ExprPcH5Path =  R"(expression/model/pcaBasis)";
        std::string m_TriangleListH5Path = R"(shape/representer/cells)";

        unsigned int m_nVertices = 28588;
	    unsigned int m_nFaces = 56572;
	    unsigned int m_nIdPcs = 199;
	    unsigned int m_nExprPcs = 100;

        double *m_rShapeCoef;
        Eigen::VectorXd m_vecShapeMu;
        Eigen::VectorXd m_vecShapeEv;
        Eigen::MatrixXd m_matShapePc;

        double *m_rTexCoef;
        Eigen::VectorXd m_vecTexMu;
        Eigen::VectorXd m_vecTexEv;
        Eigen::MatrixXd m_matTexPc;

        double *m_rExprCoef;
        Eigen::VectorXd m_vecExprMu;
        Eigen::VectorXd m_vecExprEv;
        Eigen::MatrixXd m_matExprPc;

        Eigen::VectorXi m_vecTriangleList;

        // Landmarks variables
        std::vector<int> m_vecLandmarkIndices; 


        //////////////////////////////////////////////////////////////////////////
        // Functions declaration
        //////////////////////////////////////////////////////////////////////////
        // load BFM model
        int Load_H5_file();
        // load landmarks
        int Load_LandmarkIdxs();
        // allocate memory for BFM model
        int Allocate_Memory();
        // load BFM model
        int Load_BFM_Model();

        //extract landmarks from BFM model
        void Extract_Landmarks();

        // data checker
        void check_data();


        template<typename Derived>
	    Matrix<Derived, Dynamic, 1> coef2Object(const Derived *const aCoef, 
		const VectorXd &vecMu, const MatrixXd &matPc, const VectorXd &vecEv, unsigned int nLength) const
	    {
            assert(aCoef != nullptr);
            assert(nLength >= 0);

            Matrix<Derived, Dynamic, 1> tmpCoef(nLength);
            for(auto i = 0u; i < nLength; i++)
                tmpCoef(i) = aCoef[i];

            Matrix<Derived, Dynamic, 1> tmpMu = vecMu.cast<Derived>();
            Matrix<Derived, Dynamic, 1> tmpEv = vecEv.cast<Derived>();
            Matrix<Derived, Dynamic, Dynamic> tmpPc = matPc.cast<Derived>();
            return tmpMu + tmpPc * tmpCoef.cwiseProduct(tmpEv);
	    }
};


#endif // BFM_MANAGER_H