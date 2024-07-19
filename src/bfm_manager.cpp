#include "bfm_manager.h"

using namespace H5;
using namespace std;

BFM_Manager::~BFM_Manager()
{
    //dtor
}

BFM_Manager::BFM_Manager()
{
    
    Load_H5_file();
    Load_LandmarkIdxs();
    Allocate_Memory();
    Load_BFM_Model();
    if (CHECK_BFM_DATA) {
        check_data();
    }
}

BFM_Manager::BFM_Manager(const std::string& strBFM_Path, const std::string& landmark_path, const int& BFM_Version)
: m_model_path(strBFM_Path), m_landmark_path(landmark_path), m_BFM_Version(BFM_Version)
{
    // Load BFM model
    Load_H5_file();
    Load_LandmarkIdxs();
    Allocate_Memory();
    Load_BFM_Model();
    
}

int BFM_Manager::Load_H5_file(){
    //load H% file
    std::cout << "Loading H5 file..." << std::endl;
    try {
        this->h5_file = H5File(m_model_path, H5F_ACC_RDONLY);
              

    } catch (FileIException &error) {
        error.printErrorStack();
        return -1;
    } catch (DataSetIException &error) {
        error.printErrorStack();
        return -1;
    } catch (DataSpaceIException &error) {
        error.printErrorStack();
        return -1;
    } catch (DataTypeIException &error) {
        error.printErrorStack();
        return -1;
    }
    std::cout << "H5 file loaded." << std::endl;
    return 0;
}

int BFM_Manager::Load_LandmarkIdxs(){
    std::ifstream inFile;
    inFile.open(m_landmark_path, std::ios::in);
    assert(inFile.is_open());
    int bfmIdx;
    while(inFile >> bfmIdx)
    {
        m_vecLandmarkIndices.push_back(std::move(bfmIdx));
    }
    inFile.close();
    std::cout << "Landmark indices loaded. " << m_vecLandmarkIndices.size() << std::endl;
    return 0;
}

int BFM_Manager::Allocate_Memory(){
    std::cout <<" Allocating memory..." << std::endl;
    // Allocate memory
    m_rShapeCoef = new double[m_nIdPcs];
	std::fill(m_rShapeCoef, m_rShapeCoef + m_nIdPcs, 0.0);
    m_vecShapeMu.resize(m_nVertices * 3);
    m_vecShapeEv.resize(m_nIdPcs);
    m_matShapePc.resize(m_nVertices * 3, m_nIdPcs);

    m_rTexCoef = new double[m_nIdPcs];
    std::fill(m_rTexCoef, m_rTexCoef + m_nIdPcs, 0.0);
    m_vecTexMu.resize(m_nVertices * 3);
    m_vecTexEv.resize(m_nIdPcs);
    m_matTexPc.resize(m_nVertices * 3, m_nIdPcs);

    m_rExprCoef = new double[m_nExprPcs];
    std::fill(m_rExprCoef, m_rExprCoef + m_nExprPcs, 0.0);
    m_vecExprMu.resize(m_nVertices * 3);
    m_vecExprEv.resize(m_nExprPcs);
    m_matExprPc.resize(m_nVertices * 3, m_nExprPcs);

    m_vecTriangleList.resize(m_nFaces * 3);

    std::cout << "Memory allocated." << std::endl;

    return 0;
}

int BFM_Manager::Load_BFM_Model(){
    std::cout << "Loading BFM model..." << std::endl;
    try{
        std::unique_ptr<float[]> vecShapeMu(new float[m_nVertices * 3]);
		std::unique_ptr<float[]> vecShapeEv(new float[m_nIdPcs]);
		std::unique_ptr<float[]> matShapePc(new float[m_nVertices * 3 * m_nIdPcs]);
		std::unique_ptr<float[]> vecTexMu(new float[m_nVertices * 3]);
		std::unique_ptr<float[]> vecTexEv(new float[m_nIdPcs]);
		std::unique_ptr<float[]> matTexPc(new float[m_nVertices * 3 * m_nIdPcs]);
		std::unique_ptr<float[]> vecExprMu(new float[m_nVertices * 3]);
		std::unique_ptr<float[]> vecExprEv(new float[m_nExprPcs]);
		std::unique_ptr<float[]> matExprPc(new float[m_nVertices * 3 * m_nExprPcs]);
		std::unique_ptr<unsigned short[]> vecTriangleList(new unsigned short[m_nFaces * 3]);

        // Load BFM model
        bfm_utils::LoadH5Model(h5_file.getId(), m_ShapeMuH5Path,vecShapeMu, m_vecShapeMu, H5T_NATIVE_FLOAT);
        bfm_utils::LoadH5Model(h5_file.getId(), m_ShapeEvH5Path,vecShapeEv, m_vecShapeEv, H5T_NATIVE_FLOAT);
        bfm_utils::LoadH5Model(h5_file.getId(), m_ShapePcH5Path,matShapePc, m_matShapePc, H5T_NATIVE_FLOAT);
        bfm_utils::LoadH5Model(h5_file.getId(), m_TexMuH5Path,vecTexMu, m_vecTexMu, H5T_NATIVE_FLOAT);
        bfm_utils::LoadH5Model(h5_file.getId(), m_TexEvH5Path,vecTexEv, m_vecTexEv, H5T_NATIVE_FLOAT);
        bfm_utils::LoadH5Model(h5_file.getId(), m_TexPcH5Path,matTexPc, m_matTexPc, H5T_NATIVE_FLOAT);
        bfm_utils::LoadH5Model(h5_file.getId(), m_ExprMuH5Path,vecExprMu, m_vecExprMu, H5T_NATIVE_FLOAT);
        bfm_utils::LoadH5Model(h5_file.getId(), m_ExprEvH5Path,vecExprEv, m_vecExprEv, H5T_NATIVE_FLOAT);
        bfm_utils::LoadH5Model(h5_file.getId(), m_ExprPcH5Path,matExprPc, m_matExprPc, H5T_NATIVE_FLOAT);
        bfm_utils::LoadH5Model(h5_file.getId(), m_TriangleListH5Path,vecTriangleList, m_vecTriangleList, H5T_NATIVE_USHORT);



        std::cout << "BFM model loaded." << std::endl;
    }
    catch (std::bad_alloc& ba) {
        std::cout<<"bad_alloc caught: "<<ba.what()<<std::endl;
        return -1;
    } 

    return 0;
}

void BFM_Manager::check_data(){
    std::cout << "Checking data..." << std::endl;
    std::cout << "Triangle list: " <<std::endl;
    for (int i = 0; i < m_nFaces; i++){
        std::cout << m_vecTriangleList(3*i) << "," << m_vecTriangleList(3*i+1) << "," << m_vecTriangleList(3*i+2) << std::endl;
    }
    std::cout << "vector size: " <<std::endl;
    std::cout << m_vecTriangleList.size() << std::endl;
}


