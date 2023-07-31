#include "pca.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <algorithm>

using namespace std;

PCA::PCA(vector<vector<float>> imagen)
{

    matX = creaMatrizX(imagen);

    std::vector<vector<float>> matT = transpose(matX);

    matrizM = multiplyMatrices(matX, matT);

    
}

std::vector<float> PCA::transfCaracteristica(vector<vector<float>> imagen, int cantComps)
{

    std::tuple<std::vector<vector<float>>, std::vector<float>> autoVecsautoVals = metodoDeflacionconPotencia(matrizM, cantComps);

    std::vector<vector<float>> autoVectors = std::get<0>(autoVecsautoVals);

    std::vector<vector<float>> vta = multiplyMatrices(transpose(autoVectors), matX);

    std::vector<vector<float>> autoVecs = dividePorAutovals(vta, std::get<1>(autoVecsautoVals));

    vector<float> result(cantComps);

    for (int i = 0; i < cantComps; i++)
    {
        vector<vector<float>> multi = multiplyMatrices(autoVecs, transpose(imagen));
        result[i] = multi[i][0];
        
    }

    return result;
}

std::vector<vector<float>> PCA::getAutovecs(int cantComps)
{

    std::tuple<std::vector<vector<float>>, std::vector<float>> autoVecsautoVals = metodoDeflacionconPotencia(matrizM, cantComps);

    std::vector<vector<float>> autoVectors = std::get<0>(autoVecsautoVals);

    std::vector<vector<float>> vta = multiplyMatrices(transpose(autoVectors), matX);

    std::vector<vector<float>> autoVecs = dividePorAutovals(vta, std::get<1>(autoVecsautoVals));

    return autoVecs;
}

std::vector<float> vectorDivide(const std::vector<float> vector, float scalar)
{
    std::vector<float> result(vector.size());

    
    for (size_t i = 0; i < vector.size(); i++)
    {
        result[i] = vector[i] / scalar;
    }

    return result;
}

std::vector<vector<float>> dividePorAutovals(std::vector<vector<float>> mat, std::vector<float> autoVals)
{

    for (int i = 0; i < mat.size(); i++)
    {
        mat[i] = vectorDivide(mat[i], std::sqrt(autoVals[i]));
        
    }

    return mat;
}

std::vector<float> vectorSum(const std::vector<float> vector1, const std::vector<float> vector2)
{
    
    if (vector1.size() != vector2.size())
    {
        throw std::runtime_error("Vectors must have the same size.");
    }

    std::vector<float> result(vector1.size());

    
    for (size_t i = 0; i < vector1.size(); i++)
    {
        result[i] = vector1[i] + vector2[i];
    }

    return result;
}

std::vector<float> vectorSub(const std::vector<float> vector1, const std::vector<float> vector2)
{
    
    if (vector1.size() != vector2.size())
    {
        throw std::runtime_error("Vectors must have the same size.");
    }

    std::vector<float> result(vector1.size());

    
    for (size_t i = 0; i < vector1.size(); i++)
    {
        result[i] = vector1[i] - vector2[i];
    }

    return result;
}

std::vector<std::vector<float>> multiplyMatrices(const std::vector<std::vector<float>> matrix1, const std::vector<std::vector<float>> matrix2)
{
    
    if (matrix1[0].size() != matrix2.size())
    {
        //Matrices tienen dimensiones incompatibles
        throw std::runtime_error("Cant mult");

        return {};
    }

    
    size_t numRows = matrix1.size();
    size_t numCols = matrix2[0].size();
    size_t innerSize = matrix2.size();

    
    std::vector<std::vector<float>> result(numRows, std::vector<float>(numCols, 0.0));

    
    for (size_t i = 0; i < numRows; i++)
    {
        for (size_t j = 0; j < numCols; j++)
        {
            for (size_t k = 0; k < innerSize; k++)
            {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return result;
}

std::vector<vector<float>> armaMatriz(std::vector<vector<float>> vectorvector)
{

    std::vector<vector<float>> mat(vectorvector.size()); // cols es el tam de un vector, rows es la cantidad de imagenees que quiero usar

    for (int i = 0; i < vectorvector.size(); i++)
    {
        for (int h = 0; h < vectorvector[i].size(); h++)
        {
            mat[i].push_back(vectorvector[i][h]);
        }
    }
    return mat;
}

std::vector<vector<float>> creaMatrizX(std::vector<vector<float>> mat)
{

    std::vector<float> rowVector(mat[0].size());
    
    for (int i = 0; i < (mat.size()); i++)
    {
        rowVector = vectorSum(rowVector, mat[i]);
    }

    rowVector = vectorDivide(rowVector, mat.size());

    for (int i = 0; i < (mat.size()); i++)
    {
        mat[i] = vectorSub(mat[i], rowVector);
    }
    
    float raizn = std::sqrt(mat.size() - 1);

    for (int i = 0; i < (mat.size()); i++)
    {
        mat[i] = vectorDivide(mat[i], raizn);
    }

    
    return mat;
}

float norma2(std::vector<vector<float>> vector)
{

    float acum = 0;
    for (int i = 0; i < vector.size(); i++)
    {
        acum += vector[i][0] * vector[i][0];
    }

    if (acum == 0)
    {
        return 1;
    }

    return std::sqrt(acum);
}

std::vector<vector<float>> transpose(vector<vector<float>> mat)
{

    std::vector<vector<float>> matT(mat[0].size());
    std::vector<float> vectorcero(mat.size());

    for (int i = 0; i < matT.size(); i++)
    {
        matT[i] = vectorcero;
    }

    for (int i = 0; i < mat.size(); i++)
    {
        for (int h = 0; h < mat[i].size(); h++)
        {
            matT[h][i] = mat[i][h];
        }
    }

    return matT;
}

std::vector<std::vector<float>> subtractMatrices(std::vector<std::vector<float>> matrix1,
                                                 std::vector<std::vector<float>> matrix2)
{
    int rows = matrix1.size();
    int cols = matrix1[0].size();

    std::vector<std::vector<float>> result(rows, std::vector<float>(cols));

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result[i][j] = matrix1[i][j] - matrix2[i][j];
        }
    }

    return result;
}

std::vector<std::vector<float>> multiplyMatrixByScalar(std::vector<std::vector<float>> matrix,
                                                       float scalar)
{
    int rows = matrix.size();
    int cols = matrix[0].size();

    std::vector<std::vector<float>> result(rows, std::vector<float>(cols));

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result[i][j] = matrix[i][j] * scalar;
        }
    }

    return result;
}

std::tuple<std::vector<vector<float>>, float> metodoPotencia(std::vector<vector<float>> mat)
{

    float epsilon = 0.001;
    vector<float> vacio(1);
    vector<vector<float>> initVector;
    for (int i = 0; i < mat[0].size(); i++)
    {
        initVector.push_back(vacio);
    }

    // Valor seed
    srand((unsigned)time(NULL));

    for (int i = 0; i < initVector.size(); i++)
    {
        initVector[i][0] = (1); // rand() % 101);
    }

    
    float autoVal1 = 0;
    float autoVal2 = 0;
    int i = 0;
    do
    {
        vector<vector<float>> previousVector = initVector;
        vector<vector<float>> vectorMult = multiplyMatrices(mat, initVector);

        initVector = vectorMult;
        for (int h = 0; h < initVector.size(); h++)
        {
            initVector[h] = vectorDivide(initVector[h], norma2(vectorMult));
        }

        std::vector<vector<float>> autoVecxEscalar1 = multiplyMatrices(mat, initVector);
        autoVal1 = 0;
        int contadorceros = 0;
        for (int h = 0; h < initVector.size(); h++)
        {
            if (initVector[h][0] == 0)
            {
                contadorceros++;
            }
            else
            {
                autoVal1 += autoVecxEscalar1[h][0] / initVector[h][0];
            }
        }
        if (contadorceros != autoVecxEscalar1.size())
        {
            autoVal1 = autoVal1 / (autoVecxEscalar1.size() - contadorceros);
        }
        contadorceros = 0;

        std::vector<vector<float>> autoVecxEscalar2 = multiplyMatrices(mat, previousVector);
        autoVal2 = 0;
        for (int h = 0; h < previousVector.size(); h++)
        {
            if (previousVector[h][0] == 0)
            {
                contadorceros++;
            }
            else
            {
                autoVal2 += autoVecxEscalar2[h][0] / previousVector[h][0];
            }
        }
        if (contadorceros != autoVecxEscalar2.size())
        {
            autoVal2 = autoVal2 / autoVecxEscalar2.size();
        }

        i++;
    } while (std::abs(autoVal1 - autoVal2) > epsilon);

    std::tuple<std::vector<vector<float>>, float> autoVecautoVal(initVector, autoVal1);
    return autoVecautoVal;
}

std::tuple<std::vector<vector<float>>, std::vector<float>> metodoDeflacionconPotencia(std::vector<vector<float>> mat, int cantVals)
{

    std::vector<vector<float>> autovecsMat(mat[0].size());

    std::vector<float> vectorcero(cantVals);

    for (int i = 0; i < autovecsMat.size(); i++)
    {
        autovecsMat[i] = vectorcero;
    }

    std::vector<float> autovalsMat;

    for (int i = 0; i < cantVals; i++)
    {
        std::tuple<std::vector<vector<float>>, float> autoVecautoVal = metodoPotencia(mat);
        std::vector<vector<float>> autoVec = std::get<0>(autoVecautoVal);
        for (int h = 0; h < autoVec.size(); h++)
        {
            autovecsMat[h][i] = autoVec[h][0];
        }
        float autoVal = std::get<1>(autoVecautoVal);
        //std::cout << autoVal << " autoval";
        autovalsMat.push_back(autoVal);
        // A' = A - λ1 * v1 * v1^T
        vector<vector<float>> matVector = multiplyMatrices(autoVec, transpose(autoVec));
        matVector = multiplyMatrixByScalar(matVector, autoVal);
        mat = subtractMatrices(mat, matVector);
    }

    std::tuple<std::vector<vector<float>>, std::vector<float>> autoVecautoVal(autovecsMat, autovalsMat);

    return autoVecautoVal;
}
