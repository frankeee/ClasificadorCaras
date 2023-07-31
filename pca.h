#ifndef ARMAMATRIZ_H
#define ARMAMATRIZ_H

using namespace std;

#include <vector>

class PCA
{
private:
    std::vector<vector<float>> matrizM;
    std::vector<vector<float>> matX;

public:
    PCA(vector<vector<float>> imagenes);
    std::vector<float> transfCaracteristica(vector<vector<float>> imagen, int cantComps);
    std::vector<vector<float>> getAutovecs(int cantAutovecs);
};
std::vector<vector<float>> armaMatriz(std::vector<vector<float>> vectorvector);
std::vector<vector<float>> creaMatrizX(std::vector<vector<float>> mat);
std::vector<float> vectorDivide(const std::vector<float> vector, float scalar);
std::vector<float> vectorSum(const std::vector<float> vector1, const std::vector<float> vector2);
std::vector<float> vectorSub(const std::vector<float> vector1, const std::vector<float> vector2);
std::vector<std::vector<float>> multiplyMatrices(const std::vector<std::vector<float>> matrix1, const std::vector<std::vector<float>> matrix2);
std::tuple<std::vector<vector<float>>, float> metodoPotencia(std::vector<vector<float>> mat);
float norma2(std::vector<vector<float>> vector);
std::tuple<std::vector<vector<float>>, std::vector<float>> metodoDeflacionconPotencia(std::vector<vector<float>> mat, int cantAutovals);
std::vector<vector<float>> transpose(vector<vector<float>> mat);
std::vector<std::vector<float>> subtractMatrices(std::vector<std::vector<float>> matrix1,
                                                 std::vector<std::vector<float>> matrix2);
std::vector<std::vector<float>> multiplyMatrixByScalar(std::vector<std::vector<float>> matrix,
                                                       float scalar);
std::vector<vector<float>> dividePorAutovals(std::vector<vector<float>> mat, std::vector<float> autoVals);

#endif // ARMAMATRIZ_H