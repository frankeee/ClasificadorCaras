#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>
#include <unistd.h>
#include "pgmconverter.h"
using namespace std;

std::vector<float> readPGMFile(const std::string &directory, const std::string &fileName)
{
    // std::vector<unsigned char> imageData;

    
    std::string filePath = directory + "/" + fileName;

    // Open the PGM file
    std::ifstream file(filePath, std::ios::binary);

    if (!file)
    {
        std::cerr << "No se pudo abrir: " << fileName << std::endl;
        return {};
    }

    std::string format;
    int width, height, maxGrayValue;

    file >> format >> width >> height >> maxGrayValue;

    if (format != "P5")
    {
        std::cerr << "Formato invalido: " << format << std::endl;
        return {};
    }

    std::vector<float> pixelValues;
    pixelValues.reserve(width * height);

    char pixelValue;
    int z = 0;
    while (file.read(&pixelValue, 1))
    {
        if (z != 0)
        {
            pixelValues.push_back(static_cast<float>(pixelValue));
        }
        z++;
    }

    return pixelValues;
}
