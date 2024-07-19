#ifndef DEBUG_H
#define DEBUG_H

#include <iostream>
#include <unordered_map>
#include <array>

class Debug {
public:
    void initKey(const std::string key, int size) {
        float* data = new float[size];
        debugData[key] = data;
    }

    // override the [] operator to access the data
    float* operator[](const std::string key) {
        // throw an error if the key does not exist
        return debugData[key];
    }

    // get the debug element
    static Debug& getInstance() {
        static Debug instance;
        return instance;
    }

    void saveToFile(const std::string key, const std::string filename, int size) {
        std::ofstream file(filename);
        for (int i = 0; i < size; i++) {
            file << debugData[key][i] << std::endl;
        }
        file.close();
    }
    
    // deconstructor to free memory
    ~Debug() {
        for (auto it = debugData.begin(); it != debugData.end(); it++) {
            delete it->second;
        }
    }

private:
    std::unordered_map<std::string, float*> debugData;
};

#endif // DEBUG_H