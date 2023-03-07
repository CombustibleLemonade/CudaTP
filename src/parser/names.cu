#include "names.h"

#include <sstream>
#include <fstream>

#include <iostream>
#include <iomanip>

__host__ __device__ TextInput32::TextInput32(const char* _text){
    bool has_string_terminated = false;

    for (int i = 0; i < 32; i++) {
        text[i] = has_string_terminated ? ' ' : _text[i];
        if (_text[i] == '\0') has_string_terminated = true;
    }
}

__host__ __device__ void TextInput32::embed(float* output) {
    for (int i = 0; i < 32; i++) {
        output[i] = float(int(text[i] - 97)) / 10;
    }
}

Names Names::names;

Names::Names() {
    std::ifstream file_stream("./data/names/names");
    std::string line;

    std::cout << "Loading names" << std::endl;
    while (std::getline(file_stream, line)) {
        std::size_t split = line.find(":");

        std::string key = line.substr(0, split);
        std::string value = line.substr(split + 1);

        dictionary[key] = value;
    }
}

std::string Names::operator[](int idx){
    std::string key;
    std::stringstream stream;

    stream << std::setfill('0') << std::setw(8) << std::hex << idx;
    stream >> key;
    key = "0x" + key;

    // std::cout << key << std::endl;

    if (dictionary.find(key)!=dictionary.end()){
        return dictionary[key];
    } else {
        throw;
    }
}