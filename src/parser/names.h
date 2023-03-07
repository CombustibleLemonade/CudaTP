#ifndef NAMES
#define NAMES

#include <map>
#include <string>

class TextInput32 {
public:
    char text[32];

    __host__ __device__ TextInput32(const char* text);

    __host__ __device__ void embed(float* output);
};

class Names{
    std::map<std::string, std::string> dictionary;
public:
    static Names names;

    Names();

    std::string operator[](int idx);    
};

#endif