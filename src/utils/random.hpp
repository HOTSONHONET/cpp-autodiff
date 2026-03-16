#pragma once
#include <random>

struct RNG {
    static std::mt19937& generator() {
        static std::mt19937 gen(42); // default seed
        return gen;
    }

    static void seed(uint32_t s){
        generator().seed(s);
    }
};