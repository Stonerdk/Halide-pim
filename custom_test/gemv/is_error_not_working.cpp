#include "HalideRuntime.h"
#include "iostream"

using namespace std;

extern int halide_error_bad_type(void *user_context, const char *func_name,
                                 uint32_t type_given, uint32_t correct_type); 

int main() {
    int32_t _47 = halide_error_bad_type(nullptr, "Input buffer weight", 737, 73728);
    cout << _47 << endl;
}

//g++ is_error_not_working.cpp -g -std=c++17 -L $HALIDE_DIR/bin/build -I $HALIDE_DIR/include -ljpeg -lpthread -lcurses -ldl -lrt -lz -lm  -o is_error_not_working