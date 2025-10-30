#include <tblis.h>
#include "marray/marray.hpp"
#include <iostream>

using namespace std;
using namespace MArray;
using namespace MArray::slice;



int main() {
    auto A = marray<float,2>({10,10},1);
    auto B = marray<float,2>({10,10},1);
    auto C = marray<float,2>({10,10});
    label_vector idx_A, idx_B, idx_C;
    
    // mult(1.0,A,"ij",B,"jk",C,"ij");

    cout << A << endl;
    cout << B << endl;
    cout << C << endl;

    return 0;
}