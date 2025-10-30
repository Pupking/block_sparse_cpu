extern "C" {
    #include <../tblis/tblis.h>
}

int main() {
    double data_A[10*10];
    for(int i = 0; i < 100; ++i) data_A[i] = 1.0;
    //tblis_tensor A;
    // tblis_init_tensor_d(&A, 2, (len_type[]){10, 10},
    //                     data_A, (stride_type[]){1, 10});

    // double data_B[10*10];
    // for(int i = 0; i < 100; ++i) data_B[i] = 1.0;
    // tblis_tensor B;
    // tblis_init_tensor_d(&B, 2, (len_type[]){10,10},
    //                     data_B, (stride_type[]){1, 10});

    // double data_C[10*10];
    // for(int i = 0; i < 100; ++i) data_C[i] = 0.0;
    // tblis_tensor C;
    // tblis_init_tensor_d(&C, 2, (len_type[]){10,10},
    //                     data_C, (stride_type[]){1,10});

    // // initialize data_A and data_B...

    // // this computes C[abcd] += A[cebf] B[afed]
    // tblis_tensor_mult(NULL, NULL, &A, "ab", &B, "bd", &C, "ad");
    // printf("C = \n");
    // for (int i = 0; i < 10; i++) {
    //     for (int j = 0; j < 10; j++) {
    //         printf("%8.3f ", data_C[i*10 + j]);
    //     }
    //     printf("\n");
    // }
    return 0;
}