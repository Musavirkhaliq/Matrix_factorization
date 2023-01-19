#include<iostream>
#include<stdio.h>
#include<math.h>
#include<vector>
#include<cmath>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <algorithm>

using namespace std;
// integrality_gap_elastic
double integrality_gap_elastic(double e, double kappa, double lambda) {
    return min(kappa * abs(e) + lambda * pow(e, 2), kappa * abs(e - 1) + lambda * pow(e - 1, 2));
}

// regularizer_elbmf
template<typename T>
double regularizer_elbmf(const vector<T> &x, double l1reg, double l2reg) {
    double sum = 0;
    for (const auto &e : x) {
        sum += integrality_gap_elastic(e, l1reg, l2reg);
    }
    return sum;
}


// proxel_1
double proxel_1(double x, double k, double l) {
    if (x <= 0.5) {
        return (x - k * copysign(1, x)) / (1 + l);
    } else {
        return (x - k * copysign(1, x - 1) + l) / (1 + l);
    }
}


// proxelp
double proxelp(double x, double k, double l) {
    return max(proxel_1(x, k, l), 0.0);
}


// prox_elbmf
template<typename T>
void prox_elbmf(vector<T> &X, double k, double l) {
    for (auto &x : X) {
        x = proxelp(x, k, l);
    }
}


// proxelb
double proxelb(double x, double k, double l) {
    return min(max(proxel_1(x, k, l), 0.0), 1.0);
}

// prox_elbmf_box
template<typename T>
void prox_elbmf_box(vector<T> &X, double k, double l) {
    for (auto &x : X) {
        x = proxelb(x, k, l);
    }
}



// CPU to GPU

// void cpu(float* X, size_t size) {
//     cudaError_t error;
//     float* d_X;
//     error = cudaMalloc((void**)&d_X, size*sizeof(float));
//     if (error != cudaSuccess) {
//         // handle error
//     }
//     error = cudaMemcpy(d_X, X, size*sizeof(float), cudaMemcpyHostToDevice);
//     if (error != cudaSuccess) {
//         // handle error
//     }
//     // Do calculations on d_X
//     error = cudaFree(d_X);
//     if (error != cudaSuccess) {
//         // handle error
//     }
// }

// GPU to CPU

template <typename T>
struct ElasticBMF {
    T l1reg;
    T l2reg;
};


struct PALM {};

template <typename T>
struct iPALM {
    T beta;
};

template<typename T>
void rounding(ElasticBMF<T>& fn, vector<T>& args) {
    for (auto &X : args) {
        cout <<"hey";
        cout << X << endl;
    }
    // for (auto &X : args) {
    //     prox_elbmf(X, 0.5, 1e20);
    //     X = round(clamp(X, 0, 1));
    // }
}

int main()
{
    /* Testing first 2 functions
    vector<int> vect{ 1, 2, 3,4}; 
    double n =regularizer_elbmf(vect, 0.1, 0.2);
    cout << n;  */

    /*Testing Proxel*/

    // double n = proxel_1(18, 0.1, 0.2);
    // cout << n;
    // double n = proxelp(18, 0.1, 0.2);
    // cout << n;
    ElasticBMF<double> a;
    a.l1reg = 0.1;
    a.l2reg = 0.2;
    vector<double> A{ 1.0,2.4,3.4}; 
    vector<double> U{ 1.0,2.0,3.0}; 
    vector<double> V{ 1.5,6.3,1,2}; 
    vector<vector<double>> vec{A,U,V};
    rounding(a,A);
    // for (int i = 0; i < vec.size(); i++) {
        // for (int j = 0; j < vec[i].size(); j++) {
            cout <<A<< " ";
        // }
        cout << endl;
    // }
    // prox_elbmf_box(vect, 0.1, 0.2);
    // for(double i:vect)
    // {
    //     cout << i << endl;
    // }
    // // cout<<proxelb(0.5, 0.1, 0.2);
}
