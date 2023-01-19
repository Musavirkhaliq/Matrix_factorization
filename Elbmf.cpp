#include <random>
#include <algorithm>
// #include <cuda_runtime.h>
// #include <cusparse.h>

using namespace std;
// using namespace cusparse;
// using namespace cuda;

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
    return max(proxel_1(x, k, l), 0);
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
    return min(max(proxel_1(x, k, l), 0), 1);
}

// prox_elbmf_box
template<typename T>
void prox_elbmf_box(vector<T> &X, double k, double l) {
    for (auto &x : X) {
        x = proxelb(x, k, l);
    }
}


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
        prox_elbmf(X, 0.5, 1e20);
        X = round(clamp(X, 0, 1));
    }
}

template<typename T>
void rounding(T& fn, vector<T>& args) {
    // do nothing
}


template<typename T>
void apply_rate(ElasticBMF<T> &fn, ElasticBMF<T> &fn0, T nu) {
    fn.l2reg = fn0.l2reg * nu;
}

template<typename T>
void apply_rate(T &fn, T &fn0, T nu) {
    // do nothing
}


template<typename T>
cuda::array<T> gpu(vector<T> &X) {
    return cuda::array<T>(X);
}

template<typename T>
vector<T> cpu(cuda::array<T> &X) {
    return vector<T>(X);
}


template<typename T>
void reducemf_impl(T (*prox)(vector<T> &, T), PALM, vector<T> &A, vector<T> &U, vector<T> &V) {
    auto VVt = V * V;
    auto AVt = A * V;
    auto grad = [](vector<T> &x) {return x * VVt - AVt;};
    T L = max(norm(VVt), 1e-4);

    T step_size = 1 / (1.1 * L);
    U = U - grad(U) * step_size;
    prox(U, step_size);
}

template<typename T>
void reducemf_impl(T (*prox)(vector<T> &, T), iPALM<T> &opt, vector<T> &A, vector<T> &U, vector<T> &V, vector<T> &U_) {
    auto VVt = V * V;
    auto AVt = A * V;
    auto grad = [](vector<T> &x) {return x * VVt - AVt;};
    T L = max(norm(VVt), 1e-4);

    for (int i = 0; i < U.size(); i++) {
        U[i] += opt.beta * (U[i] - U_[i]);
        U_[i] = U[i];
    }

    T step_size = 2 * (1 - opt.beta) / (1 + 2 * opt.beta) / L;
    U = U - grad(U) * step_size;
    prox(U, step_size);
}

template<typename T>
void reducemf(ElasticBMF<T> &fn, PALM opt, vector<T> &A, vector<T> &U, vector<T> &V) {
    auto prox = [fn](vector<T> &x, T alpha) {prox_elbmf(x, fn.l1reg * alpha, fn.l2reg * alpha);};
    reducemf_impl(prox, opt, A, U, V);
}

template<typename T>
void reducemf(ElasticBMF<T> &fn, iPALM<T> &opt, vector<T> &A, vector<T> &U, vector<T> &V, vector<T> &U_) {
    auto prox = [fn](vector<T> &x, T alpha) {prox_elbmf(x, fn.l1reg * alpha, fn.l2reg * alpha);};
    reducemf_impl(prox, opt, A, U, V, U_);
}

template<typename T>
pair<vector<T>, vector<T>> factorize_palm(ElasticBMF<T> &fn, vector<T> &X, vector<T> &U, vector<T> &V, function<T(T)> regularization_rate, int maxiter, T tol, function<void(pair<vector<T>, vector<T>>, T)> callback = nullptr) {
    T ell = numeric_limits<T>::max();
    ElasticBMF<T> fn0 = fn;

    for (int t = 0; t < maxiter; t++) {
        fn.l2reg = fn0.l2reg * regularization_rate(t);

        reducemf(fn, PALM(), X, U, V);
        reducemf(fn, PALM(), transpose(X), transpose(V), transpose(U));

        T ell0 = ell;
        ell = norm(X - U * V, 2);

        if (callback) {callback({U, V}, ell);}
        if (abs(ell - ell0) < tol) {break;}
    }
    fn = fn0;
    return {U, V};
}


template<typename T>
pair<vector<T>, vector<T>> factorize_ipalm(ElasticBMF<T> &fn, vector<T> &X, vector<T> &U, vector<T> &V, function<T(T)> regularization_rate, int maxiter, T tol, T beta, function<void(pair<vector<T>, vector<T>>, T)> callback = nullptr) {
    if (beta == 0) {
        return factorize_palm(fn, X, U, V, regularization_rate, maxiter, tol, callback);
    }

    T ell = numeric_limits<T>::max();
    ElasticBMF<T> fn0 = fn;

    iPALM<T> ipalm = {beta};
    vector<T> U_ = U;
    vector<T> Vt_ = transpose(V);

    for (int t = 0; t < maxiter; t++) {
        fn.l2reg = fn0.l2reg * regularization_rate(t);

        reducemf(fn, ipalm, X, U, V, U_);
        reducemf(fn, ipalm, transpose(X), transpose(V), transpose(U), Vt_);

        T ell0 = ell;
        ell = norm(X - U * V, 2);

        if (callback) {callback({U, V}, ell);}
        if (abs(ell - ell0) < tol) {break;}
    }
    fn = fn0;
    return {U, V};
}


template<typename T>
void batched_factorize_ipalm(ElasticBMF<T> &fn, vector<T> &X, vector<T> &U, vector<T> &V, 
                              function<T(int)> regularization_rate, int maxiter, T tolerance, T beta, int batchsize, 
                              function<void(pair<vector<T>, vector<T>>, ElasticBMF<T>&)> callback = nullptr) {
    T ell = numeric_limits<T>::max();
    ElasticBMF<T> fn0 = fn;
    vector<T> U_, H, Ht_;
    if (beta != 0) {
        U_ = U;
    }
    H = V;
    Ht_ = H;

    for (int t = 0; t < maxiter; t++) {
        ell = 0;
        fn.l2reg = fn0.l2reg * regularization_rate(t);

        for (int i = 0; i < X.size(); i += batchsize) {
            auto batch = make_pair(i, min(i + batchsize, (int)X.size()));
            vector<T> A = vector<T>(X.begin() + batch.first, X.begin() + batch.second);
            vector<T> W = vector<T>(U.begin() + batch.first, U.begin() + batch.second);
            if (beta != 0) {
                vector<T> W_ = vector<T>(U_.begin() + batch.first, U_.begin() + batch.second);
                reducemf(fn, iPALM<T>{beta}, A, W, H, W_);
                reducemf(fn, iPALM<T>{beta}, A, W, H, W_);
                copy(W_.begin(), W_.end(), U_.begin() + batch.first);
            } else {
                reducemf(fn, PALM, A, W, H);
                reducemf(fn, PALM, A, W, H);
            }
            ell += norm(A - W * H)^2;
            copy(W.begin(), W.end(), U.begin() + batch.first);
        }

        if (callback != nullptr) {
            callback(make_pair(U, H), fn);
        }
        if (abs(ell - ell0) < tolerance) {
            break;
        }
    }
    V = H;
    fn = fn0;
}






