#ifndef MODEL_TEST_HELPERS_H
#define MODEL_TEST_HELPERS_H
#include <Eigen/Dense>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <string>
#include <cstddef>
#include <fstream>

struct MMapVector {
    int fd = -1;
    size_t size = 0;
    double* data = nullptr;
};

// Convert std::vector to Eigen::VectorXd
Eigen::VectorXd vectorToEigen(const std::vector<double>& v);

Eigen::Map<const Eigen::VectorXd> vectorToEigen(const MMapVector& mv);

// Function to calculate the Pearson correlation coefficient
double correlation_coefficient(const std::vector<double>& x, const std::vector<double>& y);

template <typename VectorTypeX, typename VectorTypeY>
double correlation_coefficient(const VectorTypeX& x, const VectorTypeY& y) {
    if (x.size != y.size || x.size == 0) {
        throw std::invalid_argument("Vectors must be of same size and non-empty");
    }

    auto X = vectorToEigen(x);
    auto Y = vectorToEigen(y);

    double mean_X = X.mean();
    double mean_Y = Y.mean();

    Eigen::VectorXd X_centered = X.array() - mean_X;
    Eigen::VectorXd Y_centered = Y.array() - mean_Y;

    double covariance = (X_centered.dot(Y_centered)) / (X.size() - 1);
    double stddev_X = std::sqrt(X_centered.squaredNorm() / (X.size() - 1));
    double stddev_Y = std::sqrt(Y_centered.squaredNorm() / (Y.size() - 1));

    if (stddev_X == 0 || stddev_Y == 0) return 0;

    return covariance / (stddev_X * stddev_Y);
}

double latentDistanceCalculator(const std::vector<double>& a, const std::vector<double>& b, int latDim);

double latentMinDistanceCalculator(const std::vector<double>& a, const std::vector<double>& b, int latDim);

double latentMinDistanceCalculatorEigen(const Eigen::VectorXf& a, const Eigen::VectorXf& b, int latDim);


template <typename T>
std::vector<double> flattenAndRemoveNAN(const std::vector<std::vector<T>>& matrix) {
    std::vector<double> out;
    for (const auto& row : matrix) {
        for (const auto& val : row) {
            if (!std::isnan(static_cast<double>(val))) {
                out.push_back(static_cast<double>(val));
            }
        }
    }
    return out;
}

template <typename T>
std::vector<double> flattenAndRemoveNANAndFree(std::vector<std::vector<T>>& matrix) {
    std::vector<double> out;
    for (auto& row : matrix) {
        for (auto& val : row) {
            double d = static_cast<double>(val);
            if (!std::isnan(d)) {
                out.push_back(d);
            }
        }
        std::vector<T>().swap(row);
    }
    std::vector<std::vector<T>>().swap(matrix);
    return out;
}

template <typename T>
std::vector<std::vector<double>> to_double_vector(const std::vector<std::vector<T>>& v) {
    std::vector<std::vector<double>> result;
    result.reserve(v.size());
    for (const auto& row : v) {
        std::vector<double> new_row;
        new_row.reserve(row.size());
        for (const T& val : row) {
            new_row.push_back(static_cast<double>(val));
        }
        result.push_back(std::move(new_row));
    }
    return result;
}

template <typename T>
std::vector<std::vector<double>> to_double_vector_remove_flipped(const std::vector<std::vector<T>>& v, size_t latDim) {
    std::vector<std::vector<double>> result;
    result.reserve(v.size());
    for (const auto& row : v) {
        std::vector<double> new_row;
        new_row.reserve(latDim);
        for (std::size_t i = 0; i < latDim && i < row.size(); ++i) {
            new_row.push_back(static_cast<double>(row[i]));
        }
        result.push_back(std::move(new_row));
    }
    return result;
}

void writeVectorToDisk(const std::vector<double>& data, const std::string& filename);
void saveEncodedToDisk(std::vector<std::vector<double>>& encoded_latent, std::string& filename);
std::vector<std::vector<double>> loadEncodedFromDisk(const std::string& filename);



MMapVector mmapVectorOpen(const std::string& filename);

void mmapVectorClose(MMapVector& mv);

inline const double& mmapVectorAt(const MMapVector& mv, size_t i) {
    return mv.data[i];
}

#endif