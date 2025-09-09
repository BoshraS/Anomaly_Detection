#include <Eigen/Dense>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <string>
#include <cstddef>
#include <fstream>

#include "modelTestHelpers.h"

// Convert std::vector to Eigen::VectorXd
Eigen::VectorXd vectorToEigen(const std::vector<double>& v) {
    return Eigen::VectorXd::Map(v.data(), v.size());
}

Eigen::Map<const Eigen::VectorXd> vectorToEigen(const MMapVector& mv) {
    return Eigen::Map<const Eigen::VectorXd>(mv.data, mv.size);
}

double correlation_coefficient(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.empty()) {
        throw std::invalid_argument("Vectors must be of same size and non-empty");
    }

    Eigen::VectorXd X = vectorToEigen(x);
    Eigen::VectorXd Y = vectorToEigen(y);

    double mean_X = X.mean();
    double mean_Y = Y.mean();

    Eigen::VectorXd X_centered = X.array() - mean_X;
    Eigen::VectorXd Y_centered = Y.array() - mean_Y;

    double covariance = (X_centered.dot(Y_centered)) / (X.size() - 1);  // Using (N-1) for sample covariance
    double stddev_X = std::sqrt(X_centered.squaredNorm() / (X.size() - 1));
    double stddev_Y = std::sqrt(Y_centered.squaredNorm() / (Y.size() - 1));

    if (stddev_X == 0 || stddev_Y == 0) return 0; // Avoid division by zero
    
    return covariance / (stddev_X * stddev_Y);
}

double latentDistanceCalculator(const std::vector<double>& a, const std::vector<double>& b, int latDim) {
    double sum = 0.0;
    for (int i = 0; i < latDim; ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

double latentMinDistanceCalculator(const std::vector<double>& a, const std::vector<double>& b, int latDim) {
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;
    double sum4 = 0.0;
    for (int i = 0; i < latDim; ++i) {
        double diff1 = a[i] - b[i];
        sum1 += diff1 * diff1;

        double diff2 = a[i] - b[latDim+i];
        sum2 += diff2 * diff2;

        double diff3 = a[latDim+i] - b[i];
        sum3 += diff3 * diff3;

        double diff4 = a[latDim+i] - b[latDim+i];
        sum4 += diff4 * diff4;
    }
    return std::sqrt(std::min({sum1, sum2, sum3, sum4}));
}

void writeVectorToDisk(const std::vector<double>& data, const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary | std::ios::out);
    ofs.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(double));
}



MMapVector mmapVectorOpen(const std::string& filename) {
    MMapVector mv;
    mv.fd = open(filename.c_str(), O_RDONLY);
    if (mv.fd == -1) throw std::runtime_error("Failed to open file");
    struct stat sb;
    if (fstat(mv.fd, &sb) == -1) throw std::runtime_error("fstat failed");
    mv.size = sb.st_size / sizeof(double);
    mv.data = static_cast<double*>(
        mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, mv.fd, 0)
    );
    if (mv.data == MAP_FAILED) throw std::runtime_error("mmap failed");
    return mv;
}

void mmapVectorClose(MMapVector& mv) {
    if (mv.data) {
        munmap(mv.data, mv.size * sizeof(double));
        mv.data = nullptr;
    }
    if (mv.fd != -1) {
        close(mv.fd);
        mv.fd = -1;
    }
    mv.size = 0;
}