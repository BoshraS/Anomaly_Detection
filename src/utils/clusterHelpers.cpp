
#include "utils/clusterHelpers.h"

NIBR::StreamlineBatch getRandomBatch(size_t batchId, size_t batchSize, NIBR::Tractogram &tracObj, std::vector<size_t> &randomList) {
    size_t startIdx = batchId * batchSize;
    size_t loopCount = std::min(batchSize, tracObj.size() - startIdx);
    NIBR::StreamlineBatch thisBatch(loopCount);
    for(size_t i = 0; i < loopCount; ++i){
        thisBatch[i] = tracObj[randomList[startIdx+i]];
    }
    return thisBatch;
}

NIBR::StreamlineBatch getOrderedBatch(size_t batchId, size_t batchSize, NIBR::Tractogram &tracObj) {
    size_t startIdx = batchId * batchSize;
    size_t loopCount = std::min(batchSize, tracObj.size() - startIdx);
    NIBR::StreamlineBatch thisBatch(loopCount);
    for(size_t i = 0; i < loopCount; ++i){
        thisBatch[i] = tracObj[startIdx+i];
    }
    return thisBatch;
}

double computeMDFSquared(NIBR::Streamline &s1, NIBR::Streamline &s2) {
    double sum_forward = 0.0;
    double sum_reverse = 0.0;
    const size_t n = s1.size();

    for (size_t i = 0; i < n; ++i) {
        // forwa
        double dx = s1[i][0] - s2[i][0];
        double dy = s1[i][1] - s2[i][1];
        double dz = s1[i][2] - s2[i][2];
        sum_forward += dx*dx + dy*dy + dz*dz;

        // reverse
        dx = s1[i][0] - s2[n - 1 - i][0];
        dy = s1[i][1] - s2[n - 1 - i][1];
        dz = s1[i][2] - s2[n - 1 - i][2];
        sum_reverse += dx*dx + dy*dy + dz*dz;
    }

    double mean_forward = sum_forward / double(n);
    double mean_reverse = sum_reverse / double(n);
    return std::min(mean_forward, mean_reverse);
}

double computeMDF(NIBR::Streamline &s1, NIBR::Streamline &s2) {
    double sum_forward = 0.0;
    double sum_reverse = 0.0;
    const size_t n = s1.size();

    for (size_t i = 0; i < n; ++i) {
        // forwa
        double dx = s1[i][0] - s2[i][0];
        double dy = s1[i][1] - s2[i][1];
        double dz = s1[i][2] - s2[i][2];
        sum_forward += dx*dx + dy*dy + dz*dz;

        // reverse
        dx = s1[i][0] - s2[n - 1 - i][0];
        dy = s1[i][1] - s2[n - 1 - i][1];
        dz = s1[i][2] - s2[n - 1 - i][2];
        sum_reverse += dx*dx + dy*dy + dz*dz;
    }

    double mean_forward = sum_forward / double(n);
    double mean_reverse = sum_reverse / double(n);
    return std::sqrt(std::min(mean_forward, mean_reverse));
}
