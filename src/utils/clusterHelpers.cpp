
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

double computeMDF(NIBR::Streamline &s1, NIBR::Streamline &s2) {
    double sum = 0.0f;
    for (size_t i = 0; i < s1.size(); ++i) {
        double dx = s1[i][0] - s2[i][0];
        double dy = s1[i][1] - s2[i][1];
        double dz = s1[i][2] - s2[i][2];
        sum += dx*dx + dy*dy + dz*dz;
    }
    return sum / double(s1.size());
}

