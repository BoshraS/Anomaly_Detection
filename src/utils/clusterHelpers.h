#ifndef MODEL_CLUSTER_HELPERS_H
#define MODEL_CLUSTER_HELPERS_H

#include "dMRI/tractography/tractogram.h"


NIBR::StreamlineBatch getRandomBatch(size_t batchId, size_t batchSize, NIBR::Tractogram &tracObj, std::vector<size_t> &randomList);

NIBR::StreamlineBatch getOrderedBatch(size_t batchId, size_t batchSize, NIBR::Tractogram &tracObj);

double computeMDF(NIBR::Streamline &s1, NIBR::Streamline &s2);

#endif