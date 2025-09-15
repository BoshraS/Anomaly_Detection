#pragma once 

#include "conf/config.h"
#include "conf/aux.h"
#include "utils/utils.h"

void modelTest(CLI::App* app);
void modelTest_precalc(CLI::App* app);
void encode(CLI::App* app);
void decode(CLI::App* app);
void findClusterCenters(CLI::App* app);
void score(CLI::App* app);
void toTrack(CLI::App* app);
void toImg(CLI::App* app);

void compareSpeed(CLI::App* app);
void measureDistance(CLI::App* app);
void findClusterCenters_euclidean(CLI::App* app);
void findClusterCenters_latentBrute(CLI::App* app);