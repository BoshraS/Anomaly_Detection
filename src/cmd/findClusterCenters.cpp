#include "cmd.h"

using namespace NIBR;

namespace CMDARGS_FINDCLUSTERCENTERS {
    std::string  inp_path;
    std::string  out_path;

    std::tuple<std::string, int, int, std::string, double> model_spec("", 0, 0, "", 1.0); // module_path, inp_dim, lat_dim, data type, distance scaler

    float  maxDist;
    int    maxIteration      = 0;

    int    batchSize         = 1000000;
    int    miniBatchSize     = 1000;
    bool   randomize         = false;
    bool   shuffle           = false;
    bool   useLeaderAlg      = false;
    bool   useCPU            = false;
    
    int numberOfThreads      =  0;
    std::string verbose      = "info";
    bool force               = false;
}

using namespace CMDARGS_FINDCLUSTERCENTERS;

void run_findClusterCenters()
{ 

    parseCommon(numberOfThreads,verbose);
    if (!parseForceOutput(out_path,force)) return;

    if (getFileExtension(out_path) != "clc") {
        disp(MSG_ERROR,"Output cluster center file must have .clc extension.");
        return;
    }

    bool isFile = false;
    
    std::string ext = getFileExtension(inp_path);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == "bin") isFile = true;
    

    bool isFolder = existsFolder(inp_path);

    if (!isFile && !isFolder) {
        disp(MSG_ERROR,"Unknown input");
        return;
    }

    // Check input limits
    if (batchSize < 1) {
        disp(MSG_ERROR,"Minimum batchSize is 1");
        return;
    }

    if (miniBatchSize < 1) {
        disp(MSG_ERROR,"Minimum miniBatchSize is 1");
        return;
    }

    if (miniBatchSize > batchSize) {
        disp(MSG_ERROR,"Max miniBatchSize is batchSize");
        return;
    }

    // Set model
    StreamlineAutoencoder model = StreamlineAutoencoder(model_spec, useCPU);
    if (!model.isReady()) return;

    // Open binary files to read encoded streamlines
    std::vector<std::ifstream*> file;
    std::vector<int> scnt;
    std::vector<int> scnt_cumSum;
    int totalCnt = 0;
     
    // Add files to the list to process
    // keeps the count of streamlines
    auto addToProcess = [&](std::string fname) -> bool {
        std::ifstream* ifs = new std::ifstream(fname, std::ios::binary | std::ios::ate | std::ios::in);
        if (!ifs->is_open()) {
            disp(MSG_ERROR,"Failed to open file: %s", fname.c_str());
            return false;
        }
        std::streampos fileSize = ifs->tellg();
        int streamlineCount     = fileSize / (sizeof(float) * 2 * model.latDim);
        totalCnt               += streamlineCount;
        ifs->seekg(0, std::ios::beg);  // Move the file pointer back to the beginning
        
        file.push_back(ifs);
        scnt.push_back(streamlineCount);
        scnt_cumSum.push_back(totalCnt);

        disp(MSG_DETAIL,"Including encoded streamlines from: %s", fname.c_str());
        return true;
    };

    if (isFile) { // If the input is a single file
        if(!addToProcess(inp_path)) return;
    } else { // If the input is a directory
        std::vector<std::string> binFiles = getMatchingFiles(inp_path + "/*.bin");
        for (auto& b : binFiles) {
            if(!addToProcess(b)) return;
        }
    }
    // ------

    if (totalCnt < batchSize) batchSize = totalCnt;

    // Compute number of streamlines to fetch from each file for each batch
    std::vector<int> scnt_batch;
    int tmpTotal = 0;
    for (auto& c : scnt) {
        float ratio = float(c) / float(totalCnt);
        int cnt = ratio * float(batchSize);
        if (cnt == 0) cnt = 1;
        tmpTotal += cnt;
        scnt_batch.push_back(cnt);
    }

    // Adjust the scnt_batch to ensure the total number is exactly batchSize
    int currentTotal = tmpTotal;
    while (currentTotal != batchSize) {
        if (currentTotal < batchSize) {
            // Add streamlines to the files
            for (size_t i = 0; i < scnt_batch.size(); ++i) {
                if (currentTotal == batchSize) break;
                if (scnt_batch[i] < scnt[i]) {  // Ensure we do not exceed the number of available streamlines
                    scnt_batch[i]++;
                    currentTotal++;
                }
            }
        } else if (currentTotal > batchSize) {
            // Remove streamlines from the files with higher counts
            for (size_t i = 0; i < scnt_batch.size(); ++i) {
                if (currentTotal == batchSize) break;
                if (scnt_batch[i] > 1) {  // Ensure that we keep at least one streamline per file
                    scnt_batch[i]--;
                    currentTotal--;
                }
            }
        }
    }
    // ------

    // Readers for latent representation of streamlines
    NIBR::RandomDoer rand;
    rand.init_uniform_int(totalCnt-1);

    std::function<std::vector<Eigen::VectorXf>(int)> getRandomBatch = [&](int readCnt) -> std::vector<Eigen::VectorXf> {

        int startInd = rand.uniform_int();

        // Find the file to start reading from
        int fileIndex = 0;
        while (fileIndex < int(scnt_cumSum.size()) && startInd >= scnt_cumSum[fileIndex]) {
            fileIndex++;
        }
        if (fileIndex == int(scnt_cumSum.size())) return std::vector<Eigen::VectorXf>();

        // Calculate local start index in the selected file
        int localStartInd = startInd - (fileIndex == 0 ? 0 : scnt_cumSum[fileIndex - 1]);

        // Determine how many streamlines we can read from this file
        int streamlinesToRead = std::min(readCnt, scnt[fileIndex] - localStartInd);

        // Create the output vector
        std::vector<Eigen::VectorXf> out;
        out.reserve(readCnt);

        // Read streamlines from the selected file
        std::vector<float> fileBatch(streamlinesToRead * 2 * model.latDim);
        file[fileIndex]->seekg(localStartInd * sizeof(float) * 2 * model.latDim, std::ios::beg);
        file[fileIndex]->read(reinterpret_cast<char*>(fileBatch.data()), sizeof(float) * 2 * model.latDim * streamlinesToRead);

        for (int i = 0; i < streamlinesToRead; ++i) {
            Eigen::VectorXf eigenVec = Eigen::VectorXf::Map(fileBatch.data() + i * 2 * model.latDim, 2 * model.latDim);
            out.push_back(std::move(eigenVec));
        }

        // If we could not read enough streamlines, read the remaining from another random file
        if (streamlinesToRead < readCnt) {
            int remaining = readCnt - streamlinesToRead;
            auto remainingBatch = getRandomBatch(remaining);  // Recursively read remaining batch

            if (int(remainingBatch.size()) != remaining) return std::vector<Eigen::VectorXf>();
            out.insert(out.end(), remainingBatch.begin(), remainingBatch.end());
        }

        return (int(out.size()) == readCnt) ? out : std::vector<Eigen::VectorXf>();
    };


    std::function<std::vector<Eigen::VectorXf>(int)> getRegularBatch = [&](int readCnt) -> std::vector<Eigen::VectorXf> {
        std::vector<Eigen::VectorXf> out;
        out.reserve(readCnt);

        for (size_t i = 0; i < scnt_batch.size(); ++i) {
            int streamlinesToRead = std::min(scnt_batch[i], readCnt - int(out.size()));
            if (streamlinesToRead <= 0) break;

            std::vector<float> fileBatch(streamlinesToRead * 2 * model.latDim);
            file[i]->read(reinterpret_cast<char*>(fileBatch.data()), sizeof(float) * 2 * model.latDim * streamlinesToRead);

            for (int s = 0; s < streamlinesToRead; s++) {
                Eigen::VectorXf eigenVec = Eigen::VectorXf::Map(fileBatch.data() + s * 2 * model.latDim, 2 * model.latDim);
                out.push_back(std::move(eigenVec));
            }
        }

        return out;
    };


    // Compute maxIteration if needed
    if (randomize) {
        if (maxIteration < 1) maxIteration = 1;
    } else {
        if (maxIteration < 1) maxIteration = std::ceil(float(totalCnt) / float(batchSize));    
    }

    // Do the clustering
    std::vector<Eigen::VectorXf> clusterCenters;
    std::vector<int>             clusterCounts;      // To store the count of streamlines in each cluster
    std::mutex                   clusterUpdateMutex; // To safely update clusters from multiple threads

    
    // Build an empty KD-Tree
    PointCloud cloud;
    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud>,PointCloud,-1> KDTree;
    KDTree kdtree(model.latDim, cloud, nanoflann::KDTreeSingleIndexAdaptorParams( ((256/model.latDim) > 10) ? 10 : (256/model.latDim) ) ) ;
    kdtree.buildIndex();

    // Adjust maxDist based on model's distance scaling factor
    float adjMaxDist = (maxDist/model.distScaler) * (maxDist/model.distScaler);

    disp(MSG_INFO,"Clustering...");
    int totRead = 0;
    for (int iter = 0; iter < maxIteration; iter++) {
        
        // Get a batch
        std::vector<Eigen::VectorXf> batch;

        if (randomize) {
            int curTot = 0;
            while (curTot < batchSize) {
                disp(MSG_DETAIL,"Getting random batch of size %d ...", batchSize);
                auto miniBatch = getRandomBatch(batchSize - curTot);
                curTot += miniBatch.size();
                batch.insert(batch.end(), miniBatch.begin(), miniBatch.end());
                disp(MSG_DETAIL,"Done");
            }
        } else {
            disp(MSG_DETAIL,"Getting regular batch of size %d ...", batchSize);
            batch = getRegularBatch(batchSize);
            totRead += batch.size();
            if (totRead > totalCnt) batchSize = totalCnt - totRead;
            disp(MSG_DETAIL,"Done");
        }

        disp(MSG_DETAIL,"Current batchSize: %d", batchSize);

        // Shuffle the batch if needed
        if (shuffle) std::shuffle(batch.begin(), batch.end(), rand.getGen());
        
        int unassignedCnt = 0;
        std::vector<Eigen::VectorXf> localClusterCenters;
        std::vector<int> newLocalCounts;

        std::vector<std::atomic<bool>> unassigned(batch.size());
        for (size_t i = 0; i < batch.size(); ++i) unassigned[i] = false;

        // Perform clustering operations on the batch
        // This will be done in five steps
        // STEP 1: If a streamline is far from all existing cluster centers, keep it as "unassigned". Otherwise, assign it to an existing cluster center.
        // STEP 2: (Online k-means only) Update the main cluster centers by adding the contributions from the assigned streamlines in the previous step.
        // STEP 3: Within the batch, locally cluster all the "unassigned" streamlines, and form localClusterCenters.
        // STEP 4: Add the new clusters to the global list and update their counts
        // STEP 5: Rebuild the KD-Tree with the updated and new centers

        // Reusable lambda to run the leader algorithm on a set of indices
        auto runLeaderOnUnassigned = [&](const std::vector<size_t>& indices) -> std::vector<Eigen::VectorXf> {
            
            int unaCnt = indices.size();
            if (unaCnt == 0) return {};

            disp(MSG_DETAIL, "Clustering %d streamlines with leader algorithm...", unaCnt);

            std::vector<Eigen::VectorXf> finalLeaders;
            PointCloud localCloud;
            KDTree localKdtree(model.latDim, localCloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
            localKdtree.buildIndex();

            std::vector<Eigen::VectorXf> pendingLeaders;
            size_t taskOffset = 0;
            std::mutex mx;

            auto findAndAddLeader = [&](NIBR::MT::TASK task) -> void {
                size_t rInd = indices[task.no + taskOffset];

                // Check against already merged leaders from previous chunks
                if (!finalLeaders.empty()) {
                    size_t closestCenterIndex = 0;
                    float  squaredDist;

                    nanoflann::KNNResultSet<float> resultSet1(1);
                    resultSet1.init(&closestCenterIndex, &squaredDist);
                    localKdtree.findNeighbors(resultSet1, batch[rInd].data(), nanoflann::SearchParameters());
                    if (squaredDist < adjMaxDist) return;

                    nanoflann::KNNResultSet<float> resultSet2(1);
                    resultSet2.init(&closestCenterIndex, &squaredDist);
                    localKdtree.findNeighbors(resultSet2, batch[rInd].data() + model.latDim, nanoflann::SearchParameters());
                    if (squaredDist < adjMaxDist) return;
                }

                // Lock and check against pending leaders from the current chunk
                mx.lock();
                for (const auto& center : pendingLeaders) {
                    float sum1 = 0, sum2 = 0;
                    for (int i = 0; i < model.latDim; i++) {
                        float d1 = (batch[rInd][i] - center[i]);
                        float d2 = (batch[rInd][i] - center[i + model.latDim]);
                        sum1 += d1 * d1;
                        sum2 += d2 * d2;
                    }
                    if (std::min(sum1, sum2) < adjMaxDist) {
                        mx.unlock();
                        return;
                    }
                }
                pendingLeaders.push_back(batch[rInd]);
                mx.unlock();
            };

            int splitSize;
            if      (unaCnt <= 100000)  splitSize = 1000;
            else if (unaCnt <= 1000000) splitSize = 10000;
            else                        splitSize = 100000;

            int begInd = 0;
            int endInd = 0;
            while (endInd != unaCnt) {
                begInd = endInd;
                endInd = ((begInd + splitSize) <= unaCnt) ? (begInd + splitSize) : unaCnt;
                int curSplitSize = endInd - begInd;
                taskOffset = begInd;
                NIBR::MT::MTRUN(curSplitSize, "Finding chunk leaders", findAndAddLeader);
                finalLeaders.insert(finalLeaders.end(), pendingLeaders.begin(), pendingLeaders.end());
                localCloud.points = finalLeaders;
                localKdtree.buildIndex();
                pendingLeaders.clear();
            }

            return finalLeaders;
        };

        //====

        if (!useLeaderAlg) {
            
            // ONLINE K-MEANS ALGORITHM

            // STEP 1: Assign streamlines to existing clusters or mark them unassigned
            std::vector<Eigen::VectorXf> batch_clusterUpdates;
            std::vector<int>             batch_clusterCounts;
            if (!clusterCenters.empty()) {
                batch_clusterUpdates.resize(clusterCenters.size(), Eigen::VectorXf::Zero(2 * model.latDim));
                batch_clusterCounts.resize(clusterCenters.size(), 0);
            }

            auto findClosestAndUpdate = [&](NIBR::MT::TASK task) -> void {
                if (clusterCenters.empty()) {
                    unassigned[task.no].store(true);
                    return;
                }
                
                size_t closestCenterIndex = 0;
                float  minSquaredDist     = std::numeric_limits<float>::max();
                nanoflann::KNNResultSet<float> resultSet(1);
                size_t tempIndex;
                float  tempDist;

                resultSet.init(&tempIndex, &tempDist);
                kdtree.findNeighbors(resultSet, batch[task.no].data(), nanoflann::SearchParameters());
                if (tempDist < minSquaredDist) {
                    minSquaredDist      = tempDist;
                    closestCenterIndex  = tempIndex;
                }

                resultSet.init(&tempIndex, &tempDist);
                kdtree.findNeighbors(resultSet, batch[task.no].data() + model.latDim, nanoflann::SearchParameters());
                if (tempDist < minSquaredDist) {
                    minSquaredDist      = tempDist;
                    closestCenterIndex  = tempIndex;
                }

                if (minSquaredDist < adjMaxDist) {
                    std::lock_guard<std::mutex> lock(clusterUpdateMutex);
                    batch_clusterUpdates[closestCenterIndex] += batch[task.no];
                    batch_clusterCounts[closestCenterIndex]++;
                } else {
                    unassigned[task.no].store(true);
                }
            };
            NIBR::MT::MTRUN( batch.size(), "Assigning clusters " + to_string_with_precision(iter+1,0) + " / " + to_string_with_precision(maxIteration,0), findClosestAndUpdate);

            // STEP 2: Apply the collected updates to the main cluster centers
            disp(MSG_DETAIL, "Applying batch updates to cluster centers...");
            for (size_t i = 0; i < clusterCenters.size(); ++i) {
                if (batch_clusterCounts[i] > 0) {
                    clusterCenters[i] = (clusterCenters[i] * clusterCounts[i] + batch_clusterUpdates[i]) / (clusterCounts[i] + batch_clusterCounts[i]);
                    clusterCounts[i] += batch_clusterCounts[i];
                }
            }
            disp(MSG_DETAIL, "Done applying updates.");
            
            // STEP 3: Cluster the unassigned streamlines to form new centers
            std::vector<size_t> unaInd;
            for (size_t r = 0; r < unassigned.size(); r++) {
                if (unassigned[r]) unaInd.push_back(r);
            }
            unassignedCnt = unaInd.size();

            disp(MSG_DETAIL, "%d are outside of existing cluster reach.",unassignedCnt);

            if (unassignedCnt > 0) {
                
                // Pass 1: Find initial leaders using the reusable leader algorithm function
                localClusterCenters = runLeaderOnUnassigned(unaInd);
                disp(MSG_DETAIL, "Found %d local leaders after merging.", localClusterCenters.size());

                // Pass 2: Assign all unassigned streamlines to the nearest leader and compute the average.
                if (!localClusterCenters.empty()) {

                    disp(MSG_DETAIL, "Pass 2: Assigning streamlines to leaders and averaging...");
                    newLocalCounts.assign(localClusterCenters.size(), 0);
                    std::vector<Eigen::VectorXf> localClusterUpdates(localClusterCenters.size(), Eigen::VectorXf::Zero(2 * model.latDim));

                    PointCloud localCloud;
                    localCloud.points = localClusterCenters;
                    KDTree localKdtree(model.latDim, localCloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
                    localKdtree.buildIndex();
                    std::mutex assignMutex;

                    auto assignToLeaders = [&](NIBR::MT::TASK task) -> void {
                        size_t rInd                 = unaInd[task.no];
                        size_t closestCenterIndex   = 0; 
                        float  minSquaredDist       = std::numeric_limits<float>::max();
                        
                        nanoflann::KNNResultSet<float> resultSet(1);
                        size_t tempIndex; 
                        float tempDist;
                        resultSet.init(&tempIndex, &tempDist);
                        localKdtree.findNeighbors(resultSet, batch[rInd].data(), nanoflann::SearchParameters());
                        minSquaredDist      = tempDist; 
                        closestCenterIndex  = tempIndex;
                        
                        resultSet.init(&tempIndex, &tempDist);
                        localKdtree.findNeighbors(resultSet, batch[rInd].data() + model.latDim, nanoflann::SearchParameters());

                        if (tempDist < minSquaredDist) { closestCenterIndex = tempIndex; }
                        
                        std::lock_guard<std::mutex> lock(assignMutex);
                        localClusterUpdates[closestCenterIndex] += batch[rInd];
                        newLocalCounts[closestCenterIndex]++;
                    };
                    NIBR::MT::MTRUN(unassignedCnt, "Assigning to leaders", assignToLeaders);

                    for (size_t i = 0; i < localClusterCenters.size(); ++i) {
                        if (newLocalCounts[i] > 0) {
                            localClusterCenters[i] = localClusterUpdates[i] / newLocalCounts[i];
                        }
                    }
                    disp(MSG_DETAIL, "Done averaging local clusters.");
                }
            }

        } else {

            // LEADER ALGORITHM
            
            // STEP 1: Assign streamlines to existing clusters or mark them unassigned
            auto addToGlobalCluster = [&](NIBR::MT::TASK task) -> void {
                if (!clusterCenters.empty()) {
                    size_t closestCenterIndex = 0;
                    float  squaredDistToClosestClusterCenter;
                    nanoflann::KNNResultSet<float> resultSet1(1);
                    resultSet1.init(&closestCenterIndex, &squaredDistToClosestClusterCenter);
                    kdtree.findNeighbors(resultSet1, batch[task.no].data(),             nanoflann::SearchParameters());
                    if (squaredDistToClosestClusterCenter < adjMaxDist) return;
                    nanoflann::KNNResultSet<float> resultSet2(1);
                    resultSet2.init(&closestCenterIndex, &squaredDistToClosestClusterCenter);
                    kdtree.findNeighbors(resultSet2, batch[task.no].data()+model.latDim, nanoflann::SearchParameters());
                    if (squaredDistToClosestClusterCenter < adjMaxDist) return;
                }
                unassigned[task.no].store(true);
            };
            NIBR::MT::MTRUN( batch.size(), "Assigning clusters " + to_string_with_precision(iter+1,0) + " / " + to_string_with_precision(maxIteration,0), addToGlobalCluster);

            // STEP 2: Main cluster centers are not updated in the leader algorithm
            
            // STEP 3: Within the batch, locally cluster all the "unassigned" streamlines using the reusable leader algorithm function.
            std::vector<size_t> unaInd;
            for (size_t r = 0; r < unassigned.size(); r++) {
                if (unassigned[r]) unaInd.push_back(r);
            }
            unassignedCnt = unaInd.size();
            
            if (unassignedCnt > 0) {
                localClusterCenters = runLeaderOnUnassigned(unaInd);
            }
        }

        // STEP 4: Add the new clusters to the global list and update their counts
        clusterCenters.insert(clusterCenters.end(), localClusterCenters.begin(), localClusterCenters.end());
        if (!useLeaderAlg) {
            clusterCounts.insert(clusterCounts.end(), newLocalCounts.begin(), newLocalCounts.end());    // Use the real counts
        } else {
            clusterCounts.insert(clusterCounts.end(), localClusterCenters.size(), 1);                   // Each new leader has a count of 1
        }

        // STEP 5: Rebuild the KD-Tree with the updated and new centers
        cloud.points = clusterCenters;
        kdtree.buildIndex();

        disp(MSG_INFO,"Unassigned streamlines: %d - New clusters found: %d - Total clusters: %d", unassignedCnt, localClusterCenters.size(), clusterCenters.size());

    }
    disp(MSG_INFO,"Done");

    // Close all the files
    for (auto f : file) {
        f->close();
        delete f;
    }


    // Open binary file for writing the latent representation of the cluster centers
    std::ofstream ofs(out_path, std::ios::binary);
    if (!ofs) {
        disp(MSG_ERROR,"Failed to open output file.");
        return;
    }

    for (const auto& c : clusterCenters) {
        ofs.write(reinterpret_cast<const char*>(c.data()), 2 * model.latDim * sizeof(float));
    }
    ofs.close();


    return;
       
}          
    
     
void findClusterCenters(CLI::App* app)   
{ 

    app->formatter(std::make_shared<CustomHelpFormatter>());

    app->description("finds streamline cluster centers based on their latent space representation");

    app->add_option("<input>",               inp_path,           "Input path. Either a file (.bin) or a folder.")
        ->required();
     
    app->add_option("<output>",              out_path,           "Output latent representations of cluster centers (.clc).")
        ->required();

    app->add_option("--model,-m",            model_spec,         "Input model, specified with the path to the Torch script file, followed by the input dimensions, latent space dimensions, data type (float or double), and distance scaling factor of the model. E.g. /model/test_model.pt 256 64 float 0.08. The distance scaler factor of the model can be obtained using the \"modelTest\" command.");

    app->add_option("--maxDist, -d",         maxDist,            "Maximum distance from any cluster center.")
        ->required();

    app->add_option("--maxIteration, -i",    maxIteration,       "Limits maximum number of iterations in constrast to the default, which iterates until all streamlines processed.");
    
    app->add_option("--batchSize, -b",       batchSize,          "Batch size. Number of streamlines to process at each iteration. Default: 1000000");
    app->add_flag("--shuffle, -s",           shuffle,            "Shuffles streamlines within batches that leads to different cluster centers at each run.");
    app->add_flag("--useLeaderAlg, -l",      useLeaderAlg,       "Uses the leader algorithm for clustering instead of the default online k-means approach. Default: OFF.");
    
    app->add_flag("--randomize, -r",         randomize,          "Cluster using randomized batches instead of the default, regularly fetched, batches.");
    app->add_option("--miniBatchSize",       miniBatchSize,      "When using random batches, each batch is split into mini batches fetched contigously from a single file, miniBatchSize sets that value. Default: 1000");

    app->add_flag("--useCPU, -c",            useCPU,             "Use only CPU without checking any available GPUs.");

    app->add_option("--numberOfThreads, -n", numberOfThreads,    "Number of threads.");
    app->add_option("--verbose, -v",         verbose,            "Verbose level. Options are \"quite\",\"fatal\",\"error\",\"warn\",\"info\" and \"debug\". Default=info");
    app->add_flag("--force, -f",             force,              "Force overwriting of existing file");

    app->callback(run_findClusterCenters);  
     
}

                       