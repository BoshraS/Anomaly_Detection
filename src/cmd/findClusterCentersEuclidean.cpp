#include "cmd.h"
#include "dMRI/tractography/tractogram.h"

using namespace NIBR;

namespace CMDARGS_FINDCLUSTERCENTERS_EUC {
    std::string  inp_path;
    std::string  out_path;

    std::tuple<std::string, int, int, std::string, double> model_spec("", 0, 0, "", 1.0); // module_path, inp_dim, lat_dim, data type, distance scaler

    float  maxDist;
    int    maxIteration     = 0;

    int    batchSize        = 1000000;
    bool   randomize        = false;
    bool   shuffle          = false;
    bool   useCPU           = false;
    
    int numberOfThreads     =  0;
    std::string verbose     = "info";
    bool force              = false;
}

using namespace CMDARGS_FINDCLUSTERCENTERS_EUC;

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

void run_findClusterCenters_euclidean()
{ 

    parseCommon(numberOfThreads,verbose);
    if (!parseForceOutput(out_path,force)) return;

    if (getFileExtension(out_path) != "clce") {
        disp(MSG_ERROR,"Output cluster center file must have .clce extension.");
        return;
    }
    
    bool isValid = ensureVTKorTCK(inp_path);

    if (!isValid) {
        disp(MSG_ERROR,"Unknown tractogram type");
        return;
    }

    // Check input limits
    if (batchSize < 1) {
        disp(MSG_ERROR,"Minimum batchSize is 1");
        return;
    }
    // Open tractogram reader
    NIBR::TractogramReader tractogram(inp_path, false);
    NIBR::Tractogram tracObj = tractogram.getTractogram();

    std::vector<int> scnt;
    std::vector<int> scnt_cumSum;
    size_t totalCnt = tractogram.getNumberOfStreamlines();

    //Create random order
    
    std::vector<size_t> randomList;
    if(randomize){
        randomList.resize(totalCnt);
        for (size_t i = 0; i < totalCnt; ++i) {
            randomList[i] = i;
        }
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(randomList.begin(), randomList.end(), g);
    }

    // Compute maxIteration if needed
    if (randomize) {
        if (maxIteration < 1) maxIteration = 1;
    } else {
        if (maxIteration < 1) maxIteration = std::ceil(float(totalCnt) / float(batchSize));    
    }
    

    // Do the clustering
    // Could use NIBR::StreamlineBatch but I think it's clearer like this
    std::vector<NIBR::Streamline> clusterCenters;

    disp(MSG_INFO,"Clustering...");
    auto clusteringStartTime = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < maxIteration; iter++) {
        
        // Get a batch
        NIBR::StreamlineBatch batch;

        if (randomize) {
            batch = getRandomBatch(iter, batchSize, tracObj, randomList);
        } else {
            batch = getOrderedBatch(iter, batchSize, tracObj);
        }

        disp(MSG_DETAIL,"Current batchSize: %d", batchSize);

        // Shuffle the batch if needed

        // Perform clustering operations on the batch
        // This will be done in two steps
        // Step 1. If a streamline is far from all existing cluster centers, keep it as "unassigned".
        // Step 2. Within in batch, locally cluster all the "unassigned" streamlines, and form localClusterCenters.
        // Step 3. Append localClusterCenters to clusterCenters

        // For efficiency split the unassigned streamlines into smaller batches of:
        // 10000 streamlines in the first iteration when there will be many new clusters,
        // and 1000 streamlines in the other iteration when there will be many existing clusters, and less new clusters.

        std::vector<NIBR::Streamline> localClusterCenters;       // New clusters found in this batch.
        
        std::vector<std::atomic<bool>> unassigned(batchSize);    // True if a streamline in the batch did not belong to any existing cluster
        for (int i = 0; i < batchSize; ++i) unassigned[i] = false;

        std::vector<size_t> unaInd;                              // Indices of the streamlines, which did not belong to any existing cluster
        std::vector<NIBR::Streamline> unassignedClusterCenters;   // Clusters of the unassigned streamlines that were not clusered within a batch
        size_t taskOffset = 0;

        auto addToGlobalCluster = [&](NIBR::MT::TASK task) -> void {

            if (!clusterCenters.empty()) {
                bool tooClose = false;
                for (size_t c = 0; c < clusterCenters.size(); ++c) {
                    double dist = computeMDF(batch[task.no], clusterCenters[c]);
                    if (dist < maxDist) {
                        tooClose = true;
                        break;
                    }
                }
                if (tooClose) return;
            }

            unassigned[task.no].store(true);

        };

        std::mutex mx;

        auto addToLocalCluster = [&](NIBR::MT::TASK task) -> void {

            size_t rInd = unaInd[task.no+taskOffset];

            if (!localClusterCenters.empty()) {
                bool tooClose = false;
                for (size_t c = 0; c < localClusterCenters.size(); ++c) {
                    double dist = computeMDF(batch[task.no], localClusterCenters[c]);
                    if (dist < maxDist) {
                        tooClose = true;
                        break;
                    }
                }
                if (tooClose) return;
            }

            {
                {
                    std::lock_guard<std::mutex> guard(mx);
                    for (size_t ind = 0; ind < unassignedClusterCenters.size(); ++ind) {
                        float dist = computeMDF(batch[rInd], unassignedClusterCenters[ind]);
                        if (dist < maxDist) return;
                    }

                    unassignedClusterCenters.push_back(batch[rInd]);
                }
                return;                
            }   

        };


        // Add the newly found localClusterCenters in the global clusterCenters
        auto doUnassigned = [&]() -> int {

            unaInd.clear();

            for (size_t r = 0; r < unassigned.size(); r++) {
                if (unassigned[r]) unaInd.push_back(r);
            }
            int unaCnt     = unaInd.size();
            

            int splitSize;
            if      (unaCnt <= 100000)  splitSize = 1000;
            else if (unaCnt <= 1000000) splitSize = 10000;
            else                        splitSize = 100000;

            taskOffset     = 0;
            int begInd     = 0;
            int endInd     = 0;
            int splitCnt = std::max(1, unaCnt / splitSize);
            int curSplitNo = 0;
            std::string preamble = "\033[1;32mNIBRARY::INFO: \033[0;32m";
            disp(MSG_INFO,"Clustering unassigned streamlines...");
            float progressScaler = 100.0f/float(splitCnt);                
            while (endInd != unaCnt) {
                begInd = endInd;
                endInd = ((begInd + splitSize) <= unaCnt) ? (begInd + splitSize) : unaCnt;
                int curSplitSize = endInd - begInd;
                NIBR::MT::MTRUN(curSplitSize, addToLocalCluster);
                localClusterCenters.insert(localClusterCenters.end(), unassignedClusterCenters.begin(), unassignedClusterCenters.end());
                unassignedClusterCenters.clear();
                taskOffset += curSplitSize;
                if (NIBR::VERBOSE()>=VERBOSE_INFO) {std::cout << "\r\033[K" << std::flush;}
                if (NIBR::VERBOSE()>=VERBOSE_INFO) {std::cout << preamble << "Clustering unassigned streamlines: " << std::fixed << std::setprecision(2) << (++curSplitNo)*progressScaler << "%" << "\033[0m" << std::flush;}
            }
            if (NIBR::VERBOSE()>=VERBOSE_INFO) {std::cout << "\r\033[K" << preamble << "Clustering unassigned streamlines: 100%" << std::endl;}

            return unaCnt;

        }; 

        
        // Assign streamlines into existing clusters, and find "unassigned" streamlines, which were not assinged to any cluster
        NIBR::MT::MTRUN( batchSize, "Assigning clusters " + to_string_with_precision(iter+1,0) + " / " + to_string_with_precision(maxIteration,0), addToGlobalCluster);

        // Find localClusterCenters that are the cluster centers of the "unassigned" streamlines, 
        int unassignedCnt = doUnassigned();

        // Append the localClusterCenters to global clusterCenters
        clusterCenters.insert(clusterCenters.end(), localClusterCenters.begin(), localClusterCenters.end());

        disp(MSG_INFO,"Unassigned streamlines: %d - New clusters found: %d - Total clusters: %d", unassignedCnt, localClusterCenters.size(), clusterCenters.size());

    }
    auto clusteringEndTime = std::chrono::high_resolution_clock::now();
    auto latentMinduration = std::chrono::duration_cast<std::chrono::seconds>(clusteringEndTime - clusteringStartTime);
    disp(MSG_INFO,"Done. Time taken: %li seconds", latentMinduration);



    // Open binary file for writing Euclidean cluster centers
    std::ofstream ofs(out_path, std::ios::binary);
    if (!ofs) {
        disp(MSG_ERROR, "Failed to open output file.");
        return;
    }

    // Loop over cluster centers
    for (const auto& c : clusterCenters) {
        ofs.write(reinterpret_cast<const char*>(c.data()), c.size() * 3 * sizeof(float));
    }

    ofs.close();


    return;
       
}          
    
     
void findClusterCenters_euclidean(CLI::App* app)   
{ 

    app->formatter(std::make_shared<CustomHelpFormatter>());

    app->description("finds streamline cluster centers in eunclidean space");

    app->add_option("<input>",               inp_path,           "Input path. Tractogram")
        ->required();
     
    app->add_option("<output>",              out_path,           "Output latent representations of cluster centers (.clce).")
        ->required();

    app->add_option("--maxDist, -d",         maxDist,            "Maximum distance from any cluster center.")
        ->required();

    app->add_option("--maxIteration, -i",    maxIteration,       "Limits maximum number of iterations in constrast to the default, which iterates until all streamlines processed.");
    
    app->add_option("--batchSize, -b",       batchSize,          "Batch size. Number of streamlines to process at each iteration. Default: 1000000");

    app->add_flag("--randomize, -r",         randomize,          "Shuffle the streamlines before starting");
    app->add_flag("--useCPU, -c",            useCPU,             "Use only CPU without checking any available GPUs.");

    app->add_option("--numberOfThreads, -n", numberOfThreads,    "Number of threads.");
    app->add_option("--verbose, -v",         verbose,            "Verbose level. Options are \"quite\",\"fatal\",\"error\",\"warn\",\"info\" and \"debug\". Default=info");
    app->add_flag("--force, -f",             force,              "Force overwriting of existing file");

    app->callback(run_findClusterCenters_euclidean);  
     
}

                       