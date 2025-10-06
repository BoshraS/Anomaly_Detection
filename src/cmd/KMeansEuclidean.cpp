#include "cmd.h"
#include "dMRI/tractography/tractogram.h"
#include "utils/clusterHelpers.h"

using namespace NIBR;

namespace CMDARGS_KMEANS_EUC {
    std::string  inp_path;
    std::string  out_path;

    std::tuple<std::string, int, int, std::string, double> model_spec("", 0, 0, "", 1.0); // module_path, inp_dim, lat_dim, data type, distance scaler

    size_t  numClusters;
    int    maxIteration     = 0;

    int    batchSize        = 1000000;
    bool   randomize        = false;
    bool   shuffle          = false;
    bool   useCPU           = false;
    
    int numberOfThreads     =  0;
    std::string verbose     = "info";
    bool force              = false;
}

using namespace CMDARGS_KMEANS_EUC;

void run_kmeans_euclidean()
{ 

    parseCommon(numberOfThreads,verbose);
    if (!parseForceOutput(out_path,force)) return;

    if (getFileExtension(out_path) != "ccke") {
        disp(MSG_ERROR,"Output cluster center file must have .ccke extension.");
        return;
    }
    
    bool isValid = ensureVTKorTCK(inp_path);

    if (!isValid) {
        disp(MSG_ERROR,"Unknown tractogram type");
        return;
    }

    if (numClusters < 1) {
        disp(MSG_ERROR,"Minimum numClusters is 1");
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

    std::vector<NIBR::Streamline> clusterCenters;
    clusterCenters.reserve(numClusters);

    // Random initialization of cluster centers
    {
        std::vector<size_t> randomList(tracObj.size());
        std::iota(randomList.begin(), randomList.end(), 0);
        std::shuffle(randomList.begin(), randomList.end(), std::mt19937{std::random_device{}()});
        for (size_t i = 0; i < numClusters; ++i)
            clusterCenters.push_back(tracObj[randomList[i % tracObj.size()]]);
    }
    

    disp(MSG_INFO,"Clustering with k-means...");
    auto clusteringStartTime = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < maxIteration; iter++) {

        disp(MSG_INFO, "Iteration %d / %d", iter + 1, maxIteration);

        std::vector<std::vector<std::vector<float>>> accum(numClusters);
        std::vector<int> counts(numClusters, 0);
        for (size_t c = 0; c < numClusters; ++c)
            accum[c].resize(clusterCenters[0].size(), std::vector<float>(3, 0.0f));

        NIBR::StreamlineBatch batch;
        if (randomize)
            batch = getRandomBatch(iter, batchSize, tracObj, randomList);
        else
            batch = getOrderedBatch(iter, batchSize, tracObj);

        disp(MSG_DETAIL, "Current batchSize: %d", batchSize);

        // find best cluster center for this streamline
        std::vector<int> batchAssignments(batchSize, -1);

        auto assignTask = [&](NIBR::MT::TASK task) -> void {
            size_t i = task.no;
            double bestDist = std::numeric_limits<double>::max();
            int bestIdx = -1;

            for (size_t c = 0; c < numClusters; ++c) {
                double d = computeMDF(batch[i], clusterCenters[c]);
                if (d < bestDist) {
                    bestDist = d;
                    bestIdx = c;
                }
            }
            batchAssignments[i] = bestIdx;
        };

        NIBR::MT::MTRUN(batchSize, "Assigning clusters", assignTask);

        // get changes of new streamlines to their cluster centers
        for (int i = 0; i < batchSize; ++i) {
            int c = batchAssignments[i];
            NIBR::Streamline& thisStreamline = batch[i];
            for (size_t p = 0; p < thisStreamline.size(); ++p) {
                accum[c][p][0] += thisStreamline[p][0];
                accum[c][p][1] += thisStreamline[p][1];
                accum[c][p][2] += thisStreamline[p][2];
            }
            counts[c]++;
        }

        // update cluster centers from new streamlines
        auto updateTask = [&](NIBR::MT::TASK task) -> void {
            int c = task.no;
            if (counts[c] == 0) return;
            NIBR::Streamline meanStreamline(clusterCenters[0].size());
            for (size_t p = 0; p < meanStreamline.size(); ++p) {
                meanStreamline[p][0] = accum[c][p][0] / counts[c];
                meanStreamline[p][1] = accum[c][p][1] / counts[c];
                meanStreamline[p][2] = accum[c][p][2] / counts[c];
            }
            clusterCenters[c] = meanStreamline;
        };

        NIBR::MT::MTRUN(numClusters, "Updating cluster centers", updateTask);
    }
    auto clusteringEndTime = std::chrono::high_resolution_clock::now();
    auto latentMinduration = std::chrono::duration_cast<std::chrono::seconds>(clusteringEndTime - clusteringStartTime);
    disp(MSG_INFO, "K-Means clustering complete. Final clusters: %d. Time taken: %li", numClusters, latentMinduration);



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
    
     
void kmeans_euclidean(CLI::App* app)   
{ 

    app->formatter(std::make_shared<CustomHelpFormatter>());

    app->description("finds streamline cluster centers in eunclidean space");

    app->add_option("<input>",               inp_path,           "Input path. Tractogram")
        ->required();
     
    app->add_option("<output>",              out_path,           "Output latent representations of cluster centers (.ccke).")
        ->required();

    app->add_option("--numClusters, -d",         numClusters,            "Maximum amount of cluster centers")
        ->required();

    app->add_option("--maxIteration, -i",    maxIteration,       "Limits maximum number of iterations in constrast to the default, which iterates until all streamlines processed.");
    
    app->add_option("--batchSize, -b",       batchSize,          "Batch size. Number of streamlines to process at each iteration. Default: 1000000");

    app->add_flag("--randomize, -r",         randomize,          "Shuffle the streamlines before starting");
    app->add_flag("--useCPU, -c",            useCPU,             "Use only CPU without checking any available GPUs.");

    app->add_option("--numberOfThreads, -n", numberOfThreads,    "Number of threads.");
    app->add_option("--verbose, -v",         verbose,            "Verbose level. Options are \"quite\",\"fatal\",\"error\",\"warn\",\"info\" and \"debug\". Default=info");
    app->add_flag("--force, -f",             force,              "Force overwriting of existing file");

    app->callback(run_kmeans_euclidean);  
     
}

                       