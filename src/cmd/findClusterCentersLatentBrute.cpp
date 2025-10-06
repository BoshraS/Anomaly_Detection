#include "cmd.h"
#include "dMRI/tractography/tractogram.h"
#include "utils/modelTestHelpers.h"
#include <cmath>
#include <string>

using namespace NIBR;

namespace CMDARGS_FINDCLUSTERCENTERS_LATB {
    std::string  inp_path;
    std::string  out_path;
    std::string encodedPath = "";

    std::tuple<std::string, int, int, std::string, double> model_spec("", 0, 0, "", 1.0); // module_path, inp_dim, lat_dim, data type, distance scaler

    float  maxDist;
    int    maxIteration     = 0;

    int    batchSize        = 1000000;
    int    encodeBatchSize  = 10000;
    bool   randomize        = false;
    bool   shuffle          = false;
    bool   useCPU           = false;
    bool    skipResampleBool = false;
    
    int numberOfThreads     =  0;
    std::string verbose     = "info";
    bool force              = false;

}

using namespace CMDARGS_FINDCLUSTERCENTERS_LATB;

std::vector<std::vector<double>>  getRandomBatch(size_t batchId, size_t batchSize, std::vector<std::vector<double>>  &encStreamlines, std::vector<size_t> &randomList) {
    size_t startIdx = batchId * batchSize;
    size_t loopCount = std::min(batchSize, encStreamlines.size() - startIdx);
    std::vector<std::vector<double>>  thisBatch(loopCount);
    for(size_t i = 0; i < loopCount; ++i){
        thisBatch[i] = encStreamlines[randomList[startIdx+i]];
    }
    return thisBatch;
}

std::vector<std::vector<double>>  getOrderedBatch(size_t batchId, size_t batchSize, std::vector<std::vector<double>>  &encStreamlines) {
    size_t startIdx = batchId * batchSize;
    size_t loopCount = std::min(batchSize, encStreamlines.size() - startIdx);
    std::vector<std::vector<double>>  thisBatch(loopCount);
    for(size_t i = 0; i < loopCount; ++i){
        thisBatch[i] = encStreamlines[startIdx+i];
    }
    return thisBatch;
}

void run_findClusterCenters_latentBrute()
{ 

    parseCommon(numberOfThreads,verbose);
    if (!parseForceOutput(out_path,force)) return;

    if (getFileExtension(out_path) != "clcl") {
        disp(MSG_ERROR,"Output cluster center file must have .clcl extension.");
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

    if (encodeBatchSize < 1) {
        disp(MSG_ERROR,"Minimum batchSize is 1");
        return;
    }



    // Open tractogram reader
    NIBR::TractogramReader tractogram(inp_path, false);
    

    // Set model
    StreamlineAutoencoder model = StreamlineAutoencoder(model_spec, useCPU);
    if (!model.isReady()) return;
    
    NIBR::StreamlineBatch streamlines;
    


    

    // Latent space representations
    std::vector<std::vector<double>>              enc_streamlines;
    
    auto run_test_with_type = [&](auto type_placeholder) {
        using T = decltype(type_placeholder);
        auto lat_streamlines = encodeStreamlines<T>(streamlines,model,encodeBatchSize);         // Encode streamlines in latent space
        auto enc_streamlines_new      =  to_double_vector_remove_flipped<T>(lat_streamlines, model.latDim);                     // Convert latent space representation to double type for analysis

        enc_streamlines.insert(enc_streamlines.end(), std::make_move_iterator(enc_streamlines_new.begin()), std::make_move_iterator(enc_streamlines_new.end()));
    };

    if(encodedPath != "") {
        enc_streamlines = loadEncodedFromDisk(encodedPath);
    } else {

        size_t resampleBatchAmount = 1000000;
        size_t resampleBatchCount = std::ceil((tractogram.getNumberOfStreamlines() + resampleBatchAmount - 1) / resampleBatchAmount);

        NIBR::Tractogram tracObj = tractogram.getTractogram();

        for(size_t i = 0; i < resampleBatchCount; ++i)
        {
            disp(MSG_INFO,"Starting resample/encode batch %d.", i);
            size_t startPos = i * resampleBatchAmount;
            size_t endPos = std::min(((i + 1) * resampleBatchAmount), tractogram.getNumberOfStreamlines());
            NIBR::StreamlineBatch slice(tracObj.begin() + startPos,tracObj.begin() + endPos);

            if(!skipResampleBool) {
                streamlines = NIBR::resampleTractogram_withStepCount(slice, model.inpDim);
                disp(MSG_DETAIL,"Resampled streamlines for %d points.", model.inpDim);
            } else {
                streamlines = slice;
            }

            // Dispatch to the generic lambda with the correct type
            if (model.dtype == torch::kFloat)       { run_test_with_type(float{});   } 
            else if (model.dtype == torch::kDouble) { run_test_with_type(double{});  } 
            else if (model.dtype == torch::kHalf)   { run_test_with_type(at::Half{});} 
            else { disp(MSG_ERROR, "Unsupported data type for modelTest: %s", c10::toString(model.dtype)); }
        }

        NIBR::Tractogram().swap(tracObj);
        

        



        

        std::string enc_filename = "encoded_" + std::to_string(enc_streamlines.size()) + "_dim_" + std::to_string(model.latDim) + ".bin";
        saveEncodedToDisk(enc_streamlines, enc_filename);
        
    }

    float adjMaxDist = (maxDist/model.distScaler);


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
    
    if (randomize) {
        if (maxIteration < 1) maxIteration = 1;
    } else {
        if (maxIteration < 1) maxIteration = std::ceil(float(totalCnt) / float(batchSize));    
    }

    // Do the clustering
    std::vector<std::vector<double>>  clusterCenters;

    disp(MSG_INFO,"Clustering...");
    auto clusteringStartTime = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < maxIteration; iter++) {
        
        // Get a batch
        std::vector<std::vector<double>>  batch;

        if (randomize) {
            batch = getRandomBatch(iter, batchSize, enc_streamlines, randomList);
        } else {
            batch = getOrderedBatch(iter, batchSize, enc_streamlines);
        }

        disp(MSG_DETAIL,"Current batchSize: %d", batchSize);


        std::vector<std::vector<double>>  localClusterCenters;       // New clusters found in this batch.
        
        std::vector<std::atomic<bool>> unassigned(batchSize);    // True if a streamline in the batch did not belong to any existing cluster
        for (int i = 0; i < batchSize; ++i) unassigned[i] = false;

        std::vector<size_t> unaInd;                              // Indices of the streamlines, which did not belong to any existing cluster
        std::vector<std::vector<double>>  unassignedClusterCenters;   // Clusters of the unassigned streamlines that were not clusered within a batch
        size_t taskOffset = 0;

        auto addToGlobalCluster = [&](NIBR::MT::TASK task) -> void {

            if (!clusterCenters.empty()) {
                bool tooClose = false;
                for (size_t c = 0; c < clusterCenters.size(); ++c) {
                    double dist = latentMinDistanceCalculator(batch[task.no], clusterCenters[c], model.latDim);
                    if (dist < adjMaxDist) {
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
                    double dist = latentMinDistanceCalculator(batch[task.no], localClusterCenters[c], model.latDim);
                    if (dist < adjMaxDist) {
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
                        float dist = latentMinDistanceCalculator(batch[rInd], unassignedClusterCenters[ind], model.latDim);
                        if (dist < adjMaxDist) return;
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



    // Open binary file for writing cluster centers
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
    
     
void findClusterCenters_latentBrute(CLI::App* app)   
{ 

    app->formatter(std::make_shared<CustomHelpFormatter>());

    app->description("finds streamline cluster centers based on their latent space representation using bruteforce");

    app->add_option("<input>",               inp_path,           "Input path. Tractogram")
        ->required();
    
    app->add_option("<model>",               model_spec,     "Input model, specified with the path to the Torch script file, followed by the input dimensions, latent space dimensions, data type (float or double), and distance scaling factor of the model. E.g. /model/test_model.pt 256 64 float 0.08. ")
        ->required();
     
    app->add_option("<output>",              out_path,           "Output latent representations of cluster centers (.clcl).")
        ->required();

    app->add_option("--maxDist, -d",         maxDist,            "Maximum distance from any cluster center.")
        ->required();

    app->add_option("--encoded, -e", encodedPath, "A .bin file which contains the encoded streamlines. If given skips the encoding part and uses these directly.");

    app->add_option("--maxIteration, -i",    maxIteration,       "Limits maximum number of iterations in constrast to the default, which iterates until all streamlines processed.");
    
    app->add_option("--batchSize, -b",       batchSize,          "Batch size. Number of streamlines to process at each iteration. Default: 1000000");

    app->add_flag("--skipResample, -s", skipResampleBool, "Won't resample the tractogram. If you enable this you are responsible for resampling the tractogram to fit the given model.");

    app->add_flag("--randomize, -r",         randomize,          "Shuffle the streamlines before starting");
    app->add_flag("--useCPU, -c",            useCPU,             "Use only CPU without checking any available GPUs.");

    app->add_option("--numberOfThreads, -n", numberOfThreads,    "Number of threads.");
    app->add_option("--verbose, -v",         verbose,            "Verbose level. Options are \"quite\",\"fatal\",\"error\",\"warn\",\"info\" and \"debug\". Default=info");
    app->add_flag("--force, -f",             force,              "Force overwriting of existing file");

    app->callback(run_findClusterCenters_latentBrute);  
     
}

                       