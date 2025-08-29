#include "cmd.h"
#include <chrono>

namespace CMDARGS_COMPARESPEED {
    std::string  inp_path;

    std::string  bin_path;
    
    std::tuple<std::string, int, int, std::string> inp_model_spec("", 0, 0, ""); // module_path, inp_dim, lat_dim, data type

    int  batchSize          = 512;
    bool useCPU             = false;
    
    int numberOfThreads     =  0;
    std::string verbose     = "info";
    bool force              = false;

    std::string  clc_path;
}

using namespace CMDARGS_COMPARESPEED;

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

// Reads latent representations of streamlines from .bin file
std::vector<Eigen::VectorXf> readClusterCenters_(const std::string& fname, StreamlineAutoencoder& model) {
    std::ifstream ifs(fname, std::ios::binary | std::ios::ate | std::ios::in);
    if (!ifs.is_open()) {
        disp(MSG_ERROR,"Failed to open file: %s", fname.c_str());
        return std::vector<Eigen::VectorXf>();
    }

    std::streampos fileSize = ifs.tellg(); // Move the file pointer to the end of file
    int streamlineCount = fileSize / (sizeof(float) * 2 * model.latDim);
    ifs.seekg(0, std::ios::beg);           // Move the file pointer back to the beginning

    std::vector<Eigen::VectorXf> latent;
    latent.reserve(streamlineCount); // Reserve space for streamlineCount elements

    // Read all streamline data in a single operation
    Eigen::VectorXf buffer(2 * model.latDim * streamlineCount);
    ifs.read(reinterpret_cast<char*>(buffer.data()), sizeof(float) * 2 * model.latDim * streamlineCount);

    for (int s = 0; s < streamlineCount; s++) {
        // Map the first model.latDim elements to Eigen::VectorXf
        Eigen::VectorXf lat = buffer.segment(s * 2 * model.latDim, model.latDim);
        latent.push_back(std::move(lat));
    }
    ifs.close();

    return latent;
}


void run_compareSpeed()
{ 
    parseCommon(numberOfThreads,verbose);

    bool isValid = ensureVTKorTCK(inp_path);

    if (!isValid) {
        disp(MSG_ERROR,"Unknown tractogram type");
        return;
    }

    // Set model
    StreamlineAutoencoder model = StreamlineAutoencoder(inp_model_spec, useCPU);
    if (!model.isReady()) return;

     // Prepare tractogram
    NIBR::TractogramReader tractogram(inp_path, false);

    // Original input streamline
    NIBR::Tractogram  streamlines = resampleTractogram_withStepCount(tractogram.getTractogram(), model.inpDim);

    disp(MSG_DETAIL,"Resampled streamlines for %d points.", model.inpDim);


    // Latent space representations
    std::vector<std::vector<double>>              enc_streamlines;

    // Decoded streamlines from the latent space representations
    NIBR::Tractogram  dec_streamlines;

    auto run_test_with_type = [&](auto type_placeholder) {
        using T = decltype(type_placeholder);
        auto lat_streamlines = encodeStreamlines<T>(streamlines,model,batchSize);         // Encode streamlines in latent space
        dec_streamlines      = decodeStreamlines<T>(lat_streamlines, model, batchSize);   // Decode the encoded streamlines
        enc_streamlines      = to_double_vector<T>(lat_streamlines);                     // Convert latent space representation to double type for analysis
        // disp(MSG_INFO,"Encoding completed.");
    };
     
    // Dispatch to the generic lambda with the correct type
    if (model.dtype == torch::kFloat)       { run_test_with_type(float{});   } 
    else if (model.dtype == torch::kDouble) { run_test_with_type(double{});  } 
    else if (model.dtype == torch::kHalf)   { run_test_with_type(at::Half{});} 
    else { disp(MSG_ERROR, "Unsupported data type for modelTest: %s", c10::toString(model.dtype)); }

    std::vector<std::vector<double>> hau_dist         (streamlines.size(), std::vector<double>(streamlines.size(),NAN));
    std::vector<std::vector<double>> mdf_dist         (streamlines.size(), std::vector<double>(streamlines.size(),NAN));


    std::vector<double> enc_dec_hau_dist (streamlines.size());
    std::vector<double> enc_dec_mdf_dist (streamlines.size());

    

    auto getDistances_mdf = [&](NIBR::MT::TASK task) -> void {
        for (size_t i = 0; i < task.no; i++) {
            mdf_dist[task.no][i]            = getMDFDistance(streamlines[task.no], streamlines[i]);
        }
        enc_dec_mdf_dist[task.no] = getMDFDistance(streamlines[task.no], dec_streamlines[task.no]);
    };

    auto getDistances_hausdorf = [&](NIBR::MT::TASK task) -> void {
        for (size_t i = 0; i < task.no; i++) {
            hau_dist[task.no][i]            = getHausdorffDistance(streamlines[task.no], streamlines[i]);
        }
        enc_dec_hau_dist[task.no] = getHausdorffDistance(streamlines[task.no], dec_streamlines[task.no]);
    };


    auto mdfStartTime = std::chrono::high_resolution_clock::now();

    NIBR::MT::MTRUN(streamlines.size(), "Computing mdf", getDistances_mdf);

    auto mdfEndTime = std::chrono::high_resolution_clock::now();

    auto mdfDuration = std::chrono::duration_cast<std::chrono::seconds>(mdfEndTime - mdfStartTime);

    std::cout << "mdf execution time: " << mdfDuration.count() << " seconds" << std::endl;



    auto hausdorfStartTime = std::chrono::high_resolution_clock::now();

    NIBR::MT::MTRUN(streamlines.size(), "Computing hausdorf", getDistances_hausdorf);

    auto hausdorfEndTime = std::chrono::high_resolution_clock::now();

    auto hausdorfDuration = std::chrono::duration_cast<std::chrono::seconds>(hausdorfEndTime - hausdorfStartTime);

    std::cout << "hausdorf execution time: " << hausdorfDuration.count() << " seconds" << std::endl;


    // kd tree starts here
    

    // Read latent space representation of cluster centers
    disp(MSG_DETAIL,"Reading cluster centers");
    PointCloud cloud;
    cloud.points = readClusterCenters_(clc_path,model);


    // Build the KD-Tree
    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud>,PointCloud,-1> KDTree;
    KDTree kdtree(model.latDim, cloud, nanoflann::KDTreeSingleIndexAdaptorParams( ((256/model.latDim) > 10) ? 10 : (256/model.latDim) ) ) ;
    kdtree.buildIndex();

    // Initialize file reader for input latent representations
    std::ifstream ifs(bin_path, std::ios::binary | std::ios::ate | std::ios::in);
    if (!ifs.is_open()) {
        disp(MSG_ERROR,"Failed to open file: %s", bin_path.c_str());
        return;
    }
    int N = ifs.tellg() / (sizeof(float) * 2 * model.latDim); // Number of streamlines in the input file
    ifs.seekg(0, std::ios::beg);

    // Compute batch count
    int batchCnt = (N < batchSize) ? 1 : (N + batchSize - 1) / batchSize;

    std::vector<float>  scores(N,0);
    std::vector<int>    clusters(N,0);

    disp(MSG_DETAIL,"Starting scoring of %d streamlines using %d cluster centers", N, cloud.points.size());

    // Iterate through the whole tractogram in batches

    std::cout << "batchcount: " << batchCnt << std::endl;
    auto kdtreeTimerStart = std::chrono::high_resolution_clock::now();

    for (int batchNo = 0; batchNo < batchCnt; batchNo++) {

        int curBatchSize = ((batchNo+1)*batchSize < N) ? batchSize : (N-batchNo*batchSize);

        // Batch has twice the number of elements, one for the direct, one for the flipped representations
        std::vector<Eigen::VectorXf> batch(2*curBatchSize);

        Eigen::VectorXf buffer(2 * model.latDim*curBatchSize);
        ifs.read(reinterpret_cast<char*>(buffer.data()), 2 * model.latDim*sizeof(float)*curBatchSize);

        // Read latent representations from the input file
        for (int i = 0; i < curBatchSize; i++) {                
            // First model.latDim elements of the 2 * model.latDim-element segment
            Eigen::VectorXf rep1 = buffer.segment(2 * model.latDim * i, model.latDim);
            batch[2 * i] = std::move(rep1);

            // Second model.latDim elements of the 2 * model.latDim-element segment
            Eigen::VectorXf rep2 = buffer.segment(2 * model.latDim * i + model.latDim, model.latDim);
            batch[2 * i + 1] = std::move(rep2);
        }

        auto calcScore = [&](NIBR::MT::TASK task) -> void {

            size_t closestCenterIndex1, closestCenterIndex2;
            float  dist1, dist2;

            nanoflann::KNNResultSet<float> resultSet1(1);
            resultSet1.init(&closestCenterIndex1, &dist1);
            kdtree.findNeighbors(resultSet1,   batch[2*task.no].data(), nanoflann::SearchParameters());

            nanoflann::KNNResultSet<float> resultSet2(1);
            resultSet2.init(&closestCenterIndex2, &dist2);
            kdtree.findNeighbors(resultSet2, batch[2*task.no+1].data(), nanoflann::SearchParameters());

            // Approximate physical space distance by multiplying the average Euclidean distance by 5
            if (dist1 < dist2) {
                scores  [task.no + batchNo * batchSize] = std::sqrt(dist1) * model.distScaler;
                clusters[task.no + batchNo * batchSize] = closestCenterIndex1 + 1; // Adding one so min cluster label is 1
            } else {
                scores  [task.no + batchNo * batchSize] = std::sqrt(dist2) * model.distScaler;
                clusters[task.no + batchNo * batchSize] = closestCenterIndex2 + 1; // Adding one so min cluster label is 1
            }

        };

        

        NIBR::MT::MTRUN(curBatchSize, "Computing anomaly scores " + to_string_with_precision(batchNo+1) + " / " + to_string_with_precision(batchCnt) , calcScore);
        

        

    }
    ifs.close();
    auto kdtreeTimerEnd = std::chrono::high_resolution_clock::now();

    auto kdtreeDuration = std::chrono::duration_cast<std::chrono::seconds>(kdtreeTimerEnd - kdtreeTimerStart);

    std::cout << "kdtree execution time: " << kdtreeDuration.count() << " seconds" << std::endl;

}

void compareSpeed(CLI::App* app)   
{ 

    app->formatter(std::make_shared<CustomHelpFormatter>());

    const std::string info = "Compare speed of k-d tree with hausdorf and mdf";

    app->add_option("<input tractogram>",    inp_path,           "Input tractogram (.vtk, .tck)")
        ->required();

    app->add_option("<model>",               inp_model_spec,     "Input model, specified with the path to the Torch script file, followed by the input dimensions, latent space dimensions, data type (float or double), and distance scaling factor of the model. E.g. /model/test_model.pt 256 64 float 0.08. ")
        ->required();
    
    app->add_option("<cluster centers>",       clc_path,           "Path to reference cluster centers (.clc).")
        ->required();

    app->add_option("<input bin>",                 bin_path,           "Input latent representations (.bin)")
        ->required();

    app->add_flag("--useCPU, -c",            useCPU,             "Use only CPU without checking any available GPUs.");

    app->add_option("--numberOfThreads, -n", numberOfThreads,    "Number of threads.");
    app->add_option("--verbose, -v",         verbose,            "Verbose level. Options are \"quite\",\"fatal\",\"error\",\"warn\",\"info\" and \"debug\". Default=info");
    app->add_flag("--force, -f",             force,              "Force overwriting of existing file");


    app->callback(run_compareSpeed);  
     
}