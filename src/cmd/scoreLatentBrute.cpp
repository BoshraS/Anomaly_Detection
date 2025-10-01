#include "cmd.h"
#include "utils/modelTestHelpers.h"

using namespace NIBR;

namespace CMDARGS_SCORE_LATENT {
    std::string  inp_path;
    std::string  out_path;
    std::string  out_labels = "";
    std::string  clc_path;

    std::tuple<std::string, int, int, std::string, double> model_spec("", 0, 0, "", 1.0); // module_path, inp_dim, lat_dim, data type, distance scaler

    int  batchSize          = 1000000;
    bool useCPU             = false;
    
    int numberOfThreads     =  0;
    std::string verbose     = "info";
    bool force              = false;
}

using namespace CMDARGS_SCORE_LATENT;

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




 
void run_score_latent()
{ 

    parseCommon(numberOfThreads,verbose);

    // Check input/output
    if (!parseForceOutput(out_path,force)) return;

    if (getFileExtension(inp_path) != "bin") {
        disp(MSG_ERROR,"Input latent representation must have .bin extension.");
        return;
    }

    if (getFileExtension(clc_path) != "clc") {
        disp(MSG_ERROR,"Input cluster centers must have .clc extension.");
        return;
    }

    if (getFileExtension(out_path) != "ano") {
        disp(MSG_ERROR,"Output anomaly score file must have .ano extension.");
        return;
    }

    if ((out_labels != "") &&  (getFileExtension(out_labels) != "clb")) {
        disp(MSG_ERROR,"Output cluster label file must have .clb extension.");
        return;
    }

    // Set model
    StreamlineAutoencoder model = StreamlineAutoencoder(model_spec, useCPU);
    if (!model.isReady()) return;

    // Initialize file reader for input latent representations
    std::ifstream ifs(inp_path, std::ios::binary | std::ios::ate | std::ios::in);
    if (!ifs.is_open()) {
        disp(MSG_ERROR,"Failed to open file: %s", inp_path.c_str());
        return;
    }
    int N = ifs.tellg() / (sizeof(float) * 2 * model.latDim); // Number of streamlines in the input file
    ifs.seekg(0, std::ios::beg);


    // Read latent space representation of cluster centers
    disp(MSG_DETAIL,"Reading cluster centers");
    PointCloud cloud;
    cloud.points = readClusterCenters_(clc_path,model);

    // Build the KD-Tree
    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud>,PointCloud,-1> KDTree;
    KDTree kdtree(model.latDim, cloud, nanoflann::KDTreeSingleIndexAdaptorParams( ((256/model.latDim) > 10) ? 10 : (256/model.latDim) ) ) ;
    kdtree.buildIndex();

    // Compute batch count
    int batchCnt = (N < batchSize) ? 1 : (N + batchSize - 1) / batchSize;

    std::vector<float>  scores(N,0);
    std::vector<int>    clusters(N,0);

    disp(MSG_DETAIL,"Starting scoring of %d streamlines using %d cluster centers", N, cloud.points.size());

    auto bruteForceNN = [&](const Eigen::VectorXf& query, size_t& bestIndex, double& bestDist) {
        bestDist  = std::numeric_limits<double>::max();
        bestIndex = 0;
        for (size_t i = 0; i < cloud.points.size(); i++) {
            double d = latentMinDistanceCalculatorEigen(query, cloud.points[i], model.latDim);
            if (d < bestDist) {
                bestDist  = d;
                bestIndex = i;
            }
        }
    };

    // Iterate through the whole tractogram in batches


    auto clusteringStartTime = std::chrono::high_resolution_clock::now();
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
            size_t closestCenterIndex;
            double dist;
            bruteForceNN(batch[2*task.no], closestCenterIndex, dist);
            scores  [task.no + batchNo * batchSize] = dist * model.distScaler;
            clusters[task.no + batchNo * batchSize] = closestCenterIndex + 1;
        };

        NIBR::MT::MTRUN(curBatchSize, "Computing anomaly scores " + to_string_with_precision(batchNo+1) + " / " + to_string_with_precision(batchCnt) , calcScore);
        
    }
    auto clusteringEndTime = std::chrono::high_resolution_clock::now();
    auto latentMinduration = std::chrono::duration_cast<std::chrono::seconds>(clusteringEndTime - clusteringStartTime);
    disp(MSG_INFO,"Done. Time taken: %li seconds", latentMinduration);
    ifs.close();


    // Open binary file for writing the anomaly scores
    disp(MSG_DETAIL,"Writing scores");
    std::ofstream ofs(out_path, std::ios::binary);
    if (!ofs) {
        disp(MSG_ERROR, "Failed to open output score file.");
        return;
    }
    ofs.write(reinterpret_cast<char*>(scores.data()), scores.size() * sizeof(float));
    ofs.close();
    disp(MSG_DETAIL,"Done");

    if (out_labels != "") {
        // Open binary file for writing the index of closest cluster label
        disp(MSG_DETAIL,"Writing labels");
        std::ofstream ofs(out_labels, std::ios::binary);
        if (!ofs) {
            disp(MSG_ERROR, "Failed to open output label file.");
            return;
        }
        ofs.write(reinterpret_cast<char*>(clusters.data()), clusters.size() * sizeof(int));
        ofs.close();
        disp(MSG_DETAIL,"Done");
    }

    return;
       
}          
    
     
void score_latent(CLI::App* app)   
{ 

    app->formatter(std::make_shared<CustomHelpFormatter>());

    const std::string info = "This function computes an anomaly score for each streamline. The anomaly score is the distance to the closest cluster center. The function can optionally save the index of the closest cluster.";

    setInfo(app,info);

    app->description("assigns an anomaly score to each streamline");

    app->add_option("<input>",                 inp_path,           "Input latent representations (.bin)")
        ->required();

    app->add_option("<cluster centers>",       clc_path,           "Path to reference cluster centers (.clc).")
        ->required();
     
    app->add_option("<output anomaly scores>", out_path,           "Output anomaly scores (.ano).")
        ->required();

    app->add_option("<output cluster labels>", out_labels,         "Optional cluster label output (.clb).");

    app->add_option("--model,-m",              model_spec,         "Input model, specified with the path to the Torch script file, followed by the input dimensions, latent space dimensions, and distance scaling factor of the model. E.g. /model/test_model.pt 256 64 0.08. The distance scaler factor of the model can be obtained using the \"modelTest\" command.");

    app->add_option("--batchSize, -b",         batchSize,          "Batch size. Default: 1000000.");
    app->add_flag("--useCPU, -c",              useCPU,             "Use only CPU without checking any available GPUs.");

    app->add_option("--numberOfThreads, -n",   numberOfThreads,    "Number of threads.");
    app->add_option("--verbose, -v",           verbose,            "Verbose level. Options are \"quite\",\"fatal\",\"error\",\"warn\",\"info\" and \"debug\". Default=info");
    app->add_flag("--force, -f",               force,              "Force overwriting of existing file");

    app->callback(run_score_latent);  
     
}

                       