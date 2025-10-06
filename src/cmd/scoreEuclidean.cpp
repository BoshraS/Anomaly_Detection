#include "base/verbose.h"
#include "cmd.h"
#include "dMRI/tractography/tractogram.h"
#include <ATen/core/Dimname.h>

using namespace NIBR;

namespace CMDARGS_SCORE_EUC {
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

    size_t points_per_streamline        = 8;
    bool skip_resample = false;

    size_t limit_centers = 0;
}

using namespace CMDARGS_SCORE_EUC;

std::vector<NIBR::Streamline> readClusterCentersEuclidean(const std::string& fname, size_t points_per_streamline, size_t limit_centers) {
    std::ifstream ifs(fname, std::ios::binary | std::ios::ate | std::ios::in);
    if (!ifs) {
        disp(MSG_ERROR,"Failed to open file: %s", fname.c_str());
        return std::vector<NIBR::Streamline>();
    }
    ifs.seekg(0, std::ios::end);
    size_t file_size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    size_t bytes_per_streamline = points_per_streamline * 3 * sizeof(float);
    if (file_size % bytes_per_streamline != 0) {
        disp(MSG_ERROR,"File size doesn't match pointsPerStreamline, filesize: %d, bytes_per_streamline: %d", file_size, bytes_per_streamline);
    }

    size_t n_streamlines = file_size / bytes_per_streamline;

    n_streamlines = limit_centers != 0  && limit_centers < n_streamlines ? limit_centers : n_streamlines;

    std::vector<NIBR::Streamline> clusterCenters;
    clusterCenters.reserve(n_streamlines);

    disp(MSG_INFO,"Starting read cluster centers for %d streamlines", n_streamlines);

    for (size_t i = 0; i < n_streamlines; ++i) {
        NIBR::Streamline streamline(points_per_streamline);
        ifs.read(reinterpret_cast<char*>(streamline.data()),bytes_per_streamline);
        clusterCenters.push_back(std::move(streamline));
    }

    return clusterCenters;
}

double computeMDF_(NIBR::Streamline &s1, NIBR::Streamline &s2) {
    double sum = 0.0f;
    for (size_t i = 0; i < s1.size(); ++i) {
        double dx = s1[i][0] - s2[i][0];
        double dy = s1[i][1] - s2[i][1];
        double dz = s1[i][2] - s2[i][2];
        sum += dx*dx + dy*dy + dz*dz;
    }
    return sum / double(s1.size());
}



 
void run_score_euc()
{ 

    parseCommon(numberOfThreads,verbose);

    // Check input/output
    if (!parseForceOutput(out_path,force)) return;

    if (getFileExtension(clc_path) != "clce") {
        disp(MSG_ERROR,"Input cluster centers must have .clce extension.");
        return;
    }

    if (getFileExtension(out_path) != "anoe") {
        disp(MSG_ERROR,"Output anomaly score file must have .anoe extension.");
        return;
    }

    if ((out_labels != "") &&  (getFileExtension(out_labels) != "clbe")) {
        disp(MSG_ERROR,"Output cluster label file must have .clbe extension.");
        return;
    }

    NIBR::TractogramReader tractogram(inp_path, false);
    //NIBR::Tractogram tracObj = tractogram.getTractogram();

    int N = tractogram.numberOfStreamlines;

    auto clusterCenters = readClusterCentersEuclidean(clc_path, points_per_streamline, limit_centers);

    // Compute batch count
    int batchCnt = (N < batchSize) ? 1 : (N + batchSize - 1) / batchSize;

    std::vector<float>  scores(N,0);
    std::vector<int>    clusters(N,0);

    auto bruteForceEucNN = [&](NIBR::Streamline query, size_t& bestIndex, double& bestDist) {
        bestDist  = std::numeric_limits<double>::max();
        bestIndex = 0;
        for (size_t i = 0; i < clusterCenters.size(); i++) {
            double d = computeMDF_(query, clusterCenters[i]);
            if (d < bestDist) {
                bestDist  = d;
                bestIndex = i;
            }
        }
    };

    disp(MSG_INFO,"Starting scoring of %d streamlines using %d cluster centers", N, clusterCenters.size());

    // Iterate through the whole tractogram in batches
    auto clusteringStartTime = std::chrono::high_resolution_clock::now();
    for (int batchNo = 0; batchNo < batchCnt; batchNo++) {

        int curBatchSize = ((batchNo+1)*batchSize < N) ? batchSize : (N-batchNo*batchSize);

        NIBR::StreamlineBatch thisBatch(batchSize);
        for(int i = 0; i < batchSize; ++i) {
            auto new_streamline = tractogram.getNextStreamline();
            if (std::get<0>(new_streamline))
                thisBatch[i] = std::get<1>(new_streamline);
            else {
                --i;
            }
             
        }

        auto calcScore = [&](NIBR::MT::TASK task) -> void {
            size_t closestCenterIndex;
            double dist;
            bruteForceEucNN(thisBatch[task.no], closestCenterIndex, dist);
            scores  [task.no + batchNo * batchSize] = dist;
            clusters[task.no + batchNo * batchSize] = closestCenterIndex + 1;
        };

        NIBR::MT::MTRUN(curBatchSize, "Computing anomaly scores " + to_string_with_precision(batchNo+1) + " / " + to_string_with_precision(batchCnt) , calcScore);
        
    }
    auto clusteringEndTime = std::chrono::high_resolution_clock::now();
    auto latentMinduration = std::chrono::duration_cast<std::chrono::seconds>(clusteringEndTime - clusteringStartTime);
    disp(MSG_INFO,"Done. Time taken: %li seconds", latentMinduration);


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
    
     
void score_euc(CLI::App* app)   
{ 

    app->formatter(std::make_shared<CustomHelpFormatter>());

    const std::string info = "This function computes an anomaly score for each streamline. The anomaly score is the distance to the closest cluster center. The function can optionally save the index of the closest cluster.";

    setInfo(app,info);

    app->description("assigns an anomaly score to each streamline");

    app->add_option("<input>",                 inp_path,           "Tractogram")
        ->required();

    app->add_option("<cluster centers>",       clc_path,           "Path to reference cluster centers (.clce).")
        ->required();
     
    app->add_option("<output anomaly scores>", out_path,           "Output anomaly scores (.anoe).")
        ->required();

    app->add_option("<output cluster labels>", out_labels,         "Optional cluster label output (.clbe).");

    app->add_option("--pointsPer, -p", points_per_streamline, "How many points there are per streamline in the cluster centers. Default. 8");

    app->add_option("--limitCenters", limit_centers, "Only take x amount of cluster centers. 0 takes all and is the default.");

    app->add_option("--batchSize, -b",         batchSize,          "Batch size. Default: 1000000.");
    app->add_flag("--useCPU, -c",              useCPU,             "Use only CPU without checking any available GPUs.");

    app->add_option("--numberOfThreads, -n",   numberOfThreads,    "Number of threads.");
    app->add_option("--verbose, -v",           verbose,            "Verbose level. Options are \"quite\",\"fatal\",\"error\",\"warn\",\"info\" and \"debug\". Default=info");
    app->add_flag("--force, -f",               force,              "Force overwriting of existing file");

    app->callback(run_score_euc);  
     
}

                       