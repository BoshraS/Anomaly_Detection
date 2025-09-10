#include "cmd.h"
#include "dMRI/tractography/tractogram.h"
#include "dMRI/tractography/utility/streamline_operators.h"
#include <chrono>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <string>
#include <cstddef>


#ifdef _HAS_MATPLOT_
#include <matplot/matplot.h>
#endif

using namespace NIBR;

namespace CMDARGS_MODELTEST_PRECALC {
    std::string  inp_path;
    
    std::tuple<std::string, int, int, std::string> inp_model_spec("", 0, 0, ""); // module_path, inp_dim, lat_dim, data type

    int  batchSize          = 512;
    bool useCPU             = false;
    
    int numberOfThreads     =  0;
    std::string verbose     = "info";
    bool force              = false;

    std::string enc1_path    = "";
    std::string enc2_path    = "";
    std::string hau_path    = "";
    std::string mdf_path    = "";
    std::string lat1_path    = "";
    std::string lat2_path    = "";

}

using namespace CMDARGS_MODELTEST_PRECALC; 

#include "utils/modelTestHelpers.h"

/*
struct MMapVector {
    int fd = -1;
    size_t size = 0;
    double* data = nullptr;
};

// Convert std::vector to Eigen::VectorXd
Eigen::VectorXd vectorToEigen(const std::vector<double>& v) {
    return Eigen::VectorXd::Map(v.data(), v.size());
}

Eigen::Map<const Eigen::VectorXd> vectorToEigen(const MMapVector& mv) {
    return Eigen::Map<const Eigen::VectorXd>(mv.data, mv.size);
}

// Function to calculate the Pearson correlation coefficient
double correlation_coefficient(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.empty()) {
        throw std::invalid_argument("Vectors must be of same size and non-empty");
    }

    Eigen::VectorXd X = vectorToEigen(x);
    Eigen::VectorXd Y = vectorToEigen(y);

    double mean_X = X.mean();
    double mean_Y = Y.mean();

    Eigen::VectorXd X_centered = X.array() - mean_X;
    Eigen::VectorXd Y_centered = Y.array() - mean_Y;

    double covariance = (X_centered.dot(Y_centered)) / (X.size() - 1);  // Using (N-1) for sample covariance
    double stddev_X = std::sqrt(X_centered.squaredNorm() / (X.size() - 1));
    double stddev_Y = std::sqrt(Y_centered.squaredNorm() / (Y.size() - 1));

    if (stddev_X == 0 || stddev_Y == 0) return 0; // Avoid division by zero
    
    return covariance / (stddev_X * stddev_Y);
}

template <typename VectorTypeX, typename VectorTypeY>
double correlation_coefficient(const VectorTypeX& x, const VectorTypeY& y) {
    if (x.size != y.size || x.size == 0) {
        throw std::invalid_argument("Vectors must be of same size and non-empty");
    }

    auto X = vectorToEigen(x);
    auto Y = vectorToEigen(y);

    double mean_X = X.mean();
    double mean_Y = Y.mean();

    Eigen::VectorXd X_centered = X.array() - mean_X;
    Eigen::VectorXd Y_centered = Y.array() - mean_Y;

    double covariance = (X_centered.dot(Y_centered)) / (X.size() - 1);
    double stddev_X = std::sqrt(X_centered.squaredNorm() / (X.size() - 1));
    double stddev_Y = std::sqrt(Y_centered.squaredNorm() / (Y.size() - 1));

    if (stddev_X == 0 || stddev_Y == 0) return 0;

    return covariance / (stddev_X * stddev_Y);
}

double latentDistanceCalculator(const std::vector<double>& a, const std::vector<double>& b, int latDim) {
    double sum = 0.0;
    for (int i = 0; i < latDim; ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

double latentMinDistanceCalculator(const std::vector<double>& a, const std::vector<double>& b, int latDim) {
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;
    double sum4 = 0.0;
    for (int i = 0; i < latDim; ++i) {
        double diff1 = a[i] - b[i];
        sum1 += diff1 * diff1;

        double diff2 = a[i] - b[latDim+i];
        sum2 += diff2 * diff2;

        double diff3 = a[latDim+i] - b[i];
        sum3 += diff3 * diff3;

        double diff4 = a[latDim+i] - b[latDim+i];
        sum4 += diff4 * diff4;
    }
    return std::sqrt(std::min({sum1, sum2, sum3, sum4}));
}


template <typename T>
std::vector<double> flattenAndRemoveNAN(const std::vector<std::vector<T>>& matrix) {
    std::vector<double> out;
    for (const auto& row : matrix) {
        for (const auto& val : row) {
            if (!std::isnan(static_cast<double>(val))) {
                out.push_back(static_cast<double>(val));
            }
        }
    }
    return out;
}

template <typename T>
std::vector<double> flattenAndRemoveNANAndFree(std::vector<std::vector<T>>& matrix) {
    std::vector<double> out;
    for (auto& row : matrix) {
        for (auto& val : row) {
            double d = static_cast<double>(val);
            if (!std::isnan(d)) {
                out.push_back(d);
            }
        }
        std::vector<T>().swap(row);
    }
    std::vector<std::vector<T>>().swap(matrix);
    return out;
}

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

void writeVectorToDisk(const std::vector<double>& data, const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary | std::ios::out);
    ofs.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(double));
}



MMapVector mmapVectorOpen(const std::string& filename) {
    MMapVector mv;
    mv.fd = open(filename.c_str(), O_RDONLY);
    if (mv.fd == -1) throw std::runtime_error("Failed to open file");
    struct stat sb;
    if (fstat(mv.fd, &sb) == -1) throw std::runtime_error("fstat failed");
    mv.size = sb.st_size / sizeof(double);
    mv.data = static_cast<double*>(
        mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, mv.fd, 0)
    );
    if (mv.data == MAP_FAILED) throw std::runtime_error("mmap failed");
    return mv;
}

void mmapVectorClose(MMapVector& mv) {
    if (mv.data) {
        munmap(mv.data, mv.size * sizeof(double));
        mv.data = nullptr;
    }
    if (mv.fd != -1) {
        close(mv.fd);
        mv.fd = -1;
    }
    mv.size = 0;
}

inline const double& mmapVectorAt(const MMapVector& mv, size_t i) {
    return mv.data[i];
}
*/

#ifdef _HAS_MATPLOT_
void plotScatter(const std::vector<double>& x, const std::vector<double>& y, const std::string& xlabel, const std::string& ylabel, const std::string& title) {
    using namespace matplot;
    auto fig = figure(true);
     fig->size(1200, 800);
    auto ax = fig->add_axes();
    ax->scatter(x, y);
    ax->title(title);
    ax->xlabel(xlabel);
    ax->ylabel(ylabel);
    ax->grid(true);
    fig->draw();
}

void plotDistancesSideBySide(const std::vector<double>& hau, const std::vector<double>& mdf, const std::vector<double>& enc1, const std::vector<double>& enc2, const std::vector<double>& edh, const std::vector<double>& edm) {
    plotScatter(hau,  enc1, "Hausdorff distance", "1-sided Euc. dist in latent space", "Hausdorff vs. 1-sided Euc. in Latent, r="+ to_string_with_precision(correlation_coefficient(hau,enc1),6));
    plotScatter(mdf,  enc1, "MDF distance", "1-sided Euc. dist in latent space", "MDF vs. 1-sided Euc. in Latent, r="+ to_string_with_precision(correlation_coefficient(mdf,enc1),6));
    plotScatter(hau,  mdf,  "Hausdorff distance", "MDF distance", "Hausdorff vs. MDF, r="+ to_string_with_precision(correlation_coefficient(hau,mdf),6));
    plotScatter(enc1, enc2, "1-sided Euc. dist in latent space", "2-sided Euc. dist in latent space", "1-sided vs. 2-sided Euc. in Latent, r="+ to_string_with_precision(correlation_coefficient(enc1,enc2),6));
    plotScatter(edh,  edm,  "Hausdorff distance between input and reconstructed", "MDF distance between input and reconstructed", "Auto-encoder error, hau="+ to_string_with_precision(vectorToEigen(edh).mean(),4)+" mm, mdf="+to_string_with_precision(vectorToEigen(edm).mean(),4)+" mm");
}
#endif
 
void run_modelTest_precalc()
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
    
    // Prepare tractogram reader
    NIBR::TractogramReader tractogram(inp_path, false);


    auto tracObj = tractogram.getTractogram();

    // Original input streamline
    auto streamlines = NIBR::resampleTractogram_withStepCount(tracObj, model.inpDim);


    disp(MSG_DETAIL,"Resampled streamlines for %d points.", model.inpDim);

    // Latent space representations
    std::vector<std::vector<double>>              enc_streamlines;

    // Decoded streamlines from the latent space representations
    NIBR::Tractogram  dec_streamlines;

    auto run_test_with_type = [&](auto type_placeholder) {
        using T = decltype(type_placeholder);
        auto lat_streamlines = encodeStreamlines<T>(streamlines,model,batchSize);         // Encode streamlines in latent space
        dec_streamlines      = decodeStreamlines<T>(lat_streamlines, model, batchSize);   // Decode the encoded streamlines
        enc_streamlines      =  to_double_vector<T>(lat_streamlines);                     // Convert latent space representation to double type for analysis
        // disp(MSG_INFO,"Encoding completed.");
    };
     
    // Dispatch to the generic lambda with the correct type
    if (model.dtype == torch::kFloat)       { run_test_with_type(float{});   } 
    else if (model.dtype == torch::kDouble) { run_test_with_type(double{});  } 
    else if (model.dtype == torch::kHalf)   { run_test_with_type(at::Half{});} 
    else { disp(MSG_ERROR, "Unsupported data type for modelTest: %s", c10::toString(model.dtype)); }

    // return;

    // Define one-sided and two-sided distance functions in the latent space


    // Calculate execution times one to one

    auto mdfStartTime = std::chrono::high_resolution_clock::now();
    double temp_result_mdf = getMDFDistance(streamlines[0], streamlines[1]);
    auto mdfEndTime = std::chrono::high_resolution_clock::now();
    auto mdfDuration = std::chrono::duration_cast<std::chrono::microseconds>(mdfEndTime - mdfStartTime);
    std::cout << "mdf execution time: " << mdfDuration.count() << " microseconds | result: " << temp_result_mdf << std::endl;

    auto hausdorffStartTime = std::chrono::high_resolution_clock::now();
    double temp_result_hau = getHausdorffDistance(streamlines[0], streamlines[1]);
    auto hausdorffEndTime = std::chrono::high_resolution_clock::now();
    auto hausdorffduration = std::chrono::duration_cast<std::chrono::microseconds>(hausdorffEndTime - hausdorffStartTime);
    std::cout << "hausdorff execution time: " << hausdorffduration.count() << " microseconds | result: " << temp_result_hau << std::endl;


    auto latentSimpleStartTime = std::chrono::high_resolution_clock::now();
    double temp_result_ls = 0.067612 * latentDistanceCalculator(enc_streamlines[0], enc_streamlines[1], model.latDim);
    auto latentSimpleEndTime = std::chrono::high_resolution_clock::now();
    auto latentSimpleduration = std::chrono::duration_cast<std::chrono::microseconds>(latentSimpleEndTime - latentSimpleStartTime);
    std::cout << "latent simple execution time: " << latentSimpleduration.count() << " microseconds | result: " << temp_result_ls << std::endl;


    auto latentMinStartTime = std::chrono::high_resolution_clock::now();
    double temp_result_lm = 0.067612 * latentMinDistanceCalculator(enc_streamlines[0], enc_streamlines[1], model.latDim);
    auto latentMinEndTime = std::chrono::high_resolution_clock::now();
    auto latentMinduration = std::chrono::duration_cast<std::chrono::microseconds>(latentMinEndTime - latentMinStartTime);
    std::cout << "latent simple execution time: " << latentMinduration.count() << " microseconds | result: " << temp_result_lm << std::endl;

    // Calculate execution times for one to all

    disp(MSG_INFO,"Calculating execution times for one to all\n");

    std::vector<std::vector<double>> temp_mdf(1024, std::vector<double>(tracObj.size(),NAN));
    auto oneAllMDF= [&](NIBR::MT::TASK task) -> void {
        for (size_t i = 0; i < task.no; i++) {
            temp_mdf[task.no][i] = getMDFDistance(tracObj[0], tracObj[task.no]);
        }

    };
    auto mdfOneToAllStartTime = std::chrono::high_resolution_clock::now();
    NIBR::MT::MTRUN(1024, "Computing MDF distances for 1024 to all", oneAllMDF); 
    auto mdfOneToAllEndTime = std::chrono::high_resolution_clock::now();
    auto mdfOneToAllDuration = std::chrono::duration_cast<std::chrono::microseconds>(mdfOneToAllEndTime - mdfOneToAllStartTime);
    disp(MSG_INFO,"MDF one to all duration: %.i microseconds\n", mdfOneToAllDuration);

    std::vector<std::vector<double>> temp_hau(1024, std::vector<double>(tracObj.size(),NAN));
    auto oneAllHau= [&](NIBR::MT::TASK task) -> void {
        for (size_t i = 0; i < task.no; i++) {
            temp_hau[task.no][i] = getHausdorffDistance(streamlines[task.no], streamlines[i]);
        }
    };
    auto hauOneToAllStartTime = std::chrono::high_resolution_clock::now();
    NIBR::MT::MTRUN(1024, "Computing Haussdorff distances for 1024 to all", oneAllHau); 
    auto hauOneToAllEndTime = std::chrono::high_resolution_clock::now();
    auto hauOneToAllDuration = std::chrono::duration_cast<std::chrono::microseconds>(hauOneToAllEndTime - hauOneToAllStartTime);
    disp(MSG_INFO,"Haussdorf one to all duration: %.i microseconds\n", hauOneToAllDuration);

    
    
    
    
    

    std::vector<double> enc_dec_hau_dist (streamlines.size());
    std::vector<double> enc_dec_mdf_dist (streamlines.size());

    disp(MSG_INFO,"Starting getDistances");

    auto getEnc1 = [&](NIBR::MT::TASK task) -> void {
        
        enc_dec_hau_dist[task.no] = getHausdorffDistance(streamlines[task.no], dec_streamlines[task.no]);
        enc_dec_mdf_dist[task.no] = getMDFDistance(streamlines[task.no], dec_streamlines[task.no]);
    };
    NIBR::MT::MTRUN(streamlines.size(), "Computing enc-dec distances", getEnc1);


    auto edh  = enc_dec_hau_dist;
    auto edm  = enc_dec_mdf_dist;

    #ifdef _HAS_MATPLOT_
    plotDistancesSideBySide(hau_mmap, mdf_mmap, enc1_mmap, enc2_mmap, edh, edm);
    std::this_thread::sleep_for(std::chrono::seconds(5));
    #endif

    MMapVector enc1_mmap;
    MMapVector enc2_mmap;
    MMapVector hau_mmap;
    MMapVector mdf_mmap;
    MMapVector lat1_mmap;
    MMapVector lat2_mmap;
    // mmap files which were given
    if(enc1_path != "")
    {
        enc1_mmap = mmapVectorOpen(enc1_path);
    }
    if(enc2_path != "")
    {
        enc2_mmap = mmapVectorOpen(enc2_path);
    }
    if(hau_path != "")
    {
        hau_mmap = mmapVectorOpen(hau_path);
    }
    if(mdf_path != "")
    {
        mdf_mmap = mmapVectorOpen(mdf_path);
    }
    if(lat1_path != "")
    {
        lat1_mmap = mmapVectorOpen(lat1_path);
    }
    if(lat2_path != "")
    {
        lat2_mmap = mmapVectorOpen(lat2_path);
    }

    double mean_mdf = -1.0;
    double min_mdf = -1.0;
    double max_mdf = -1.0;
    if (mdf_path != "") {
        double sum_mdf = std::accumulate(mdf_mmap.data, mdf_mmap.data + mdf_mmap.size, 0.0);
        mean_mdf = sum_mdf / mdf_mmap.size;
        auto [min_it_mdf, max_it_mdf] = std::minmax_element(mdf_mmap.data, mdf_mmap.data + mdf_mmap.size);
        min_mdf = *min_it_mdf;
        max_mdf = *max_it_mdf;
    }
    

    

    
    double mean_hau = -1.0;
    double min_hau = -1.0;
    double max_hau = -1.0;
    if(hau_path != "") {
        double sum_hau = std::accumulate(hau_mmap.data, hau_mmap.data + hau_mmap.size, 0.0);
        mean_hau = sum_hau / hau_mmap.size;
        auto [min_it_hau, max_it_hau] = std::minmax_element(hau_mmap.data, hau_mmap.data + hau_mmap.size);
        min_hau = *min_it_hau;
        max_hau = *max_it_hau;
    }


    


    disp(MSG_INFO,"");
    disp(MSG_INFO,"MDF Minimum distance between streamlines: %.6f", min_mdf);
    disp(MSG_INFO,"MDF Maximum distance between streamlines: %.6f", max_mdf);
    disp(MSG_INFO,"MDF Average distance between streamlines: %.6f", mean_mdf);
    //disp(MSG_INFO,"MDF Median distance between streamlines: %.6f", median_mdf);

    disp(MSG_INFO,"");
    disp(MSG_INFO,"Haussdorff Minimum distance between streamlines: %.6f", min_hau);
    disp(MSG_INFO,"Haussdorff Maximum distance between streamlines: %.6f", max_hau);
    disp(MSG_INFO,"Haussdorff Average distance between streamlines: %.6f", mean_hau);
    //disp(MSG_INFO,"Haussdorff Median distance between streamlines: %.6f", median_hau);
   
    
    if(hau_path != "" && mdf_path != "" && enc1_path != "" && enc2_path != ""){
        disp(MSG_INFO,"");
        disp(MSG_INFO,"Pearson correlation coefficients:");
        disp(MSG_INFO,"Hausdorff and (one-sided) Euc. distance in latent space: %.6f", correlation_coefficient(hau_mmap,enc1_mmap ));
        disp(MSG_INFO,"MDF and (one-sided) Euc. distance in latent space:       %.6f", correlation_coefficient(mdf_mmap,enc1_mmap ));
        disp(MSG_INFO,"Hausdorff and MDF:                                       %.6f", correlation_coefficient(hau_mmap,mdf_mmap  ));
        disp(MSG_INFO,"One-sided and two-sided Euc. distance in latent space:   %.6f", correlation_coefficient(enc1_mmap,enc2_mmap));

    } else {
        disp(MSG_INFO, "");
        disp(MSG_INFO, "Skipped correlation coefficient. Requires hausdorff, mdf, enc1 and enc2");
    }
    
    disp(MSG_INFO,"");
    disp(MSG_INFO,"Auto-encoder error");
    disp(MSG_INFO,"Mean Hausdorff distance between input and reconstructed: %.6f mm", vectorToEigen(edh).mean());
    disp(MSG_INFO,"Mean MDF distance between input and reconstructed:       %.6f mm", vectorToEigen(edm).mean());

    if(hau_path != "" && mdf_path != ""){
        disp(MSG_INFO,"");
        disp(MSG_INFO,"Distance scaling factor to match Hausdorff distance:     %.6f", (vectorToEigen(hau_mmap).array() / vectorToEigen(enc1_mmap).array()).mean() );
        disp(MSG_INFO,"Distance scaling factor to match MDF distance:           %.6f", (vectorToEigen(mdf_mmap).array() / vectorToEigen(enc1_mmap).array()).mean() );

    } else {
        disp(MSG_INFO,"");
        disp(MSG_INFO,"Sjipped scaling factor. Requires hausdorff and mdf");
    }
    
    disp(MSG_INFO,"");
    

    #ifdef _HAS_MATPLOT_
    wait("Done ");
    #endif


    
    
    
    
    
    

    if(enc1_path != "")
    {
        mmapVectorClose(enc1_mmap);
    }
    if(enc2_path != "")
    {
        mmapVectorClose(enc2_mmap);
    }
    if(hau_path != "")
    {
        mmapVectorClose(hau_mmap);
    }
    if(mdf_path != "")
    {
        mmapVectorClose(mdf_mmap);
    }
    if(lat1_path != "")
    {
        mmapVectorClose(lat1_mmap);
    }
    if(lat2_path != "")
    {
        mmapVectorClose(lat2_mmap);
    }

    return;
       
}          
    
     
void modelTest_precalc(CLI::App* app)   
{ 

    app->formatter(std::make_shared<CustomHelpFormatter>());

    const std::string info = "Testing a model involves encoding each streamline in a tractogram and comparing the pair-wise distance in latent space agains the distance computed using two-sided Hausdorff and minimum average direct-flip (MDF) distances. This version doesn't compute the distances but instead reads them from files.";

    app->description("tests how well a model encodes a tractogram");

    app->add_option("<input tractogram>",    inp_path,           "Input tractogram (.vtk, .tck)")
        ->required();

    app->add_option("<model>",               inp_model_spec,     "Input model, specified with the path to the Torch script file, followed by the input dimensions, latent space dimensions, data type (float or double), and distance scaling factor of the model. E.g. /model/test_model.pt 256 64 float 0.08. ")
        ->required();

    app->add_option("--enc1", enc1_path, "Path to a enc1.bin");

    app->add_option("--enc2", enc2_path, "Path to a enc2.bin");

    app->add_option("--hau", hau_path, "Path to a hau.bin");

    app->add_option("--mdf", mdf_path, "Path to a mdf.bin");

    app->add_option("--lat1", lat1_path, "Path to a lat1.bin");

    app->add_option("--lat2", lat2_path, "Path to a lat2.bin");

    app->add_flag("--useCPU, -c",            useCPU,             "Use only CPU without checking any available GPUs.");

    app->add_option("--numberOfThreads, -n", numberOfThreads,    "Number of threads.");
    app->add_option("--verbose, -v",         verbose,            "Verbose level. Options are \"quite\",\"fatal\",\"error\",\"warn\",\"info\" and \"debug\". Default=info");
    app->add_flag("--force, -f",             force,              "Force overwriting of existing file");

    app->callback(run_modelTest_precalc);  
     
}

                       