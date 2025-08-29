#include "cmd.h"
#include "dMRI/tractography/tractogram.h"


#ifdef _HAS_MATPLOT_
#include <matplot/matplot.h>
#endif

using namespace NIBR;

namespace CMDARGS_MODELTEST {
    std::string  inp_path;
    
    std::tuple<std::string, int, int, std::string> inp_model_spec("", 0, 0, ""); // module_path, inp_dim, lat_dim, data type

    int  batchSize          = 512;
    bool useCPU             = false;
    
    int numberOfThreads     =  0;
    std::string verbose     = "info";
    bool force              = false;
}

using namespace CMDARGS_MODELTEST; 

// Convert std::vector to Eigen::VectorXd
Eigen::VectorXd vectorToEigen(const std::vector<double>& v) {
    return Eigen::VectorXd::Map(v.data(), v.size());
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
 
void run_modelTest()
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
    auto getOneSidedEncodedDistance = [&](size_t idx1, size_t idx2) -> double {
        double sum1 = 0;
        double sum2 = 0;
        for (int i = 0; i < model.latDim; i++) {
            double d1 = (enc_streamlines[idx1][i] - enc_streamlines[idx2][i]);
            double d2 = (enc_streamlines[idx1][i] - enc_streamlines[idx2][i+model.latDim]);
            sum1   += d1 * d1;
            sum2   += d2 * d2;
        }

        return std::min(std::sqrt(sum2), std::sqrt(sum1));
    };

    auto getTwoSidedEncodedDistance = [&](size_t idx1, size_t idx2) -> double {
        double sum1 = 0;
        double sum2 = 0;
        double sum3 = 0;
        double sum4 = 0;
        for (int i = 0; i < model.latDim; i++) {
            double d1 = (enc_streamlines[idx1][i]              - enc_streamlines[idx2][i]);
            double d2 = (enc_streamlines[idx1][i]              - enc_streamlines[idx2][i+model.latDim]);
            double d3 = (enc_streamlines[idx1][i+model.latDim] - enc_streamlines[idx2][i]);
            double d4 = (enc_streamlines[idx1][i+model.latDim] - enc_streamlines[idx2][i+model.latDim]);
            sum1     += d1 * d1;
            sum2     += d2 * d2;
            sum3     += d3 * d3;
            sum4     += d4 * d4;
        }

        return std::min(std::sqrt(sum4), std::min(std::sqrt(sum3), std::min(std::sqrt(sum2), std::sqrt(sum1))));
    };


    // Compute pair-wise distances
    std::vector<std::vector<double>> enc_dist1        (streamlines.size(), std::vector<double>(streamlines.size(),NAN));
    std::vector<std::vector<double>> enc_dist2        (streamlines.size(), std::vector<double>(streamlines.size(),NAN));
    std::vector<std::vector<double>> hau_dist         (streamlines.size(), std::vector<double>(streamlines.size(),NAN));
    std::vector<std::vector<double>> mdf_dist         (streamlines.size(), std::vector<double>(streamlines.size(),NAN));

    std::vector<double> enc_dec_hau_dist (streamlines.size());
    std::vector<double> enc_dec_mdf_dist (streamlines.size());

    auto getDistances = [&](NIBR::MT::TASK task) -> void {
        for (size_t i = 0; i < task.no; i++) {
            enc_dist1[task.no][i]           = getOneSidedEncodedDistance(task.no,i);
            enc_dist2[task.no][i]           = getTwoSidedEncodedDistance(task.no,i);
            hau_dist[task.no][i]            = getHausdorffDistance(streamlines[task.no], streamlines[i]);
            mdf_dist[task.no][i]            = getMDFDistance(streamlines[task.no], streamlines[i]);
        }
        enc_dec_hau_dist[task.no] = getHausdorffDistance(streamlines[task.no], dec_streamlines[task.no]);
        enc_dec_mdf_dist[task.no] = getMDFDistance(streamlines[task.no], dec_streamlines[task.no]);
    };
    NIBR::MT::MTRUN(streamlines.size(), "Computing pair-wise distances", getDistances);

    auto enc1 = flattenAndRemoveNAN(enc_dist1);
    auto enc2 = flattenAndRemoveNAN(enc_dist2);
    auto hau  = flattenAndRemoveNAN(hau_dist );
    auto mdf  = flattenAndRemoveNAN(mdf_dist );

    auto edh  = enc_dec_hau_dist;
    auto edm  = enc_dec_mdf_dist;

    #ifdef _HAS_MATPLOT_
    plotDistancesSideBySide(hau, mdf, enc1, enc2, edh, edm);
    std::this_thread::sleep_for(std::chrono::seconds(5));
    #endif
   
    
    
    disp(MSG_INFO,"");
    disp(MSG_INFO,"Pearson correlation coefficients:");
    disp(MSG_INFO,"Hausdorff and (one-sided) Euc. distance in latent space: %.6f", correlation_coefficient(hau,enc1 ));
    disp(MSG_INFO,"MDF and (one-sided) Euc. distance in latent space:       %.6f", correlation_coefficient(mdf,enc1 ));
    disp(MSG_INFO,"Hausdorff and MDF:                                       %.6f", correlation_coefficient(hau,mdf  ));
    disp(MSG_INFO,"One-sided and two-sided Euc. distance in latent space:   %.6f", correlation_coefficient(enc1,enc2));

    disp(MSG_INFO,"");
    disp(MSG_INFO,"Auto-encoder error");
    disp(MSG_INFO,"Mean Hausdorff distance between input and reconstructed: %.6f mm", vectorToEigen(edh).mean());
    disp(MSG_INFO,"Mean MDF distance between input and reconstructed:       %.6f mm", vectorToEigen(edm).mean());

    disp(MSG_INFO,"");
    disp(MSG_INFO,"Distance scaling factor to match Hausdorff distance:     %.6f", (vectorToEigen(hau).array() / vectorToEigen(enc1).array()).mean() );
    disp(MSG_INFO,"Distance scaling factor to match MDF distance:           %.6f", (vectorToEigen(mdf).array() / vectorToEigen(enc1).array()).mean() );

    disp(MSG_INFO,"");

    #ifdef _HAS_MATPLOT_
    wait("Done ");
    #endif

    return;
       
}          
    
     
void modelTest(CLI::App* app)   
{ 

    app->formatter(std::make_shared<CustomHelpFormatter>());

    const std::string info = "Testing a model involves encoding each streamline in a tractogram and comparing the pair-wise distance in latent space agains the distance computed using two-sided Hausdorff and minimum average direct-flip (MDF) distances.";

    app->description("tests how well a model encodes a tractogram");

    app->add_option("<input tractogram>",    inp_path,           "Input tractogram (.vtk, .tck)")
        ->required();

    app->add_option("<model>",               inp_model_spec,     "Input model, specified with the path to the Torch script file, followed by the input dimensions, latent space dimensions, data type (float or double), and distance scaling factor of the model. E.g. /model/test_model.pt 256 64 float 0.08. ")
        ->required();

    app->add_flag("--useCPU, -c",            useCPU,             "Use only CPU without checking any available GPUs.");

    app->add_option("--numberOfThreads, -n", numberOfThreads,    "Number of threads.");
    app->add_option("--verbose, -v",         verbose,            "Verbose level. Options are \"quite\",\"fatal\",\"error\",\"warn\",\"info\" and \"debug\". Default=info");
    app->add_flag("--force, -f",             force,              "Force overwriting of existing file");

    app->callback(run_modelTest);  
     
}

                       