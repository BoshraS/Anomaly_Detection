#include "cmd.h"


#ifdef _HAS_MATPLOT_
#include <matplot/matplot.h>
#endif

using namespace NIBR;

namespace CMDARGS_MODELTEST {
    std::string  inp_path;
    
    std::tuple<std::string, int, int> inp_model_spec("", 0, 0); // module_path, inp_dim, lat_dim

    int  batchSize          = 512;
    bool useCPU             = false;
    
    int numberOfThreads     =  0;
    std::string verbose     = "info";
    bool force              = false;
}

using namespace CMDARGS_MODELTEST; 

// Convert std::vector to Eigen::VectorXf
Eigen::VectorXf vectorToEigen(const std::vector<float>& v) {
    return Eigen::VectorXf::Map(v.data(), v.size());
}

// Function to calculate the Pearson correlation coefficient
float correlation_coefficient(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size() || x.empty()) {
        throw std::invalid_argument("Vectors must be of same size and non-empty");
    }

    Eigen::VectorXf X = vectorToEigen(x);
    Eigen::VectorXf Y = vectorToEigen(y);

    float mean_X = X.mean();
    float mean_Y = Y.mean();

    Eigen::VectorXf X_centered = X.array() - mean_X;
    Eigen::VectorXf Y_centered = Y.array() - mean_Y;

    float covariance = (X_centered.dot(Y_centered)) / (X.size() - 1);  // Using (N-1) for sample covariance
    float stddev_X = std::sqrt(X_centered.squaredNorm() / (X.size() - 1));
    float stddev_Y = std::sqrt(Y_centered.squaredNorm() / (Y.size() - 1));

    return covariance / (stddev_X * stddev_Y);
}

std::vector<float> flattenAndRemoveNAN(const std::vector<std::vector<float>>& matrix) {
    std::vector<float> out;
    for (const auto& row : matrix) {
        for (const auto& val : row) {
            if (!std::isnan(val)) {
                out.push_back(val);
            }
        }
    }
    return out;
}


std::vector<double> to_double_vector(const std::vector<float>& v) {
    return std::vector<double>(v.begin(), v.end());
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

void plotDistancesSideBySide(const std::vector<float>& hau, const std::vector<float>& mdf, const std::vector<float>& enc1, const std::vector<float>& enc2, const std::vector<float>& edh, const std::vector<float>& edm) {
    // Convert float vectors to double vectors
    std::vector<double> hau_double  = to_double_vector(hau);
    std::vector<double> mdf_double  = to_double_vector(mdf);
    std::vector<double> enc1_double = to_double_vector(enc1);
    std::vector<double> enc2_double = to_double_vector(enc2);
    std::vector<double> edh_double  = to_double_vector(edh);
    std::vector<double> edm_double  = to_double_vector(edm);

    plotScatter(hau_double,  enc1_double, "Hausdorff distance", "1-sided Euc. dist in latent space", "Hausdorff vs. 1-sided Euc. in Latent, r="+ to_string_with_precision(correlation_coefficient(hau,enc1),6));
    plotScatter(mdf_double,  enc1_double, "MDF distance", "1-sided Euc. dist in latent space", "MDF vs. 1-sided Euc. in Latent, r="+ to_string_with_precision(correlation_coefficient(mdf,enc1),6));
    plotScatter(hau_double,  mdf_double,  "Hausdorff distance", "MDF distance", "Hausdorff vs. MDF, r="+ to_string_with_precision(correlation_coefficient(hau,mdf),6));
    plotScatter(enc1_double, enc2_double, "1-sided Euc. dist in latent space", "2-sided Euc. dist in latent space", "1-sided vs. 2-sided Euc. in Latent, r="+ to_string_with_precision(correlation_coefficient(enc1,enc2),6));
    plotScatter(edh_double,  edm_double,  "Hausdorff distance between input and reconstructed", "MDF distance between input and reconstructed", "Auto-encoder error, hau="+ to_string_with_precision(vectorToEigen(edh).mean(),4)+" mm, mdf="+to_string_with_precision(vectorToEigen(edm).mean(),4)+" mm");
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

    // Prepare tractogram
    NIBR::TractogramReader tractogram(inp_path);
    std::vector<std::vector<std::vector<float>>> streamlines = resampleTractogram_withStepCount(&tractogram, model.inpDim);


    // Encode streamlines in latent space
    auto enc_streamlines = encodeStreamlines(streamlines,     model, batchSize);

    // Decode the encoded streamlines
    auto dec_streamlines = decodeStreamlines(enc_streamlines, model, batchSize);

    // Define one-sided and two-sided distance functions in the latent space
    auto getOneSidedEncodedDistance = [&](size_t idx1, size_t idx2) -> float {
        float sum1 = 0;
        float sum2 = 0;
        for (int i = 0; i < model.latDim; i++) {
            float d1 = (enc_streamlines[idx1][i] - enc_streamlines[idx2][i]);
            float d2 = (enc_streamlines[idx1][i] - enc_streamlines[idx2][i+model.latDim]);
            sum1   += d1 * d1;
            sum2   += d2 * d2;
        }

        return std::min(std::sqrt(sum2), std::sqrt(sum1));
    };

    auto getTwoSidedEncodedDistance = [&](size_t idx1, size_t idx2) -> float {
        float sum1 = 0;
        float sum2 = 0;
        float sum3 = 0;
        float sum4 = 0;
        for (int i = 0; i < model.latDim; i++) {
            float d1 = (enc_streamlines[idx1][i]              - enc_streamlines[idx2][i]);
            float d2 = (enc_streamlines[idx1][i]              - enc_streamlines[idx2][i+model.latDim]);
            float d3 = (enc_streamlines[idx1][i+model.latDim] - enc_streamlines[idx2][i]);
            float d4 = (enc_streamlines[idx1][i+model.latDim] - enc_streamlines[idx2][i+model.latDim]);
            sum1   += d1 * d1;
            sum2   += d2 * d2;
            sum3   += d3 * d3;
            sum4   += d4 * d4;
        }

        return std::min(std::sqrt(sum4), std::min(std::sqrt(sum3), std::min(std::sqrt(sum2), std::sqrt(sum1))));
    };


    // Compute pair-wise distances
    std::vector<std::vector<float>> enc_dist1        (streamlines.size(), std::vector<float>(streamlines.size(),NAN));
    std::vector<std::vector<float>> enc_dist2        (streamlines.size(), std::vector<float>(streamlines.size(),NAN));
    std::vector<std::vector<float>> hau_dist         (streamlines.size(), std::vector<float>(streamlines.size(),NAN));
    std::vector<std::vector<float>> mdf_dist         (streamlines.size(), std::vector<float>(streamlines.size(),NAN));
    std::vector<float> enc_dec_hau_dist (streamlines.size());
    std::vector<float> enc_dec_mdf_dist (streamlines.size());

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

    app->add_option("<model>",               inp_model_spec,     "Input model, specified with the path to the Torch script file, followed by the input dimensions and latent space dimensions. E.g. /model/test_model.pt 256 64")
        ->required();

    app->add_flag("--useCPU, -c",            useCPU,             "Use only CPU without checking any available GPUs.");

    app->add_option("--numberOfThreads, -n", numberOfThreads,    "Number of threads.");
    app->add_option("--verbose, -v",         verbose,            "Verbose level. Options are \"quite\",\"fatal\",\"error\",\"warn\",\"info\" and \"debug\". Default=info");
    app->add_flag("--force, -f",             force,              "Force overwriting of existing file");

    app->callback(run_modelTest);  
     
}

                       