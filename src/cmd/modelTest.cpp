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

namespace CMDARGS_MODELTEST {
    std::string  inp_path;
    
    std::tuple<std::string, int, int, std::string> inp_model_spec("", 0, 0, ""); // module_path, inp_dim, lat_dim, data type

    int  batchSize          = 512;
    bool useCPU             = false;
    
    int numberOfThreads     =  0;
    std::string verbose     = "info";
    bool force              = false;
    std::string preCalcLoc_path = "";
}

using namespace CMDARGS_MODELTEST; 


#include "utils/modelTestHelpers.h"

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
    if (preCalcLoc_path != "") {
        if (preCalcLoc_path.back() != '/') {
            preCalcLoc_path += '/';
        }
    }

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


    // Calculate execution times

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
    

    // Compute pair-wise distances

    // this takes an absurd amount of memory with bigger tractograms
    
    
    
    

    std::vector<double> enc_dec_hau_dist (streamlines.size());
    std::vector<double> enc_dec_mdf_dist (streamlines.size());

    disp(MSG_INFO,"Starting getDistances");

    std::vector<std::vector<double>> enc_dist1        (streamlines.size(), std::vector<double>(streamlines.size(),NAN));
    auto getEnc1 = [&](NIBR::MT::TASK task) -> void {
        for (size_t i = 0; i < task.no; i++) {
            enc_dist1[task.no][i]           = getOneSidedEncodedDistance(task.no,i);
        }
        enc_dec_hau_dist[task.no] = getHausdorffDistance(streamlines[task.no], dec_streamlines[task.no]);
        enc_dec_mdf_dist[task.no] = getMDFDistance(streamlines[task.no], dec_streamlines[task.no]);
    };
    NIBR::MT::MTRUN(streamlines.size(), "Computing enc 1 distance", getEnc1);
    auto enc1 = flattenAndRemoveNANAndFree(enc_dist1);
    writeVectorToDisk(enc1, preCalcLoc_path+"enc1.bin");
    enc1.clear(); enc1.shrink_to_fit();
    MMapVector enc1_mmap = mmapVectorOpen(preCalcLoc_path+"enc1.bin");


    std::vector<std::vector<double>> enc_dist2        (streamlines.size(), std::vector<double>(streamlines.size(),NAN));
    auto getEnc2 = [&](NIBR::MT::TASK task) -> void {
        for (size_t i = 0; i < task.no; i++) {
            enc_dist2[task.no][i]           = getTwoSidedEncodedDistance(task.no,i);
        }
    };
    NIBR::MT::MTRUN(streamlines.size(), "Computing enc 2 distance", getEnc2);
    auto enc2 = flattenAndRemoveNANAndFree(enc_dist2);
    writeVectorToDisk(enc2, preCalcLoc_path+"enc2.bin");
    enc2.clear(); enc2.shrink_to_fit();
    MMapVector enc2_mmap = mmapVectorOpen(preCalcLoc_path+"enc2.bin");



/*
    std::vector<std::vector<double>> hau_dist         (streamlines.size(), std::vector<double>(streamlines.size(),NAN));

    auto getHauDist= [&](NIBR::MT::TASK task) -> void {
        for (size_t i = 0; i < task.no; i++) {
            hau_dist[task.no][i]            = getHausdorffDistance(streamlines[task.no], streamlines[i]);
        }
    };
    NIBR::MT::MTRUN(streamlines.size(), "Computing hau distance", getHauDist);
    auto hau = flattenAndRemoveNANAndFree(hau_dist);
    writeVectorToDisk(hau, preCalcLoc_path+"hau.bin");
    hau.clear(); hau.shrink_to_fit();
    MMapVector hau_mmap = mmapVectorOpen(preCalcLoc_path+"hau.bin");
*/



    std::vector<std::vector<double>> mdf_dist         (streamlines.size(), std::vector<double>(streamlines.size(),NAN));
    auto getMDFDist= [&](NIBR::MT::TASK task) -> void {
        for (size_t i = 0; i < task.no; i++) {
            mdf_dist[task.no][i]            = getMDFDistance(streamlines[task.no], streamlines[i]);
        }
    };
    NIBR::MT::MTRUN(streamlines.size(), "Computing mdf distance", getMDFDist);
    auto mdf = flattenAndRemoveNANAndFree(mdf_dist);
    writeVectorToDisk(mdf, preCalcLoc_path+"mdf.bin");
    mdf.clear(); mdf.shrink_to_fit();
    MMapVector mdf_mmap = mmapVectorOpen(preCalcLoc_path+"mdf.bin");


    auto edh  = enc_dec_hau_dist;
    auto edm  = enc_dec_mdf_dist;

    #ifdef _HAS_MATPLOT_
    plotDistancesSideBySide(hau_mmap, mdf_mmap, enc1_mmap, enc2_mmap, edh, edm);
    std::this_thread::sleep_for(std::chrono::seconds(5));
    #endif

    double latentScalingFactorMdf = (vectorToEigen(mdf_mmap).array() / vectorToEigen(enc1_mmap).array()).mean();

    disp(MSG_INFO,"Starting lat distance calculations");

    std::vector<std::vector<double>> lat_dist       (streamlines.size(), std::vector<double>(streamlines.size(),NAN));
    

    auto calcLatDistancesSimple = [&](NIBR::MT::TASK task) -> void {
        for(size_t i = 0; i < task.no; ++i) {
            lat_dist[task.no][i] = latentScalingFactorMdf * latentDistanceCalculator(enc_streamlines[task.no], enc_streamlines[i], model.latDim);
        }
    };

    NIBR::MT::MTRUN(enc_streamlines.size(), "Computing Lat simple distances", calcLatDistancesSimple);
    auto lat1 = flattenAndRemoveNANAndFree(lat_dist);
    writeVectorToDisk(lat1, preCalcLoc_path+"lat1.bin");
    lat1.clear(); lat1.shrink_to_fit();
    MMapVector lat1_mmap = mmapVectorOpen(preCalcLoc_path+"lat1.bin");




    std::vector<std::vector<double>> lat_min_dist   (streamlines.size(), std::vector<double>(streamlines.size(),NAN));

    auto calcLatDistancesMin = [&](NIBR::MT::TASK task) -> void {
        for(size_t i = 0; i < task.no; ++i) {
            lat_min_dist[task.no][i] = latentScalingFactorMdf * latentMinDistanceCalculator(enc_streamlines[task.no], enc_streamlines[i], model.latDim);
        }
    };

    NIBR::MT::MTRUN(enc_streamlines.size(), "Computing Lat min distances", calcLatDistancesMin);
    auto lat2 = flattenAndRemoveNANAndFree(lat_min_dist);
    writeVectorToDisk(lat2, preCalcLoc_path+"lat2.bin");
    lat2.clear(); lat2.shrink_to_fit();
    MMapVector lat2_mmap = mmapVectorOpen(preCalcLoc_path+"lat2.bin");



    double sum_mdf = std::accumulate(mdf_mmap.data, mdf_mmap.data + mdf_mmap.size, 0.0);
    double mean_mdf = sum_mdf / mdf_mmap.size;
    auto [min_it_mdf, max_it_mdf] = std::minmax_element(mdf_mmap.data, mdf_mmap.data + mdf_mmap.size);
    double min_mdf = *min_it_mdf;
    double max_mdf = *max_it_mdf;

    
/*
    double sum_hau = std::accumulate(hau_mmap.data, hau_mmap.data + hau_mmap.size, 0.0);
    double mean_hau = sum_hau / hau_mmap.size;
    auto [min_it_hau, max_it_hau] = std::minmax_element(hau_mmap.data, hau_mmap.data + hau_mmap.size);
    double min_hau = *min_it_hau;
    double max_hau = *max_it_hau;
*/


    disp(MSG_INFO,"Latent Scaling Factor MDF: %.6f", latentScalingFactorMdf);
    double mean_enc1 = -1.0;
    double min_enc1 = -1.0;
    double max_enc1 = -1.0;

    double sum_enc1 = std::accumulate(enc1_mmap.data, enc1_mmap.data + enc1_mmap.size, 0.0);
    mean_enc1 = (sum_enc1 / enc1_mmap.size) * latentScalingFactorMdf;
    auto [min_it_enc1, max_it_enc1] = std::minmax_element(enc1_mmap.data, enc1_mmap.data + enc1_mmap.size);
    min_enc1 = *min_it_enc1;
    max_enc1 = *max_it_enc1;
    min_enc1 = min_enc1 * latentScalingFactorMdf;
    max_enc1 = max_enc1 * latentScalingFactorMdf;

    disp(MSG_INFO,"");
    disp(MSG_INFO,"Encoded Minimum simple distance between streamlines: %.6f", min_enc1);
    disp(MSG_INFO,"Encoded Maximum simple distance between streamlines: %.6f", max_enc1);
    disp(MSG_INFO,"Encoded Average simple distance between streamlines: %.6f", mean_enc1);


    


    disp(MSG_INFO,"");
    disp(MSG_INFO,"MDF Minimum distance between streamlines: %.6f", min_mdf);
    disp(MSG_INFO,"MDF Maximum distance between streamlines: %.6f", max_mdf);
    disp(MSG_INFO,"MDF Average distance between streamlines: %.6f", mean_mdf);
    //disp(MSG_INFO,"MDF Median distance between streamlines: %.6f", median_mdf);

    /*
    disp(MSG_INFO,"");
    disp(MSG_INFO,"Haussdorff Minimum distance between streamlines: %.6f", min_hau);
    disp(MSG_INFO,"Haussdorff Maximum distance between streamlines: %.6f", max_hau);
    disp(MSG_INFO,"Haussdorff Average distance between streamlines: %.6f", mean_hau);
    //disp(MSG_INFO,"Haussdorff Median distance between streamlines: %.6f", median_hau);
    */
   
    
    
    disp(MSG_INFO,"");
    disp(MSG_INFO,"Pearson correlation coefficients:");
    //disp(MSG_INFO,"Hausdorff and (one-sided) Euc. distance in latent space: %.6f", correlation_coefficient(hau_mmap,enc1_mmap ));
    disp(MSG_INFO,"MDF and (one-sided) Euc. distance in latent space:       %.6f", correlation_coefficient(mdf_mmap,enc1_mmap ));
    //disp(MSG_INFO,"Hausdorff and MDF:                                       %.6f", correlation_coefficient(hau_mmap,mdf_mmap  ));
    disp(MSG_INFO,"One-sided and two-sided Euc. distance in latent space:   %.6f", correlation_coefficient(enc1_mmap,enc2_mmap));

    disp(MSG_INFO,"");
    disp(MSG_INFO,"Auto-encoder error");
    disp(MSG_INFO,"Mean Hausdorff distance between input and reconstructed: %.6f mm", vectorToEigen(edh).mean());
    disp(MSG_INFO,"Mean MDF distance between input and reconstructed:       %.6f mm", vectorToEigen(edm).mean());

    disp(MSG_INFO,"");
    //disp(MSG_INFO,"Distance scaling factor to match Hausdorff distance:     %.6f", (vectorToEigen(hau_mmap).array() / vectorToEigen(enc1_mmap).array()).mean() );
    disp(MSG_INFO,"Distance scaling factor to match MDF distance:           %.6f", (vectorToEigen(mdf_mmap).array() / vectorToEigen(enc1_mmap).array()).mean() );

    disp(MSG_INFO,"");
    

    #ifdef _HAS_MATPLOT_
    wait("Done ");
    #endif


    mmapVectorClose(mdf_mmap);
    //mmapVectorClose(hau_mmap);
    mmapVectorClose(enc1_mmap);
    mmapVectorClose(enc2_mmap);
    mmapVectorClose(lat1_mmap);
    mmapVectorClose(lat2_mmap);

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

    app->add_option("--preCalcLoc", preCalcLoc_path, "Where to store generated files. If left empty the current directory is used.");
    app->add_flag("--useCPU, -c",            useCPU,             "Use only CPU without checking any available GPUs.");

    app->add_option("--numberOfThreads, -n", numberOfThreads,    "Number of threads.");
    app->add_option("--verbose, -v",         verbose,            "Verbose level. Options are \"quite\",\"fatal\",\"error\",\"warn\",\"info\" and \"debug\". Default=info");
    app->add_flag("--force, -f",             force,              "Force overwriting of existing file");

    app->callback(run_modelTest);  
     
}

                       