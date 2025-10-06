#include "cmd.h"
#include "dMRI/tractography/tractogram.h"
#include "dMRI/tractography/utility/streamline_operators.h"
#include <algorithm>
#include <chrono>

#include <numeric>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <cstddef>
#include <fstream>
#include <iostream>


#ifdef _HAS_MATPLOT_
#include <matplot/matplot.h>
#endif

using namespace NIBR;

namespace CMDARGS_MEASUREDISTANCES {
    std::string  inp_path;
    
    std::tuple<std::string, int, int, std::string> inp_model_spec("", 0, 0, ""); // module_path, inp_dim, lat_dim, data type

    int  batchSize          = 512;
    bool useCPU             = false;
    
    int numberOfThreads     =  0;
    std::string verbose     = "info";
    bool force              = false;
    std::string preCalcLoc_path = "";
}

using namespace CMDARGS_MEASUREDISTANCES; 


#include "utils/modelTestHelpers.h"

 
void run_measureDistances()
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
    auto streamlines_resampled = NIBR::resampleTractogram_withStepCount(tracObj, model.inpDim);


    disp(MSG_DETAIL,"Resampled streamlines for %d points.", model.inpDim);

    // Latent space representations
    std::vector<std::vector<double>>              enc_streamlines;

    // Decoded streamlines from the latent space representations
    NIBR::Tractogram  dec_streamlines;

    auto run_test_with_type = [&](auto type_placeholder) {
        using T = decltype(type_placeholder);
        auto lat_streamlines = encodeStreamlines<T>(streamlines_resampled,model,batchSize);         // Encode streamlines in latent space
        dec_streamlines      = decodeStreamlines<T>(lat_streamlines, model, batchSize);   // Decode the encoded streamlines
        enc_streamlines      =  to_double_vector_remove_flipped<T>(lat_streamlines, model.latDim);                     // Convert latent space representation to double type for analysis
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

    auto mdfStartTime = std::chrono::high_resolution_clock::now();
    double temp_result_mdf = getMDFDistance(tracObj[0], tracObj[1]);
    auto mdfEndTime = std::chrono::high_resolution_clock::now();
    auto mdfDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(mdfEndTime - mdfStartTime);
    std::cout << "mdf execution time: " << mdfDuration.count() << " nanoseconds | result: " << temp_result_mdf << std::endl;

    auto hausdorffStartTime = std::chrono::high_resolution_clock::now();
    double temp_result_hau = getHausdorffDistance(tracObj[2], tracObj[3]);
    auto hausdorffEndTime = std::chrono::high_resolution_clock::now();
    auto hausdorffduration = std::chrono::duration_cast<std::chrono::nanoseconds>(hausdorffEndTime - hausdorffStartTime);
    std::cout << "hausdorff execution time: " << hausdorffduration.count() << " nanoseconds | result: " << temp_result_hau << std::endl;

    auto latentSimpleStartTime = std::chrono::high_resolution_clock::now();
    double temp_result_ls = 0.067612 * latentDistanceCalculator(enc_streamlines[4], enc_streamlines[5], model.latDim);
    auto latentSimpleEndTime = std::chrono::high_resolution_clock::now();
    auto latentSimpleduration = std::chrono::duration_cast<std::chrono::nanoseconds>(latentSimpleEndTime - latentSimpleStartTime);
    std::cout << "latent simple execution time: " << latentSimpleduration.count() << " nanoseconds | result: " << temp_result_ls << std::endl;


    auto latentMinStartTime = std::chrono::high_resolution_clock::now();
    double temp_result_lm = 0.067612 * latentMinDistanceCalculator(enc_streamlines[6], enc_streamlines[7], model.latDim);
    auto latentMinEndTime = std::chrono::high_resolution_clock::now();
    auto latentMinduration = std::chrono::duration_cast<std::chrono::nanoseconds>(latentMinEndTime - latentMinStartTime);
    std::cout << "latent min execution time: " << latentMinduration.count() << " nanoseconds | result: " << temp_result_lm << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, tractogram.getNumberOfStreamlines()-1);

    size_t testAmount = 1000000;


    std::vector<long long> durationsVectorMdf(testAmount);
    std::vector<long long> durationsVectorHau(testAmount);

    std::vector<double> mdfResults(testAmount);
    std::vector<double> hauResults(testAmount);

    std::vector<long long> durationsVectorLat1(testAmount);
    std::vector<long long> durationsVectorLat2(testAmount);

    std::vector<double> Lat1Results(testAmount);
    std::vector<double> Lat2Results(testAmount);

    auto oneToOneTimesMdf = [&](NIBR::MT::TASK task) -> void {
        size_t random1 = dist(gen);
        size_t random2 = dist(gen);
        auto thisTaskTimeStart = std::chrono::high_resolution_clock::now();
        mdfResults[task.no] = getMDFDistance(tracObj[random1], tracObj[random2]);
        auto thisTaskTimeEnd = std::chrono::high_resolution_clock::now();
        auto thisTaskTimeDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(thisTaskTimeEnd - thisTaskTimeStart);
        durationsVectorMdf[task.no] = thisTaskTimeDuration.count();
    };

    auto oneToOneTimesHau = [&](NIBR::MT::TASK task) -> void {
        size_t random1 = dist(gen);
        size_t random2 = dist(gen);
        auto thisTaskTimeStart = std::chrono::high_resolution_clock::now();
        hauResults[task.no] = getHausdorffDistance(tracObj[random1], tracObj[random2]);
        auto thisTaskTimeEnd = std::chrono::high_resolution_clock::now();
        auto thisTaskTimeDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(thisTaskTimeEnd - thisTaskTimeStart);
        durationsVectorHau[task.no] = thisTaskTimeDuration.count();
    };

    auto oneToOneTimesLat1 = [&](NIBR::MT::TASK task) -> void {
        size_t random1 = dist(gen);
        size_t random2 = dist(gen);
        auto thisTaskTimeStart = std::chrono::high_resolution_clock::now();
        Lat1Results[task.no] = 0.067612 * latentDistanceCalculator(enc_streamlines[random1], enc_streamlines[random2], model.latDim);
        auto thisTaskTimeEnd = std::chrono::high_resolution_clock::now();
        auto thisTaskTimeDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(thisTaskTimeEnd - thisTaskTimeStart);
        durationsVectorLat1[task.no] = thisTaskTimeDuration.count();
    };

    auto oneToOneTimesLat2 = [&](NIBR::MT::TASK task) -> void {
        size_t random1 = dist(gen);
        size_t random2 = dist(gen);
        auto thisTaskTimeStart = std::chrono::high_resolution_clock::now();
        Lat2Results[task.no] = 0.067612 * latentMinDistanceCalculator(enc_streamlines[random1], enc_streamlines[random2], model.latDim);
        auto thisTaskTimeEnd = std::chrono::high_resolution_clock::now();
        auto thisTaskTimeDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(thisTaskTimeEnd - thisTaskTimeStart);
        durationsVectorLat2[task.no] = thisTaskTimeDuration.count();
    };

    NIBR::MT::MTRUN(testAmount, "Calculating random 1 to 1 streamlines mdf", oneToOneTimesMdf);

    auto averageDistMdf = std::accumulate(mdfResults.begin(), mdfResults.end(), 0.0);

    auto minMaxMdf = std::minmax_element(durationsVectorMdf.begin(), durationsVectorMdf.end());

    auto totalTimeMdf = std::accumulate(durationsVectorMdf.begin(), durationsVectorMdf.end(),0LL);

    double averageTimeMdf = static_cast<double>(totalTimeMdf) / durationsVectorMdf.size();

    double sqDiffSumMdf = 0.0;
    for (auto t : durationsVectorMdf) {
        double diff = t - averageTimeMdf;
        sqDiffSumMdf += diff * diff;
    }
    double stdDeviationMdf = std::sqrt(sqDiffSumMdf / testAmount);


    disp(MSG_INFO,"");
    disp(MSG_INFO,"Average time mdf 1 to 1: %lf ns", averageTimeMdf);
    disp(MSG_INFO,"Min time mdf 1 to 1: %lld ns",static_cast<long>(*minMaxMdf.first));
    disp(MSG_INFO,"Max time mdf 1 to 1: %lld ns",static_cast<long>(*minMaxMdf.second));
    disp(MSG_INFO,"Standard deviation mdf: %lf ", static_cast<float>(stdDeviationMdf));
    disp(MSG_INFO,"AverageDist: %lf", averageDistMdf/testAmount);

    std::string mdfFilename = "mdf_" + std::to_string(tracObj[0].size()) + ".csv";
    std::ofstream mdfFile(mdfFilename);
    if (!mdfFile.is_open()) {
        std::cerr << "Error: could not open file " << mdfFilename << " for writing\n";
        return;
    }
    for (auto d : durationsVectorMdf) {
        mdfFile << d << "\n";
    }
    mdfFile.close();

    
    NIBR::MT::MTRUN(testAmount, "Calculating random 1 to 1 streamlines hausdorff", oneToOneTimesHau);

    auto averageDistHau = std::accumulate(hauResults.begin(), hauResults.end(), 0.0);

    auto minMaxHau = std::minmax_element(durationsVectorHau.begin(), durationsVectorHau.end());

    auto totalTimeHau = std::accumulate(durationsVectorHau.begin(), durationsVectorHau.end(),0LL);

    double averageTimeHau = static_cast<double>(totalTimeHau) / durationsVectorHau.size();

    disp(MSG_INFO,"");
    disp(MSG_INFO,"Average time hau 1 to 1: %lf ns", averageTimeHau);
    disp(MSG_INFO,"Min time hau 1 to 1: %lld ns",static_cast<long>(*minMaxHau.first));
    disp(MSG_INFO,"Max time hau 1 to 1: %lld ns",static_cast<long>(*minMaxHau.second));
    disp(MSG_INFO,"AverageDist: %lf", averageDistHau/testAmount);
    


    NIBR::MT::MTRUN(testAmount, "Calculating random 1 to 1 latent simple", oneToOneTimesLat1);

    auto averageDistLat1 = std::accumulate(Lat1Results.begin(), Lat1Results.end(), 0.0);

    auto minMaxLat1 = std::minmax_element(durationsVectorLat1.begin(), durationsVectorLat1.end());

    auto totalTimeLat1 = std::accumulate(durationsVectorLat1.begin(), durationsVectorLat1.end(),0LL);

    double averageTimeLat1 = static_cast<double>(totalTimeLat1) / durationsVectorLat1.size();

    disp(MSG_INFO,"");
    disp(MSG_INFO,"Average time latent simple 1 to 1: %lf ns", averageTimeLat1);
    disp(MSG_INFO,"Min time latent simple 1 to 1: %lld ns",static_cast<long>(*minMaxLat1.first));
    disp(MSG_INFO,"Max time latent simple 1 to 1: %lld ns",static_cast<long>(*minMaxLat1.second));
    disp(MSG_INFO,"AverageDist: %lf", averageDistLat1/testAmount);
    


    NIBR::MT::MTRUN(testAmount, "Calculating random 1 to 1 latent min", oneToOneTimesLat2);

    auto averageDistLat2 = std::accumulate(Lat2Results.begin(), Lat2Results.end(), 0.0);

    auto minMaxLat2 = std::minmax_element(durationsVectorLat2.begin(), durationsVectorLat2.end());

    auto totalTimeLat2 = std::accumulate(durationsVectorLat2.begin(), durationsVectorLat2.end(),0LL);

    double averageTimeLat2 = static_cast<double>(totalTimeLat2) / durationsVectorLat2.size();

    double sqDiffSumLat2 = 0.0;
    for (auto t : durationsVectorLat2) {
        double diff = t - averageTimeLat2;
        sqDiffSumLat2 += diff * diff;
    }
    double stdDeviationLat2 = std::sqrt(sqDiffSumLat2 / testAmount);

    disp(MSG_INFO,"");
    disp(MSG_INFO,"Average time latent min 1 to 1: %lf ns", averageTimeLat2);
    disp(MSG_INFO,"Min time latent min 1 to 1: %lld ns",static_cast<long>(*minMaxLat2.first));
    disp(MSG_INFO,"Max time latent min 1 to 1: %lld ns",static_cast<long>(*minMaxLat2.second));
    disp(MSG_INFO,"Standard deviation latent min: %lf ", static_cast<float>(stdDeviationLat2));
    disp(MSG_INFO,"AverageDist: %lf", averageDistLat2/testAmount);

    std::string lat2Filename = "lat2_" + std::to_string(model.latDim) + ".csv";
    std::ofstream lat2File(lat2Filename);
    if (!lat2File.is_open()) {
        std::cerr << "Error: could not open file " << lat2Filename << " for writing\n";
        return;
    }
    for (auto d : durationsVectorLat2) {
        lat2File << d << "\n";
    }
    lat2File.close();

    return;





    std::vector<std::vector<double>> enc_dist1        (tracObj.size(), std::vector<double>(tracObj.size(),NAN));
    auto getEnc1 = [&](NIBR::MT::TASK task) -> void {
        for (size_t i = 0; i < task.no; i++) {
            enc_dist1[task.no][i]           = getOneSidedEncodedDistance(task.no,i);
        }
    };
    NIBR::MT::MTRUN(tracObj.size(), "Computing enc 1 distance", getEnc1);
    auto enc1 = flattenAndRemoveNANAndFree(enc_dist1);
    writeVectorToDisk(enc1, preCalcLoc_path+"enc1.bin");
    enc1.clear(); enc1.shrink_to_fit();
    MMapVector enc1_mmap = mmapVectorOpen(preCalcLoc_path+"enc1.bin");


    std::vector<std::vector<double>> enc_dist2        (tracObj.size(), std::vector<double>(tracObj.size(),NAN));
    auto getEnc2 = [&](NIBR::MT::TASK task) -> void {
        for (size_t i = 0; i < task.no; i++) {
            enc_dist2[task.no][i]           = getTwoSidedEncodedDistance(task.no,i);
        }
    };
    NIBR::MT::MTRUN(tracObj.size(), "Computing enc 2 distance", getEnc2);
    auto enc2 = flattenAndRemoveNANAndFree(enc_dist2);
    writeVectorToDisk(enc2, preCalcLoc_path+"enc2.bin");
    enc2.clear(); enc2.shrink_to_fit();
    MMapVector enc2_mmap = mmapVectorOpen(preCalcLoc_path+"enc2.bin");




    std::vector<std::vector<double>> hau_dist         (tracObj.size(), std::vector<double>(tracObj.size(),NAN));

    auto getHauDist= [&](NIBR::MT::TASK task) -> void {
        for (size_t i = 0; i < task.no; i++) {
            hau_dist[task.no][i]            = getHausdorffDistance(tracObj[task.no], tracObj[i]);
        }
    };
    NIBR::MT::MTRUN(tracObj.size(), "Computing hau distance", getHauDist);
    auto hau = flattenAndRemoveNANAndFree(hau_dist);
    writeVectorToDisk(hau, preCalcLoc_path+"hau.bin");
    hau.clear(); hau.shrink_to_fit();
    MMapVector hau_mmap = mmapVectorOpen(preCalcLoc_path+"hau.bin");




    std::vector<std::vector<double>> mdf_dist         (tracObj.size(), std::vector<double>(tracObj.size(),NAN));
    auto getMDFDist= [&](NIBR::MT::TASK task) -> void {
        for (size_t i = 0; i < task.no; i++) {
            mdf_dist[task.no][i]            = getMDFDistance(tracObj[task.no], tracObj[i]);
        }
    };
    NIBR::MT::MTRUN(tracObj.size(), "Computing mdf distance", getMDFDist);
    auto mdf = flattenAndRemoveNANAndFree(mdf_dist);
    writeVectorToDisk(mdf, preCalcLoc_path+"mdf.bin");
    mdf.clear(); mdf.shrink_to_fit();
    MMapVector mdf_mmap = mmapVectorOpen(preCalcLoc_path+"mdf.bin");

    #ifdef _HAS_MATPLOT_
    plotDistancesSideBySide(hau_mmap, mdf_mmap, enc1_mmap, enc2_mmap, edh, edm);
    std::this_thread::sleep_for(std::chrono::seconds(5));
    #endif

    double latentScalingFactorMdf = (vectorToEigen(mdf_mmap).array() / vectorToEigen(enc1_mmap).array()).mean();

    disp(MSG_INFO,"Starting lat distance calculations");

    std::vector<std::vector<double>> lat_dist       (tracObj.size(), std::vector<double>(tracObj.size(),NAN));
    

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




    std::vector<std::vector<double>> lat_min_dist   (tracObj.size(), std::vector<double>(tracObj.size(),NAN));

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

    

    double sum_hau = std::accumulate(hau_mmap.data, hau_mmap.data + hau_mmap.size, 0.0);
    double mean_hau = sum_hau / hau_mmap.size;
    auto [min_it_hau, max_it_hau] = std::minmax_element(hau_mmap.data, hau_mmap.data + hau_mmap.size);
    double min_hau = *min_it_hau;
    double max_hau = *max_it_hau;


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

    disp(MSG_INFO,"");
    disp(MSG_INFO,"Haussdorff Minimum distance between streamlines: %.6f", min_hau);
    disp(MSG_INFO,"Haussdorff Maximum distance between streamlines: %.6f", max_hau);
    disp(MSG_INFO,"Haussdorff Average distance between streamlines: %.6f", mean_hau);
    //disp(MSG_INFO,"Haussdorff Median distance between streamlines: %.6f", median_hau);
   
    
    
    disp(MSG_INFO,"");
    disp(MSG_INFO,"Pearson correlation coefficients:");
    disp(MSG_INFO,"Hausdorff and (one-sided) Euc. distance in latent space: %.6f", correlation_coefficient(hau_mmap,enc1_mmap ));
    disp(MSG_INFO,"MDF and (one-sided) Euc. distance in latent space:       %.6f", correlation_coefficient(mdf_mmap,enc1_mmap ));
    disp(MSG_INFO,"Hausdorff and MDF:                                       %.6f", correlation_coefficient(hau_mmap,mdf_mmap  ));
    disp(MSG_INFO,"One-sided and two-sided Euc. distance in latent space:   %.6f", correlation_coefficient(enc1_mmap,enc2_mmap));

    disp(MSG_INFO,"");
    disp(MSG_INFO,"Distance scaling factor to match Hausdorff distance:     %.6f", (vectorToEigen(hau_mmap).array() / vectorToEigen(enc1_mmap).array()).mean() );
    disp(MSG_INFO,"Distance scaling factor to match MDF distance:           %.6f", (vectorToEigen(mdf_mmap).array() / vectorToEigen(enc1_mmap).array()).mean() );

    disp(MSG_INFO,"");
    

    #ifdef _HAS_MATPLOT_
    wait("Done ");
    #endif


    mmapVectorClose(mdf_mmap);
    mmapVectorClose(hau_mmap);
    mmapVectorClose(enc1_mmap);
    mmapVectorClose(enc2_mmap);
    mmapVectorClose(lat1_mmap);
    mmapVectorClose(lat2_mmap);

    return;
       
}          
    
     
void measureDistance(CLI::App* app)   
{ 

    app->formatter(std::make_shared<CustomHelpFormatter>());

    const std::string info = "Basically same as model test but doesn't resample the tractogram.";

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

    app->callback(run_measureDistances);  
     
}

                       