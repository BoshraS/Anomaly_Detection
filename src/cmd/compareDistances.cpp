#include "cmd.h"
#include "dMRI/tractography/tractogram.h"
#include "dMRI/tractography/utility/streamline_operators.h"
#include <algorithm>
#include <chrono>

#include <limits>
#include <numeric>
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

namespace CMDARGS_COMPAREDISTANCE {
    std::string  inp_path;
    
    std::tuple<std::string, int, int, std::string> inp_model_spec("", 0, 0, ""); // module_path, inp_dim, lat_dim, data type

    int  batchSize          = 512;
    bool useCPU             = false;
    
    int numberOfThreads     =  0;
    std::string verbose     = "info";
    bool force              = false;
    std::string preCalcLoc_path = "";
    std::string mdf_base_path = "";
    std::string hau_base_path = "";
    double lat_scaling_factor_inp = 1.0;
    std::string lat_base_path = "";
}

using namespace CMDARGS_COMPAREDISTANCE; 


#include "utils/modelTestHelpers.h"

 
void run_compareDistance()
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

    if(mdf_base_path == "" && hau_base_path =="") {
        disp(MSG_ERROR,"MDF base and Hausdorff base are not set. Atleast one needs to be given.");
        return;
    }

    // Set model
    StreamlineAutoencoder model = StreamlineAutoencoder(inp_model_spec, useCPU);
    if (!model.isReady()) return;
    
    // Prepare tractogram reader
    NIBR::TractogramReader tractogram(inp_path, false);


    auto tracObj = tractogram.getTractogram();


    if(lat_base_path != "") {
        auto streamlines_resampled = NIBR::resampleTractogram_withStepCount(tracObj, model.inpDim);
        disp(MSG_DETAIL,"Resampled streamlines for %d points.", model.inpDim);
        std::vector<std::vector<double>>              enc_streamlines;

        double lat_model_scaling_factor = lat_scaling_factor_inp;

        auto run_test_with_type = [&](auto type_placeholder) {
            using T = decltype(type_placeholder);
            auto lat_streamlines = encodeStreamlines<T>(streamlines_resampled,model,batchSize);         // Encode streamlines in latent space
            enc_streamlines      =  to_double_vector_remove_flipped<T>(lat_streamlines, model.latDim);                     // Convert latent space representation to double type for analysis
        };
        
        // Dispatch to the generic lambda with the correct type
        if (model.dtype == torch::kFloat)       { run_test_with_type(float{});   } 
        else if (model.dtype == torch::kDouble) { run_test_with_type(double{});  } 
        else if (model.dtype == torch::kHalf)   { run_test_with_type(at::Half{});} 
        else { disp(MSG_ERROR, "Unsupported data type for modelTest: %s", c10::toString(model.dtype)); }


        MMapVector lat_base_mmap = mmapVectorOpen(lat_base_path);

        std::vector<std::vector<double>> lat_dist         (tracObj.size(), std::vector<double>(tracObj.size(),NAN));

        auto getLatDist= [&](NIBR::MT::TASK task) -> void {
            for (size_t i = 0; i < task.no; i++) {
                lat_dist[task.no][i] = lat_model_scaling_factor * latentMinDistanceCalculator(enc_streamlines[task.no], enc_streamlines[i], model.latDim);
            }
        };
        NIBR::MT::MTRUN(enc_streamlines.size(), "Computing min lat distance", getLatDist);
        auto lat = flattenAndRemoveNANAndFree(lat_dist);

        std::vector<double> perThreadAverageLat(numberOfThreads, 0.0);
        std::vector<std::vector<double>> perThreadDiffLat(numberOfThreads);
        std::vector<double> perThreadMaxLat(numberOfThreads, std::numeric_limits<double>::min());
        std::vector<double> perThreadMinLat(numberOfThreads, std::numeric_limits<double>::max());
        
        auto compareToBaseLat = [&](NIBR::MT::TASK task) -> void {
            double tempDiff = std::abs(lat_base_mmap.data[task.no] - lat[task.no]);
            perThreadAverageLat[task.threadId] += tempDiff;
            perThreadDiffLat[task.threadId].push_back(tempDiff);
            if (tempDiff < perThreadMinLat[task.threadId]) perThreadMinLat[task.threadId] = tempDiff;
            if (tempDiff > perThreadMaxLat[task.threadId]) perThreadMaxLat[task.threadId] = tempDiff;
        };

        NIBR::MT::MTRUN(lat_base_mmap.size, "Comparing base to Lat", compareToBaseLat);

        double hauAverage = (std::accumulate(perThreadAverageLat.begin(), perThreadAverageLat.end(), 0.0)) / lat_base_mmap.size;
        double hauMin = *std::min_element(perThreadMinLat.begin(), perThreadMinLat.end());
        double hauMax = *std::max_element(perThreadMaxLat.begin(), perThreadMaxLat.end());

        disp(MSG_INFO,"");
        disp(MSG_INFO,"Hau average difference: %f", hauAverage);
        disp(MSG_INFO,"Hau min difference: %f", hauMin);
        disp(MSG_INFO,"Hau max difference: %f", hauMax);

        std::string latFilename = preCalcLoc_path + "lat_distance_diffs_64v" + std::to_string(model.latDim) + ".csv";
        std::ofstream latFile(latFilename);
        if (!latFile.is_open()) {
            std::cerr << "Error: could not open file " << latFilename << " for writing\n";
            return;
        }
        for (auto v : perThreadDiffLat) {
            for(auto d: v)
                latFile << d << "\n";
        }
        latFile.close();
    }

    


    if(hau_base_path != "") {
        MMapVector hau_base_mmap = mmapVectorOpen(hau_base_path);
        std::vector<std::vector<double>> hau_dist         (tracObj.size(), std::vector<double>(tracObj.size(),NAN));

        auto getHauDist= [&](NIBR::MT::TASK task) -> void {
            for (size_t i = 0; i < task.no; i++) {
                hau_dist[task.no][i]            = getHausdorffDistance(tracObj[task.no], tracObj[i]);
            }
        };
        NIBR::MT::MTRUN(tracObj.size(), "Computing hau distance", getHauDist);
        auto hau = flattenAndRemoveNANAndFree(hau_dist);

        std::vector<double> perThreadAverageHau(numberOfThreads, 0.0);
        std::vector<std::vector<double>> perThreadDiffHau(numberOfThreads);
        std::vector<double> perThreadMaxHau(numberOfThreads, std::numeric_limits<double>::min());
        std::vector<double> perThreadMinHau(numberOfThreads, std::numeric_limits<double>::max());
        
        auto compareToBaseHau = [&](NIBR::MT::TASK task) -> void {
            double tempDiff = std::abs(hau_base_mmap.data[task.no] - hau[task.no]);
            perThreadAverageHau[task.threadId] += tempDiff;
            perThreadDiffHau[task.threadId].push_back(tempDiff);
            if (tempDiff < perThreadMinHau[task.threadId]) perThreadMinHau[task.threadId] = tempDiff;
            if (tempDiff > perThreadMaxHau[task.threadId]) perThreadMaxHau[task.threadId] = tempDiff;
        };

        NIBR::MT::MTRUN(hau_base_mmap.size, "Comparing base to Hausdorff", compareToBaseHau);

        double hauAverage = (std::accumulate(perThreadAverageHau.begin(), perThreadAverageHau.end(), 0.0)) / hau_base_mmap.size;
        double hauMin = *std::min_element(perThreadMinHau.begin(), perThreadMinHau.end());
        double hauMax = *std::max_element(perThreadMaxHau.begin(), perThreadMaxHau.end());

        disp(MSG_INFO,"");
        disp(MSG_INFO,"Hau average difference: %f", hauAverage);
        disp(MSG_INFO,"Hau min difference: %f", hauMin);
        disp(MSG_INFO,"Hau max difference: %f", hauMax);

        std::string hauFilename = preCalcLoc_path + "hau_distance_diffs_256v" + std::to_string(tracObj[0].size()) + ".csv";
        std::ofstream hauFile(hauFilename);
        if (!hauFile.is_open()) {
            std::cerr << "Error: could not open file " << hauFilename << " for writing\n";
            return;
        }
        for (auto v : perThreadDiffHau) {
            for(auto d: v)
                hauFile << d << "\n";
        }
        hauFile.close();


    }


    if(mdf_base_path != "") {
        MMapVector mdf_base_mmap = mmapVectorOpen(mdf_base_path);
        std::vector<std::vector<double>> mdf_dist         (tracObj.size(), std::vector<double>(tracObj.size(),NAN));
        auto getMDFDist= [&](NIBR::MT::TASK task) -> void {
            for (size_t i = 0; i < task.no; i++) {
                mdf_dist[task.no][i]            = getMDFDistance(tracObj[task.no], tracObj[i]);
            }
        };
        NIBR::MT::MTRUN(tracObj.size(), "Computing mdf distance", getMDFDist);
        auto mdf = flattenAndRemoveNANAndFree(mdf_dist);

        std::vector<double> perThreadAverageMdf(numberOfThreads, 0.0);
        std::vector<std::vector<double>> perThreadDiffMdf(numberOfThreads);
        std::vector<double> perThreadMaxMdf(numberOfThreads, 0.0);
        std::vector<double> perThreadMinMdf(numberOfThreads, std::numeric_limits<double>::max());


        auto compareToBaseMDF = [&](NIBR::MT::TASK task) -> void {
            double tempDiff = std::abs(mdf_base_mmap.data[task.no] - mdf[task.no]);
            perThreadAverageMdf[task.threadId] += tempDiff;
            perThreadDiffMdf[task.threadId].push_back(tempDiff);
            if (tempDiff < perThreadMinMdf[task.threadId]) perThreadMinMdf[task.threadId] = tempDiff;
            if (tempDiff > perThreadMaxMdf[task.threadId]) perThreadMaxMdf[task.threadId] = tempDiff;
        };

        NIBR::MT::MTRUN(mdf_base_mmap.size, "Comparing base to MDF", compareToBaseMDF);

        double mdfAverage = (std::accumulate(perThreadAverageMdf.begin(), perThreadAverageMdf.end(), 0.0)) / mdf_base_mmap.size;
        double mdfMin = *std::min_element(perThreadMinMdf.begin(), perThreadMinMdf.end());
        double mdfMax = *std::max_element(perThreadMaxMdf.begin(), perThreadMaxMdf.end());

        disp(MSG_INFO,"");
        disp(MSG_INFO,"MDF average difference: %f", mdfAverage);
        disp(MSG_INFO,"MDF min difference: %f", mdfMin);
        disp(MSG_INFO,"MDF max difference: %f", mdfMax);

        std::string mdfFilename = preCalcLoc_path + "mdf_distance_diffs_256v" + std::to_string(tracObj[0].size()) + ".csv";
        std::ofstream mdfFile(mdfFilename);
        if (!mdfFile.is_open()) {
            std::cerr << "Error: could not open file " << mdfFilename << " for writing\n";
            return;
        }
        for (auto v : perThreadDiffMdf) {
            for(auto d: v)
                mdfFile << d << "\n";
        }
        mdfFile.close();
    }
    

    return;
       
}          
    
     
void compareDistance(CLI::App* app)   
{ 

    app->formatter(std::make_shared<CustomHelpFormatter>());

    const std::string info = "Basically same as model test but doesn't resample the tractogram.";

    app->description("tests how well a model encodes a tractogram");

    app->add_option("<input tractogram>",    inp_path,           "Input tractogram (.vtk, .tck)")
        ->required();

    app->add_option("<model>",               inp_model_spec,     "Input model, specified with the path to the Torch script file, followed by the input dimensions, latent space dimensions, data type (float or double), and distance scaling factor of the model. E.g. /model/test_model.pt 256 64 float 0.08. ")
        ->required();

    app->add_option("--mdfBase", mdf_base_path, "A mdf .bin file generated by modelTest");
    
    app->add_option("--hauBase", hau_base_path, "A hau .bin file generated by modelTest");

    app->add_option("--latBase", lat_base_path, "A lat .bin file. Also requires the use of --latScaling or the scaling will be set to 1.");
    app->add_option("--latScaling", lat_scaling_factor_inp, "Scaling factor of the given model. Required if using --latScaling.");

    app->add_option("--preCalcLoc", preCalcLoc_path, "Where to store generated files. If left empty the current directory is used.");
    app->add_flag("--useCPU, -c",            useCPU,             "Use only CPU without checking any available GPUs.");

    app->add_option("--numberOfThreads, -n", numberOfThreads,    "Number of threads.");
    app->add_option("--verbose, -v",         verbose,            "Verbose level. Options are \"quite\",\"fatal\",\"error\",\"warn\",\"info\" and \"debug\". Default=info");
    app->add_flag("--force, -f",             force,              "Force overwriting of existing file");

    app->callback(run_compareDistance);  
     
}

                       