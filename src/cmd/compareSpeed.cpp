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

#include "utils/modelTestHelpers.h"


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

    std::vector<std::vector<double>> temp_hau(8196, std::vector<double>(tracObj.size(),NAN));
    auto oneAllHau= [&](NIBR::MT::TASK task) -> void {
        for (size_t i = 0; i < task.no; i++) {
            temp_hau[task.no][i] = getHausdorffDistance(streamlines[task.no], streamlines[i]);
        }
    };
    auto hauOneToAllStartTime = std::chrono::high_resolution_clock::now();
    NIBR::MT::MTRUN(8196, "Computing Haussdorff distances for 8196 to all", oneAllHau); 
    auto hauOneToAllEndTime = std::chrono::high_resolution_clock::now();
    auto hauOneToAllDuration = std::chrono::duration_cast<std::chrono::microseconds>(hauOneToAllEndTime - hauOneToAllStartTime);
    disp(MSG_INFO,"Haussdorf one to all duration: %.i microseconds\n", hauOneToAllDuration);

    std::vector<std::vector<double>> temp_mdf(8196, std::vector<double>(tracObj.size(),NAN));
    auto oneAllMDF= [&](NIBR::MT::TASK task) -> void {
        for (size_t i = 0; i < task.no; i++) {
            temp_mdf[task.no][i] = getMDFDistance(tracObj[0], tracObj[task.no]);
        }

    };
    auto mdfOneToAllStartTime = std::chrono::high_resolution_clock::now();
    NIBR::MT::MTRUN(8196, "Computing MDF distances for 8196 to all", oneAllMDF); 
    auto mdfOneToAllEndTime = std::chrono::high_resolution_clock::now();
    auto mdfOneToAllDuration = std::chrono::duration_cast<std::chrono::microseconds>(mdfOneToAllEndTime - mdfOneToAllStartTime);
    disp(MSG_INFO,"MDF one to all duration: %.i microseconds\n", mdfOneToAllDuration);


    std::vector<std::vector<double>> temp_lat1(8196, std::vector<double>(enc_streamlines.size(),NAN));
    auto oneAllLat1= [&](NIBR::MT::TASK task) -> void {
        for (size_t i = 0; i < task.no; i++) {
            temp_hau[task.no][i] = latentDistanceCalculator(enc_streamlines[task.no], enc_streamlines[i], model.latDim);
        }
    };
    auto lat1OneToAllStartTime = std::chrono::high_resolution_clock::now();
    NIBR::MT::MTRUN(8196, "Computing Latent distances simple for 8196 to all", oneAllLat1); 
    auto lat1OneToAllEndTime = std::chrono::high_resolution_clock::now();
    auto lat1OneToAllDuration = std::chrono::duration_cast<std::chrono::microseconds>(lat1OneToAllEndTime - lat1OneToAllStartTime);
    disp(MSG_INFO,"Lat simple one to all duration: %.i microseconds\n", lat1OneToAllDuration);

    std::vector<std::vector<double>> temp_lat2(8196, std::vector<double>(enc_streamlines.size(),NAN));
    auto oneAllLat2= [&](NIBR::MT::TASK task) -> void {
        for (size_t i = 0; i < task.no; i++) {
            temp_hau[task.no][i] = latentMinDistanceCalculator(enc_streamlines[task.no], enc_streamlines[i], model.latDim);
        }
    };
    auto lat2OneToAllStartTime = std::chrono::high_resolution_clock::now();
    NIBR::MT::MTRUN(8196, "Computing Latent distances min for 8196 to all", oneAllLat2); 
    auto lat2OneToAllEndTime = std::chrono::high_resolution_clock::now();
    auto lat2OneToAllDuration = std::chrono::duration_cast<std::chrono::microseconds>(lat2OneToAllEndTime - lat2OneToAllStartTime);
    disp(MSG_INFO,"Lat min one to all duration: %.i microseconds\n", lat2OneToAllDuration);

    

}

void compareSpeed(CLI::App* app)   
{ 

    app->formatter(std::make_shared<CustomHelpFormatter>());

    const std::string info = "Compare speed";

    app->add_option("<input tractogram>",    inp_path,           "Input tractogram (.vtk, .tck)")
        ->required();

    app->add_option("<model>",               inp_model_spec,     "Input model, specified with the path to the Torch script file, followed by the input dimensions, latent space dimensions, data type (float or double), and distance scaling factor of the model. E.g. /model/test_model.pt 256 64 float 0.08. ")
        ->required();

    app->add_flag("--useCPU, -c",            useCPU,             "Use only CPU without checking any available GPUs.");

    app->add_option("--numberOfThreads, -n", numberOfThreads,    "Number of threads.");
    app->add_option("--verbose, -v",         verbose,            "Verbose level. Options are \"quite\",\"fatal\",\"error\",\"warn\",\"info\" and \"debug\". Default=info");
    app->add_flag("--force, -f",             force,              "Force overwriting of existing file");


    app->callback(run_compareSpeed);  
     
}