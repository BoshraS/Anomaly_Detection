#include "cmd.h"
#include <chrono>

namespace CMDARGS_CHECKFLIPPING {
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

using namespace CMDARGS_CHECKFLIPPING;

#include "utils/modelTestHelpers.h"


void run_checkFlipping()
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


    disp(MSG_INFO,"Resampled streamlines for %d points.", model.inpDim);

    disp(MSG_INFO,"Flipping streamlines");

    NIBR::StreamlineBatch streamlines_flipped;
    streamlines_flipped.reserve(streamlines.size());

    for(auto streamline : streamlines) {
        NIBR::Streamline flipped = streamline;
        std::reverse(flipped.begin(), flipped.end());
        streamlines_flipped.push_back(std::move(flipped));
    }

    for( auto point : streamlines[0]) {
        std::cout << "x: " << point[0] << " | y: " << point[1] << " | z: " << point[2] << std::endl;
    }

    std::cout << "--------" << std::endl;

    for( auto point : streamlines_flipped[0]) {
        std::cout << "x: " << point[0] << " | y: " << point[1] << " | z: " << point[2] << std::endl;
    }

    // Latent space representations
    std::vector<std::vector<double>>              enc_streamlines;
    std::vector<std::vector<double>>              enc_streamlines_flipped;

    // Decoded streamlines from the latent space representations
    NIBR::Tractogram  dec_streamlines;


    auto lat_streamlines = encodeStreamlines<float>(streamlines,model,batchSize); 
    enc_streamlines      =  to_double_vector<float>(lat_streamlines);

    auto lat_streamlines_flipped = encodeStreamlines<float>(streamlines_flipped,model,batchSize); 
    enc_streamlines_flipped      =  to_double_vector<float>(lat_streamlines_flipped);


    //std::cout << streamlines[0] << std::endl;
    //std::cout << streamlines_flipped[0] << std::endl;


    std::cout << "og\tflip\tflipflip" << std::endl;
    for ( size_t i = 0; i < enc_streamlines[0].size(); ++i) {
        std::cout << enc_streamlines[0][i] << "\t" << enc_streamlines_flipped[0][i] << "\t" << enc_streamlines_flipped[0][(enc_streamlines_flipped[0].size()-1)-i] << std::endl;
    }
    //std::cout << enc_streamlines[0] << std::endl;
    //std::cout << enc_streamlines_flipped[0] << std::endl;

    

}

void checkFlipping(CLI::App* app)   
{ 

    app->formatter(std::make_shared<CustomHelpFormatter>());

    const std::string info = "Check Flipping";

    app->add_option("<input tractogram>",    inp_path,           "Input tractogram (.vtk, .tck)")
        ->required();

    app->add_option("<model>",               inp_model_spec,     "Input model, specified with the path to the Torch script file, followed by the input dimensions, latent space dimensions, data type (float or double), and distance scaling factor of the model. E.g. /model/test_model.pt 256 64 float 0.08. ")
        ->required();

    app->add_flag("--useCPU, -c",            useCPU,             "Use only CPU without checking any available GPUs.");

    app->add_option("--numberOfThreads, -n", numberOfThreads,    "Number of threads.");
    app->add_option("--verbose, -v",         verbose,            "Verbose level. Options are \"quite\",\"fatal\",\"error\",\"warn\",\"info\" and \"debug\". Default=info");
    app->add_flag("--force, -f",             force,              "Force overwriting of existing file");


    app->callback(run_checkFlipping);  
     
}