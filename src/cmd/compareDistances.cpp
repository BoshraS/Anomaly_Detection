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
        enc_streamlines      =  to_double_vector<T>(lat_streamlines);                     // Convert latent space representation to double type for analysis
        // disp(MSG_INFO,"Encoding completed.");
    };
     
    // Dispatch to the generic lambda with the correct type
    if (model.dtype == torch::kFloat)       { run_test_with_type(float{});   } 
    else if (model.dtype == torch::kDouble) { run_test_with_type(double{});  } 
    else if (model.dtype == torch::kHalf)   { run_test_with_type(at::Half{});} 
    else { disp(MSG_ERROR, "Unsupported data type for modelTest: %s", c10::toString(model.dtype)); }

    MMapVector hau_mmap;
    MMapVector mdf_mmap;

    if(hau_base_path != "") {
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
        hau_mmap = mmapVectorOpen(preCalcLoc_path+"hau.bin");
    }

    

    if(mdf_base_path != "") {
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
        mdf_mmap = mmapVectorOpen(preCalcLoc_path+"mdf.bin");
    }

    






    MMapVector hau_base_mmap;
    MMapVector mdf_base_mmap;
    if(hau_base_path != "")
        hau_base_mmap = mmapVectorOpen(hau_base_path);

    if(mdf_base_path != "")
        mdf_base_mmap = mmapVectorOpen(mdf_base_path);

    
    std::vector<double> perThreadAverageMdf(numberOfThreads);
    std::vector<double> perThreadMaxMdf(numberOfThreads, std::numeric_limits<double>::min());
    std::vector<double> perThreadMinMdf(numberOfThreads, std::numeric_limits<double>::max());


    auto compareToBaseMDF = [&](NIBR::MT::TASK task) -> void {
        double tempDiff = std::abs(mdf_base_mmap.data[task.no] - mdf_mmap.data[task.no]);
        perThreadAverageMdf[task.threadId] += tempDiff;
        if (tempDiff < perThreadMinMdf[task.threadId]) perThreadMinMdf[task.threadId] = tempDiff;
        if (tempDiff > perThreadMaxMdf[task.threadId]) perThreadMaxMdf[task.threadId] = tempDiff;
    };

    std::vector<double> perThreadAverageHau(numberOfThreads);
    std::vector<double> perThreadMaxHau(numberOfThreads, std::numeric_limits<double>::min());
    std::vector<double> perThreadMinHau(numberOfThreads, std::numeric_limits<double>::max());
    
    auto compareToBaseHau = [&](NIBR::MT::TASK task) -> void {
        double tempDiff = std::abs(hau_base_mmap.data[task.no] - hau_mmap.data[task.no]);
        perThreadAverageHau[task.threadId] += tempDiff;
        if (tempDiff < perThreadMinHau[task.threadId]) perThreadMinHau[task.threadId] = tempDiff;
        if (tempDiff > perThreadMaxHau[task.threadId]) perThreadMaxHau[task.threadId] = tempDiff;
    };

    if(mdf_base_path != ""){
        NIBR::MT::MTRUN(mdf_base_mmap.size, "Comparing base to MDF", compareToBaseMDF);

        double mdfAverage = (std::accumulate(perThreadAverageMdf.begin(), perThreadAverageMdf.end(), 0.0)) / mdf_base_mmap.size;
        double mdfMin = *std::min_element(perThreadMinMdf.begin(), perThreadMinMdf.end());
        double mdfMax = *std::max_element(perThreadMaxMdf.begin(), perThreadMaxMdf.end());

        disp(MSG_INFO,"");
        disp(MSG_INFO,"MDF average difference: %d", mdfAverage);
        disp(MSG_INFO,"MDF min difference: %d", mdfMin);
        disp(MSG_INFO,"MDF max difference: %d", mdfMax);

    }

    if(hau_base_path != ""){
        NIBR::MT::MTRUN(hau_base_mmap.size, "Comparing base to Hausdorff", compareToBaseHau);

        double hauAverage = (std::accumulate(perThreadAverageHau.begin(), perThreadAverageHau.end(), 0.0)) / mdf_base_mmap.size;
        double hauMin = *std::min_element(perThreadMinHau.begin(), perThreadMinHau.end());
        double hauMax = *std::max_element(perThreadMaxHau.begin(), perThreadMaxHau.end());

        disp(MSG_INFO,"");
        disp(MSG_INFO,"MDF average difference: %d", hauAverage);
        disp(MSG_INFO,"MDF min difference: %d", hauMin);
        disp(MSG_INFO,"MDF max difference: %d", hauMax);
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

    app->add_option("--preCalcLoc", preCalcLoc_path, "Where to store generated files. If left empty the current directory is used.");
    app->add_flag("--useCPU, -c",            useCPU,             "Use only CPU without checking any available GPUs.");

    app->add_option("--numberOfThreads, -n", numberOfThreads,    "Number of threads.");
    app->add_option("--verbose, -v",         verbose,            "Verbose level. Options are \"quite\",\"fatal\",\"error\",\"warn\",\"info\" and \"debug\". Default=info");
    app->add_flag("--force, -f",             force,              "Force overwriting of existing file");

    app->callback(run_compareDistance);  
     
}

                       