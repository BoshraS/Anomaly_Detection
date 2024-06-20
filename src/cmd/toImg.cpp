#include "cmd.h"

using namespace NIBR;

namespace CMDARGS_TOIMG {
    std::string  inp_tractogram;
    std::string  inp_scores;
    std::string  out_img;

    std::string  weight     = "";

    std::string  temp_img   = "";
    float        voxDim     = 1.0f;
    float        thresh     = 0.0f;
    bool         medFilt    = false;
    bool         aveFilt    = false;
    
    int numberOfThreads     =  0;
    std::string verbose     = "info";
    bool force              = false;
}

using namespace CMDARGS_TOIMG;

 
void run_toImg()
{ 

    parseCommon(numberOfThreads,verbose);

    if (!parseForceOutput(out_img,force)) return;
    if(!ensureVTKorTCK(inp_tractogram)) return;
    
    if (getFileExtension(inp_scores) != "ano") {
        disp(MSG_ERROR,"Input anomaly scores must have .ano extension.");
        return;
    }

    // Read number of streamlines from tractogram
    NIBR::TractogramReader tractogram(inp_tractogram);
    int N = tractogram.numberOfStreamlines; // Number of streamlines

    // Check number of streamlines in the score file
    std::ifstream ifs_scores(inp_scores, std::ios::binary | std::ios::ate | std::ios::in);
    if (!ifs_scores.is_open()) {
        disp(MSG_ERROR,"Failed to open file: %s", inp_scores.c_str());
        return;
    }
    int N_fromScoreFile = ifs_scores.tellg() / sizeof(float) ; // Number of streamlines in the in the score file
    ifs_scores.seekg(0, std::ios::beg);

    if (N_fromScoreFile != N) {
        disp(MSG_ERROR,"Streamline count between input tractogram and score file does not match.");
        ifs_scores.close();
        return;
    }

    // Check the weight file if provided
    std::ifstream ifs_weights;
    if (weight != "") {
        ifs_weights.open(weight, std::ios::binary | std::ios::ate | std::ios::in);
        if (!ifs_weights.is_open()) {
            disp(MSG_ERROR,"Failed to open file: %s", weight.c_str());
            return;
        }
        int N_fromWeights = ifs_weights.tellg() / sizeof(float) ; // Number of streamlines in the in the weıght file
        ifs_weights.seekg(0, std::ios::beg);

        if (N_fromWeights != N) {
            disp(MSG_ERROR,"Streamline count between input tractogram and weight file does not match.");
            ifs_weights.close();
            return;
        }
    }

    // Read the scores
    std::vector<float> scores;
    scores.resize(N);
    ifs_scores.read(reinterpret_cast<char*>(scores.data()), sizeof(float) * N);
    ifs_scores.close();

    if (thresh > 0) {
        for (int n = 0; n < N; n++) {
            if (scores[n] <= thresh) scores[n] = 0.0f;
        }
    }

    // Read the weights and scale scores
    if (weight != "") {
        std::vector<float> tmp;
        tmp.resize(N);
        ifs_weights.read(reinterpret_cast<char*>(tmp.data()), sizeof(float) * N);
        ifs_weights.close();
        for (int n = 0; n < N; n++) {
            scores[n] *= tmp[n];
        }
    }
    
    // Map to anomaly image
    disp(MSG_DETAIL,"Computing anomaly image");
    NIBR::Image<float> img_ano;
    NIBR::Image<float> img_ref;

    if (temp_img!="") {
        NIBR::Image<float> tempImg(temp_img);
        img_ano.createFromTemplate(tempImg,false);
        img_ref.createFromTemplate(tempImg,false);
    } else {
        std::vector<float> bb = getTractogramBBox(&tractogram);
        if (voxDim <= 0) voxDim = 1.0f;
        img_ano.createFromBoundingBox(3,bb,voxDim,false);
        img_ref.createFromBoundingBox(3,bb,voxDim,false);
    }
    

    Tractogram2ImageMapper<float> gridder_ano(&tractogram,&img_ano);
    // gridder_ano.setWeights(scores, STREAMLINE_WEIGHT);
    gridder_ano.setWeights(scores, STREAMLINE_WEIGHT);
    allocateGrid_4segmentLength(&gridder_ano);
    gridder_ano.run(processor_4segmentLength_weighted<float>, outputCompiler_4segmentLength<float>);
    deallocateGrid_4segmentLength(&gridder_ano);

    Tractogram2ImageMapper<float> gridder_ref(&tractogram,&img_ref);
    allocateGrid_4segmentLength(&gridder_ref);
    gridder_ref.run(processor_4segmentLength<float>,          outputCompiler_4segmentLength<float>);
    deallocateGrid_4segmentLength(&gridder_ref);

    for (int n = 0; n < img_ref.numel; n++) {
        if (img_ref.data[n] != 0) {
            img_ano.data[n] /= img_ref.data[n];
        }
    }

    if (medFilt) {
        Image<float> tmp;
        imgMedFilt(tmp,img_ano,CONN27);
        img_ano = tmp;
    }

    if (aveFilt) {
        Image<float> tmp;
        imgAveFilt(tmp,img_ano,CONN27);
        img_ano = tmp;
    }
    
    img_ano.write(out_img);

    return;
       
}          
    
     
void toImg(CLI::App* app)   
{ 

    app->formatter(std::make_shared<CustomHelpFormatter>());

    app->description("maps anomaly scores on an image");

    app->add_option("<input tractogram>",    inp_tractogram,     "Input tractogram (.vtk,.tck).")
        ->required();

    app->add_option("<input scores>",        inp_scores,         "Path to anomaly score file (.ano).")
        ->required();
     
    app->add_option("<output image>",        out_img,            "Output anomaly image (.nii,.nii.gz).")
        ->required();

    app->add_option("--weight",              weight,             "Streamline weights to scale anomaly scores.");
    app->add_option("--thresh",              thresh,             "Anomaly threshold. Streamlines with anomaly below this level will be ignored. This can be set to the same value as the --maxDist parameter used in findClusterCenters.");
    app->add_flag  ("--medFilt",             medFilt,            "Median filtered image. (Applied before aveFilt if set.)");
    app->add_flag  ("--aveFilt",             aveFilt,            "Average filtered image. (Applied after medFilt if set.)");
    app->add_option("--template",            temp_img,           "Template image for the anomaly mapping (.nii,.nii.gz)");
    app->add_option("--voxDim",              voxDim,             "Voxel size of the output anomaly mapping.");

    app->add_option("--numberOfThreads, -n", numberOfThreads,    "Number of threads.");
    app->add_option("--verbose, -v",         verbose,            "Verbose level. Options are \"quite\",\"fatal\",\"error\",\"warn\",\"info\" and \"debug\". Default=info");
    app->add_flag("--force, -f",             force,              "Force overwriting of existing file");

    app->callback(run_toImg);  
     
}

                       