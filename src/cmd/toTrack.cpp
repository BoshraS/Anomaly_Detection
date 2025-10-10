#include "cmd.h"

using namespace NIBR;

namespace CMDARGS_TOTRACK {
    std::string  inp_tractogram;
    std::string  inp_scores = "";
    std::string  inp_labels = "";

    std::string  out_tractogram;
    
    int numberOfThreads     =  0;
    std::string verbose     = "info";
    bool force              = false;
}

using namespace CMDARGS_TOTRACK;

 
void run_toTrack()
{ 

    parseCommon(numberOfThreads,verbose);

    if (!parseForceOutput(out_tractogram,force)) return;
    if(!ensureVTKorTCK(inp_tractogram)) return;
    if(!ensureVTK(out_tractogram)) return;
    
    if (getFileExtension(inp_scores) != "ano") {
        disp(MSG_ERROR,"Input anomaly scores must have .ano extension.");
        return;
    }

    if (getFileExtension(inp_labels) != "clb") {
        disp(MSG_ERROR,"Input anomaly scores must have .clb extension.");
        return;
    }

    if ((inp_scores == "") && (inp_labels == "")) {
        disp(MSG_ERROR,"Missing score or cluster label file.");
        return;
    }

    // Read number of streamlines from tractogram
    NIBR::TractogramReader tractogram(inp_tractogram);
    int N = tractogram.numberOfStreamlines; // Number of streamlines

    // Check number of streamlines in the score file
    std::ifstream ifs_scores;
    if (inp_scores != "") {
        ifs_scores.open(inp_scores, std::ios::binary | std::ios::ate | std::ios::in);
        if (!ifs_scores.is_open()) {
            disp(MSG_ERROR,"Failed to open file: %s", inp_scores.c_str());
            return;
        }
        int N_fromScoreFile = ifs_scores.tellg() / sizeof(float) ; // Number of streamlines in the in the score file
        ifs_scores.seekg(0, std::ios::beg);

        if (N_fromScoreFile != N) {
            disp(MSG_ERROR,"Streamline count between input tractogram and scores does not match.");
            ifs_scores.close();
            return;
        }
    }

    // Check number of streamlines in the cluster label file
    std::ifstream ifs_labels;
    if (inp_labels != "") {
        ifs_labels.open(inp_labels, std::ios::binary | std::ios::ate | std::ios::in);
        if (!ifs_labels.is_open()) {
            disp(MSG_ERROR,"Failed to open file: %s", inp_scores.c_str());
            return;
        }
        int N_fromClusterLabelFile = ifs_labels.tellg() / sizeof(int); // Number of streamlines in the in the score file
        ifs_labels.seekg(0, std::ios::beg);

        if (N_fromClusterLabelFile != N) {
            disp(MSG_ERROR,"Streamline count between input tractogram and cluster labels does not match.");
            ifs_labels.close();
            return;
        }
    }

    // Read all streamlines
    disp(MSG_DETAIL,"Reading input tractogram");
    NIBR::Tractogram allStreamlines = tractogram.getTractogram();

    // Prepare tractogram field vector
    std::vector<TractogramField> fields;
    float** scoreData = NULL;
    int**   labelData = NULL;

    // Read scores
    if (inp_scores != "") {

        scoreData = new float*[N];
        for (int n = 0; n < N; n++) {
            scoreData[n]    = new float[1];
            ifs_scores.read(reinterpret_cast<char*>(scoreData[n]), sizeof(float));
        }
        ifs_scores.close();

        TractogramField f;
        f.owner     = STREAMLINE_OWNER;
        f.name      = "score";
        f.datatype  = FLOAT32_DT;
        f.dimension = 1;
        f.data      = (void*)scoreData;
        
        fields.push_back(f);        
    }

    // Read labels
    if (inp_labels != "") {

        labelData = new int*[N];
        for (int n = 0; n < N; n++) {
            labelData[n]    = new int[1];
            ifs_labels.read(reinterpret_cast<char*>(labelData[n]), sizeof(int));
        }
        ifs_labels.close();

        TractogramField f;
        f.owner     = STREAMLINE_OWNER;
        f.name      = "label";
        f.datatype  = INT32_DT;
        f.dimension = 1;
        f.data      = (void*)labelData;
        
        fields.push_back(f);        
    }

    // Write output tractogram
    writeTractogram(out_tractogram,allStreamlines,fields);

    // Delete field data
    if (inp_scores != "") {
        for (int n = 0; n < N; n++) {
            delete[] scoreData[n];
        }
        delete[] scoreData;
    }

    if (inp_labels != "") {
        for (int n = 0; n < N; n++) {
            delete[] labelData[n];
        }
        delete[] labelData;
    }

    return;
       
}          
    
     
void toTrack(CLI::App* app)   
{ 

    app->formatter(std::make_shared<CustomHelpFormatter>());

    app->description("maps anomaly scores and/or cluster labels on streamlines");

    app->add_option("<input>",               inp_tractogram,     "Input tractogram (.vtk,.tck).")
        ->required();

    app->add_option("--scores,-s",           inp_scores,         "Path to anomaly score file (.ano).");
    app->add_option("--labels,-l",           inp_labels,         "Path to cluster label file (.clb).");
     
    app->add_option("<output>",              out_tractogram,     "Output tractogram (.vtk).")
        ->required();

    app->add_option("--verbose, -v",         verbose,            "Verbose level. Options are \"quite\",\"fatal\",\"error\",\"warn\",\"info\" and \"debug\". Default=info");
    app->add_flag("--force, -f",             force,              "Force overwriting of existing file");

    app->callback(run_toTrack);  
     
}

                       