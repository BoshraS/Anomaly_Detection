#include "cmd/cmd.h"
#include "utils.h"

using namespace NIBR;

bool decodeAndSave(std::string inp, std::string out, bool force, StreamlineAutoencoder& model, int batchSize)
{
    
    if (existsFile(out) && !force) return true;

    // Initialize file reader for input latent representations
    std::ifstream ifs(inp, std::ios::binary | std::ios::ate | std::ios::in);
    if (!ifs.is_open()) {
        disp(MSG_ERROR,"Failed to open file: %s", inp.c_str());
        return false;
    }
    int N = ifs.tellg() / (sizeof(float) * 2 * model.latDim); // Number of streamlines in the input file
    ifs.seekg(0, std::ios::beg);

    std::vector<at::Tensor> batches;

    int begInd = 0;
    int endInd = 0;

    while (endInd != N) {

        begInd = endInd;
        endInd = ((begInd + batchSize) <= N) ? (begInd + batchSize) : N;

        int curBatchSize = endInd - begInd;

        // Create blob having only one representation of the cluster (removing the flipped version)
        std::vector<float> blob;
        blob.reserve(curBatchSize * model.latDim);

        for (int i = 0; i < curBatchSize; i++) {
            blob.resize(blob.size() + model.latDim);
            ifs.read(reinterpret_cast<char*>(blob.data() + i * model.latDim), model.latDim * sizeof(float));
            ifs.seekg(model.latDim * sizeof(float), std::ios::cur);
        }

        auto curBatch = torch::from_blob(blob.data(), {curBatchSize, model.latDim}, torch::kFloat).to(model.device).clone();

        batches.push_back(std::move(curBatch));
    }

    // Concatenate all batches into one tensor
    auto input = torch::cat(batches, 0);

    // Prepare the input as a vector of torch::jit::IValue
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
        
    // Decode the cluster centers
    at::Tensor decoded = model.module.get_method("decode")(inputs).toTensor().to(torch::kCPU).contiguous();

    // Copy decoded tensor data to decodedCenters vector
    std::vector<std::vector<std::vector<float>>> streamlines(N);

    for (int i = 0; i < N; ++i) {
        std::vector<float> tmp(3 * model.inpDim);
        std::memcpy(tmp.data(), decoded[i].data_ptr<float>(), 3 * model.inpDim * sizeof(float));

        std::vector<std::vector<float>> trk(model.inpDim, std::vector<float>(3));
        for (int j = 0; j < model.inpDim; ++j) {
            trk[j][0] = tmp[j];
            trk[j][1] = tmp[j + model.inpDim];
            trk[j][2] = tmp[j + model.inpDim * 2];
        }
        streamlines[i] = trk;
    }

    writeTractogram(out,streamlines);
    return true;

}