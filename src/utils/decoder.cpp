#include "cmd/cmd.h"
#include "utils.h"

using namespace NIBR;

std::vector<std::vector<std::vector<float>>> decodeStreamlines(             // output tractogram <number of streamlines x (fixed) inpDim x 3>
    const std::vector<std::vector<float>>& latent,                          // input latent representations <number of streamlines x latDim>
    StreamlineAutoencoder& model,                                           // model
    int batchSize                                                           // batch-size
    )
{
    int N = latent.size();
    int begInd = 0;
    int endInd = 0;

    std::vector<at::Tensor> batches;

    while (endInd != N) {
        begInd = endInd;
        endInd = ((begInd + batchSize) <= N) ? (begInd + batchSize) : N;
        int curBatchSize = endInd - begInd;

        // Create blob having only one representation of the cluster (removing the flipped version)
        std::vector<float> blob(curBatchSize * model.latDim);

        for (int i = 0; i < curBatchSize; ++i) {
            std::copy(latent[begInd + i].begin(), latent[begInd + i].begin() + model.latDim, blob.begin() + i * model.latDim);
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
        auto decoded_data = decoded[i].data_ptr<float>();
        std::vector<std::vector<float>> trk(model.inpDim, std::vector<float>(3));
        for (int j = 0; j < model.inpDim; ++j) {
            trk[j][0] = decoded_data[j];
            trk[j][1] = decoded_data[j + model.inpDim];
            trk[j][2] = decoded_data[j + 2 * model.inpDim];
        }
        streamlines[i] = std::move(trk);
    }

    return streamlines;
}
