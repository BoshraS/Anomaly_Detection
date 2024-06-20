#include "cmd/cmd.h"
#include "utils.h"

using namespace NIBR;

// Function to flatten and flip the streamline data (defined below)
std::vector<float> flatten_and_flip_streamlines(const std::vector<std::vector<std::vector<float>>>& streamlines, int idx, int bas);

std::vector<std::vector<float>> encodeStreamlines(                          // output latent representations <number of streamlines x latDim>
    const std::vector<std::vector<std::vector<float>>>& streamlines,        // input tractogram <number of streamlines x (variable) number of points x 3>
    StreamlineAutoencoder& model,                                           // model
    int batchSize                                                           // batch-size
    )
{   

    int N         = streamlines.size();
    int batchCnt = (N < batchSize) ? 1 : (N + batchSize - 1) / batchSize;

    // A streamline is representated in a latDim-dimension latent space
    // We also flip the streamline and save the flipped representation
    // Therefore, for a single streamline we keep 2*latDim float32 values
    std::vector<std::vector<float>> latent(N, std::vector<float>(2*model.latDim));

    // Iterate throught the whole tractogram
    auto run = [&](NIBR::MT::TASK task) -> void {

        int bas = ((int(task.no)+1)*batchSize < N) ? batchSize : (N-int(task.no)*batchSize);
        
        int idx = task.no * batchSize;

        // Flatten and flip the streamlines data into a contiguous block
        std::vector<float> flattened_data = flatten_and_flip_streamlines(streamlines, idx, bas);

        // Create a tensor from the contiguous block of data
        auto input = torch::from_blob(flattened_data.data(), {2 * bas, 3, model.inpDim}, torch::kFloat).to(model.device).clone();

        // Prepare the input as a vector of torch::jit::IValue
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);

        // Encode this batch
        at::Tensor encoded = model.module.get_method("encode")(inputs).toTensor().to(torch::kCPU).contiguous();

        // Copy encoded tensor data to latent vector
        for (int i = 0; i < bas; i++) {
            int idx = i + int(task.no) * batchSize;
            latent[idx].resize(2*model.latDim);
            std::memcpy(latent[idx].data(), encoded[2*i].data_ptr<float>(), 2*model.latDim * sizeof(float));
        }
        
    };
    NIBR::MT::MTRUN(batchCnt, "Encoding streamlines", run);

    return latent;

}

std::vector<float> flatten_and_flip_streamlines(const std::vector<std::vector<std::vector<float>>>& streamlines, int idx, int bas) {

    std::vector<float> flattened(2 * bas * 3 * streamlines[0].size());

    size_t index = 0;

    for (int i = idx; i < idx+bas; i++) {

        // Append streamline 
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < int(streamlines[0].size()); ++k) {
                flattened[index++] = streamlines[i][k][j];
            }
        }

        // Flip and append streamline
        for (int j = 0; j < 3; ++j) {
            for (int k = int(streamlines[0].size()-1); k > -1; --k) {
                flattened[index++] = streamlines[i][k][j];
            }
        }

    }
    return flattened;
}