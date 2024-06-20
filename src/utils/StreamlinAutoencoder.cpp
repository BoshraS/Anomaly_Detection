#include "cmd/cmd.h"
#include "utils.h"

using namespace NIBR;

// Common initialization method
void StreamlineAutoencoder::init(const std::string& _moduleFile, int _inpDim, int _latDim, float _distScaler, bool _useCPU) {

    moduleFile  = _moduleFile;
    inpDim      = _inpDim;
    latDim      = _latDim;
    distScaler  = _distScaler;
    useCPU      = _useCPU;
    ready       = false;

    device      = torch::kCPU;
    if (!useCPU && torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        disp(MSG_DETAIL,"Using CUDA");
    } else {
        disp(MSG_DETAIL,"Using CPU");
    }

    if (moduleFile.empty()) {
        std::filesystem::path curPath = std::filesystem::absolute(__FILE__);
        std::filesystem::path parentPath = curPath.parent_path().parent_path().parent_path();
        std::filesystem::path default_model = parentPath / "models" / "conv_autoencoder_scripted.pt";
        moduleFile  = default_model.string();
        inpDim      = 256;
        latDim      = 64;
        distScaler  = 0.08;
    } else {
        if (getFileExtension(moduleFile) != "pt") {
            disp(MSG_ERROR,"Torch script module file must have extension .pt");
            ready = false;
            return;
        }
    }

    if (!std::filesystem::exists(moduleFile)) {
        disp(MSG_ERROR,"Model not found");
        ready = false;
        return;
    }

    try {
        disp(MSG_DETAIL,"Loading model %s", moduleFile.c_str());
        module = torch::jit::load(moduleFile);
        module.to(device);
        ready = true;
    } catch (const c10::Error& e) {
        disp(MSG_DETAIL,"Error loading the model");
        ready = false;
    }
}


StreamlineAutoencoder::StreamlineAutoencoder(const std::tuple<std::string, int, int>& moduleSpec, bool _useCPU)
    : device(torch::kCPU), useCPU(_useCPU), ready(false) {
    init(std::get<0>(moduleSpec), std::get<1>(moduleSpec), std::get<2>(moduleSpec), 1.0f, _useCPU);
}

StreamlineAutoencoder::StreamlineAutoencoder(const std::tuple<std::string, int, int, float>& moduleSpec, bool _useCPU)
    : device(torch::kCPU), useCPU(_useCPU), ready(false) {
    init(std::get<0>(moduleSpec), std::get<1>(moduleSpec), std::get<2>(moduleSpec), std::get<3>(moduleSpec), _useCPU);
}