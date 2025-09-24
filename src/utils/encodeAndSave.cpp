#include "cmd/cmd.h"
#include "dMRI/tractography/algorithms/ptt/algorithm_ptt.h"
#include "dMRI/tractography/tractogram.h"
#include "utils.h"

using namespace NIBR;

bool encodeAndSave(std::string inp, std::string out, bool force, StreamlineAutoencoder& model, int batchSize, bool skipResample)
{
    if (existsFile(out) && !force) return true;

    // Create tractogram reader and preload it
    NIBR::TractogramReader _tractogram(inp, true);

    NIBR::MT::SETMAXNUMBEROFTHREADS(1);

    // Template that deduces the type T (float, double, at::Half) from its argument.
    auto process_with_type = [&](auto type_placeholder) -> bool {
        
        using T = decltype(type_placeholder);

        // Write to a temporary file to prevent corruption on failure
        std::string tmp_out = out + ".tmp";
        std::ofstream ofs(tmp_out, std::ios::binary);
        if (!ofs) {
            disp(MSG_ERROR, "Failed to open temporary output file for writing.");
            return false;
        }

        int N = _tractogram.numberOfStreamlines;
        int batchCnt = (N < batchSize) ? 1 : (N + batchSize - 1) / batchSize;




        auto run = [&](NIBR::MT::TASK task) -> void {
            std::string tmpBatchFile = out + ".batch" + std::to_string(task.no) + ".tmp";

            {
                std::ifstream ifs(tmpBatchFile, std::ios::binary);
                if (ifs) {
                    ifs.seekg(-3, std::ios::end);
                    char readCheckBytes[3] = {};
                    ifs.read(readCheckBytes, 3);
                    if (std::string(readCheckBytes) == "OK") {
                        disp(MSG_DETAIL, "Skipping completed batch file: %s", tmpBatchFile.c_str());
                        return;
                    }
                }
            }

            int bas = std::min(batchSize, N - (int)task.no * batchSize);
            NIBR::StreamlineBatch streamlines(bas);
            for (int i = 0; i < bas; i++) {
                int idx = i + (int)task.no * batchSize;
                auto tmp = _tractogram.getStreamline(idx);
                if (!skipResample)
                    tmp = resampleStreamline_withStepCount(tmp, model.inpDim);

                streamlines[i] = std::move(tmp);
            }

            auto encoded = encode_batch<T>(streamlines, model);
            std::ofstream ofs(tmpBatchFile, std::ios::binary);
            if (!ofs) {
                disp(MSG_ERROR, "Failed to open batch file for writing: %s", tmpBatchFile.c_str());
                return;
            }

            for (const auto& encoded_streamline : encoded) {
                ofs.write(reinterpret_cast<const char*>(encoded_streamline.data()), 2 * model.latDim * sizeof(T));
            }

            const char endCheckBytes[] = "OK";
            ofs.write(endCheckBytes, sizeof(endCheckBytes));
            ofs.close();

        };

        NIBR::MT::MTRUN(batchCnt, "Encoding streamlines", run);

        std::cout << "encoding finished "<< std::endl;



        std::ofstream finalOfs(out, std::ios::binary);
        for (int i = 0; i < batchCnt; ++i) {
            std::ifstream ifs(out + ".batch" + std::to_string(i) + ".tmp", std::ios::binary);
            finalOfs << ifs.rdbuf();
            ifs.close();
        }
        finalOfs.close();


        return true;
    };

    // Dispatch to the generic lambda with the correct type
    std::cout << "dtype: " << model.dtype << std::endl;
    if (model.dtype == torch::kFloat) {
        return process_with_type(float{});
    } else if (model.dtype == torch::kDouble) {
        return process_with_type(double{});
    } else if (model.dtype == torch::kHalf) {
        return process_with_type(at::Half{});
    } else {
        disp(MSG_ERROR, "Unsupported data type for encoding: %s", c10::toString(model.dtype));
        return false;
    }
}

