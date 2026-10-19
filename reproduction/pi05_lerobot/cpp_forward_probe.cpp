// Fixed-input diagnostic. Reuses pinned official classes without modifying them.
#include "pi05.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <cstdlib>
#include <dlfcn.h>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>

static std::vector<float> reference_rope_frequency;
static std::unordered_map<std::string, std::vector<float>> rope_factor_inputs;

// Optional precision-only interposition for diagnosis. Model graphs and weights
// are unchanged; explicit F32 is a supported GGML matmul precision setting.
extern "C" ggml_tensor *ggml_mul_mat(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b) {
    using Original = ggml_tensor *(*)(ggml_context *, ggml_tensor *, ggml_tensor *);
    static auto original = reinterpret_cast<Original>(dlsym(RTLD_NEXT, "ggml_mul_mat"));
    if (!original) throw std::runtime_error("Cannot resolve original ggml_mul_mat");
    auto result = original(ctx, a, b);
    if (std::getenv("ACTQUANT_PROBE_F32")) ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    return result;
}

// Frequency factors are a public RoPE input. Match the source model's actual
// BF16-rounded nonpersistent buffer instead of silently reconstructing ideal F32.
extern "C" ggml_tensor *ggml_rope_ext(ggml_context *ctx, ggml_tensor *a, ggml_tensor *positions,
        ggml_tensor *factors, int n_dims, int mode, int n_ctx_orig, float freq_base,
        float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    using Original = ggml_tensor *(*)(ggml_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *,
        int, int, int, float, float, float, float, float, float);
    static auto original = reinterpret_cast<Original>(dlsym(RTLD_NEXT, "ggml_rope_ext"));
    if (!original) throw std::runtime_error("Cannot resolve original ggml_rope_ext");
    if (std::getenv("ACTQUANT_PROBE_MATCH_ROPE")) {
        if (factors || n_dims != 256 || mode != 2 || reference_rope_frequency.size()!=128)
            throw std::runtime_error("Unsupported RoPE configuration for frequency match");
        factors = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_dims/2);
        static int serial = 0;
        ggml_set_name(factors, ("probe.rope.factors."+std::to_string(serial++)).c_str());
        ggml_set_input(factors);
        std::vector<float> data(n_dims/2);
        float theta_scale = std::pow(freq_base, -2.0f/n_dims);
        for (int i=0; i<n_dims/2; ++i)
            data[i] = std::pow(theta_scale, static_cast<float>(i)) / reference_rope_frequency[i];
        rope_factor_inputs[factors->name] = std::move(data);
    }
    return original(ctx,a,positions,factors,n_dims,mode,n_ctx_orig,freq_base,freq_scale,
                    ext_factor,attn_factor,beta_fast,beta_slow);
}

extern "C" ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t scheduler, ggml_cgraph *graph) {
    using Original = ggml_status (*)(ggml_backend_sched_t, ggml_cgraph *);
    static auto original = reinterpret_cast<Original>(dlsym(RTLD_NEXT, "ggml_backend_sched_graph_compute"));
    if (!original) throw std::runtime_error("Cannot resolve original graph compute");
    int written = 0;
    for (const auto &item : rope_factor_inputs) {
        if (auto input = ggml_graph_get_tensor(graph,item.first.c_str())) {
            ggml_backend_tensor_set(input,item.second.data(),0,item.second.size()*sizeof(float));
            ++written;
        }
    }
    if (written) fprintf(stderr,"Probe: wrote %d RoPE frequency inputs\n",written);
    return original(scheduler,graph);
}

namespace fs = std::filesystem;

template <typename T> std::vector<T> read(const fs::path &path) {
    std::vector<T> result;
    if (!load_file_to_vector(result, path.string()))
        throw std::runtime_error("Cannot read " + path.string());
    return result;
}
template <typename T> void write(const fs::path &path, const std::vector<T> &data) {
    if constexpr (std::is_floating_point_v<T>)
        for (auto value : data)
            if (!std::isfinite(value)) throw std::runtime_error("Non-finite output: " + path.string());
    if (!save_vector_to_file(data, path.string()))
        throw std::runtime_error("Cannot write " + path.string());
}

int main(int argc, char **argv) {
    try {
        if (argc != 5) throw std::runtime_error("Usage: probe model.gguf tokenizer.model inputs_dir new_output_dir");
        fs::path input(argv[3]), output(argv[4]);
        if (!fs::create_directory(output)) throw std::runtime_error("Output directory exists");
        if (std::getenv("ACTQUANT_PROBE_MATCH_ROPE"))
            reference_rope_frequency = read<float>(input / "rope_inv_freq.f32");
        ContextParams params;
        params.device_name = "CUDA0";
        params.n_threads = 4;
        params.max_nodes = 8192;
        std::vector<float> actual_prefix;
        {
            Pi05VisionEncoder vision(argv[1], argv[1], params);
            for (int i=0; i<2; ++i) {
                auto pixels = read<float>(input / ("image" + std::to_string(i) + ".f32"));
                if (pixels.size() != 3*224*224) throw std::runtime_error("Invalid image input size");
                std::vector<float> embedding;
                if (!vision.encode(pixels, embedding)) throw std::runtime_error("Vision inference failed");
                if (embedding.size() != 256*2048) throw std::runtime_error("Invalid image embedding size");
                write(output / ("image_embedding" + std::to_string(i) + ".f32"), embedding);
                actual_prefix.insert(actual_prefix.end(), embedding.begin(), embedding.end());
            }
        }
        {
            Pi05TextEmbed text;
            if (!text.init(argv[1], argv[2])) throw std::runtime_error("Text embedding initialization failed");
            auto tokens = read<int32_t>(input / "tokens.i32");
            auto embedding = text.get_text_embeddings(tokens);
            write(output / "text_embedding.f32", embedding);
            actual_prefix.insert(actual_prefix.end(), embedding.begin(), embedding.end());
            std::ifstream file(input / "prompt.txt");
            std::string prompt((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            auto encoded = text.tokenize(prompt, true);
            write(output / "raw_prompt_tokens.i32", encoded);
        }
        write(output / "prefix_cpp.f32", actual_prefix);
        {
            Pi05ActionExpertRunner runner(argv[1], params);
            auto prefix = read<float>(input / "prefix_reference.f32");
            auto actions = read<float>(input / "noisy_actions.f32");
            auto timestep = read<float>(input / "timestep_embedding.f32");
            auto moment = read<float>(input / "time.f32");
            if (prefix.size() != actual_prefix.size() || prefix.size()%2048 || actions.size()!=50*32 || timestep.size()!=1024 || moment.size()!=1)
                throw std::runtime_error("Invalid fixed-input tensor dimensions");
            std::vector<float> computed_time;
            runner.fill_sinusoidal_embedding(moment[0], 1024, computed_time);
            write(output / "timestep_embedding_cpp.f32", computed_time);
            runner.set_prefix_length(prefix.size()/2048);
            std::vector<float> velocity;
            if (!runner.run(prefix, actions, timestep, velocity)) throw std::runtime_error("Fixed-prefix joint forward failed");
            write(output / "velocity_reference_prefix.f32", velocity);
            if (!runner.run(actual_prefix, actions, timestep, velocity)) throw std::runtime_error("C++-prefix joint forward failed");
            write(output / "velocity_cpp_prefix.f32", velocity);
            if (!runner.compute_kv_cache(actual_prefix)) throw std::runtime_error("Prefix cache failed");
            if (!runner.run_with_cache(actions, timestep, velocity)) throw std::runtime_error("Cached suffix forward failed");
            write(output / "velocity_cached_cpp_prefix.f32", velocity);
            if (const char *steps_string = std::getenv("ACTQUANT_PROBE_ODE_STEPS")) {
                int steps = std::stoi(steps_string);
                if (steps < 1 || steps > 20) throw std::runtime_error("ODE steps must be in [1,20]");
                auto sampled = read<float>(input / "noise.f32");
                if (sampled.size()!=50*32) throw std::runtime_error("Invalid fixed initial noise");
                std::vector<float> trajectory;
                float dt = -1.0f/steps;
                for (int step=0; step<steps; ++step) {
                    // Same sinusoidal formula as the reference: double arithmetic
                    // then cast to float. Both baseline and quantized use this path.
                    double t = 1.0 - static_cast<double>(step)/steps;
                    std::vector<float> embedding(1024);
                    for (int i=0; i<512; ++i) {
                        double period = 0.004*std::pow(1000.0, static_cast<double>(i)/511);
                        double angle = t/period*2.0*std::acos(-1.0);
                        embedding[i] = static_cast<float>(std::sin(angle));
                        embedding[512+i] = static_cast<float>(std::cos(angle));
                    }
                    if (!runner.run_with_cache(sampled, embedding, velocity)) throw std::runtime_error("ODE suffix forward failed");
                    for (size_t i=0; i<sampled.size(); ++i) sampled[i] += dt*velocity[i];
                    trajectory.insert(trajectory.end(),sampled.begin(),sampled.end());
                }
                write(output / "ode_trajectory.f32", trajectory);
                write(output / "sampled_actions.f32", sampled);
            }
        }
        std::cout << "Fixed-input C++ probe completed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Probe error: " << error.what() << '\n';
        return 1;
    }
}
