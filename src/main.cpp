#include "utils/Logger.h"
#include "utils/Config.h"
#include "utils/Timer.h"
#include "utils/CSVWriter.h"
#include "tests/TestSuite.h"

#include "core/RandomInitializer.h"
#include "data/LabelEncoder.h"
#include "data/ImageFolderDatasetLoader.h"
#include "layers/Dense.h"
#include "layers/ReLU.h"
#include "layers/SoftmaxCrossEntropy.h"
#include "training/OptimizerSGD.h"
#include "training/Metrics.h"

#include <filesystem>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;
using namespace flowercnn;

struct ToySample {
    std::vector<double> x;
    int y;
};

static std::vector<ToySample> buildToyFlowerLikeDataset() {
    return {
        {{0.10, 0.20, 0.15}, 0}, {{0.12, 0.18, 0.20}, 0}, {{0.08, 0.25, 0.16}, 0}, {{0.11, 0.22, 0.19}, 0},
        {{0.85, 0.20, 0.10}, 1}, {{0.82, 0.18, 0.14}, 1}, {{0.88, 0.24, 0.12}, 1}, {{0.80, 0.22, 0.18}, 1},
        {{0.20, 0.80, 0.15}, 2}, {{0.18, 0.82, 0.20}, 2}, {{0.22, 0.78, 0.14}, 2}, {{0.24, 0.84, 0.18}, 2},
        {{0.15, 0.25, 0.90}, 3}, {{0.18, 0.22, 0.86}, 3}, {{0.12, 0.28, 0.88}, 3}, {{0.20, 0.26, 0.92}, 3},
        {{0.70, 0.70, 0.25}, 4}, {{0.74, 0.68, 0.20}, 4}, {{0.68, 0.75, 0.22}, 4}, {{0.72, 0.72, 0.28}, 4}
    };
}

static void saveDatasetSummary(const ImageDataset& dataset, const LabelEncoder& encoder) {
    fs::create_directories("results");
    std::ofstream file("results/dataset_resumen.txt");
    if (!file.is_open()) return;

    file << "Resumen de carga del dataset real\n";
    file << "=================================\n";
    file << "Total de imagenes cargadas: " << dataset.size() << "\n\n";

    auto counts = dataset.countByClass();
    for (const auto& cls : encoder.classes()) {
        file << cls << ": " << counts[cls] << "\n";
    }

    if (!dataset.empty()) {
        const auto& sample = dataset.samples().front();
        file << "\nPrimera imagen cargada:\n";
        file << "Archivo: " << sample.filePath << "\n";
        file << "Clase: " << sample.className << "\n";
        file << "Etiqueta: " << sample.label << "\n";
        file << "Tensor: " << sample.image.shape() << "\n";
        file << "Min: " << sample.image.minValue() << "\n";
        file << "Max: " << sample.image.maxValue() << "\n";
        file << "Media: " << sample.image.meanValue() << "\n";
    }
}

static void runMiniNeuralNetworkDemo(int epochs, double learningRate, unsigned int seed) {
    Logger::title("Mini red base: Dense -> ReLU -> Dense -> Softmax");
    Logger::info("Esta prueba valida las piezas base antes de implementar Conv2D y MaxPool.");

    fs::create_directories("results");
    CSVWriter csv("results/prueba_bloque1.csv");
    csv.writeRow({"epoch", "loss", "accuracy"});

    RandomInitializer::setSeed(seed);

    Dense dense1(3, 12);
    ReLU relu1;
    Dense dense2(12, 5);
    SoftmaxCrossEntropy criterion;
    OptimizerSGD optimizer(learningRate);

    auto dataset = buildToyFlowerLikeDataset();
    Timer timer;
    timer.start();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double totalLoss = 0.0;
        std::vector<int> predictions;
        std::vector<int> labels;

        for (const auto& sample : dataset) {
            auto z1 = dense1.forward(sample.x);
            auto a1 = relu1.forward(z1);
            auto z2 = dense2.forward(a1);
            double loss = criterion.forward(z2, sample.y);
            auto gradZ2 = criterion.backward();
            auto gradA1 = dense2.backward(gradZ2);
            auto gradZ1 = relu1.backward(gradA1);
            dense1.backward(gradZ1);

            optimizer.step(dense2);
            optimizer.step(dense1);

            totalLoss += loss;
            predictions.push_back(Metrics::argmax(criterion.probabilities()));
            labels.push_back(sample.y);
        }

        double avgLoss = totalLoss / static_cast<double>(dataset.size());
        double acc = Metrics::accuracy(predictions, labels);

        csv.writeRow({std::to_string(epoch), std::to_string(avgLoss), std::to_string(acc)});

        int printEvery = std::max(1, epochs / 10);
        if (epoch == 1 || epoch == epochs || epoch % printEvery == 0) {
            std::ostringstream oss;
            oss << "Epoca " << std::setw(3) << epoch
                << " | loss=" << std::fixed << std::setprecision(6) << avgLoss
                << " | accuracy=" << std::fixed << std::setprecision(2) << (acc * 100.0) << "%";
            Logger::info(oss.str());
        }
    }

    double ms = timer.elapsedMilliseconds();
    std::ofstream summary("results/resumen_bloque1.txt");
    if (summary.is_open()) {
        summary << "Bloque 1 ejecutado correctamente\n";
        summary << "================================\n";
        summary << "Prueba: mini red Dense -> ReLU -> Dense -> Softmax\n";
        summary << "Epocas: " << epochs << "\n";
        summary << "Learning rate: " << learningRate << "\n";
        summary << "Tiempo total ms: " << ms << "\n";
        summary << "Archivo de entrenamiento: results/prueba_bloque1.csv\n";
        summary << "\nPendiente para siguientes bloques:\n";
        summary << "- Conv2D forward/backward\n";
        summary << "- MaxPool2D forward/backward\n";
        summary << "- Flatten\n";
        summary << "- CNN completa\n";
        summary << "- OpenMP\n";
    }

    std::ostringstream oss;
    oss << "Mini red finalizada en " << std::fixed << std::setprecision(2) << ms << " ms.";
    Logger::ok(oss.str());
}

int main() {
    Logger::title("FlowerCNN - Bloque 1");

    Config config;
    if (config.load("config/config.txt")) {
        Logger::ok("Configuracion cargada desde config/config.txt");
    } else {
        Logger::warn("No se pudo leer config/config.txt. Se usaran valores por defecto.");
    }

    int imageWidth = config.getInt("image.width", 64);
    int imageHeight = config.getInt("image.height", 64);
    int maxImagesPerClass = config.getInt("max.images.per.class", 3);
    int epochs = config.getInt("epochs", 80);
    double learningRate = config.getDouble("learning.rate", 0.05);
    unsigned int seed = static_cast<unsigned int>(config.getInt("seed", 42));
    std::string datasetPath = config.getString("dataset.path", "dataset/flowers");

    if (!TestSuite::runAll()) {
        Logger::warn("Se recomienda revisar las pruebas antes de continuar.");
    }

    Logger::title("Carga del dataset real de flores");
    LabelEncoder encoder;
    ImageFolderDatasetLoader loader(
        static_cast<std::size_t>(imageHeight),
        static_cast<std::size_t>(imageWidth),
        maxImagesPerClass
    );

    ImageDataset imageDataset = loader.load(datasetPath, encoder);
    saveDatasetSummary(imageDataset, encoder);

    if (imageDataset.empty()) {
        Logger::warn("No se cargaron imagenes. Copia los JPG dentro de dataset/flowers/<clase>/.");
        Logger::warn("El Bloque 1 continuara con la prueba interna de red neuronal.");
    } else {
        std::ostringstream oss;
        oss << "Dataset cargado: " << imageDataset.size() << " imagen(es) procesada(s) a "
            << imageHeight << "x" << imageWidth << "x3.";
        Logger::ok(oss.str());
    }

    runMiniNeuralNetworkDemo(epochs, learningRate, seed);

    Logger::title("Fin del Bloque 1");
    Logger::ok("Base lista para implementar Conv2D, MaxPool, Flatten y OpenMP en los siguientes bloques.");

    return 0;
}
