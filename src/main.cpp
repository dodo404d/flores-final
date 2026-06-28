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
#include "layers/Conv2D.h"
#include "layers/MaxPool2D.h"
#include "layers/Flatten.h"

#include <filesystem>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <vector>
#include <fstream>
#include <string>

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

static void saveSequentialCNNModel(
    const std::string& path,
    const Conv2D& conv1,
    const Dense& dense1,
    const Dense& dense2,
    int epochs,
    double learningRate,
    std::size_t datasetSize
) {
    fs::create_directories("models");

    std::ofstream out(path);

    if (!out.is_open()) {
        Logger::warn("No se pudo guardar el modelo en " + path);
        return;
    }

    out << "FlowerCNNSequentialModelV1\n";
    out << "image_height 64\n";
    out << "image_width 64\n";
    out << "image_channels 3\n";
    out << "classes daisy dandelion rose sunflower tulip\n";
    out << "epochs " << epochs << "\n";
    out << "learning_rate " << learningRate << "\n";
    out << "dataset_size " << datasetSize << "\n";

    conv1.save(out);
    dense1.save(out);
    dense2.save(out);

    Logger::ok("Modelo guardado en " + path);
}


static void runSequentialCNNTraining(
    const ImageDataset& imageDataset,
    int epochs,
    double learningRate,
    unsigned int seed
) {
    Logger::title("CNN secuencial - Entrenamiento completo");

    if (imageDataset.empty()) {
        Logger::warn("No hay imagenes cargadas para entrenar la CNN secuencial.");
        return;
    }

    fs::create_directories("results");

    CSVWriter csv("results/cnn_secuencial.csv");
    csv.writeRow({"epoch", "loss", "accuracy", "time_ms"});

    RandomInitializer::setSeed(seed);

    Conv2D conv1(3, 8, 3, 1, 0);
    ReLU reluConv;
    MaxPool2D pool1(2, 2);
    Flatten flatten;

    Dense dense1(31 * 31 * 8, 64);
    ReLU reluDense;
    Dense dense2(64, 5);

    SoftmaxCrossEntropy criterion;

    saveSequentialCNNModel(
        "models/cnn_secuencial_inicial.txt",
        conv1,
        dense1,
        dense2,
        epochs,
        learningRate,
        imageDataset.size()
    );

    Timer totalTimer;
    totalTimer.start();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        Timer epochTimer;
        epochTimer.start();

        double totalLoss = 0.0;
        std::vector<int> predictions;
        std::vector<int> labels;

        for (const auto& sample : imageDataset.samples()) {
            /*
                Forward
            */
            Tensor zConv = conv1.forward(sample.image);
            Tensor aConv = reluConv.forward(zConv);
            Tensor pooled = pool1.forward(aConv);
            std::vector<double> flat = flatten.forward(pooled);

            std::vector<double> zDense1 = dense1.forward(flat);
            std::vector<double> aDense1 = reluDense.forward(zDense1);
            std::vector<double> logits = dense2.forward(aDense1);

            double loss = criterion.forward(logits, sample.label);

            predictions.push_back(Metrics::argmax(criterion.probabilities()));
            labels.push_back(sample.label);

            /*
                Backward
            */
            std::vector<double> gradLogits = criterion.backward();

            std::vector<double> gradA1 = dense2.backward(gradLogits);
            std::vector<double> gradZ1 = reluDense.backward(gradA1);

            std::vector<double> gradFlat = dense1.backward(gradZ1);
            Tensor gradPooled = flatten.backward(gradFlat);

            Tensor gradAConv = pool1.backward(gradPooled);
            Tensor gradZConv = reluConv.backward(gradAConv);

            conv1.backward(gradZConv);

            /*
                Actualizacion de pesos
            */
            dense2.applyGradients(learningRate);
            dense1.applyGradients(learningRate);
            conv1.applyGradients(learningRate);

            totalLoss += loss;
        }

        double avgLoss = totalLoss / static_cast<double>(imageDataset.size());
        double acc = Metrics::accuracy(predictions, labels);
        double epochMs = epochTimer.elapsedMilliseconds();

        csv.writeRow({
            std::to_string(epoch),
            std::to_string(avgLoss),
            std::to_string(acc),
            std::to_string(epochMs)
        });

        std::ostringstream oss;
        oss << "Epoca " << std::setw(3) << epoch
            << " | loss=" << std::fixed << std::setprecision(6) << avgLoss
            << " | accuracy=" << std::fixed << std::setprecision(2) << (acc * 100.0) << "%"
            << " | tiempo=" << std::fixed << std::setprecision(2) << epochMs << " ms";

        Logger::info(oss.str());
    }

    double totalMs = totalTimer.elapsedMilliseconds();

    std::ofstream summary("results/resumen_cnn_secuencial.txt");
    if (summary.is_open()) {
        summary << "CNN secuencial ejecutada correctamente\n";
        summary << "=====================================\n";
        summary << "Arquitectura:\n";
        summary << "Input 64x64x3\n";
        summary << "Conv2D 8 filtros 3x3\n";
        summary << "ReLU\n";
        summary << "MaxPool2D 2x2\n";
        summary << "Flatten 7688\n";
        summary << "Dense 64\n";
        summary << "ReLU\n";
        summary << "Dense 5\n";
        summary << "SoftmaxCrossEntropy\n\n";
        summary << "Epocas: " << epochs << "\n";
        summary << "Learning rate: " << learningRate << "\n";
        summary << "Imagenes usadas: " << imageDataset.size() << "\n";
        summary << "Tiempo total ms: " << totalMs << "\n";
        summary << "Metricas CSV: results/cnn_secuencial.csv\n";
    }


    saveSequentialCNNModel(
        "models/cnn_secuencial_final.txt",
        conv1,
        dense1,
        dense2,
        epochs,
        learningRate,
        imageDataset.size()
    );

    std::ostringstream done;
    done << "Entrenamiento CNN secuencial finalizado en "
         << std::fixed << std::setprecision(2) << totalMs << " ms.";

    Logger::ok(done.str());
    Logger::ok("Metricas guardadas en results/cnn_secuencial.csv");
}

int main() {
    Logger::title("FlowerCNN - CNN secuencial");

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

    /*
        Para pruebas pequeñas usamos pocas epocas.
        Luego, para resultados finales, pueden aumentar este valor
        y compilar con -O2.
    */
    int cnnEpochs = 20;  //NÚMERO DE EPOCAS
    double cnnLearningRate = 0.001;

    runSequentialCNNTraining(imageDataset, cnnEpochs, cnnLearningRate, seed);

    Logger::title("Fin del Bloque 2 - CNN secuencial");
    Logger::ok("CNN secuencial lista como linea base para la version paralela con OpenMP.");

    return 0;
}
