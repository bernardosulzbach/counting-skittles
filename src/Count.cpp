#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using U8 = uint8_t;

using Color = std::array<float, 3>;

float addToAverage(float average, float value, float samplesSoFar) {
  return average * (samplesSoFar / (samplesSoFar + 1.0f)) + value / (samplesSoFar + 1.0f);
}

std::string colorToString(const Color color) {
  std::stringstream stream;
  stream << std::fixed << std::setprecision(3);
  stream << "(";
  stream << color[0] << ", ";
  stream << color[1] << ", ";
  stream << color[2];
  stream << ")";
  return stream.str();
}

struct Position {
  int i = 0;
  int j = 0;

  [[nodiscard]] std::string toString() const {
    return "(" + std::to_string(i) + ", " + std::to_string(j) + ")";
  }
};

constexpr int MedianFilterKernelSide = 21;

constexpr float CannyThreshold = 50.0f;

constexpr int DilationSize = 3;
constexpr int DilationKernelSize = 2 * DilationSize + 1;

constexpr int EdgeThreshold = 100;

static const char *const Extension = ".jpg";

constexpr std::array<Color, 5> ReferenceColors = {{{0.094f, 0.107f, 0.299f},
                                                   {0.076f, 0.226f, 0.576f},
                                                   {0.048f, 0.419f, 0.579f},
                                                   {0.082f, 0.308f, 0.152f},
                                                   {0.091f, 0.088f, 0.113f}}};

constexpr std::array<std::string_view, 5> ReferenceColorNames = {{"Red", "Orange", "Yellow", "Green", "Purple"}};

int getReferenceColorIndex(const Color color) {
  int bestMatch = -1;
  auto bestMatchDistance = std::numeric_limits<float>::infinity();
  const auto sourceColor = cv::Vec3f(color[0], color[1], color[2]);
  for (int i = 0; i < static_cast<int>(ReferenceColors.size()); i++) {
    const auto referenceColor = cv::Vec3f(ReferenceColors[i][0], ReferenceColors[i][1], ReferenceColors[i][2]);
    const auto thisDistance = cv::norm(referenceColor - sourceColor);
    if (thisDistance < bestMatchDistance) {
      bestMatch = i;
      bestMatchDistance = thisDistance;
    }
  }
  return bestMatch;
}

struct Object {
  Position center;
  Color averageColor;
  int pixels = 0;

  [[nodiscard]] std::string toString() const {
    auto string = "Object of size " + std::to_string(pixels);
    string += " centered at " + center.toString();
    string += " with average color " + colorToString(averageColor);
    string += " (";
    string += ReferenceColorNames[getReferenceColorIndex(averageColor)];
    string += ")";
    return string;
  }
};

cv::Mat dilate(const cv::Mat &image) {
  cv::Mat dilatedImage = image;
  const auto kernelSize = cv::Size(DilationKernelSize, DilationKernelSize);
  const auto structure = cv::getStructuringElement(cv::MORPH_RECT, kernelSize);
  cv::dilate(image, dilatedImage, structure);
  return dilatedImage;
}

std::vector<std::vector<bool>> makeBackgroundMask(const cv::Mat &image, const cv::Mat &edges) {
  const auto rows = image.size[0];
  const auto columns = image.size[1];
  std::vector<std::vector<bool>> background(rows, std::vector<bool>(columns));
  std::vector<std::vector<bool>> enqueued(rows, std::vector<bool>(columns));
  std::queue<Position> queue;
  queue.push({0, 0});
  enqueued[0][0] = true;
  while (!queue.empty()) {
    const auto head = queue.front();
    queue.pop();
    const auto i = head.i;
    const auto j = head.j;
    if (edges.at<U8>(i, j) > EdgeThreshold) {
      continue;
    }
    background[i][j] = true;
    const auto enqueue = [&enqueued, &queue, rows, columns](int cI, int cJ) {
      if (cI < 0 || cI >= rows || cJ < 0 || cJ >= columns || enqueued[cI][cJ]) {
        return;
      }
      queue.push({cI, cJ});
      enqueued[cI][cJ] = true;
    };
    enqueue(i - 1, j);
    enqueue(i, j - 1);
    enqueue(i, j + 1);
    enqueue(i + 1, j);
  }
  return background;
}

void selectForeground(cv::Mat &image, const cv::Mat &edges) {
  const auto rows = image.size[0];
  const auto columns = image.size[1];
  const auto background = makeBackgroundMask(image, edges);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      if (background[i][j]) {
        for (int c = 0; c < 3; c++) {
          image.at<cv::Vec3b>(i, j)[c] = 0;
        }
      }
    }
  }
}

std::vector<Object> findObjects(const cv::Mat &selection) {
  const auto rows = selection.size[0];
  const auto columns = selection.size[1];
  std::vector<Object> objects;
  std::vector<std::vector<bool>> used(rows, std::vector<bool>(columns));
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      if (used[i][j]) {
        continue;
      }
      if (selection.at<cv::Vec3b>(i, j) == cv::Vec3b(0, 0, 0)) {
        continue;
      }
      float centerI = 0.0;
      float centerJ = 0.0;
      Color averageColor = {};
      float pixels = 0;
      std::queue<Position> queue;
      queue.push({i, j});
      used[i][j] = true;
      while (!queue.empty()) {
        const auto head = queue.front();
        queue.pop();
        const auto pI = head.i;
        const auto pJ = head.j;
        if (selection.at<cv::Vec3b>(pI, pJ) == cv::Vec3b(0, 0, 0)) {
          continue;
        }
        centerI = addToAverage(centerI, pI, pixels);
        centerJ = addToAverage(centerJ, pJ, pixels);
        for (int c = 0; c < 3; c++) {
          const auto colorInC = static_cast<float>(selection.at<cv::Vec3b>(pI, pJ)[c]);
          averageColor[c] = addToAverage(averageColor[c], colorInC / 255.0f, pixels);
        }
        pixels += 1.0f;
        const auto enqueue = [&used, &queue, rows, columns](int eI, int eJ) {
          if (eI < 0 || eI >= rows || eJ < 0 || eJ >= columns || used[eI][eJ]) {
            return;
          }
          queue.push({eI, eJ});
          used[eI][eJ] = true;
        };
        enqueue(pI - 1, pJ);
        enqueue(pI, pJ - 1);
        enqueue(pI, pJ + 1);
        enqueue(pI + 1, pJ);
      }
      const auto roundedCenterI = static_cast<int>(std::round(centerI));
      const auto roundedCenterJ = static_cast<int>(std::round(centerJ));
      const auto center = Position{roundedCenterI, roundedCenterJ};
      objects.push_back(Object{center, averageColor, static_cast<int>(pixels)});
    }
  }
  return objects;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Should pass the number of the input image." << '\n';
    return 0;
  }
  const std::string inputIdentifier = std::string(argv[1]);
  if (inputIdentifier.size() != 2) {
    std::cout << "Input identifiers should have length 2." << '\n';
    return 0;
  }
  const auto inputPath = std::filesystem::path("..") / "input";
  const auto inputImagePath = (inputPath / "images" / inputIdentifier).concat(Extension);
  const auto input = cv::imread(inputImagePath.native());
  cv::Mat smoothed;
  cv::medianBlur(input, smoothed, MedianFilterKernelSide);
  const auto outputPath = std::filesystem::path("..") / "output";
  const auto smoothedOutputPath = (outputPath / "smoothed" / inputIdentifier).concat(Extension);
  cv::imwrite(smoothedOutputPath.native(), smoothed);
  cv::Mat edges;
  cv::Canny(smoothed, edges, CannyThreshold, CannyThreshold / 2.0f);
  const auto edgesOutputPath = (outputPath / "edges" / inputIdentifier).concat(Extension);
  cv::imwrite(edgesOutputPath.native(), edges);
  const auto dilatedEdges = dilate(edges);
  const auto dilatedEdgesOutputPath = (outputPath / "dilated-edges" / inputIdentifier).concat(Extension);
  cv::imwrite(dilatedEdgesOutputPath.native(), dilatedEdges);
  cv::Mat selection = input;
  selectForeground(selection, edges);
  const auto selectionOutputPath = (outputPath / "selection" / inputIdentifier).concat(Extension);
  cv::imwrite(selectionOutputPath.native(), selection);
  for (const auto object : findObjects(selection)) {
    std::cout << object.toString() << "." << '\n';
  }
  return 0;
}
