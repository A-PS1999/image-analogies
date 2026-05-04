#include "../util/arg_parser.h"
#include "../algo/image_analogy.h"

int main(int argc, char **argv)
{
    Util::ArgParser::ArgumentParser parser;
    Util::ArgParser::addBasicArguments(parser);

    if (!parser.parseInput(argc, argv))
    {
        parser.printUsage();
        return 1;
    }

    const std::string imageAPath = parser.getArgument("imageA")->value;
    const std::string imageAPrimePath = parser.getArgument("imageAPrime")->value;
    const std::string imageBPath = parser.getArgument("imageB")->value;
    float coherenceWeight = Util::ArgParser::parseCoherenceWeight(parser);

    ImageAnalogy::ImageAnalogyMaker analogyMaker(
        imageAPath,
        imageAPrimePath,
        imageBPath,
        coherenceWeight
    );

    return 0;
}