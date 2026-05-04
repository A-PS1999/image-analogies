#include "opencv2/core.hpp"
#include "../util/arg_parser.h"

int main(int argc, char** argv) {    
    Util::ArgParser::ArgumentParser parser;
    Util::ArgParser::addBasicArguments(parser);

    if (!parser.parseInput(argc, argv))
    {
        parser.printUsage();
        return 1;
    }
    
    return 0;
}