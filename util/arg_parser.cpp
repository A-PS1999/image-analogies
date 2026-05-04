#include "arg_parser.h"
#include <iostream>

namespace Util
{
    namespace ArgParser
    {
        const float DEFAULT_COHERENCE_WEIGHT = 5.0f;
        
        void ArgumentParser::addArgument(const std::string &name, const std::string &description, bool required, std::string defaultValue)
        {
            auto arg = std::make_unique<Argument>();
            arg->name = name;
            arg->description = description;
            arg->required = required;
            if (!defaultValue.empty())
            {
                arg->value = defaultValue;
            }

            if (required)
            {
                requiredArgsCount++;
            }
            arguments.push_back(std::move(arg));
            argMap.insert({name, arguments.back().get()});
        }

        bool ArgumentParser::parseInput(int argc, char **argv)
        {
            bool shouldBeValue = false;
            std::map<std::string, Argument *>::const_iterator lastArg = argMap.end();

            programName = std::string(argv[0]);

            for (int i = 1; i < argc; ++i)
            {
                std::string input{argv[i]};

                if (shouldBeValue)
                {
                    lastArg->second->value = input;
                    shouldBeValue = false;
                }
                else if (input.compare(0, 2, "--") == 0)
                {
                    input.erase(0, 2);
                    lastArg = argMap.find(input);
                    if (lastArg == argMap.end())
                    {
                        std::cerr << "Unknown argument: --" << input << std::endl;
                        return false;
                    }
                    shouldBeValue = true;
                    parsedArgsCount++;
                }
                else
                {
                    std::cerr << "Unexpected value: " << input << std::endl;
                    return false;
                }
            }

            return (parsedArgsCount >= requiredArgsCount) && !shouldBeValue;
        }

        const Argument *ArgumentParser::getArgument(const std::string &name) const
        {
            auto argObj = argMap.find(name);
            if (argObj != argMap.end())
            {
                return argObj->second;
            }
            return nullptr;
        }

        void ArgumentParser::printUsage() const
        {
            std::cout << "Usage: " << programName << " [--<argument> <value>]" << std::endl;
            for (const auto &arg : arguments)
            {
                std::cout << " --" << arg->name << ": " << arg->description << std::endl;
            }
        }

        void addBasicArguments(ArgumentParser &parser)
        {
            parser.addArgument("imageA", "Path to the unfiltered source image A", true);
            parser.addArgument("imageAPrime", "Path to the filtered source image A'", true);
            parser.addArgument("imageB", "Path to the unfiltered target image B", true);
            parser.addArgument("coherence", "The coherence weight K", false, "5.0");
        }

        float parseCoherenceWeight(const ArgumentParser &parser)
        {
            auto *coherenceArg = parser.getArgument("coherence");
            if (coherenceArg && !coherenceArg->value.empty())
            {
                try
                {
                    return std::stof(coherenceArg->value);
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Invalid coherence weight value: " << coherenceArg->value << ". Using default value of 5.0." << std::endl;
                    return DEFAULT_COHERENCE_WEIGHT;
                }
            }
            return DEFAULT_COHERENCE_WEIGHT; // Default value
        }
    } // namespace ArgParser
} // namespace Util